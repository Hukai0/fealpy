from fealpy.backend import backend_manager as bm
from fealpy.sparse import csr_matrix,coo_matrix,CSRTensor
from fealpy.solver import spsolve,GAMGSolver
from scipy import sparse

class AMSSolver:
    
    def __init__(self):
        """AMS预条件器求解类初始化。"""
        # 初始化内部数据结构
        self.A = None               # 边空间刚度矩阵 (稀疏矩阵)
        self.G = None               # 离散梯度矩阵 G
        # 插值矩阵按方向分解
        self.Pi_x = None
        self.Pi_y = None
        self.Pi_z = None
        self.dim = None             # 空间维度 (2 或 3)
        self.n_nodes = 0
        self.n_edges = 0
        self.vertices = None        # 顶点坐标数组
        self.edges = None           # 每条边的顶点索引对 (取向已定)
        # 平滑器与循环参数
        self.smoother = 'GaussSeidel'
        self.smooth_iters = 1
        self.jacobi_weight = 1.0    # Jacobi 平滑的松弛因子
        self.cycle_type = 'V'       # 多重网格循环类型 ('V' 或 'W')
        # 子空间矩阵
        self.A_G = None             # 梯度子空间矩阵 A_G = G^T * A * G
        self.A_Pi = None            # 插值子空间矩阵 A_Pi = Pi^T * A * Pi

    def AMSFEISetup(self, A, vertices, edges, dim=None):
        """
        初始化有限元结构，包括网格拓扑和自由度编号。
        参数:
        - A: H(curl)空间的刚度矩阵 (scipy.sparse 矩阵).
        - vertices: 顶点坐标数组，形状 (num_vertices, dim).
        - edges: 边的顶点索引对列表 (长度 num_edges，每个元素为 (v1, v2)).
        - dim: 空间维度 (2 或 3)，可选参数，不提供则根据 vertices 列数推断。
        本函数存储网格和矩阵信息，并确定每条边的有向表示（统一以较小顶点索引指向较大顶点索引）。
        """
        # 设置维度
        if dim is None:
            dim = vertices.shape[1]
        self.dim = dim
        # 存储矩阵和几何信息
        self.A = A
        self.vertices = vertices
        self.n_nodes = self.vertices.shape[0]
        self.edges = bm.array(edges, dtype=int)
        self.n_edges = self.edges.shape[0]


    def AMSComputeGPi(self):
        """
        构造离散梯度矩阵 G
        """
        num_edges = self.n_edges
        num_nodes = self.n_nodes

        edges = self.edges
        ei = edges[:, 0].astype(int)            # 所有边的起点索引
        ej = edges[:, 1].astype(int)            # 所有边的终点索引
        row = bm.repeat(bm.arange(num_edges, dtype=int), 2)  
        col = bm.empty(2 * num_edges, dtype=int)
        col[0::2] = ei
        col[1::2] = ej
        data = bm.empty(2 * num_edges, dtype=float)
        data[0::2] = -1.0
        data[1::2] =  1.0
        G = csr_matrix((data, (row, col)), shape=(num_edges, num_nodes))
        self.G = G
        self.A_G = (self.G.T)@(self.A@(self.G))
            
    def AMSSetAGradient(self, A_grad):
        """
        直接设置梯度子空间矩阵 A_G。
        """
        self.A_G = A_grad

    def AMSComputePi(self):
        
        num_edges = self.n_edges
        num_nodes = self.n_nodes
        dim = self.dim
        coords = self.vertices

        data = []
        rows = []
        cols = []

        for e, (i, j) in enumerate(self.edges):
            dx = coords[j,0] - coords[i,0]
            dy = coords[j,1] - coords[i,1]
            dz = (coords[j,2] - coords[i,2]) if dim == 3 else 0.0

            half = [0.5*dx, 0.5*dy] + ([0.5*dz] if dim == 3 else [])

            for v in (i, j):
                for d in range(dim):
                    rows.append(e)
                    cols.append(dim*v + d)     
                    data.append(half[d])

        self.Pi = csr_matrix((bm.array(data), (bm.array(rows), bm.array(cols))), shape=(num_edges, dim*num_nodes))
        self.A_Pi = self.Pi.T@(self.A@self.Pi)


            
        
    def AMSSetAPi(self, A_pi):
        """
        直接设置插值子空间矩阵 A_Pi。
        """
        self.A_Pi = A_pi
    


    def AMSSetCycle(self, cycle_type='V'):
        """
        设置多重网格循环策略。
        可选 'V'（V-循环）或 'W'（W-循环）等。默认使用 V-cycle。。
        """
        self.cycle_type = cycle_type

    def AMSSolve(self, b, tol=1e-6, maxiter=100, x0=None):
        """
        使用AMS预条件器求解线性系统 A x = b。
        参数:
        - b: 右端项向量 (长度 = 边自由度数).
        - tol: 收敛残差容许阈值 (L2范数).
        - maxiter: 最大迭代次数.
        - x0: 初始解 (可选，不提供则使用零向量).
        返回: (解向量 x, 实际迭代次数)。
        """
        # 初始解
        if x0 is None:
            x = bm.zeros(self.n_edges,dtype=self.A.dtype)
        else:
            x = x0.copy()
        
        b = bm.array(b)
        
        for it in range(maxiter):
            # 计算残差 r = b - A x
            r = b - self.A@x
            res_norm = bm.linalg.norm(r)
            if res_norm < tol:
                return x, it  # 提前收敛
            el = spsolve(self.A.tril(), r,'scipy')
            
            for _ in range(self.smooth_iters):
                el += spsolve(self.A.tril(), r - self.A @ el,'scipy')       
            x = x + el
            r = b - self.A@x  # 更新残差
            # 2. 梯度子空间校正：求解 A_G e_g = G^T r，然后 x <- x + G e_g

            r_g = self.G.T@r  # 将残差限制到节点梯度空间
        
            e_g = spsolve(self.A_G, r_g,'scipy')

            x = x + self.G@e_g
            r = b - self.A@x  
            el = spsolve(self.A.tril(), r,'scipy')
            for _ in range(self.smooth_iters):
                el += spsolve(self.A.tril(), r - self.A @ el,'scipy')
            x = x + el
            r = b - self.A@x  # 更新残差
            r_pi_combined = self.Pi.T@r
            e_pi_combined = bm.zeros(self.Pi.shape[0])
            ml = GAMGSolver(isolver='MG', ptype='V', sstep=3, theta=0.25)
            ml.setup(self.A_Pi)
            e_pi_combined, info = ml.solve(r_pi_combined)
            print(info)
            x = x + self.Pi@e_pi_combined
            r = b - self.A@x
            el = spsolve(self.A.triu(), r,'scipy')
            
            for _ in range(self.smooth_iters):
                el += spsolve(self.A.triu(), r - self.A @ el,'scipy')
            x = x + el
            # 检查收敛
            r = b - self.A@x
            res_norm = bm.linalg.norm(r)
            print("iter:", it, "res_norm:", res_norm)
            if res_norm < tol:
                return x, it+1
        # 未在 maxiter 内收敛，返回当前解
        return x, maxiter
    
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import FirstNedelecFESpace, LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import ScalarMassIntegrator, CurlCurlIntegrator, BoundaryFaceMassIntegrator,DiffusionIntegrator, ScalarSourceIntegrator
from fealpy.fem import CurlJumpPenaltyIntergrator 
from fealpy.fem import BoundaryFaceSourceIntegrator, VectorSourceIntegrator, DirichletBC      
from fealpy.fem import BilinearForm, LinearForm

from fealpy.pde.maxwell_2d import SinData as PDE2d
pde = PDE2d()
mesh = TriangleMesh.from_box(pde.domain(), nx=32, ny=32) 

p = 0
space= FirstNedelecFESpace(mesh, p=p)
Eh = space.function()
LDOF = space.number_of_local_dofs()
GDOF = space.number_of_global_dofs()

D = CurlCurlIntegrator(coef=1, q=p+3)
M = ScalarMassIntegrator(coef=1, q=p+3)
 
beform = BilinearForm(space)
beform.add_integrator(D)
beform.add_integrator(M)
A = beform.assembly() 
print(A.shape)

f = VectorSourceIntegrator(pde.source, q=p+3)
# # Vr = BoundaryFaceSourceIntegrator(pde.robin, q=p+2)

leform = LinearForm(space)
leform.add_integrator(f)
# leform.add_integrator(Vr)
F = leform.assembly()
bc = DirichletBC(space, gd=pde.dirichlet)
A, F = bc.apply(A, F)

space3 = LagrangeFESpace(mesh, p=1)
space2 = TensorFunctionSpace(space3,(-1,2))

blform = BilinearForm(space2)
blform.add_integrator(ScalarMassIntegrator(q=4))
blform.add_integrator(DiffusionIntegrator(coef=1, q=4))
A_pi = blform.assembly()

blform2 = BilinearForm(space3)
blform2.add_integrator(DiffusionIntegrator(coef=1, q=4))
A_G = blform2.assembly()

node = mesh.entity('node')
edge = mesh.entity('edge')
print(edge)

amg = GAMGSolver()
amg.setup(A)
x0, info = amg.solve(F)
print(info)

# ams = AMSSolver()
# ams.AMSFEISetup(A, node, edge, dim=2)
# ams.AMSComputeGPi()
# ams.AMSComputePi()
# # ams.AMSSetAGradient(A_G)
# # ams.AMSSetAPi(A_pi)

# x0,iters = ams.AMSSolve(F)
# print(x0)
# print("iters:", iters)

Eh[:] = x0
error = mesh.error(pde.solution, Eh.value)
print("error:", error)

x2 = spsolve(A, F,'scipy')
Eh[:] = x2
error2 = mesh.error(pde.solution, Eh.value)
print("error2:", error2)



