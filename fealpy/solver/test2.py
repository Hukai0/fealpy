
from typing import Optional, TypeVar, Union, Generic, Callable
from fealpy.typing import TensorLike, Index, _S, Threshold

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.decorator import barycentric, cartesian

from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator, ScalarMassIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.model import PDEModelManager
from fealpy.pde.poisson_2d import CosCosData
from fealpy.pde.poisson_3d import CosCosCosData

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

import time
import sys
import sympy as sp

from scipy.sparse.linalg import cg
from pyamg.relaxation.relaxation import gauss_seidel
import pyamg


def build_vertex_patch_dofs(space):
    mesh = space.mesh

    # 每个单元的顶点编号 (NC, nvc)；三角形 nvc=3，四面体 nvc=4
    c2n = mesh.entity("cell")
    NC, nvc = c2n.shape

    NN = mesh.number_of_nodes()

    # 每个单元对应的全局 dof 编号
    c2d = space.dof.cell_to_dof()


    # 1) 构建 vertex->cell 邻接矩阵 v2c，大小 (NN, NC)
    # v2c[v, k] = 1 表示 顶点 v 属于 单元 k
    rows = c2n.reshape(-1)  # 顶点编号
    cols = bm.repeat(bm.arange(NC, dtype=bm.int32), nvc)  # 对应单元编号
    data = bm.ones_like(rows, dtype=bm.int8)
    v2c = csr_matrix((data, (rows, cols)), shape=(NN, NC))

    # 2) 对每个顶点 v：取相邻单元 cells，然后对这些 cells 的 c2d 做并集 unique
    patch_dofs = []
    for v in range(NN):
        start, end = v2c.indptr[v], v2c.indptr[v+1]
        cells = v2c.indices[start:end]  # 顶点 v 的相邻单元列表

        dofs = bm.unique(c2d[cells].reshape(-1))  # 并集 + 去重
        patch_dofs.append(dofs)

    return patch_dofs, v2c

class ASPForPoisson(object):
    """
    Add Schwarz Preconditioner for Poisson equation.
    Space decomposition based on vertices:
                        V = V_0 + (V_1 + V_2 + ... + V_N)
    V_0 is the linear finite element space. V_i is the subspace associated with
    vertex i.

    TODO: More efficient implementation with static condensation.
    """

    def __init__(self, space, A):
        """
        Parameters
            space : LagrangeFESpace with degree p
            A     : Stiffness matrix, must be in scipy.sparse.csr_matrix format
        """
        self.space = space
        self.A = A

        self.pre = self.generate_preconditioner()

    def solve(self, r):
        A = self.A
        iter_count = 0
        def callback(xk):
            nonlocal iter_count
            iter_count += 1
        t0 = time.time()
        e = cg(A, r, M = self.pre, callback=callback, rtol=1e-8)[0]
        t1 = time.time()
        print(f"cg iter count: {iter_count}", f"solve time: {t1-t0}")
        return e

    def vertex_preconditioner(self):
        """
        Vertex-based additive Schwarz preconditioner
        """
        A     = self.A
        space = self.space

        p  = space.p
        TD = space.mesh.TD
        NN = space.mesh.number_of_nodes()
        gdof = space.number_of_global_dofs()

        c2d = space.dof.cell_to_dof()
        c2n = space.mesh.entity("cell")

        patch_dofs, v2c = build_vertex_patch_dofs(space)

        A_diag = []
        for v, dofs in enumerate(patch_dofs):
        # 取子矩阵 A_sub = A[dofs, dofs]
            A_sub = A[dofs, :][:, dofs]   # A 是 scipy.sparse 矩阵时可用
            A_sub_dense = A_sub.toarray()

            invA = bm.linalg.inv(A_sub_dense)  # 小块才建议直接求逆
            A_diag.append((invA, dofs))

        def preconditioner(r):
            r = r.astype(bm.float64)
            e = bm.zeros_like(r)
            for i in range(NN):
                A_i, dofs = A_diag[i]
                e[dofs] += A_i @ r[dofs]
            return e
        return LinearOperator(shape=A.shape, matvec=preconditioner, rmatvec=preconditioner)

    def generate_preconditioner(self):
        A     = self.A
        space = self.space

        space0 = FirstNedelecFESpace(space.mesh, p=p)
        node = space.mesh.entity('node')
        edge = space.mesh.entity('edge')
        PI = self.projection(space0)

        A0 = ((PI.T)@A@PI)
        mask = bm.abs(A0.data) < 1e-15
        A0.data[mask] = 0
        A0.eliminate_zeros()

        P1 = AMSSolver()
        P1.AMSFEISetup(A0, node, edge, dim=2)
        P1.AMSComputeGPi()
        P1.AMSComputePi()
        P0 = self.vertex_preconditioner()

        def mix_preconditioner(r1, P0, P1, PI, A):
            r1  = r1.astype(bm.float64)
            print(P1.AMSSolve(PI.T@r1))
            e1  = PI@(P1.AMSSolve(PI.T@r1)[0])
            e1 += 0.5*P0@r1
            return e1 

        pre_fun = lambda r: mix_preconditioner(r, P0, P1, PI, A)
        pre = LinearOperator(shape=A.shape, matvec=pre_fun, rmatvec=pre_fun)
        return pre

    def projection(self, space0):
        """
        L2 projection from space0 to space1
        @TODO: There exists a more efficient implementation.
        """

        space = self.space
        p = space.p
        q = p + 3

        mesh = space0.mesh

        qf = mesh.quadrature_formula(q, "cell") 
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi  = space.basis(bcs) # (NC, NQ, ldof0)
        phi0 = space0.basis(bcs)

        cm = mesh.entity_measure("cell")
        F  = bm.einsum('cqld, cqmd, q, c-> clm', phi0, phi, ws, cm)
        M  = bm.einsum('cqld, cqmd, q, c-> clm', phi, phi, ws, cm)
        Minv = bm.linalg.inv(M)
        F = bm.einsum('clm, cmd->cld', F, Minv)

        c2d0   = space0.cell_to_dof()
        gdof0  = space0.number_of_global_dofs()
        c2d  = space.cell_to_dof()
        gdof = space.number_of_global_dofs()

        I = bm.broadcast_to(c2d[:, None], F.shape).reshape(-1)
        J = bm.broadcast_to(c2d0[..., None], F.shape).reshape(-1)
        data = F.reshape(-1)

        IJ = bm.concatenate([I[None, :], J[None, :]], axis=0)
        unique_IJ, index = bm.unique(IJ, axis=1, return_index=True)

        unique_data = data[index]
        M = csr_matrix((unique_data, (unique_IJ[0], unique_IJ[1])), 
                       shape=(gdof, gdof0), dtype=phi.dtype)
        return M.tocsr()

def test_pre(n: int = 10, p: int = 2, dim=2):
    # 创建PDE模型
    if dim == 2:
        pde = CosCosData() 
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=n, ny=n)
    else:
        pde = CosCosCosData()
        mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
    space = LagrangeFESpace(mesh, p=p)

    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator(q=p+3, method='fast'))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(pde.source, q=p+3))
    b = lform.assembly()

    # 边界条件
    gdof = space.number_of_global_dofs()
    A, b = DirichletBC(space, gd=pde.solution).apply(A, b)

    A  = A.to_scipy()
    uh = space.function()

    print("Constructing ASP preconditioner...")
    Solver = ASPForPoisson(space, A)
    print("Solving linear system with ASP preconditioner...")
    uh[:] = Solver.solve(b)
    print("Solving done.")

    error0 = mesh.error(uh, pde.solution)
    error1 = mesh.error(uh.grad_value, pde.gradient)
    print(f"n = {n}, p = {p}, L2 error: {error0}, H1 error: {error1}")
    return error0, error1

from fealpy.backend import backend_manager as bm
from fealpy.solver import spsolve,GAMGSolver

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
        self.G = csr_matrix((data, (row, col)), shape=(num_edges, num_nodes))
        print(type(self.A))
        self.A_G = (self.G.T).dot(self.A.dot(self.G))
            
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
                return x
        # 未在 maxiter 内收敛，返回当前解
        return x
    
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import FirstNedelecFESpace, LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import ScalarMassIntegrator, CurlCurlIntegrator, BoundaryFaceMassIntegrator,DiffusionIntegrator, ScalarSourceIntegrator
from fealpy.fem import CurlJumpPenaltyIntergrator 
from fealpy.fem import BoundaryFaceSourceIntegrator, VectorSourceIntegrator, DirichletBC      
from fealpy.fem import BilinearForm, LinearForm

from fealpy.pde.maxwell_2d import SinData as PDE2d
pde = PDE2d()
mesh = TriangleMesh.from_box(pde.domain(), nx=10, ny=10) 

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

# space3 = LagrangeFESpace(mesh, p=1)
# space2 = TensorFunctionSpace(space3,(-1,2))

# blform = BilinearForm(space2)
# blform.add_integrator(ScalarMassIntegrator(q=4))
# blform.add_integrator(DiffusionIntegrator(coef=1, q=4))
# A_pi = blform.assembly()

# blform2 = BilinearForm(space3)
# blform2.add_integrator(DiffusionIntegrator(coef=1, q=4))
# A_G = blform2.assembly()
Solver = ASPForPoisson(space, A.to_scipy())

node = mesh.entity('node')
edge = mesh.entity('edge')
print(edge)

ams = AMSSolver()
ams.AMSFEISetup(A, node, edge, dim=2)
ams.AMSComputeGPi()
ams.AMSComputePi()
# ams.AMSSetAGradient(A_G)
# ams.AMSSetAPi(A_pi)

x0,iters = ams.AMSSolve(F)
print("iters:", iters)

Eh[:] = x0
error = mesh.error(pde.solution, Eh.value)
print("error:", error)

x2 = spsolve(A, F,'scipy')
Eh[:] = x2
error2 = mesh.error(pde.solution, Eh.value)
print("error2:", error2)