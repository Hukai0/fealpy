from fealpy.backend import backend_manager as bm
from fealpy.sparse import csr_matrix, CSRTensor
from fealpy.solver import spsolve, GAMGSolver
from fealpy.mesh import TriangleMesh, TetrahedronMesh
import numpy as np

class ADSSolver:
    """
    2D (三角形) / 3D (四面体) 的 Auxiliary Space Divergence Solver (ADS)
    适用于最低阶 Raviart-Thomas (RT0) 空间
    """
    
    def __init__(self, dim: int = 3, split_pi: bool = False, cycle_type: int = 1):
        """
        dim:       2 或 3（会自动检测，也可手动指定）
        split_pi:  True → 使用独立 Pix/Piy(/Piz)
                   False → monolithic Pi，推荐用于 3D 或需要更高效率的情况
        """
        self.dim = dim
        self.split_pi = split_pi
        self.cycle_type = cycle_type
        
        # 几何与拓扑
        self.vertices = None      # (NV, GD)
        self.cells = None         # (NC, 3) 或 (NC, 4)
        self.faces = None         # (NF, dim+1)  2D:三角形, 3D:三角形面
        self.edges = None         # (NE, 2)
        self.face2edge = None     # (NF, dim+1)
        self.edge_sign = None     # (NF, dim+1)  +1/-1
        
        # 矩阵
        self.A = None             # RT 刚度矩阵 (NF x NF)
        self.C = None             # 离散旋度: face → edge (NF x NE)
        self.G = None             # 离散梯度: edge → node (NE x NV)
        self.PiND = []            # [PiNDx, PiNDy, (PiNDz)]  Nédélec 插值分量
        
        self.Pi = None            # monolithic (NF x dim*NV)
        self.Pix = self.Piy = self.Piz = None
        
        # 子空间
        self.A_C = None
        self.A_Pi = None
        self.solver_C = None
        self.solver_Pi = None
        self.solver_Pix = self.solver_Piy = self.solver_Piz = None
        
        # 参数
        self.smoother = 'GaussSeidel'
        self.smooth_iters = 1
        self.jacobi_weight = 1.0
        self.maxiter = 50
        self.tol = 1e-8

    # ========================================================
    # 1. 自动构建网格拓扑（兼容 2D/3D）
    # ========================================================
    def setup_mesh(self, vertices: np.ndarray, cells: np.ndarray):
        """
        vertices: (NV, GD)
        cells:    (NC, 3) for 2D 或 (NC, 4) for 3D
        """
        vertices = np.array(vertices, dtype=np.float64)
        cells = np.array(cells, dtype=np.int32)
        
        self.dim = vertices.shape[1]
        self.vertices = bm.array(vertices)
        self.cells = bm.array(cells)
        
        if self.dim == 2:
            mesh = TriangleMesh(vertices, cells)
        else:
            mesh = TetrahedronMesh(vertices, cells)
        
        # 统一提取 face (2D: cell 本身, 3D: 四面体面)
        self.faces = bm.array(mesh.entity('face'))           # (NF, dim+1)
        self.edges = bm.array(mesh.ds.edge)                  # (NE, 2)
        self.face2edge = bm.array(mesh.ds.face_to_edge())    # (NF, dim+1)
        
        # 计算 edge 在每个 face 上的方向符号
        NF = self.faces.shape[0]
        nfp = self.dim + 1  # nodes per face
        edge_sign = bm.ones((NF, nfp), dtype=bm.float64)
        
        for k in range(nfp):
            v0 = self.faces[:, k]
            v1 = self.faces[:, (k+1) % nfp]
            # edge 的正向与 (v0→v1) 一致 → +1，否则 -1
            match_forward = (self.edges[:, 0] == v0) & (self.edges[:, 1] == v1)
            match_backward = (self.edges[:, 0] == v1) & (self.edges[:, 1] == v0)
            sign = bm.where(match_forward, 1.0, -1.0)
            sign = bm.where(match_backward, -1.0, sign)
            edge_sign[:, k] = sign[self.face2edge[:, k]]
        
        self.edge_sign = edge_sign

    # ========================================================
    # 2. 构建拓扑矩阵 G, C, PiND
    # ========================================================
    def build_topological_matrices(self):
        NV = self.vertices.shape[0]
        NE = self.edges.shape[0]
        NF = self.faces.shape[0]
        
        # --- G: edge → node 离散梯度 (NE x NV) ---
        data, row, col = [], [], []
        for e, (i, j) in enumerate(self.edges):
            row += [e, e]
            col += [i, j]
            data += [-1.0, 1.0]
        self.G = csr_matrix((bm.array(data), (bm.array(row), bm.array(col))), shape=(NE, NV))
        
        # --- C: face → edge 离散旋度 (NF x NE) ---
        data, row, col = [], [], []
        for f in range(NF):
            for k in range(self.dim + 1):
                e = self.face2edge[f, k]
                s = self.edge_sign[f, k]
                row.append(f)
                col.append(e)
                data.append(s)
        self.C = csr_matrix((bm.array(data), (bm.array(row), bm.array(col))), shape=(NF, NE))
        
        # --- PiNDx, PiNDy, (PiNDz): 最低阶 Nédélec 插值 ---
        coords = self.vertices
        self.PiND = []
        for d in range(self.dim):
            data_d, row_d, col_d = [], [], []
            for e, (i, j) in enumerate(self.edges):
                diff = coords[j, d] - coords[i, d]
                half = 0.5 * diff
                for v in (i, j):
                    row_d.append(e)
                    col_d.append(v)
                    data_d.append(half)
            PiND_d = csr_matrix((bm.array(data_d), (bm.array(row_d), bm.array(col_d))), shape=(NE, NV))
            self.PiND.append(PiND_d)

    # ========================================================
    # 3. 构建 Π 插值（核心！完全复现 HYPRE ADSComputePi/Pixyz）
    # ========================================================
    def build_Pi(self):
        if self.C is None or self.G is None or len(self.PiND) == 0:
            raise RuntimeError("请先调用 build_topological_matrices()")
        
        NV = self.vertices.shape[0]
        NF = self.C.shape[0]
        coords = [self.vertices[:, d] for d in range(self.dim)]
        
        # 计算常向量场 (1,0,..), (0,1,..) 在 RT 空间的表示
        RT_const = []
        for d in range(self.dim):
            # RT_e = C * PiND_d * coord_d
            RT_const.append(self.C @ (self.PiND[d] @ coords[d]))
        
        # face-to-vertex 稀疏模式
        F2V_pattern = bm.abs(self.C) @ bm.abs(self.G)
        F2V_pattern.eliminate_zeros()
        indptr = F2V_pattern.indptr
        indices = F2V_pattern.indices
        
        if not self.split_pi:
            # monolithic Pi (NF x dim*NV)
            data, row, col = [], [], []
            for f in range(NF):
                start, end = indptr[f], indptr[f+1]
                verts = indices[start:end]
                for v in verts:
                    for d in range(self.dim):
                        row.append(f)
                        col.append(self.dim * v + d)
                        data.append(RT_const[d][f])
            self.Pi = csr_matrix((bm.array(data), (bm.array(row), bm.array(col))), 
                                 shape=(NF, self.dim * NV))
        else:
            # split Pix, Piy, (Piz)
            self.Pix = csr_matrix((bm.repeat(RT_const[0], bm.diff(indptr)),
                                   (bm.repeat(bm.arange(NF), bm.diff(indptr)), indices)),
                                  shape=(NF, NV))
            self.Piy = csr_matrix((bm.repeat(RT_const[1], bm.diff(indptr)),
                                   (bm.repeat(bm.arange(NF), bm.diff(indptr)), indices)),
                                  shape=(NF, NV))
            if self.dim == 3:
                self.Piz = csr_matrix((bm.repeat(RT_const[2], bm.diff(indptr)),
                                       (bm.repeat(bm.arange(NF), bm.diff(indptr)), indices)),
                                      shape=(NF, NV))

    # ========================================================
    # 4. 建立子空间求解器
    # ========================================================
    def setup(self, A):
        self.A = A if isinstance(A, CSRTensor) else CSRTensor.from_scipy(A.to_scipy() if hasattr(A, 'to_scipy') else A)
        
        if self.C is None:
            self.build_topological_matrices()
        if (self.Pi is None) and (self.Pix is None):
            self.build_Pi()
        
        # Curl/Gradient 子空间
        self.A_C = self.C.T @ (self.A @ self.C)
        self.solver_C = GAMGSolver(isolver='MG', ptype='V', sstep=3, theta=0.25)
        self.solver_C.setup(self.A_C)
        
        # Pi 子空间
        if not self.split_pi:
            self.A_Pi = self.Pi.T @ (self.A @ self.Pi)
            self.solver_Pi = GAMGSolver(isolver='MG', ptype='V', sstep=3, theta=0.25)
            self.solver_Pi.setup(self.A_Pi)
        else:
            solvers = []
            for d in range(self.dim):
                Pi_d = [self.Pix, self.Piy, self.Piz][d]
                A_Pi_d = Pi_d.T @ (self.A @ Pi_d)
                solver = GAMGSolver(isolver='MG', ptype='V', sstep=3, theta=0.25)
                solver.setup(A_Pi_d)
                solvers.append(solver)
            self.solver_Pix, self.solver_Piy = solvers[0], solvers[1]
            if self.dim == 3:
                self.solver_Piz = solvers[2]

    # ========================================================
    # 5. 手动设置接口
    # ========================================================
    def set_C(self, C): self.C = C
    def set_G(self, G): self.G = G
    def set_PiND(self, PiND_list): self.PiND = PiND_list
    def set_Pi(self, Pi): 
        self.Pi = Pi
        self.split_pi = False
    def set_Pixyz(self, Pix, Piy, Piz=None): 
        self.Pix, self.Piy, self.Piz = Pix, Piy, Piz
        self.split_pi = True

    # ========================================================
    # 6. 求解（经典 multiplicative "01210"）
    # ========================================================
    def solve(self, b, x0=None, tol=None, maxiter=None):
        if tol is None: tol = self.tol
        if maxiter is None: maxiter = self.maxiter
        
        b = bm.array(b)
        x = bm.zeros_like(b) if x0 is None else bm.array(x0).copy()
        
        for it in range(maxiter):
            r = b - self.A @ x
            norm_r = bm.linalg.norm(r)
            print(f"Iter {it:3d}:  residual = {norm_r:.3e}")
            if norm_r < tol:
                return x, it + 1
            
            # 前平滑
            for _ in range(self.smooth_iters):
                if self.smoother == 'Jacobi':
                    dx = self.jacobi_weight * (r / self.A.diagonal())
                else:
                    dx = spsolve(self.A.tril().to_scipy(), r, 'scipy')
                x += dx
                r = b - self.A @ x
            
            # Curl/Gradient 子空间校正
            rc = self.C.T @ r
            ec, _ = self.solver_C.solve(rc)
            x += self.C @ ec
            
            # Pi 子空间校正
            if not self.split_pi:
                rp = self.Pi.T @ r
                ep, _ = self.solver_Pi.solve(rp)
                x += self.Pi @ ep
            else:
                ep_x, _ = self.solver_Pix.solve(self.Pix.T @ r)
                ep_y, _ = self.solver_Piy.solve(self.Piy.T @ r)
                x += self.Pix @ ep_x + self.Piy @ ep_y
                if self.dim == 3 and self.Piz is not None:
                    ep_z, _ = self.solver_Piz.solve(self.Piz.T @ r)
                    x += self.Piz @ ep_z
            
            # 后平滑
            for _ in range(self.smooth_iters):
                r = b - self.A @ x
                if self.smoother == 'Jacobi':
                    dx = self.jacobi_weight * (r / self.A.diagonal())
                else:
                    dx = spsolve(self.A.triu().to_scipy(), r, 'scipy')
                x += dx
        
        print("ADS 未在最大迭代次数内收敛")
        return x, maxiter
