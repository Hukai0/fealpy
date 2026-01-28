from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import FirstNedelecFESpace, LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import ScalarMassIntegrator, CurlCurlIntegrator, BoundaryFaceMassIntegrator,DiffusionIntegrator, ScalarSourceIntegrator
from fealpy.fem import CurlJumpPenaltyIntergrator 
from fealpy.fem import BoundaryFaceSourceIntegrator, VectorSourceIntegrator, DirichletBC      
from fealpy.fem import BilinearForm, LinearForm

from fealpy.pde.maxwell_2d import SinData as PDE2d
pde = PDE2d()
mesh = TriangleMesh.from_box(pde.domain(), nx=8, ny=8) 

p = 0
space= FirstNedelecFESpace(mesh, p=p)

space3 = LagrangeFESpace(mesh, p=1)
space2 = TensorFunctionSpace(space3,(-1,2))


import numpy as np
from fealpy.backend import backend_manager as bm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, cg, LinearOperator


# 获取全局自由度数
ndof = space.number_of_global_dofs()      # Nédélec DOFs
vdof = space2.number_of_global_dofs()       # 向量节点 DOFs = 3 * (p+1 阶 Lagrange DOFs)

# 准备积分规则（高阶积分，建议 2p+1 或更高）
q = 2*p + 3
qf = mesh.quadrature_formula(q)                   # 四面体积分点
bcs, ws = qf.get_quadrature_points_and_weights()   # bcs: (NQ, TD+1), ws: (NQ,)

# 计算 Nédélec 基函数在积分点上的值
# shape: (NC, NQ, ldof, GD, dim) → 通常会 squeeze
phi_nd = space.basis(bcs)   # (NC, NQ, ldof_nd, dim) 或类似

# 计算向量节点基函数在积分点上的值
phi_vec = space2.basis(bcs)   # (NC, NQ, ldof_vec, dim)

# 组装向量质量矩阵 M_ND (ndof x ndof)
# 使用 cell2dof 映射来组装稀疏矩阵
cell2dof_nd = space.cell_to_dof()          # (NC, ldof_nd)
cell2dof_vec = space2.cell_to_dof()          # (NC, ldof_vec)
cm = mesh.entity_measure("cell")
# 计算局部质量矩阵（每个单元）
# ∫ λ_i · λ_j dV ≈ sum_q ws[q] * (phi_nd[...,i,:] @ phi_nd[...,j,:])
NC = mesh.number_of_cells()
ldof_nd = phi_nd.shape[2]
ldof_vec = phi_vec.shape[2]

# 先计算局部 M_local (NC, ldof_nd, ldof_nd)
# dot = bm.einsum('cnqd, cnqd -> cnq', phi_nd, phi_nd)          # (NC, NQ)
# M_local = bm.einsum('cnq, cq, cn, cn -> cnm', dot, ws, bm.ones(NC), bm.ones(NC))
# 更高效写法：
print(phi_nd.shape, ws.shape)
M_local = bm.einsum('cnid, cnjd, n, c -> cij', phi_nd, phi_nd, ws,cm)  # (NC, ldof_nd, ldof_nd)

# 组装全局稀疏矩阵
I =bm.broadcast_to(cell2dof_nd[:, :, None], shape=M_local.shape)
J = bm.broadcast_to(cell2dof_nd[:, None, :], shape=M_local.shape)
data = M_local.flatten()

M_ND = coo_matrix((data, (I.flatten(), J.flatten())), shape=(ndof, ndof)).tocsr()

# 组装右端向量 b (vdof x ndof) —— 每一列对应一个 Nédélec 基
# b[m, k] = ∫ w_m · λ_k dV
b_local = bm.einsum('cnid, cnjd, n, c -> cij', phi_vec, phi_nd, ws, cm)  # (NC, ldof_vec, ldof_nd)

I_b = bm.repeat(cell2dof_vec[:, :, None], ldof_nd, axis=2)        # (NC, ldof_vec, ldof_nd)
J_b = bm.repeat(cell2dof_nd[:, None, :], ldof_vec, axis=1)
data_b = b_local.transpose(0, 2, 1).flatten()  # 调整顺序

b_dense = coo_matrix((data_b, (I_b.flatten(), J_b.flatten())), shape=(vdof, ndof)).toarray()
# 结果 b_dense.shape = (vdof, ndof)

# 解 M_ND @ Pi^T = b_dense^T   →  Pi^T = M_ND \ b_dense^T
# 即 Pi 的列 = Nédélec 基对向量节点基的投影系数
# 由于 M_ND 可能大，用 CG 求解每个列（推荐）

Pi_rows = []
Pi_cols = []
Pi_data = []

solver = LinearOperator((ndof, ndof), matvec=lambda x: M_ND @ x)
# 或者直接 factorize 如果规模允许

for m in range(vdof):
    rhs = b_dense[m]  # 第 m 个向量节点基对应的右端
    coef = spsolve(M_ND, rhs)  # 直接求解
    # coef, info = cg(M_ND, rhs)  # 或用 spsolve
    # if info != 0:
    #     print(f"CG not converged at dof {m}")
    print(bm.linalg.norm(M_ND @ coef - rhs))
    
    nonzero = np.abs(coef) > 1e-5
    Pi_rows.extend(np.where(nonzero)[0])
    Pi_cols.extend([m] * nonzero.sum())
    Pi_data.extend(coef[nonzero])

Pi = coo_matrix((Pi_data, (Pi_rows, Pi_cols)), shape=(ndof, vdof)).tocsr()

# 现在 Pi 就是你要的插值矩阵： Pi @ w_vec = u_edge (Nédélec 系数)
print(Pi)
