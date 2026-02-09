from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm

class PDEInUPML3D_Z():
    def __init__(self, k,
                 h1, h2, delta):       
        self.h1, self.h2 = h1, h2
        self.delta = delta
        self.mu = 1.0
        self.k = k
        self.sigma_max = 10
        self.m = 5
    @cartesian
    def eps(self, pp):
        """
        空间位置相关的 ε(x)，z<=0 为介质区，z>0 为空气
        """
        z = pp[..., 2]
        return bm.where(z <= 0, 2.25, 1.0)
    def sigma3(self, z):
        """构造 z 方向的 sigma3 分段函数"""
        sigma = bm.zeros_like(z,dtype=bm.complex128)
        mask1 = (z >= self.h1 - self.delta) & (z <= self.h1)                     # 下层 PML
        mask2 = (z >= self.h2) & (z <= self.h2 + self.delta)                     # 上层 PML
        sigma[mask1] = self.sigma_max * (bm.abs(z[mask1] - self.h1) / self.delta)**self.m
        sigma[mask2] = self.sigma_max * (bm.abs(z[mask2] - self.h2) / self.delta)**self.m
        return sigma
    def Jacobi(self, pp):
        """
        返回雅可比矩阵 F(x) = diag(1, 1, s3(z))
        仅 z 方向有变换
        """
        z = pp[..., 2]
        N = pp.shape[:-1]
        shape = N + (3, 3)
        F = bm.zeros(shape, dtype=bm.complex128)
        sigma = self.sigma3(z)
        s3 = bm.ones_like(z, dtype=bm.complex128)
        # 下层 PML
        mask1 = (z >= self.h1 - self.delta) & (z <= self.h1)
        s3[mask1] = 1 + 1j * sigma[mask1]
        # 上层 PML
        mask2 = (z >= self.h2) & (z <= self.h2 + self.delta)
        s3[mask2] = 1 + 1j * sigma[mask2] 
        F[..., 0, 0] = 1.0
        F[..., 1, 1] = 1.0
        F[..., 2, 2] = s3
        return F
    def detJacobi(self, pp):
        """行列式 J = s1*s2*s3 = s3"""
        F = self.Jacobi(pp)
        return F[..., 2, 2]  # s3
    @cartesian
    def alpha(self, pp):
        """alpha = mu^{-1} * A = mu^{-1} * (F^T F / detF)"""
        F = self.Jacobi(pp)
        detF = self.detJacobi(pp)
        FTF = bm.einsum("...ij,...ik->...jk", F, F)
        A = FTF / detF[..., None, None]
        return A / self.mu
    @cartesian
    def beta(self, pp):
        """beta = k^2 * epsilon * A^{-1}"""
        F = self.Jacobi(pp)
        detF = self.detJacobi(pp)
        FTF = bm.einsum("...ij,...ik->...jk", F, F)
        invA = detF[..., None, None] * bm.linalg.inv(FTF)
        return -(self.k ** 2) * self.eps(pp)[..., None, None] * invA

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian

class MyExtendedPDE(PDEInUPML3D_Z):
    """
    扩展的 PML 类，添加额外功能，比如构造拉伸因子 s3(z)
    """
    def __init__(self, *args, **kwargs):
        """
        调用父类构造函数，自动继承其所有参数和成员变量。
        """
        super().__init__(*args, **kwargs)  # 初始化父类的参数
    @cartesian
    def E_inc(self, pp):
        """
        入射平面波:
            E_inc(x) = p * exp(i q · x)
        其中:
            p = (1, 1, 0)
            q = (0, 0, -kappa)  (你的 self.k 就是 κ)
        """
        z = pp[..., 2]                     # 提取 z 坐标
        phase = bm.exp(-1j * self.k * z)   # exp(i q·x) = exp(-i k z)
        # p = (1, 1, 0)
        E_x = phase
        E_y = phase
        E_z = bm.zeros_like(z, dtype=bm.complex128)
        return bm.stack([E_x, E_y, E_z], axis=-1)
    @cartesian
    def source(self, pp):
        """
        在 0 <= z <= 10 区域施加入射场，其余区域为 0
        """
        z = pp[..., 2]
        # 计算入射场 E_inc(x)
        E = self.E_inc(pp)
        # 定义激励区域 0 <= z <= 10
        mask = (z >= 0) & (z <= self.h2)
        # 保持与你之前的形式一致：source = k^2 * eps * E_inc
        # 在激励区域使用 g_inc，其余地方为 0
        g = bm.where(mask[..., None], E, 0.0)
        return g
    @cartesian
    def dirichlet(self, pp, n):
        # z = pp[..., 2]
        # mask = (z < 0)
        # val1 = self.E_inc(pp)
        # val  = bm.cross(val1, n)
        # 创建与 val 相同形状的零数组
        # arr = bm.zeros_like(z)
        # 在 mask 为 True 的位置保留原来的 val
        # arr[mask] = val[mask]
        return bm.zeros(pp.shape, dtype=bm.complex128)

#!/usr/bin/env python3
import time
import sys
import argparse 
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import FirstNedelecFESpace,SecondNedelecFESpace
from fealpy import logger
logger.setLevel('WARNING')
from fealpy.backend import backend_manager as bm
# 双线性型
from fealpy.fem import BilinearForm
# 线性型
from fealpy.fem import LinearForm
# 积分子
from fealpy.fem import ScalarMassIntegrator
from fealpy.fem import CurlCurlIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.decorator import cartesian, barycentric
from fealpy.tools.show import showmultirate, show_error_table
# solver
from fealpy.solver import spsolve, gmres
from fealpy.pde.maxwell_2d import SinData as PDE2d
from fealpy.pde.maxwell_3d import BubbleData3d as PDE3d
from fealpy.utils import timer
# from model import MyExtendedPDE
mesh = TetrahedronMesh.from_box([0,1,0,1,-2,2],nx=10,ny=10,nz=10)
p = 0
space = FirstNedelecFESpace(mesh, p)
pde = MyExtendedPDE(k=2*np.pi, h1=-1, h2=1, delta=1)
# ----------------------------
# ----------------------------
# 处理边界自由度位置
# ----------------------------
# ----------------------------
edge = mesh.edge
node = mesh.node
edge_node = node[edge]
# ----------------------------
# 前后周期边界条件自由度
# ----------------------------
front_mask = bm.all(np.isclose(edge_node[:,:,0], 1), axis=1)
back_mask = bm.all(np.isclose(edge_node[:,:,0], 0), axis=1)
# 前后周期边界的自由度编号
front_mask_index = bm.where(front_mask)[0]
back_mask_index = bm.where(back_mask)[0]
front_edges = edge[front_mask]
back_edges = edge[back_mask]
# 边中点坐标计算
def edge_midpoints(edges):
    pts = node[edges]              # (Ne, 2, 3)
    mid = pts.mean(axis=1)         # (Ne, 3)
    return mid
front_mid = edge_midpoints(front_edges)[:,1:3]
back_mid  = edge_midpoints(back_edges)[:,1:3]
diff1 = front_mid[:,None,:] - back_mid[None,:,:]
dist1 = bm.linalg.norm(diff1, axis=2)
match_index1 = np.argmin(dist1, axis=1)
back_mask_index = back_mask_index[match_index1]
# ----------------------------
# 左右周期边界条件自由度
# ----------------------------
left_mask = bm.all(np.isclose(edge_node[:,:,1], 1), axis=1)
right_mask = bm.all(np.isclose(edge_node[:,:,1], 0), axis=1)
# 前后周期边界的自由度编号
left_mask_index = bm.where(left_mask)[0]
right_mask_index = bm.where(right_mask)[0]
left_edges = edge[left_mask]
right_edges = edge[right_mask]
left_mid = edge_midpoints(left_edges)[:,0:3:2]
right_mid  = edge_midpoints(right_edges)[:,0:3:2]
diff2 = left_mid[:,None,:] - right_mid[None,:,:]
dist2 = bm.linalg.norm(diff2, axis=2)
match_index2 = np.argmin(dist2, axis=1)
right_mask_index = right_mask_index[match_index2]
# 上下的自由度
up_mask = bm.all(np.isclose(edge_node[:,:,2], 2), axis=1)
down_mask = bm.all(np.isclose(edge_node[:,:,2], -2), axis=1)
# 前后周期边界的自由度编号
up_mask_index = bm.where(up_mask)[0]
down_mask_index = bm.where(down_mask)[0]
# ----------------------------
# ----------------------------
# 结束
# ----------------------------
# ----------------------------
NF = space.number_of_global_dofs()
ID1 = bm.zeros(NF, dtype=bm.bool)
ID1[up_mask_index] = True
ID1[down_mask_index] = True
c = np.unique(np.concatenate([right_mask_index, front_mask_index]))
bform = BilinearForm(space)
bform.add_integrator(ScalarMassIntegrator(coef=pde.beta, q=p+3))
bform.add_integrator(CurlCurlIntegrator(coef=pde.alpha, q=p+3))
A = bform.assembly()
lform = LinearForm(space)
lform.add_integrator(VectorSourceIntegrator(pde.source, q=p+3))
F = lform.assembly()
# Dirichlet 边界条件
uh = space.function(dtype=np.complex128)
bc = DirichletBC(space,threshold= ID1, gd=pde.dirichlet)
A, F = bc.apply(A, F)
print("开始计算")
import numpy as np
from scipy.sparse import coo_matrix
from fealpy.sparse import COOTensor, CSRTensor
def sparse_row_add_coo(I, target_indices, source_indices):
    """
    对稀疏矩阵 I (COO格式) 做行加法：I[target[i], :] += I[source[i], :]
    I: coo_matrix
    target_indices, source_indices: 等长数组
    """
    row = I.row
    col = I.col
    data = I.data
    new_rows = []
    new_cols = []
    new_data = []
    # 遍历每一对 (target, source)
    for t, s in zip(target_indices, source_indices):
        mask = (row == s)  # 找到源行 s 的所有非零元素
        new_rows.extend([t]*np.sum(mask))
        new_cols.extend(col[mask])
        new_data.extend(data[mask])
    # 构造行加法矩阵
    add_matrix = coo_matrix((new_data, (new_rows, new_cols)), shape=I.shape)
    # 返回 COO 格式
    return (I + add_matrix).tocoo()
def sparse_delete_rows_coo(mat, rows_to_delete):
    all_rows = np.arange(mat.shape[0])
    keep_rows = np.setdiff1d(all_rows, rows_to_delete)
    # mask 保留行
    mask = np.isin(mat.row, keep_rows)
    # 旧行 → 新行映射
    row_mapping = {old: new for new, old in enumerate(keep_rows)}
    new_row = np.array([row_mapping[r] for r in mat.row[mask]])
    # 构造新 COO 矩阵
    new_mat = coo_matrix(
        (mat.data[mask], (new_row, mat.col[mask])),
        shape=(len(keep_rows), mat.shape[1])
    )
    return new_mat
row = np.arange(NF)
col = np.arange(NF)
# 值
data = np.ones(NF, dtype=np.float64)
# 构造 COO 矩阵
I = coo_matrix((data, (row, col)), shape=(NF, NF))
I = sparse_row_add_coo(I, back_mask_index, front_mask_index)
I = sparse_row_add_coo(I, left_mask_index, right_mask_index)
I = sparse_delete_rows_coo(I, c)
kk1 = I.row
kk2 = I.col
kk3 = I.data
indices = np.vstack([kk1, kk2])   # shape = (2, nnz)
values = kk3                      # shape = (nnz,)
shape = (NF - len(c), NF)         # 新矩阵形状
# 创建 COOTensor
I = COOTensor(indices, values, spshape=shape)
I = I.tocsr()
A = I @ A @ I.T
print(A.toarray())
F = I @ F
val = spsolve(A, F, "mumps")
# val,_ = gmres(A, F, atol=1e-6, rtol=1e-6)
vall = I.T @ val
uh[:] = vall
print('结束计算')
NN = mesh.number_of_nodes()
cell = mesh.entity('cell')
bc = bm.array([[1/3, 1/3, 1/3, 1/3]], dtype=bm.float64)
vals = space.value(uh, bc) # (1, NF, 3)
val = np.squeeze(vals, axis=1)
mesh.celldata['Ex_real'] = val[:, 0].real
mesh.celldata['Ex_imag'] = val[:, 0].imag
mesh.celldata['Ey_real'] = val[:, 1].real
mesh.celldata['Ey_imag'] = val[:, 1].imag
mesh.celldata['Ez_real'] = val[:, 2].real
mesh.celldata['Ez_imag'] = val[:, 2].imag
# 也可以加入模值
mesh.celldata['E_mag']  = np.linalg.norm(val, axis=1).real
mesh.to_vtk(fname = 'liuzhi.vtu')