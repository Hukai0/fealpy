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

from scipy.sparse.linalg import gmres,cg
from pyamg.relaxation.relaxation import gauss_seidel
import pyamg


from fealpy.mesh import Mesh, TriangleMesh
from fealpy.functionspace import TensorFunctionSpace, LagrangeFESpace,RaviartThomasFESpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm, LinearBlockForm
from fealpy.fem import ScalarSourceIntegrator, ScalarNeumannBCIntegrator, ScalarMassIntegrator, GradPressureIntegrator,DivIntegrator,DivIntegrator2,DirichletBC    
from fealpy.solver import spsolve,GAMGSolver
from fealpy.model import PDEModelManager
from fealpy.decorator import barycentric
from fealpy.backend import backend_manager as bm




# def build_face_patch_dofs(space):
#     """
#     向量化构建面 patch DOFs
#     假设：边界面的 f2c[:,0] == f2c[:,1]（重复同一个单元），内部面不同
#     """
#     mesh = space.mesh
#     f2c = mesh.face_to_cell()          # (NF, 4)，我们只用第0和第2列
#     ledof = space.number_of_local_dofs(doftype='edge')
#     c2d = space.dof.cell_to_dof()      # (NC, ldof)
#     f2d = space.dof.edge_to_dof()      # (NF, ldof)

#     c2d = c2d[:,3*ledof:]
#     dofs = c2d[f2c[:,0:2]]                              # (NF, 2, ldof)
#     dofs = bm.stack([dofs,f2d],axis=1)
    
#     # 每行打平后去重即可（自动处理重复单元的情况）
#     patch_dofs = [bm.unique(row.reshape(-1)) for row in dofs]
    
#     return patch_dofs

# def build_face_patch_dofs(space):
#     """
#     构建每个面/边的 patch dofs:
#     patch = 相邻(最多2个)单元内部 dofs + 当前面/边 dofs，并去重
#     兼容：边界面 cellR = -1 或 cellR == cellL
#     """
#     mesh = space.mesh
#     f2c = mesh.face_to_cell()          # 常见 (NF,4): [cL, cR, lfL, lfR]，也可能 (NF,2)
#     ledof = space.number_of_local_dofs(doftype='edge')

#     c2d = space.dof.cell_to_dof()      # (NC, ldof_total)
#     e2d = space.dof.edge_to_dof()      # (NF, ledof)  (2D里face=edge)


#     cells = f2c[:, :2]


#     c2d_cell = c2d[:, 3 * ledof:]      # (NC, cdof)

#     # (NF,2,cdof) -> (NF,2*cdof)
#     cell_dofs = c2d_cell[cells].reshape(cells.shape[0], -1)

#     # (NF, 2*cdof + ledof)
#     all_dofs = bm.concatenate([cell_dofs, e2d], axis=1)

#     # 每行去重（输出 ragged list）
#     patch_dofs = [bm.unique(row) for row in all_dofs]
#     return patch_dofs


def build_vertex_patch_dofs(space):
    mesh = space.mesh

    node2cell = mesh.node_to_cell()
    node2edge = mesh.node_to_edge()

    cell2dof = space.dof.cell_to_dof()
    edge2dof = space.dof.edge_to_dof()

    ledof = space.number_of_local_dofs(doftype='edge')
    cell2dof = cell2dof[:, 3 * ledof:]  

    NN = node2cell.shape[0]
    patch_dofs = []
    for v in range(NN):
        cells = node2cell.indices[node2cell.indptr[v]:node2cell.indptr[v+1]]   # adj cells of node v
        edges = node2edge.indices[node2edge.indptr[v]:node2edge.indptr[v+1]]   # adj edges of node v

        dofs_from_cells = cell2dof[cells].ravel() 
        dofs_from_edges = edge2dof[edges].ravel() 

        patch = bm.unique(bm.concatenate((dofs_from_cells, dofs_from_edges)))
        patch_dofs.append(patch)

    return patch_dofs


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
    
    # def vertex_preconditioner(self):
    #     """
    #     Vertex-based additive Schwarz preconditioner
    #     """
    #     A     = self.A
    #     space = self.space

    #     NN = space.mesh.number_of_nodes()
    #     NF = space.mesh.number_of_faces()

    #     patch_dofs = build_face_patch_dofs(space)

    #     A_diag = []
    #     for v,dofs in enumerate(patch_dofs):
    #     # 取子矩阵 A_sub = A[dofs, dofs]
    #         print(dofs)
    #         A_sub = A[dofs, :][:, dofs]   # A 是 scipy.sparse 矩阵时可用
    #         A_sub_dense = A_sub.toarray()

    #         invA = bm.linalg.inv(A_sub_dense)  # 小块才建议直接求逆
    #         A_diag.append((invA, dofs))

    #     def preconditioner(r):
    #         r = r.astype(bm.float64)
    #         e = bm.zeros_like(r)
    #         for i in range(NN):
    #             A_i, dofs = A_diag[i]
    #             e[dofs] += A_i @ r[dofs]
    #         return e
    #     return LinearOperator(shape=A.shape, matvec=preconditioner, rmatvec=preconditioner)

    def vertex_preconditioner(self):
        """
        Vertex-based additive Schwarz preconditioner
        """
        A     = self.A
        space = self.space

        p  = space.p
        TD = space.mesh.TD
        NN = space.mesh.number_of_nodes()
        NF = space.mesh.number_of_faces()
        gdof = space.number_of_global_dofs()

        c2d = space.dof.cell_to_dof()
        c2n = space.mesh.entity("cell")

        patch_dofs = build_vertex_patch_dofs(space)

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

        space0 = RaviartThomasFESpace(space.mesh, p=0)
        PI = self.projection(space0)
        print(PI)

        A0 = ((PI.T)@A@PI)
        mask = bm.abs(A0.data) < 1e-15
        A0.data[mask] = 0
        A0.eliminate_zeros()

        P1 = pyamg.smoothed_aggregation_solver(A0.tocsr())
        P0 = self.vertex_preconditioner()

        def mix_preconditioner(r1, P0, P1, PI, A):
            r1  = r1.astype(bm.float64)
            e1  = PI@P1.solve(PI.T@r1, maxiter=1, cycle='V')
            print
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




pde = PDEModelManager('darcyforchheimer').get_example(9)
mesh = pde.init_mesh['uniform_tri'](nx=8, ny=8)
p = 10
q = p+3
unit = mesh.edge_unit_normal()
uspace = RaviartThomasFESpace(mesh, p=p)
u_bform = BilinearForm(uspace)
Mu = ScalarMassIntegrator(coef=1,q=q)
u_bform.add_integrator(Mu)
u_bform.add_integrator(DivIntegrator2(coef=1,q=q))

M = u_bform.assembly().to_scipy()

ulform = LinearForm(uspace)
ulform.add_integrator(ScalarSourceIntegrator(pde.f, q=q))
b = bm.random.rand(M.shape[0]) 
F = M@b

Solver = ASPForPoisson(uspace, M)
b = Solver.solve(F)
# ml = GAMGSolver(isolver='MG', ptype='V', sstep=3, theta=0.25)
# ml.setup(M)
# e_pi_combined, info = ml.solve(F)
# print(info)