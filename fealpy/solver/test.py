from fealpy.functionspace import functionspace,TensorFunctionSpace,LagrangeFESpace
from fealpy.fem import ScalarMassIntegrator
from fealpy.mesh import TriangleMesh
from fealpy.model import PDEModelManager
from fealpy.fem import BilinearForm
from fealpy.fem import ScalarMassIntegrator
from fealpy.backend import backend_manager as bm
from fealpy.sparse import csr_matrix
pde = PDEModelManager('darcyforchheimer').get_example(1)
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=5, ny=5)
space = LagrangeFESpace(mesh, p=2,ctype='D')
uspace = TensorFunctionSpace(space,(-1,2))
GDOF = uspace.number_of_global_dofs()
Nt = mesh.number_of_cells()
cm = mesh.entity_measure('cell')
qf = mesh.quadrature_formula(3,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cell2dof = uspace.cell_to_dof()
print(bcs.shape)
u0 = uspace.function()
u0[:] = uspace.interpolate(pde.velocity)
u = u0(bcs)
print(u.shape)

# print(u.shape)
# u0[:] = uspace.interpolate(pde.velocity)

phi = uspace.basis(bcs)
fphi = bm.einsum('cqd,cqid->cqi', u, phi)

print(fphi.shape)
coef = pde.beta/bm.sqrt(bm.sum(u0(bcs) ** 2, axis=-1))
print("coef.shape:", coef.shape)

K = bm.einsum('q,c,cqi,cqj,cq->cij', ws, cm, fphi, fphi, coef)
print("K.shape:", K.shape)
coef2 = pde.mu + pde.beta * bm.sqrt(bm.sum(u0(bcs) ** 2, axis=-1))
S = bm.einsum('q,c,cqid,cqjd,cq->cij', ws, cm, phi, phi, coef2)

J = K + S
R = bm.zeros_like(J)
for i in range(Nt):
    R[i] = bm.linalg.inv(J[i])
I = bm.broadcast_to(cell2dof[:, :, None], shape=K.shape)
J = bm.broadcast_to(cell2dof[:, None, :], shape=K.shape)
M = csr_matrix((K.ravel() + S.ravel(), (I.ravel(), J.ravel())), 
               shape=(GDOF, GDOF))
print(M.to_dense()[:20, :20])


def J(u0):

    ux = u0[0::2]    # shape (Nt,)
    uy = u0[1::2]    # shape (Nt,)

    r = bm.sqrt(ux * ux + uy * uy)    # (Nt,)
    s = pde.mu + pde.beta * r                     # (Nt,)

    a11 = (s + pde.beta * ux * ux / r)*cm           # (Nt,)
    a22 = (s + pde.beta * uy * uy / r)*cm           # (Nt,)
    a12 = (pde.beta * ux * uy / r)*cm               # (Nt,)
    # a11 = ux * ux *cm /r            # (Nt,)
    # a22 = uy * uy *cm /r         # (Nt,)
    # a12 = ux * uy *cm /r         # (Nt,)

    idx = bm.arange(Nt, dtype=bm.int64)   # (Nt,)

    rows = bm.concatenate([2 * idx, 2 * idx, 2 * idx+1, 2 * idx+1])
    cols = bm.concatenate([2 * idx, 2 * idx+1, 2 * idx, 2 * idx+1])
    data = bm.concatenate([a11, a12, a12, a22])
    J = csr_matrix((data, (rows, cols)), shape=(GDOF, GDOF))

    return J
# J = J(u0)
# print(J.to_dense()[:10, :10])

