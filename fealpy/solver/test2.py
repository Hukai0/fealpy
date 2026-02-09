from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.model import PDEModelManager
pde = PDEModelManager('poisson').get_example(2)
print(pde.__doc__)

from fealpy.mesh import TriangleMesh
domain = pde.domain()
mesh = TriangleMesh.from_box(domain, nx=2, ny=2) 
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
NE = mesh.number_of_edges()

from fealpy.functionspace import LagrangeFESpace
p = 0
space = LagrangeFESpace(mesh, p, ctype='D')
cell2dof = space.cell_to_dof()
print(cell2dof)

from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import BilinearForm, LinearForm
bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator(q=3))
A = bform.assembly()

lform = LinearForm(space)
lform.add_integrator(ScalarSourceIntegrator(pde.source, q=3))
F = lform.assembly()