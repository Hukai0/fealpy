from typing import Optional
from scipy.special import factorial

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import (
    CellOperatorIntegrator,
    enable_cache,
    assemblymethod,
    CoefLike
)

class gradmIntegrator(CellOperatorIntegrator):
    r"""The diffusion integrator for function spaces based on homogeneous meshes."""
    def __init__(self, m: int=2,coef: Optional[CoefLike]=None, q: int=3, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.m = m
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        m = self.m
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The gradmIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__}is"
                               "not a subclass of HomoMesh.")

        import ipdb
        ipdb.set_trace()
        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gmphi = space.grad_m_basis(bcs, m=m)
        return bcs, ws, gmphi, cm, index

    def assembly(self, space: _FS) -> TensorLike:
        m = self.m
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        GD = mesh.geo_dimension()
        idx = mesh.multi_index_matrix(m, GD-1)
        num = factorial(m)/bm.prod(factorial(idx),axis=1)
        bcs, ws, gmphi, cm, index = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell',
                                 index=index)
        M = bm.einsum('cqlg,cqmg,g,q,c->clm',gmphi,gmphi,num,ws,cm)
        return M
    #return bilinear_integral(gmphi1, gmphi, ws, cm, coef,
    #                             batched=self.batched)

