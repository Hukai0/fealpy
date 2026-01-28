from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import BoxMesher2d
from ...decorator import cartesian


# class Exp0001(BoxMesher2d):
#     """
#     2D Maxwell-type problem with complex Robin boundary condition:

#         curl(curl(E)) - k^2 * E = f   in Ω = (0, 1)^2
#         curl(E) cross n - i*k*E_t = g on ∂Ω

#     Exact solution:
#         E(x, y) = [x*y*(1 - x)*(1 - y), sin(πx) * sin(πy)]

#     Boundary condition mimics absorbing (impedance-type) boundary.
#     """

#     def __init__(self, options: dict = {}):
#         self.box = [0.0, 1.0, 0.0, 1.0]
#         super().__init__(box=self.box)
#         self.k = options.get('k', 1.0)

#     def geo_dimension(self) -> int:
#         return 2

#     def domain(self) -> Sequence[float]:
#         return self.box

#     @cartesian
#     def solution(self, p: TensorLike) -> TensorLike:
#         x = p[..., 0, None]
#         y = p[..., 1, None]
#         Fx = x*y*(1 - x)*(1 - y)
#         Fy = bm.sin(bm.pi*x)*bm.sin(bm.pi*y)
#         f = bm.concatenate([Fx, Fy], axis=-1) 
#         return f 

#     @cartesian
#     def curl(self, p: TensorLike) -> TensorLike:
#         x = p[..., 0]
#         y = p[..., 1]
#         pi = bm.pi
#         sin = bm.sin
#         cos = bm.cos

#         curlF = -x*y*(x - 1) - x*(1 - x)*(1 - y) + pi*sin(pi*y)*cos(pi*x)
#         return curlF
    
#     @cartesian
#     def curl_curl_solution(self, p):
#         x = p[..., 0, None]
#         y = p[..., 1, None]
#         pi = bm.pi
#         sin = bm.sin
#         cos = bm.cos

#         ccFx = 2*x*(1 - x) + pi**2*cos(pi*x)*cos(pi*y)
#         ccFy = x*y - x*(1 - y) - y*(1 - x) - (1 - y)*(x - 1) + pi**2*sin(pi*x)*sin(pi*y)
#         ccf = bm.concatenate([ccFx, ccFy] , axis=-1)
#         return ccf

#     @cartesian
#     def source(self, p):
#         return self.curl_curl_solution(p) - self.k** 2 * self.solution(p)

#     @cartesian
#     def robin(self, p, n):
#         t = bm.flip(n , axis=-1).copy()
#         t[:, 0] = -t[:, 0]
#         t = t[:, None, :]
#         a = self.curl(p)[..., None] * t
#         b = 1 * self.k * bm.einsum("eqd,eqd->eq", self.solution(p), t)[..., None] * t
#         return a - b

#     @cartesian
#     def is_robin_boundary(self, p: TensorLike) -> TensorLike:
#         x, y = p[..., 0], p[..., 1]
#         atol = 1e-12
#         return (
#             (bm.abs(x - 0.0) < atol) |
#             (bm.abs(x - 1.0) < atol) |
#             (bm.abs(y - 0.0) < atol) |
#             (bm.abs(y - 1.0) < atol)
#         )

class Exp0001(BoxMesher2d):
    """
    2D Maxwell-type problem with the curl equation:
    
        curl(E) = J   in Ω = (0, 1)^2
        E = 0 on ∂Ω (Dirichlet boundary condition)

    Exact solution:
        E(x, y) = [sin(πx) * sin(πy), 0]  (you can adapt to your own exact solution)

    Boundary condition: E = 0 on all boundaries (Dirichlet).
    """

    def __init__(self, options: dict = {}):
        self.box = [0.0, 1.0, 0.0, 1.0]  # Domain [0, 1] x [0, 1]
        super().__init__(box=self.box)
        self.pi = bm.pi

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """
        Exact solution for E(x, y), here we assume E_x = sin(πx) * sin(πy), and E_y = 0.
        """
        x = p[..., 0, None]
        y = p[..., 1, None]
        Fx = bm.sin(self.pi * x) * bm.sin(self.pi * y)  # E_x component
        Fy = bm.zeros_like(Fx)  # E_y component (0 in this example)
        f = bm.concatenate([Fx, Fy], axis=-1)
        return f

    @cartesian
    def curl(self, p: TensorLike) -> TensorLike:
        """
        The curl of E (in 2D, this is just the scalar component representing the rotation).
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = self.pi
        curlE = pi * bm.cos(pi * x) * bm.sin(pi * y)  # curl(E) = J (which is sin(pi * x) * sin(pi * y))
        return curlE

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """
        The source term J = sin(pi * x) * sin(pi * y.
        """
        x = p[..., 0]
        y = p[..., 1]
        return bm.sin(self.pi * x) * bm.sin(self.pi * y)

    @cartesian
    def robin(self, p, n):
        """
        The Robin boundary condition is not used in this problem, as it's Dirichlet.
        But we include this method for future extensibility.
        """
        return bm.zeros_like(p)  # As E = 0 on boundary, the Robin condition is trivial here.

    @cartesian
    def is_robin_boundary(self, p: TensorLike) -> TensorLike:
        """
        Here we define the boundary as where x or y is 0 or 1 (i.e., the Dirichlet boundary).
        """
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        return (
            (bm.abs(x - 0.0) < atol) |
            (bm.abs(x - 1.0) < atol) |
            (bm.abs(y - 0.0) < atol) |
            (bm.abs(y - 1.0) < atol)
        )
