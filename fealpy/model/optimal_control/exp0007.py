from typing import Optional
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...mesher import BoxMesher2d
import sympy as sp

class Exp0007(BoxMesher2d):

    def __init__(self, options: dict = {}):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(box=self.box)

        self.manager = bm._backends

        # symbolic variables
        x1, x2 = sp.symbols('x1 x2', real=True)
        self.x1 = x1
        self.x2 = x2

        # coefficients and prescribed functions
        self.a_sym = sp.exp(x1 + x2)                      # a(x) = e^{x1+x2}
        self.c_sym = 1 + x1**2 + x2**2                    # c(x)

    
        self.z = sp.cos(2*sp.pi*x1) * sp.cos(2*sp.pi*x2) # z(x)
        self.y = sp.cos(sp.pi*x1) * sp.cos(sp.pi*x2)      # y(x)


        self.u = -self.z                      # u(x)

        # fluxes: p = -a * grad(y), q = -a * grad(z)
        dy_dx1 = sp.diff(self.y, x1)
        dy_dx2 = sp.diff(self.y, x2)

        self.p0 = - self.a_sym * dy_dx1
        self.p1 = - self.a_sym * dy_dx2
        self.p = sp.Matrix([self.p0, self.p1])

        dz_dx1 = sp.diff(self.z, x1)
        dz_dx2 = sp.diff(self.z, x2)
        self.q0 = - self.a_sym * dz_dx1
        self.q1 = - self.a_sym * dz_dx2
        self.q = sp.Matrix([self.q0, self.q1])

        # divergence of p and q
        self.div_p = sp.diff(self.p0, x1) + sp.diff(self.p1, x2)
        self.div_q = sp.diff(self.q0, x1) + sp.diff(self.q1, x2)

        # source term f from: -div(a grad y) + c*y = f + u  =>  f = div(p) + c*y - u
        self.f_sym = self.div_p + self.c_sym * self.y - self.u

        # desired state y_d (you can change this; here set equal to y for testing)
        self.y_d = self.y-self.c_sym*self.z - self.div_q

        # 系数矩阵
        self.A00 = self.a_sym
        self.A11 =self.a_sym
        self.A = sp.Matrix([[self.A00, 0], [0, self.A11]])

        


    def geo_dimension(self) -> int:
        return 2

    def domain(self):
        return [0, 1, 0, 1]

    @cartesian
    def y_solution(self, space):
        """Exact state y at points in `space`."""
        f = sp.lambdify([self.x1, self.x2], self.y, 'numpy')
        return f(space[..., 0], space[..., 1])

    @cartesian
    def z_solution(self, space):
        """Adjoint state z at points in `space`."""
        f = sp.lambdify([self.x1, self.x2], self.z, 'numpy')
        return f(space[..., 0], space[..., 1])

    @cartesian
    def u_solution(self, space):
        """Control u at points in `space`."""
        f = sp.lambdify([self.x1, self.x2], self.u, 'numpy')
        return f(space[..., 0], space[..., 1])

    @cartesian
    def p_solution(self, space):
        """Flux p = -a * grad(y)."""
        x = space[..., 0]; y = space[..., 1]
        result = bm.zeros_like(space)
        p0_fun = sp.lambdify([self.x1, self.x2], self.p0, 'numpy')
        p1_fun = sp.lambdify([self.x1, self.x2], self.p1, 'numpy')
        result[..., 0] = p0_fun(x, y)
        result[..., 1] = p1_fun(x, y)
        return result

    @cartesian
    def q_solution(self, space):
        """Adjoint flux q = -a * grad(z)."""
        x = space[..., 0]; y = space[..., 1]
        result = bm.zeros_like(space)
        q0_fun = sp.lambdify([self.x1, self.x2], self.q0, 'numpy')
        q1_fun = sp.lambdify([self.x1, self.x2], self.q1, 'numpy')
        result[..., 0] = q0_fun(x, y)
        result[..., 1] = q1_fun(x, y)
        return result
    
    @cartesian
    def A_matirx(self, space): 
        """ Compute the coefficient matrix A at given points in space."""
        x = space[..., 0]
        y = space[..., 1]
        result = bm.zeros(space.shape[:-1]+(2,2)) 
        result[..., 0, 0] = bm.exp(x + y)
        result[..., 1, 1] = bm.exp(x + y)
        return result 
    
    @cartesian
    def A_inverse(self, space):
        """ Compute the inverse of the coefficient matrix A at given points in space."""
        x = space[..., 0]
        y = space[..., 1]
        result = bm.zeros(space.shape[:-1]+(2,2)) 
        result[..., 0, 0] = 1/(bm.exp(x + y))
        result[..., 1, 1] = 1/(bm.exp(x + y))
        return result 


    @cartesian
    def f_fun(self, space, index=None):
        """Right-hand side f at points in `space`."""
        f = sp.lambdify([self.x1, self.x2], self.f_sym, 'numpy')
        return f(space[..., 0], space[..., 1])

    @cartesian
    def pd_fun(self, space):
        """Return p (for prescribing p_d = p)."""
        return self.p_solution(space)

    @cartesian
    def div_pd_fun(self, space):
        """Divergence of p."""
        f = sp.lambdify([self.x1, self.x2], self.div_p, self.manager)
        return f(space[..., 0], space[..., 1])

    @cartesian
    def yd_fun(self, space):
        """Desired state y_d evaluated at points."""
        f = sp.lambdify([self.x1, self.x2], self.y_d, "numpy")
        return f(space[..., 0], space[..., 1])

    @cartesian
    def grad_dirichlet(self, p, space):
        """Placeholder for Neumann/Dirichlet gradient handling — returns zero."""
        return bm.zeros_like(p[..., 0])
    
    @cartesian
    def C_matrix(self, space):
        """Control u at points in `space`."""
        f = sp.lambdify([self.x1, self.x2], self.c_sym, 'numpy')
        return f(space[..., 0], space[..., 1])
