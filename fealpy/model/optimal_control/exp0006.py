from typing import Optional
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...mesher import BoxMesher2d
import sympy as sp

class Exp0006(BoxMesher2d):

    def __init__(self, options: dict = {}):
        super().__init__(box=[0.0, 1.0, 0.0, 1.0])
        # backend manager if needed for numeric ops later
        self.manager = bm._backends

        # 符号变量
        x1, x2, t = sp.symbols('x1 x2 t', real=True)
        self.x1, self.x2, self.t = x1, x2, t

        # 参数（例题中给定）
        xi2 = sp.Integer(1)
        xi1 = sp.Rational(1, 2) 
        # xi2 = sp.Integer(1)           
            

        # 基本基函数 phi
        phi = x1**2 * (1 - x1)**2 * x2**2 * (1 - x2)**2

        # 按 Example 定义 y, z, u
        self.y = t * phi
        self.z = -200 * (1 - t) * phi
        # u = max{ xi1, min(xi2, -z) }
        self.u = sp.Max(xi1, sp.Min(xi2, -self.z))


        self.p0 = -2 * t * x1 * (1 - x1) * (1 - 2*x1) * x2**2 * (1 - x2)**2
        self.p1 = -2 * t * x1**2 * (1 - x1)**2 * x2 * (1 - x2) * (1 - 2*x2)

        self.q0 = 400 * (1 - t) * x1 * (1 - x1) * (1 - 2*x1) * x2**2 * (1 - x2)**2
        self.q1 = 400 * (1 - t) * x1**2 * (1 - x1)**2 * x2 * (1 - x2) * (1 - 2*x2)

        self.p = sp.Matrix([self.p0, self.p1])
        self.q = sp.Matrix([self.q0, self.q1])

    # geometry / time interval
    def geo_dimension(self) -> int:
        return 2

    def domain(self):
        return [0.0, 1.0, 0.0, 1.0]

    def duration(self):
        return [0.0, 1.0]


    @cartesian
    def y_solution(self, space, time):
        """ y(x,t) """
        x1 = self.x1
        x2 = self.x2
        t = self.t
        func = sp.lambdify([x1, x2, t], self.y, "numpy")
        x = space[..., 0]; y = space[..., 1]
        return func(x, y, time)

    @cartesian
    def y_t_solution(self, space, time):
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1, x2, t], sp.diff(self.y, t), self.manager)
        return result(space[...,0], space[...,1], time)
    

    @cartesian
    def z_solution(self, space, time):
        """ z(x,t) """
        x1 = self.x1
        x2 = self.x2
        t = self.t
        func = sp.lambdify([x1, x2, t], self.z, "numpy")
        x = space[..., 0]; y = space[..., 1]
        return func(x, y, time)

    @cartesian
    def z_t_solution(self, space, time):
        """ z_t(x,t) """
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1, x2, t], sp.diff(self.z, t), self.manager)
        return result(space[...,0], space[...,1], time)

    @cartesian
    def u_solution(self, space, time):
        """ u(x,t) """
        x1 = self.x1
        x2 = self.x2
        t = self.t
        func = sp.lambdify([x1, x2, t], self.u, "numpy")
        x = space[..., 0]; y = space[..., 1]
        return func(x, y, time)

    @cartesian
    def p_solution(self, space, time):
        """ p(x,t) 返回 shape (...,2) """
        x1 = self.x1
        x2 = self.x2
        t = self.t
        p0 = sp.lambdify([x1, x2, t], self.p0, "numpy")
        p1 = sp.lambdify([x1, x2, t], self.p1, "numpy")
        x = space[..., 0]; y = space[..., 1]
        result = bm.zeros_like(space)
        result[..., 0] = p0(x, y, time)
        result[..., 1] = p1(x, y, time)
        return result

    @cartesian
    def q_solution(self, space, time):
        """ q(x,t) 返回 shape (...,2) """
        x1 = self.x1
        x2 = self.x2
        t = self.t
        q0 = sp.lambdify([x1, x2, t], self.q0, 'numpy')
        q1 = sp.lambdify([x1, x2, t], self.q1, 'numpy')
        x = space[..., 0]; y = space[..., 1]
        result = bm.zeros_like(space)
        result[..., 0] = q0(x, y, time)
        result[..., 1] = q1(x, y, time)
        return result

    @cartesian
    def f_fun(self, space, time):
        """ source f(x,t) """
        x1 = self.x1
        x2 = self.x2
        t = self.t
        self.div_p = sp.diff(self.p0, x1) + sp.diff(self.p1, x2)
        self.y_t = sp.diff(self.y, t)
        self.f = self.y_t + self.div_p - self.u
        func = sp.lambdify([x1, x2, t], self.f, "numpy")
        x = space[..., 0]; y = space[..., 1]
        return func(x, y, time)

    @cartesian
    def pd_fun(self, space, time):
        """ desired flux p_d(x,t) """
        x1 = self.x1
        x2 = self.x2
        t = self.t
        self.y_t = sp.diff(self.y, t)
        self.grad_z = sp.Matrix([sp.diff(self.z, x1), sp.diff(self.z, x2)])
        p_plus = self.p + self.q + self.grad_z
        pd0 = sp.lambdify([x1, x2, t], p_plus[0], 'numpy')
        pd1 = sp.lambdify([x1, x2, t], p_plus[1], 'numpy')
        x = space[..., 0]; y = space[..., 1]
        result = bm.zeros(space.shape[:-1] + (2,))
        result[..., 0] = pd0(x, y, time)
        result[..., 1] = pd1(x, y, time)
        return result

    @cartesian
    def yd_fun(self, space, time):
        """ desired state y_d(x,t) """
        x1 = self.x1
        x2 = self.x2
        t = self.t
        self.z_t = sp.diff(self.z, t)
        self.div_q = sp.diff(self.q0, x1) + sp.diff(self.q1, x2)
        self.yd = self.y + self.z_t - self.div_q
        func = sp.lambdify([x1, x2, t], self.yd, 'numpy')
        x = space[..., 0]; y = space[..., 1]
        return func(x, y, time)


    @cartesian
    def grad_dirichlet(self, p, space):
        return bm.zeros_like(p[..., 0])
    
    @cartesian
    def A_inverse(self, space):

        result = bm.zeros(space.shape[:-1]+(2,2)) 
        result[..., 0, 0] = 1
        result[..., 1, 1] = 1
        return result 
