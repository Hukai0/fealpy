# time_manufactured_2d.py
from typing import Optional
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...mesher import BoxMesher2d
import sympy as sp

class Exp0005(BoxMesher2d):

    def __init__(self, options: dict = {}):
        super().__init__(box=[0.0, 1.0, 0.0, 1.0])
        self.manager = bm._backends  

        # 符号变量
        x1, x2, t = sp.symbols('x1 x2 t', real=True)
        self.x1 = x1
        self.x2 = x2
        self.t = t

        # 基本表达式
        self.y = sp.cos(sp.pi * x1) * sp.cos(sp.pi * x2) * sp.exp(t)                # y(x,t)
        self.z =  sp.cos(2*sp.pi * x1) * sp.cos(2*sp.pi * x2) * (1 - t)                  # z(x,t)
        # self.u = -self.z                     # z(x,t)
        self.u = sp.Max(0, -self.z)                    # z(x,t)
        # self.u = bm.maximum(1, bm.minimum(3 ,-self.z))                            # u(x,t) = -z

        # flux p 
        self.p0 = sp.pi * sp.sin(sp.pi * x1) * sp.cos(sp.pi * x2) * sp.exp(t)
        self.p1 = sp.pi * sp.cos(sp.pi * x1) * sp.sin(sp.pi * x2) * sp.exp(t)
        self.p = sp.Matrix([self.p0, self.p1])

        # adjoint flux q
        self.q0 = 2*sp.pi * sp.sin(2*sp.pi * x1) * sp.cos(2*sp.pi * x2) * (1 - t)
        self.q1 = 2*sp.pi * sp.cos(2*sp.pi * x1) * sp.sin(2*sp.pi * x2) * (1 - t)
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
