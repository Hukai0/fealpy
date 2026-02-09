# time_manufactured_2d.py
from typing import Optional
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...mesher import BoxMesher2d
import sympy as sp

class Exp0003(BoxMesher2d):

    def __init__(self, options: dict = {}):
        super().__init__(box=[0.0, 1.0, 0.0, 1.0])
        self.manager = bm._backends  

        x1, x2, t = sp.symbols('x1 x2 t', real=True)
        self.x1 = x1
        self.x2 = x2
        self.t = t

        phi = (1 - x1)**2 * (1 - x2)**2 * x1**2 * x2**2
        self.y = phi * self.t                 # y(x,t)
        phi2 = (x1*(1 - x1)*x2*(1 - x2))**2
        self.z = phi2 * ( 1 - self.t)                     # z(x,t)

        self.u = - self.z                               # u(x,t) = -z

        self.p0 = -2 * x1 * (1 - x1) * x2**2 * (1 - 2 * x1)*(1-x2)**2 * self.t 
        self.p1 =  -2 * x2 * (1 - x2) * x1**2 * (1 - 2 * x2)*(1-x1)**2 * self.t 
        self.p = sp.Matrix([self.p0, self.p1])

        self.q0 = -(1 - 2*x1) * x2 *(1- x2) * (1-self.t) 
        self.q1 = -(1 - 2*x2) * x1 *(1- x1) * (1-self.t)
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
        func = sp.lambdify([self.x1, self.x2, self.t], self.y, self.manager)
        x = space[..., 0]; y = space[..., 1]
        return func(x, y, time)

    @cartesian
    def y_t_solution(self, space, time):

        result = sp.lambdify([self.x1,self.x2,self.t], sp.diff(self.y, self.t), self.manager)
        return result(space[...,0], space[...,1], time)
    

    @cartesian
    def z_solution(self, space, time):
        """ z(x,t) """
        func = sp.lambdify([self.x1, self.x2, self.t], self.z, self.manager)
        x = space[..., 0]; y = space[..., 1]
        return func(x, y, time)

    @cartesian
    def z_t_solution(self, space, time):
        """ z_t(x,t) """
        result = sp.lambdify([self.x1,self.x2,self.t], sp.diff(self.z, self.t), self.manager)
        return result(space[...,0], space[...,1], time)

    @cartesian
    def u_solution(self, space, time):
        """ u(x,t) """
        func = sp.lambdify([self.x1, self.x2, self.t], self.u, self.manager)
        x = space[..., 0]; y = space[..., 1]
        return func(x, y, time)

    @cartesian
    def p_solution(self, space, time):
        """ p(x,t) 返回 shape (...,2) """
        p0 = sp.lambdify([self.x1, self.x2, self.t], self.p0, self.manager)
        p1 = sp.lambdify([self.x1, self.x2, self.t], self.p1, self.manager)
        x = space[..., 0]; y = space[..., 1]
        result = bm.zeros_like(space)
        result[..., 0] = p0(x, y, time)
        result[..., 1] = p1(x, y, time)
        return result

    @cartesian
    def q_solution(self, space, time):
        """ q(x,t) 返回 shape (...,2) """
        q0 = sp.lambdify([self.x1, self.x2, self.t], self.q0, 'numpy')
        q1 = sp.lambdify([self.x1, self.x2, self.t], self.q1, 'numpy')
        x = space[..., 0]; y = space[..., 1]
        result = bm.zeros_like(space)
        result[..., 0] = q0(x, y, time)
        result[..., 1] = q1(x, y, time)
        return result

    @cartesian
    def f_fun(self, space, time):
        """ source f(x,t) """
        self.div_p = sp.diff(self.p0, self.x1) + sp.diff(self.p1, self.x2)
        self.y_t = sp.diff(self.y, self.t, 1)
        self.f = self.y_t + self.div_p - self.u
        func = sp.lambdify([self.x1, self.x2, self.t], self.f, self.manager)
        x = space[..., 0]; y = space[..., 1]
        return func(x, y, time)

    @cartesian
    def pd_fun(self, space, time):
        """ desired flux p_d(x,t) """
        return self.p_solution(space, time)

    @cartesian
    def yd_fun(self, space, time):
        """ desired state y_d(x,t) """
        self.z_t = sp.diff(self.z, self.t, 1)
        self.div_q = sp.diff(self.q0, self.x1) + sp.diff(self.q1, self.x2)
        self.yd = self.y + self.z_t - self.div_q
        func = sp.lambdify([self.x1, self.x2, self.t], self.yd, 'numpy')
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
