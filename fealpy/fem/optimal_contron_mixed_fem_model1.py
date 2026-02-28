
from typing import Optional,Union
from fealpy.model import ComputationalModel
from fealpy.model.optimal_control import OPCPDEDataT
from fealpy.fem import DirichletBC  
from fealpy.functionspace import RaviartThomasFESpace2d
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import ScalarMassIntegrator, ScalarSourceIntegrator
from fealpy.fem import DivIntegrator
from fealpy.fem import OPCSIntegrator,OPCIntegrator
from fealpy.fem import BlockForm
from fealpy.backend import backend_manager as bm
from fealpy.solver import spsolve
from fealpy.model import PDEModelManager
from fealpy.decorator import variantmethod
from fealpy.mesh import Mesh
from functools import partial
from fealpy.decorator import barycentric, cartesian
from fealpy.solver import spsolve
from fealpy.utils import timer

class TimeOPCMixedFEMModel(ComputationalModel):
    """
    OPCRTFEMModel: Optimal Control Problem Raviart-Thomas Finite Element Model

    This class implements a 2D optimal control PDE solver using the Raviart-Thomas finite element method (RT FEM).
    It is designed for optimal control problems that require accurate flux approximation and state-control coupling.
    The model supports custom optimal control PDE data, mesh handling, linear system assembly, boundary condition application,
    solution routines, and error analysis.

    Parameters
    mesh : TriangleMesh
        The 2D triangle mesh object used for finite element discretization.
    c : float
        The regularization parameter or control parameter for the optimal control problem.
    pde : OPCPDEDataT, optional, default=None
        The optimal control PDE data object, including coefficients, source terms, and boundary conditions.
        If None, a built-in example problem is used.

    Attributes
    pde : OPCPDEDataT
        The current optimal control PDE data object, including coefficients, source terms, and boundary conditions.
    mesh : TriangleMesh
        The current 2D triangle mesh object used for finite element discretization.

    Methods
    run()
        Execute the FEM solution process and return the numerical solutions of the state and control variables.
    space()
        Return the finite element spaces used in the model.
    linear_system()
        Assemble the linear system (stiffness matrix and right-hand side) for the optimal control RT FEM.
    boundary_apply()
        Apply boundary conditions to the linear system.
    solve()
        Solve the linear system and return the FEM solutions of the state and control variables.
    show_mesh()
        Visualize the current mesh structure.
    L2_error()
        Compute the L2 error between the numerical and exact solutions.
    max_error()
        Compute the maximum error between the numerical and exact solutions.

    Notes
    This class uses a mixed finite element method (Raviart-Thomas space and piecewise constant space),
    suitable for optimal control problems with flux continuity and state-control coupling.
    It supports automatic boundary condition handling and error evaluation for algorithm verification and numerical experiments.

    Examples
    >>> model = OPCRTFEMModel(mesh, c)
    >>> p, u = model.run()
    >>> error_y, error_p = model.L2_error(y, p, exact_y, exact_p)
    >>> model.show_mesh()
    """

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'])
        self.set_order(options['space_degree'])
        # self.solve.set(options['solve']) 
        self.op_type = options['op_type']
        self.t0, self.t1 = self.pde.duration()
        self.nt = 10
        self.tau = (self.t1 - self.t0) / self.nt


    def set_pde(self, pde: Union[OPCPDEDataT, str]="opc"):
        """
        Set the optimal control PDE data for the model.
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('optimal_control').get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_tri", **kwargs):
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else: 
            self.mesh = mesh
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")


    def set_order(self, p: int = 0):   
        """
        Set the polynomial degree for the Raviart-Thomas finite element space.
        Parameters
        p : int, optional
            The polynomial degree of the Raviart-Thomas space. Default is 0 (piecewise constant).
        """ 
        self.p = p    
        
    def space(self, p: Optional[int] = None):
        """
        Set the finite element spaces for the model.
        
        Parameters
        p : int, optional
            The polynomial degree of the Raviart-Thomas space. If None, use the default degree.
        """
        if p is None:
            p = self.p
        self.pspace = RaviartThomasFESpace2d(self.mesh, p=p)
        self.yspace = LagrangeFESpace(self.mesh, p=p, ctype='D')
        return self.pspace, self.yspace
    
    def assembly_Mass(self):
        """
        Assemble the linear system for the optimal control problem.
        """
        self.pspace,self.yspace = self.space()
        pLDOF = self.pspace.number_of_local_dofs()
        pGDOF = self.pspace.number_of_global_dofs()
        yLDOF = self.yspace.number_of_local_dofs()
        yGDOF = self.yspace.number_of_global_dofs()
        self.logger.info(f"Raviart-Thomas space: {self.pspace}, LDOF: {pLDOF}, GDOF: {pGDOF}")
        self.logger.info(f"Lagrange space: {self.yspace}, LDOF: {yLDOF}, GDOF: {yGDOF}")
        self.ph = self.pspace.function()
        self.yh = self.yspace.function()
        self.xh = bm.zeros((pGDOF + yGDOF,), dtype=bm.float64)
        
        bform1 = BilinearForm(self.pspace)
        bform1.add_integrator(OPCIntegrator(coef=self.pde.A_inverse,q=3))

        bform2 = BilinearForm((self.yspace,self.pspace))
        bform2.add_integrator(DivIntegrator(coef=-1, q=3))

        bform3 = BilinearForm((self.yspace,self.pspace))
        bform3.add_integrator(DivIntegrator(coef=self.tau, q=3))

        bform4 = BilinearForm(self.yspace)
        bform4.add_integrator(ScalarMassIntegrator(coef=1, q=3))
    
    
        M = BlockForm([[bform1,bform2],
                       [bform3.T,bform4]])
        M = M.assembly()
        
        return M

    def assembly_forward_b(self,y,fn,un = None):

        lform1 = LinearForm(self.pspace)
        lform2 = LinearForm(self.yspace)
        lform2.add_integrator(ScalarSourceIntegrator(source=fn))
        @barycentric
        def coef_y(bcs, index=None):
            result = y(bcs)
            return result
        lform2.add_integrator(ScalarSourceIntegrator(source=coef_y))
        if un is not None:
            @barycentric
            def source_u(bcs, index=None):
                result = self.tau*un(bcs)
                return result
            lform2.add_integrator(ScalarSourceIntegrator(source=source_u))

        F = lform2.assembly()
        G = lform1.assembly()
        b = bm.concatenate([G,F],axis=0)

        return b
    
    def assembly_backward_b(self,z,pn,pd,yn,yd):
        @barycentric
        def coef_z(bcs, index=None):
            result = z(bcs)
            return result
        lform1 = LinearForm(self.pspace)
        lform1.add_integrator(OPCSIntegrator(source=pn))
        lform1.add_integrator(OPCSIntegrator(source=pd))
        lform2 = LinearForm(self.yspace)
        lform2.add_integrator(ScalarSourceIntegrator(source=yn))
        lform2.add_integrator(ScalarSourceIntegrator(source=yd))
        lform2.add_integrator(ScalarSourceIntegrator(source=coef_z))
        F = lform2.assembly()
        G = lform1.assembly()
        F = F 
        b = bm.concatenate([G,F],axis=0)

        return b


    def apply_bc(self, A, b, gd):
        """
        Apply the boundary conditions to the linear system.
        """
        pspace, yspace = self.space()
        yGDOF = yspace.number_of_global_dofs()
        G_apply = pspace.set_neumann_bc(gd)
        F = bm.zeros(yGDOF, dtype=bm.float64)
        b_apply = bm.concatenate([G_apply,F],axis=0)
        b = b - b_apply
        return A, b

    def time_step(self, allu, allp, ally, allz, allq, A, it=0, tmr=None):
        pGDOF = self.pspace.number_of_global_dofs()

        for n in range(1, self.nt+1):
            tn = self.t0 + n * self.tau
            y = ally[n-1]

            @cartesian
            def fn(p, index=None):
                return self.tau*self.pde.f_fun(p, time=tn)

            b = self.assembly_forward_b(y, fn, allu[n])

            A_copy = A.copy()
            A_copy, b = self.apply_bc(A_copy, b, gd=partial(self.pde.y_solution, time=tn))
            self.xh[:] = spsolve(A_copy, b, solver='scipy')

            p2 = self.pspace.function()
            y2 = self.yspace.function()
            p2[:] = self.xh[:pGDOF]
            y2[:] = self.xh[pGDOF:]
            ally[n] = y2
            allp[n] = p2

        if tmr is not None:
            tmr.send(f'第{it}次迭代：正向求解时间')


        for i in bm.arange(self.nt-1, -1, -1):
            tn = self.t0 + i * self.tau
            y = ally[i+1]
            p = allp[i+1]
            z = allz[i+1]

            @barycentric
            def pn(bcs, index=None):
                return -p(bcs)

            @cartesian
            def pd(bcs, index=None):
                return self.pde.pd_fun(bcs, time=tn+self.tau)

            @barycentric
            def yn(bcs, index=None):
                return self.tau*y(bcs)

            @cartesian
            def yd(bcs, index=None):
                return -self.tau*self.pde.yd_fun(bcs, time=tn+self.tau)

            b = self.assembly_backward_b(z, pn, pd, yn, yd)

            A_copy = A.copy()
            A_copy, b = self.apply_bc(A_copy, b, gd=partial(self.pde.z_solution, time=tn+self.tau))
            self.xh[:] = spsolve(A_copy, b, solver='scipy')

            q2 = self.pspace.function()
            z2 = self.yspace.function()
            q2[:] = self.xh[:pGDOF]
            z2[:] = self.xh[pGDOF:]
            allz[i] = z2
            allq[i] = q2

        if tmr is not None:
            tmr.send(f'第{it}次迭代：反向求解时间')

        # -------- update control --------
        for i in bm.arange(1, self.nt+1):
            z = allz[i-1]
            ufunction = self.yspace.function()
            if self.op_type == 0:
                # print("op_type 0: u = max(0, -z)")
                ufunction[:] = bm.maximum(0, -z)
            elif self.op_type == 1:
                ufunction[:] = bm.maximum(0.5, bm.minimum(1,-z))
            allu[i] = ufunction
        
        if tmr is not None:
            tmr.send(f'第{it}次迭代：更新控制变量时间')

        return allu, allp, ally, allz, allq

    def run(self, tmr=None):
        maxit = 10
        allu = [None]*(self.nt+1)
        ally = [None]*(self.nt+1)
        allp = [None]*(self.nt+1)
        allz = [None]*(self.nt+1)
        allq = [None]*(self.nt+1)

        self.pspace, self.yspace = self.space()

        allz[-1] = self.yspace.function(self.yspace.interpolate(partial(self.pde.z_solution, time=self.t1)))
        ally[0]  = self.yspace.function(self.yspace.interpolate(partial(self.pde.y_solution, time=0)))
        allu[0]  = self.yspace.function(self.yspace.interpolate(partial(self.pde.u_solution, time=0)))
        allp[0]  = self.pspace.function(self.pspace.interpolation(partial(self.pde.p_solution, time=0)))
        allq[-1] = self.pspace.function(self.pspace.interpolation(partial(self.pde.q_solution, time=self.t1)))

        M = self.assembly_Mass()

        erroru0 = errorp0 = errory0 = errorz0 = errorq0 = 1e4

        for it in range(maxit):

            M_copy = M.copy()
            allu, allp, ally, allz, allq = self.time_step(allu, allp, ally, allz, allq, M_copy, it=it, tmr=tmr)
            erroru = bm.zeros(self.nt+1)
            errorp = bm.zeros(self.nt+1)
            errory = bm.zeros(self.nt+1)
            errorz = bm.zeros(self.nt+1)
            errorq = bm.zeros(self.nt+1)

            for i in range(1, self.nt):
                ti = i*self.tau
                erroru[i] = self.mesh.error(allu[i], partial(self.pde.u_solution, time=ti))
                errorp[i] = self.mesh.error(allp[i], partial(self.pde.p_solution, time=ti))
                errory[i] = self.mesh.error(ally[i], partial(self.pde.y_solution, time=ti))
                errorz[i] = self.mesh.error(allz[i], partial(self.pde.z_solution, time=ti))
                errorq[i] = self.mesh.error(allq[i], partial(self.pde.q_solution, time=ti))

            erroru1 = bm.max(erroru)
            errorp1 = bm.max(errorp)
            errory1 = bm.max(errory)
            errorz1 = bm.max(errorz)
            errorq1 = bm.max(errorq)
            
            tmr.send(f'第{it}次迭代：计算误差时间')

            if (bm.abs(errorp1 - errorp0) < 1e-10 and bm.abs(errorq1 - errorq0) < 1e-10 and
                bm.abs(errory1 - errory0) < 1e-10 and bm.abs(errorz1 - errorz0) < 1e-10 and
                bm.abs(erroru1 - erroru0) < 1e-10):
                self.logger.info(f"Convergence achieved at iteration {it+1}.")
                self.logger.info(f"p error: {errorp1}, q error: {errorq1}, y error: {errory1}, z error: {errorz1}, u error: {erroru1}")
                if self.nt == 160:
                    self.plot(allp, allq, allu, ally, allz, nt=80)
                return errorp1, errorq1, erroru1, errory1, errorz1

            erroru0, errorp0, errory0, errorz0, errorq0 = erroru1, errorp1, errory1, errorz1, errorq1

    def refine_run(self, reit=3):
        errorType = ['$|| p - p_h||_{L2}$ ',
                    '$|| q - q_h||_{L2}$ ',
                    '$|| u - u_h||_{L2}$ ',
                    '$|| y - y_h||_{L2}$ ',
                    '$|| z - z_h||_{L2}$ ']

        errorMatrix = bm.zeros((len(errorType), reit), dtype=bm.float64)

        for level in range(reit):
            tmr = timer()
            tmr_total = timer()
            next(tmr_total)
            next(tmr)
            tmr.send(f'===== level {level} 网格开始 =====')

            errorMatrix[:, level] = self.run(tmr=tmr)
            print(self.nt)

            tmr.send(f'===== level {level} 网格结束 =====')
            
            if level < reit - 1:
                self.mesh.uniform_refine(1)
                nt = self.nt * 2
                self.nt = nt
                self.tau = (self.t1 - self.t0) / self.nt
                tmr.send(f'===== level {level} 网格和时间加密时间 =====')
            next(tmr)
            tmr_total.send(f'===== level {level} 总时间 =====')
            next(tmr_total)
            
        return errorMatrix
    
    def show_p0(self,solution,title: str | None = None):
        """
        Visualize the mesh structure.
        """
        import functools
        if isinstance(solution, functools.partial):
             u1 = self.yspace.interpolate(solution)
        else:
            u1 = solution[:]
        node = self.mesh.entity('node')  # 节点坐标 (N_node, 2)
        cell = self.mesh.entity('cell')  # 单元 (N_cell, 3)
        # 假设 node, cell, u1 已经定义好
        num_nodes = len(node)
        node_values = bm.zeros(num_nodes)

        for i in range(len(cell)):
            for j in cell[i]:
                node_values[j] += u1[i]
                
        # 计算每个节点的平均值
        node_values /= bm.bincount(bm.concatenate(cell))
        from scipy.interpolate import griddata

        xi = bm.linspace(min(node[:, 0]), max(node[:, 0]))
        yi = bm.linspace(min(node[:, 1]), max(node[:, 1]))
        xi, yi = bm.meshgrid(xi, yi)

        zi = griddata((node[:, 0], node[:, 1]), node_values, (xi, yi), method='linear')
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xi, yi, zi, cmap='jet', linewidth=0, antialiased=False, edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_title(f'{title}')
        ax.xaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        ax.yaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        ax.zaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        # 设置整个图表背景为透明
        fig.colorbar(surf)
        plt.show()
    
    def show_rt(self, solution,title: str | None = None):
        """
        Visualize the mesh structure.
        """
        import matplotlib.pyplot as plt
        import functools
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        node = self.mesh.entity('node')
        edge = self.mesh.entity('edge')
        node_center = 0.5 * (node[edge[:, 0]] + node[edge[:, 1]])
        if isinstance(solution, functools.partial):
            p_solution = self.pspace.interpolation(solution)
        else:
            p_solution = solution[:]
        x = node_center[:, 0]
        y = node_center[:, 1]
        surf = plt.tricontourf(x, y, p_solution, levels=50, cmap='jet', edgecolor='none')
        ax.xaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        ax.yaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        ax.zaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        fig.colorbar(surf)  
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_title(f'{title} ')
        plt.show()
        
        
    def plot(self, allp, allq, allu, ally, allz, nt):
    
        ti = self.t0 + nt * self.tau
        tlabel = f"t={ti:.6g}"

        self.show_rt(partial(self.pde.p_solution, time=ti), title=f"p — True ({tlabel})")
        self.show_rt(partial(self.pde.q_solution, time=ti), title=f"q — True ({tlabel})")
        self.show_p0(partial(self.pde.u_solution, time=ti), title=f"u — True ({tlabel})")
        self.show_p0(partial(self.pde.y_solution, time=ti), title=f"y — True ({tlabel})")
        self.show_p0(partial(self.pde.z_solution, time=ti), title=f"z — True ({tlabel})")

        self.show_rt(allp[nt], title=f"p — Numerical ({tlabel})")
        self.show_rt(allq[nt], title=f"q — Numerical ({tlabel})")
        self.show_p0(allu[nt], title=f"u — Numerical ({tlabel})")
        self.show_p0(ally[nt], title=f"y — Numerical ({tlabel})")
        self.show_p0(allz[nt], title=f"z — Numerical ({tlabel})")
        
        p_err = allp[nt] - self.pspace.interpolation(partial(self.pde.p_solution, time=ti))
        self.show_rt(bm.abs(p_err), title=f"p — Error ({tlabel})")
        q_err = allq[nt] - self.pspace.interpolation(partial(self.pde.q_solution, time=ti))
        self.show_rt(bm.abs(q_err), title=f"q — Error ({tlabel})")
        u_err = allu[nt] - self.yspace.interpolate(partial(self.pde.u_solution, time=ti))
        self.show_p0(bm.abs(u_err), title=f"u — Error ({tlabel})")
        y_err = ally[nt] - self.yspace.interpolate(partial(self.pde.y_solution, time=ti))
        self.show_p0(bm.abs(y_err), title=f"y — Error ({tlabel})")
        z_err = allz[nt] - self.yspace.interpolate(partial(self.pde.z_solution, time=ti))
        self.show_p0(bm.abs(z_err), title=f"z — Error ({tlabel})")


    @variantmethod("direct")
    def solve(self, A, b):
        from fealpy.solver import spsolve
        self.xh[:] = spsolve(A, b, solver='mumps')
        return self.xh


