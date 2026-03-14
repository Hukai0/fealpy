from typing import Optional,Union
from fealpy.model import ComputationalModel
from fealpy.model.optimal_control import OPCPDEDataT

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

class TwoGridOPCMixedFEMModel(ComputationalModel):
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
        self.solve.set(options['solve']) 

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
            self.mesh = self.pde.init_mesh[mesh] (nx=2, ny=2)
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
        self.uspace = LagrangeFESpace(self.mesh, p=p, ctype='D')
        return self.uspace, self.pspace
    
    def linear_system(self,  p, s1, s2, s3, s4):
        """
        Assemble the linear system for the optimal control problem.
        """
        self.pspace = RaviartThomasFESpace2d(self.mesh, p=p)
        self.uspace= LagrangeFESpace(self.mesh, p=p, ctype='D')  

        uLDOF = self.uspace.number_of_local_dofs()
        uGDOF = self.uspace.number_of_global_dofs()
        pLDOF = self.pspace.number_of_local_dofs()
        pGDOF = self.pspace.number_of_global_dofs()
        self.logger.info(f"Raviart-Thomas space: {self.pspace}, LDOF: {pLDOF}, GDOF: {pGDOF}")
        self.logger.info(f"Lagrange space: {self.uspace}, LDOF: {uLDOF}, GDOF: {uGDOF}")
        self.uh = self.uspace.function()
        self.ph = self.pspace.function()
        self.xh = bm.zeros((pGDOF + uGDOF,), dtype=bm.float64)
        
        bform1 = BilinearForm(self.pspace)
        bform1.add_integrator(OPCIntegrator(coef=self.pde.A_inverse, q=3))

        bform2 = BilinearForm((self.uspace,self.pspace))
        bform2.add_integrator(DivIntegrator(coef=-1, q=3))

        bform3 = BilinearForm((self.uspace,self.pspace))
        bform3.add_integrator(DivIntegrator(coef=1, q=3))

        bform4 = BilinearForm(self.uspace)
        bform4.add_integrator(ScalarMassIntegrator(coef=self.pde.C_matrix, q=3))

        M = BlockForm([[bform1,bform2],
                       [bform3.T,bform4]])
        A = M.assembly()
        D = bform4.assembly()        
        lform1 = LinearForm(self.pspace)
        lform1.add_integrator(OPCSIntegrator(source=s1))
        lform1.add_integrator(OPCSIntegrator(source=s2))
        lform2 = LinearForm(self.uspace)
        lform2.add_integrator(ScalarSourceIntegrator(source=s3))
        lform2.add_integrator(ScalarSourceIntegrator(source=s4))
        F = lform2.assembly()
        G = lform1.assembly()
        b = bm.concatenate([G,F],axis=0)
        
        return A, b
    
    def apply_bc(self, A, b, gd):
        """
        Apply the boundary conditions to the linear system.
        """
        uspace, pspace = self.space()
        ugdof = uspace.number_of_global_dofs()
        G_apply = pspace.set_neumann_bc(gd)
        F = bm.zeros(ugdof, dtype=bm.float64)
        b_apply = bm.concatenate([G_apply,F],axis=0)
        b = b - b_apply
        return A, b
    
    @variantmethod("direct")
    def solve(self, A, b):
        from fealpy.solver import spsolve
        self.xh[:] = spsolve(A, b, solver='scipy')
        return self.xh
    
    def postprocess(self, uh, ph, solution1, solution2):
        """
        Post-process the numerical solution to compute the error in L2 norm.
        """
        ul2 = self.mesh.error(solution1, uh)
        pl2 = self.mesh.error(solution2, ph)
        return ul2, pl2
    
    def postprocess_interpolate(self, uh, ph, solution1=None, solution2=None):
        """
        Post-process the numerical solution to compute the error in max norm.
        """
        if solution1 is None or solution2 is None:
            raise ValueError("Exact solutions must be provided for error computation.")
        u1 = self.uspace.interpolate(solution1)
        p1 = self.pspace.interpolation(solution2)
        umax = self.mesh.error(u1, uh)
        pmax = self.mesh.error(p1, ph)
        return umax, pmax

    @staticmethod
    def _format_plot_ticks(ax, cbar=None):
        import matplotlib.ticker as mticker

        # Avoid overcrowded long decimal labels.
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.tick_params(axis='both', which='major', labelsize=9, pad=2)

        if hasattr(ax, 'zaxis'):
            ax.zaxis.set_major_locator(mticker.MaxNLocator(5))
            zfmt = mticker.ScalarFormatter(useMathText=True)
            zfmt.set_scientific(True)
            zfmt.set_powerlimits((-2, 2))
            ax.zaxis.set_major_formatter(zfmt)
            ax.tick_params(axis='z', which='major', labelsize=9, pad=4)

        if cbar is not None:
            cbar.locator = mticker.MaxNLocator(6)
            cfmt = mticker.ScalarFormatter(useMathText=True)
            cfmt.set_scientific(True)
            cfmt.set_powerlimits((-2, 2))
            cbar.formatter = cfmt
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=9, pad=2)
    
    def show_p0(self,solution,title: str | None = None):
        """
        Visualize the mesh structure.
        """
        import types
        if isinstance(solution, types.MethodType):
            u1 = self.uspace.interpolate(solution)
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
        cbar = fig.colorbar(surf, ax=ax, shrink=0.85, pad=0.08)
        self._format_plot_ticks(ax, cbar)
        fig.tight_layout()
        plt.show()
    
    def show_rt(self, solution,title: str | None = None):
        """
        Visualize the mesh structure.
        """
        import matplotlib.pyplot as plt
        import types

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        node = self.mesh.entity('node')
        edge = self.mesh.entity('edge')
        node_center = 0.5 * (node[edge[:, 0]] + node[edge[:, 1]])
        if isinstance(solution, types.MethodType):
            p_solution = self.pspace.interpolation(solution)
        else:
            p_solution = solution[:]
        x = node_center[:, 0]
        y = node_center[:, 1]
        surf = plt.tricontourf(x, y, p_solution, levels=50, cmap='jet', edgecolor='none')
        ax.xaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        ax.yaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        ax.zaxis._axinfo["grid"].update({"linewidth": 0.5, "linestyle": "--", "alpha": 0.5})
        cbar = fig.colorbar(surf, ax=ax, shrink=0.85, pad=0.08)
        self._format_plot_ticks(ax, cbar)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_title(f'{title} ')
        fig.tight_layout()
        plt.show()
        
        
    def plot(self, p, q, u, y, z):
    
        self.show_rt(self.pde.p_solution, title=f"p — True")
        self.show_rt(self.pde.q_solution, title=f"q — True")
        self.show_p0(self.pde.u_solution, title=f"u — True")
        self.show_p0(self.pde.y_solution, title=f"y — True")
        self.show_p0(self.pde.z_solution, title=f"z — True")

        self.show_rt(p, title=f"p — Numerical")
        self.show_rt(q, title=f"q — Numerical")
        self.show_p0(u, title=f"u — Numerical")
        self.show_p0(y, title=f"y — Numerical")
        self.show_p0(z, title=f"z — Numerical")
        
        p_err = bm.abs(p - self.pspace.interpolation(self.pde.p_solution))
        self.show_rt(p_err, title=f"p — Error ")
        q_err = bm.abs(q - self.pspace.interpolation(self.pde.q_solution))
        self.show_rt(q_err, title=f"q — Error ")
        u_err = bm.abs(u - self.uspace.interpolate(self.pde.u_solution))
        self.show_p0(u_err, title=f"u — Error ")    
        y_err = bm.abs(y - self.uspace.interpolate(self.pde.y_solution))    
        self.show_p0(y_err, title=f"y — Error")
        z_err = bm.abs(z - self.uspace.interpolate(self.pde.z_solution))
        self.show_p0(z_err, title=f"z — Error")
    
    @staticmethod
    def recover_p1_from_cell_mean(mesh, uh):
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        node = mesh.entity('node')      # (NN, 2)
        cell = mesh.entity('cell')      # (NC, 3)

        cell_node = node[cell]          # (NC, 3, 2)
        bary = bm.mean(cell_node, axis=1)   # (NC, 2)
        area = mesh.entity_measure('cell')  # (NC,)

        # node_to_cell 是 CSR：行=节点，列=单元，非零表示相邻
        node2cell = mesh.node_to_cell()     # csr_matrix
        indptr = node2cell.indptr
        indices = node2cell.indices         # 每行非零对应的列号（也就是 cell id）

        Rh_uh = bm.zeros(NN)

        for z in range(NN):
            cells = indices[indptr[z]:indptr[z+1]]  # z 这个节点相邻的单元编号列表（1D）
            if len(cells) == 0:  # 孤立点/异常情况（正常网格一般不会出现）
                Rh_uh[z] = 0.0
                continue
            
            A = []
            b = []
            for k in cells:
                x, y = bary[k]
                w = area[k]
                uh_mean = uh[k]
                A.append([w, w*x, w*y])
                b.append(uh_mean * w)

            A = bm.array(A)
            b = bm.array(b)

            coef, *_ = bm.linalg.lstsq(A, b, rcond=None)
            xz, yz = node[z]
            Rh_uh[z] = coef[0] + coef[1]*xz + coef[2]*yz

        return Rh_uh


    
