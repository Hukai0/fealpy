import argparse

from fealpy.backend import backend_manager as bm

## 参数解析
parser = argparse.ArgumentParser(description=
    """
    Solve elliptic equations using the lowest order Raviart-Thomas element and piecewise constant mixed finite element space.
    The stiffness matrix includes a transfer operator term.
    """)

parser.add_argument('--backend',
    default='numpy', type=str,
    help="Default backend is numpy. You can also choose pytorch, jax, tensorflow, etc.")

parser.add_argument('--pde',
    default=7, type=int,
    help="Name of thes PDE model, default is opc")

parser.add_argument('--init_mesh',
    default='uniform_tri', type=str,
    help="Type of initial mesh, default is uniform_tri")

parser.add_argument('--space_degree',
    default=0, type=int,
    help="Degree of Lagrange finite element space, default is 0")

parser.add_argument('--solve',
    default='direct', type=str,
    help="Type of solver, default is direct, options are direct, iterative")

parser.add_argument('--pbar_log',
    default=True, type=bool,
    help="Whether to show progress bar, default is True")

parser.add_argument('--log_level',
    default='INFO', type=str,
    help="Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL")



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

        # 你原来的 if len(cells) < 0 永远不会触发，这里如果你想区分边界点，需要别的判据
        # 这里只按“有相邻单元就做拟合/平均”处理

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


options = vars(parser.parse_args())
from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.fem import TwoGridOPCMixedFEMModel
from fealpy.decorator import barycentric, cartesian
from fealpy.utils import timer
from fealpy.functionspace import LagrangeFESpace

maxit_norm = 4
errorType = ['$|| p - p_h||_{L2}$ ',
             '$|| q - q_h||_{L2}$ ',
             '$|| u - u_h||_{L2}$ ',
             '$|| y - y_h||_{L2}$ ',
             '$|| z - z_h||_{L2}$ ',
              ]
errorMatrix = bm.zeros((len(errorType), maxit_norm), dtype=bm.float64)
NDof = bm.zeros(maxit_norm, dtype=bm.float64)
errorp = bm.zeros(maxit_norm, dtype=bm.float64)

for i in range(maxit_norm):

    tmr = timer()
    tmr_all = timer()
    next(tmr)
    next(tmr_all)

    model = TwoGridOPCMixedFEMModel(options)
    maxit = 20  
    model.mesh.uniform_refine(n=i)
    space1,space2 = model.space(p=0)
    pdof = space2.dof.number_of_global_dofs()
    ydof = space1.dof.number_of_global_dofs()
    qdof = space2.dof.number_of_global_dofs()
    zdof = space1.dof.number_of_global_dofs()
    y0 = space1.function()
    y1 = space1.function()
    p0 = space2.function()
    q0 = space2.function()
    q1 = space2.function()
    z0 = space1.function()
    z1 = space1.function()
    p1 = space2.function()
    u0 = space1.function()
    u1 = space1.function()
    xh = bm.zeros((pdof + ydof), dtype=bm.float64)
    hx = bm.zeros((qdof + zdof), dtype=bm.float64)

    # for j in range(maxit):
    #     pde = model.pde
    #     A, b_forward = model.linear_system(p=0, s1=0, s2=0, s3=pde.f_fun, s4=u0)
    #     tmr.send('第{}次迭代：正向求解组装线性系统时间'.format(j))
    #     A, b_forward = model.apply_bc(A, b_forward, gd=pde.y_solution)
    #     tmr.send('第{}次迭代：正向求解应用边界条件时间'.format(j))
    #     xh[:] = model.solve(A, b_forward)
    #     tmr.send('第{}次迭代：正向求解求解线性系统时间'.format(j))
    #     p1[:] = xh[:pdof]
    #     y1[:] = xh[pdof:]
    #     @barycentric
    #     def coef_p(bcs, index=None):
    #         result = - p1(bcs, index)
    #         return result
    #     @cartesian
    #     def coef_pd(p, index=None):
    #         result = pde.pd_fun(p)
    #         return result
    #     @barycentric
    #     def coef_y(bcs, index=None):
    #         return y1(bcs) 
    #     @cartesian
    #     def coef_yd(p, index=None):
    #         return -pde.yd_fun(p)
    #     A,b_backward = model.linear_system(p=0 ,s1=coef_p, s2=coef_pd, s3=coef_y, s4=coef_yd)
    #     tmr.send('第{}次迭代：反向求解组装线性系统时间'.format(j))
    #     A, b_backward = model.apply_bc(A, b_backward, gd=pde.z_solution)
    #     tmr.send('第{}次迭代：反向求解应用边界条件时间'.format(j))
    #     hx[:]= model.solve(A, b_backward)
    #     tmr.send('第{}次迭代：反向求解求解线性系统时间'.format(j))
    #     q1[:] = hx[:qdof]
    #     z1[:] = hx[qdof:]
    #     # 假设控制系数为1  
    #     nu = 1
    #     u1[:] = -z1
    #     mesh = model.mesh
    #     p_error = mesh.error(p0,p1)
    #     q_error = mesh.error(q0,q1)
    #     y_error = mesh.error(y0,y1)
    #     z_error = mesh.error(z0,z1)
    #     p0[:] = p1[:]
    #     q0[:] = q1[:]
    #     y0[:] = y1[:]
    #     z0[:] = z1[:]
    #     u0[:] = u1[:]
    #     if p_error < 1e-14:
    #         print('p收敛',p_error)
    #         print('q收敛',q_error)
    #         print('y收敛',y_error)
    #         print('z收敛',z_error)
    #         break

    # G = model.mesh.uniform_refine(n=1,return_cellim = True)
    # print("G.shape",G[-1].shape)
    # space1,space2 = model.space(p=0)
    # pdof = space2.dof.number_of_global_dofs()
    # ydof = space1.dof.number_of_global_dofs()
    # qdof = space2.dof.number_of_global_dofs()
    # zdof = space1.dof.number_of_global_dofs()
    # y0 = space1.function()
    # y1 = space1.function()
    # p0 = space2.function()
    # q0 = space2.function()
    # q1 = space2.function()
    # z0 = space1.function()
    # z1 = space1.function()
    # p1 = space2.function() 
    # u0 = space1.function()
    # u0[:] = G[-1] @ u1[:]
    # u1 = space1.function()
    # xh = bm.zeros((pdof + ydof), dtype=bm.float64)
    # hx = bm.zeros((qdof + zdof), dtype=bm.float64)

    for j in range(maxit):
        pde = model.pde
        A, b_forward = model.linear_system(p=0, s1=0, s2=0, s3=pde.f_fun, s4=u0)
        tmr.send('第{}次迭代：正向求解组装线性系统时间'.format(j))
        A, b_forward = model.apply_bc(A, b_forward, gd=pde.y_solution)
        tmr.send('第{}次迭代：正向求解应用边界条件时间'.format(j))
        xh[:] = model.solve(A, b_forward)
        tmr.send('第{}次迭代：正向求解求解线性系统时间'.format(j))
        p1[:] = xh[:pdof]
        y1[:] = xh[pdof:]
        @barycentric
        def coef_p(bcs, index=None):
            result = - p1(bcs, index)
            return result
        @cartesian
        def coef_pd(p, index=None):
            result = pde.pd_fun(p)
            return result
        @barycentric
        def coef_y(bcs, index=None):
            return y1(bcs) 
        @cartesian
        def coef_yd(p, index=None):
            return -pde.yd_fun(p)
        A,b_backward = model.linear_system(p=0 ,s1=coef_p, s2=coef_pd, s3=coef_y, s4=coef_yd)
        tmr.send('第{}次迭代：反向求解组装线性系统时间'.format(j))
        A, b_backward = model.apply_bc(A, b_backward, gd=pde.z_solution)
        tmr.send('第{}次迭代：反向求解应用边界条件时间'.format(j))
        hx[:]= model.solve(A, b_backward)
        tmr.send('第{}次迭代：反向求解求解线性系统时间'.format(j))
        q1[:] = hx[:qdof]
        z1[:] = hx[qdof:]
        # 假设控制系数为1  
        nu = 1
        u1[:] = -z1
        mesh = model.mesh
        p_error = mesh.error(p0,p1)
        q_error = mesh.error(q0,q1)
        y_error = mesh.error(y0,y1)
        z_error = mesh.error(z0,z1)
        p0[:] = p1[:]
        q0[:] = q1[:]
        y0[:] = y1[:]
        z0[:] = z1[:]
        u0[:] = u1[:]
        if p_error < 1e-8:
            print('p收敛',p_error)
            print('q收敛',q_error)
            print('y收敛',y_error)
            print('z收敛',z_error)
            break


    # qsolution = space2.interpolation(pde.q_solution)
    # errorq = bm.max(bm.abs(q1-qsolution))
    # print("q误差",errorq)
    # ysolution =space1.interpolate(pde.y_solution,)
    # errory = bm.max(bm.abs(y1-ysolution))
    # print("y误差",errory)
    # psolution = space2.interpolation(pde.p_solution)
    # errorp = bm.max(bm.abs(p1-psolution))
    # print("p误差",errorp)
    # zsolution = space1.interpolate(pde.z_solution)
    # errorz = bm.max(bm.abs(z1-zsolution))
    # print("z误差",errorz)

    # 后处理计算误差

    val = recover_p1_from_cell_mean(model.mesh, u1)
    spaceh = LagrangeFESpace(model.mesh, p=1)
    u3 = spaceh.function()
    u3[:] = val
    error = model.mesh.error(pde.u_solution, u3)
    errorp[i] = error


    errorl2y, errorl2p = model.postprocess(y1, p1, solution1=pde.y_solution, solution2=pde.p_solution)
    errorl2u, errorl2q = model.postprocess(u1, q1, solution1=pde.u_solution, solution2=pde.q_solution)
    errorl2z, errorl2pd = model.postprocess(z1, p1, solution1=pde.z_solution, solution2=pde.p_solution)
    print('第{}次迭代：p误差={}, q误差={}, u误差={}, y误差={}, z误差={}'
          .format(i, errorl2p, errorl2q, errorl2u, errorl2y, errorl2z))
    errorMatrix[0, i] = errorl2p
    errorMatrix[1, i] = errorl2q
    errorMatrix[2, i] = errorl2u
    errorMatrix[3, i] = errorl2y
    errorMatrix[4, i] = errorl2z

    NDof[i] = 1/2**i

    tmr.send('后处理计算误差时间')
    tmr_all.send('总时间')
    next(tmr)
    next(tmr_all)

    # model.show_p0(pde.y_solution)
    # model.show_p0(pde.u_solution)
    # model.show_p0(pde.z_solution)
    # model.show_rt(pde.q_solution)
    # model.show_rt(pde.p_solution)

# from fealpy.tools.show import showmultirate, show_error_table
# from matplotlib import pyplot as plt
# showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
# show_error_table(NDof, errorType, errorMatrix)
# plt.show()


print(errorp)
print(errorMatrix)