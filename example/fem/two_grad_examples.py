import argparse
from fealpy.backend import backend_manager as bm

parser = argparse.ArgumentParser(description="""
Solve elliptic optimal control problem using RT0 mixed FEM + piecewise constant control.
Implement a TWO-GRID algorithm:
  coarse: full optimization iteration
  fine: one-shot correction (forward + adjoint + one control update)
""")
parser.add_argument('--backend', default='numpy', type=str,
                    help="numpy/pytorch/jax/tensorflow etc.")
parser.add_argument('--pde', default=7, type=int,
                    help="PDE model id (default 7)")
parser.add_argument('--init_mesh', default='uniform_tri', type=str,
                    help="initial mesh type")
parser.add_argument('--space_degree', default=0, type=int,
                    help="degree (default 0)")
parser.add_argument('--solve', default='direct', type=str,
                    help="direct/iterative")
parser.add_argument('--pbar_log', default=True, type=bool,
                    help="progress bar")
parser.add_argument('--log_level', default='INFO', type=str,
                    help="log level")

# two-grid params
parser.add_argument('--nlevel', default=5, type=int,
                    help="number of mesh levels to test (0..nlevel-1)")
parser.add_argument('--coarse_it', default=20, type=int,
                    help="max iterations on coarse mesh")
parser.add_argument('--coarse_tol', default=1e-10, type=float,
                    help="stop tol on coarse mesh (based on p error)")
parser.add_argument('--nu', default=1.0, type=float,
                    help="Tikhonov parameter nu in u = Proj(-z/nu)")


def coarse_optimize(model, maxit=50, tol=1e-8, nu=1.0, tmr=None):

    from fealpy.decorator import barycentric, cartesian

    pde = model.pde
    space1, space2 = model.space(p=0)  # space1: P0 (cell), space2: RT0
    pdof = space2.dof.number_of_global_dofs()
    ydof = space1.dof.number_of_global_dofs()
    qdof = space2.dof.number_of_global_dofs()
    zdof = space1.dof.number_of_global_dofs()

    # unknown functions
    y0 = space1.function()
    y1 = space1.function()
    z0 = space1.function()
    z1 = space1.function()
    u0 = space1.function()
    u1 = space1.function()

    p0 = space2.function()
    p1 = space2.function()
    q0 = space2.function()
    q1 = space2.function()

    xh = bm.zeros(pdof + ydof, dtype=bm.float64)
    hx = bm.zeros(qdof + zdof, dtype=bm.float64)

    def control_update_from_z(z_fun):
        u = space1.function()
        u[:] = -(1.0/nu) * z_fun[:]
        return u

    for j in range(maxit):
        # -------- forward: (p,y) with current control u0 --------
        A, b_forward = model.linear_system(p=0, s1=0, s2=0, s3=pde.f_fun, s4=u0)
        if tmr is not None:
            tmr.send(f'[coarse iter {j}] assemble forward')
        A, b_forward = model.apply_bc(A, b_forward, gd=pde.y_solution)
        if tmr is not None:
            tmr.send(f'[coarse iter {j}] apply bc forward')
        xh[:] = model.solve(A, b_forward)
        if tmr is not None:
            tmr.send(f'[coarse iter {j}] solve forward')

        p1[:] = xh[:pdof]
        y1[:] = xh[pdof:]

        # -------- build coefficients for adjoint --------
        @barycentric
        def coef_p(bcs, index=None):
            return -p1(bcs, index)

        @cartesian
        def coef_pd(p, index=None):
            return pde.pd_fun(p)

        @barycentric
        def coef_y(bcs, index=None):
            return y1(bcs)

        @cartesian
        def coef_yd(p, index=None):
            return -pde.yd_fun(p)

        # -------- backward: (q,z) --------
        A, b_backward = model.linear_system(p=0, s1=coef_p, s2=coef_pd, s3=coef_y, s4=coef_yd)
        if tmr is not None:
            tmr.send(f'[coarse iter {j}] assemble adjoint')
        A, b_backward = model.apply_bc(A, b_backward, gd=pde.z_solution)
        if tmr is not None:
            tmr.send(f'[coarse iter {j}] apply bc adjoint')
        hx[:] = model.solve(A, b_backward)
        if tmr is not None:
            tmr.send(f'[coarse iter {j}] solve adjoint')

        q1[:] = hx[:qdof]
        z1[:] = hx[qdof:]

        # -------- control update --------
        u1 = control_update_from_z(z1)
        if tmr is not None:
            tmr.send(f'[coarse iter {j}] update control')

        # -------- convergence check --------
        mesh = model.mesh
        p_err = mesh.error(p0, p1)

        # update iterates
        p0[:] = p1[:]
        y0[:] = y1[:]
        q0[:] = q1[:]
        z0[:] = z1[:]
        u0[:] = u1[:]

        if p_err < tol:
            print(f"[coarse] converged at iter {j}, p_err={p_err}")
            break

    return p1, y1, q1, z1, u1



def fine_oneshot(model, u0_fine, nu=1.0, tmr=None):
    """
    On fine mesh: solve forward once + adjoint once + update control once.
    Return (p,y,q,z,u_star).

    If tmr is provided, we will add time stamps.
    """
    from fealpy.decorator import barycentric, cartesian

    pde = model.pde
    space1, space2 = model.space(p=0)
    pdof = space2.dof.number_of_global_dofs()
    ydof = space1.dof.number_of_global_dofs()
    qdof = space2.dof.number_of_global_dofs()
    zdof = space1.dof.number_of_global_dofs()

    y = space1.function()
    z = space1.function()
    u_star = space1.function()
    p = space2.function()
    q = space2.function()

    xh = bm.zeros(pdof + ydof, dtype=bm.float64)
    hx = bm.zeros(qdof + zdof, dtype=bm.float64)

    # forward
    A, b_forward = model.linear_system(p=0, s1=0, s2=0, s3=pde.f_fun, s4=u0_fine)
    if tmr is not None:
        tmr.send('[fine] assemble forward')
    A, b_forward = model.apply_bc(A, b_forward, gd=pde.y_solution)
    if tmr is not None:
        tmr.send('[fine] apply bc forward')
    xh[:] = model.solve(A, b_forward)
    if tmr is not None:
        tmr.send('[fine] solve forward')

    p[:] = xh[:pdof]
    y[:] = xh[pdof:]

    @barycentric
    def coef_p(bcs, index=None):
        return -p(bcs, index)

    @cartesian
    def coef_pd(x, index=None):
        return pde.pd_fun(x)

    @barycentric
    def coef_y(bcs, index=None):
        return y(bcs)

    @cartesian
    def coef_yd(x, index=None):
        return -pde.yd_fun(x)

    # adjoint
    A, b_backward = model.linear_system(p=0, s1=coef_p, s2=coef_pd, s3=coef_y, s4=coef_yd)
    if tmr is not None:
        tmr.send('[fine] assemble adjoint')
    A, b_backward = model.apply_bc(A, b_backward, gd=pde.z_solution)
    if tmr is not None:
        tmr.send('[fine] apply bc adjoint')
    hx[:] = model.solve(A, b_backward)
    if tmr is not None:
        tmr.send('[fine] solve adjoint')

    q[:] = hx[:qdof]
    z[:] = hx[qdof:]

    # control update (one-shot)
    u_star[:] = -(1.0/nu) * z[:]
    if tmr is not None:
        tmr.send('[fine] update control')

    return p, y, q, z, u_star


# -----------------------------
# one two-grid experiment on a given coarse level i
# -----------------------------
def run_two_grid_once(options, i_coarse, nu=1.0, coarse_it=50, coarse_tol=1e-8, tmr=None):
    """
    Build a model, refine to coarse level i_coarse, solve coarse optimization.
    Then refine ONE more time to get fine mesh and cell prolongation G,
    prolong u_H to u_h^0, run one-shot on fine.
    Return fine results and errors.

    If tmr is provided, we will add time stamps.
    """
    from fealpy.fem import TwoGridOPCMixedFEMModel
    from fealpy.functionspace import LagrangeFESpace

    # ---- build model on base mesh and refine to coarse level ----
    model = TwoGridOPCMixedFEMModel(options)
    if tmr is not None:
        tmr.send(f'[level {i_coarse}] build model')
    model.mesh.uniform_refine(n=i_coarse)
    if tmr is not None:
        tmr.send(f'[level {i_coarse}] refine to coarse mesh')

    # ---- coarse full optimize ----
    pH, yH, qH, zH, uH = coarse_optimize(
        model, maxit=coarse_it, tol=coarse_tol, nu=nu, tmr=tmr
    )
    if tmr is not None:
        tmr.send(f'[level {i_coarse}] coarse optimize done')

    # ---- refine ONE step and get cell prolongation mapping ----
    G = model.mesh.uniform_refine(n=1, return_cellim=True)
    if tmr is not None:
        tmr.send(f'[level {i_coarse}] refine to fine mesh + get prolongation')

    # ---- rebuild spaces on fine mesh ----
    space1h, space2h = model.space(p=0)  # on refined mesh
    uh0 = space1h.function()
    uh0[:] = G[-1] @ uH[:]
    if tmr is not None:
        tmr.send(f'[level {i_coarse}] prolong control uH -> uh0')

    # ---- fine one-shot ----
    ph, yh, qh, zh, uh_star = fine_oneshot(model, uh0, nu=nu, tmr=tmr)
    if tmr is not None:
        tmr.send(f'[level {i_coarse}] fine oneshot done')
    
    # if i_coarse == 4: 
    #     model.plot(ph, qh, uh_star, yh, zh)

    # ---- compute errors ----
    pde = model.pde
    errorl2y, errorl2p = model.postprocess(yh, ph, solution1=pde.y_solution, solution2=pde.p_solution)
    errorl2u, errorl2q = model.postprocess(uh_star, qh, solution1=pde.u_solution, solution2=pde.q_solution)
    errorl2z, _ = model.postprocess(zh, ph, solution1=pde.z_solution, solution2=pde.p_solution)
    if tmr is not None:
        tmr.send(f'[level {i_coarse}] postprocess errors')

    # extra: recover P1 from cell-mean control and compute L2 error
    val = model.recover_p1_from_cell_mean(model.mesh, uh_star)
    spaceh = LagrangeFESpace(model.mesh, p=1)
    uP1 = spaceh.function()
    uP1[:] = val
    error_recover_u = model.mesh.error(pde.u_solution, uP1)
    if tmr is not None:
        tmr.send(f'[level {i_coarse}] recover u (P1) + error')

    return model, (ph, yh, qh, zh, uh_star), (errorl2p, errorl2q, errorl2u, errorl2y, errorl2z, error_recover_u)


def main():
    options = vars(parser.parse_args())

    # set backend
    from fealpy.backend import backend_manager as bm_backend
    bm_backend.set_backend(options['backend'])

    from fealpy.utils import timer

    nlevel = options['nlevel']
    coarse_it = options['coarse_it']
    coarse_tol = options['coarse_tol']
    nu = options['nu']

    errorType = [
        '$|| p - p_h||_{L2}$',
        '$|| q - q_h||_{L2}$',
        '$|| u - u_h||_{L2}$',
        '$|| y - y_h||_{L2}$',
        '$|| z - z_h||_{L2}$',
        '$|| u - R_h u_h||_{L2}$ (recover P1)'
    ]

    errorMatrix = bm.zeros((len(errorType), nlevel), dtype=bm.float64)
    hvals = bm.zeros(nlevel, dtype=bm.float64)



    for i in range(nlevel):
        # per-level timer
        tmr_all = timer()
        next(tmr_all)
        tmr = timer()
        next(tmr)

        model, sol, errs = run_two_grid_once(
            options,
            i_coarse=i,
            nu=nu,
            coarse_it=coarse_it,
            coarse_tol=coarse_tol,
            tmr=tmr
        )

        errorl2p, errorl2q, errorl2u, errorl2y, errorl2z, error_recover_u = errs
        print(f"[two-grid] coarse level={i}, fine level={i+1}: "
              f"p={errorl2p:.3e}, q={errorl2q:.3e}, u={errorl2u:.3e}, "
              f"y={errorl2y:.3e}, z={errorl2z:.3e}, recover_u={error_recover_u:.3e}")

        errorMatrix[0, i] = errorl2p
        errorMatrix[1, i] = errorl2q
        errorMatrix[2, i] = errorl2u
        errorMatrix[3, i] = errorl2y
        errorMatrix[4, i] = errorl2z
        errorMatrix[5, i] = error_recover_u

        # a simple "h" proxy (each refine halves h); here we used fine = coarse+1
        hvals[i] = 0.2 / (2.0 ** (i + 1))

        # end per-level timer
        tmr.send(f'[level {i}] done (printed above)')
        next(tmr)

        tmr_all.send('[levels] done')
        next(tmr_all)

    print("\nerrorType:")
    for k, t in enumerate(errorType):
        print(k, t)

    print("\nh (proxy):")
    print(hvals)

    print("\nerrorMatrix:")
    print(errorMatrix)


if __name__ == "__main__":
    main()
