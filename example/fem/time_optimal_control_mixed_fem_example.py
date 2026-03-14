import argparse

from fealpy.backend import backend_manager as bm

# editable defaults in file (CLI arguments can still override)
DEFAULT_PLOT = False
DEFAULT_PLOT_WHEN_NT = 80   # -1 means plot for all nt
DEFAULT_PLOT_NT = -1.0      # -1 means nt//2; [0,1] means ratio; >1 means absolute index

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
    default=6,type=int,
    help="Name of thes PDE model, default is opc")

parser.add_argument('--init_mesh',
    default='uniform_tri', type=str,
    help="Type of initial mesh, default is uniform_tri")

parser.add_argument('--space_degree',
    default=0, type=int,
    help="Degree of Lagrange finite element space, default is 0")

parser.add_argument('--op_type', default=1, type=int)

parser.add_argument('--pbar_log',
    default=True, type=bool,
    help="Whether to show progress bar, default is True")

parser.add_argument('--log_level',
    default='INFO', type=str,
    help="Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL")

parser.add_argument('--plot', dest='plot', action='store_true',
    help="Enable plotting after convergence (override file default)")

parser.add_argument('--no-plot', dest='plot', action='store_false',
    help="Disable plotting (override file default)")

parser.add_argument('--plot_when_nt',
    default=DEFAULT_PLOT_WHEN_NT, type=int,
    help="Only plot when time steps nt equals this value; use -1 to allow all nt")

parser.add_argument('--plot_nt',
    default=DEFAULT_PLOT_NT, type=float,
    help="Plot position: -1->nt//2, [0,1]->ratio of nt, >1->absolute time index")
parser.set_defaults(plot=DEFAULT_PLOT)

# 解析参数
options = vars(parser.parse_args())

from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.fem import TimeOPCMixedFEMModel
from fealpy.decorator import barycentric, cartesian
from fealpy.utils import timer

tof = TimeOPCMixedFEMModel(options)
errorMatrix = tof.refine_run(reit=4)
print(errorMatrix)
