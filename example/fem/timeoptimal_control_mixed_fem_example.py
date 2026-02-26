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
    default=5,type=int,
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

# 解析参数
options = vars(parser.parse_args())

from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.fem import TimeOPCMixedFEMModel
from fealpy.decorator import barycentric, cartesian
from fealpy.utils import timer

tof = TimeOPCMixedFEMModel(options)
errorMatrix = tof.refine_run(reit=5)
print(errorMatrix)