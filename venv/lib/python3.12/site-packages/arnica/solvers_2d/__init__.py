"""
======================
Solver 2D (Deprecated)
======================

This package contains a solver of second partial derivative equations to treat heat conduction and heat radiation problem.
The 2nd order finit difference scheme is used to solve the inside of a computational domain and that of first order for boundaries.

This development is now hosted in a separate repository at `nitrox - calcifer <https://nitrox.cerfacs.fr/open-source/calcifer>`_
"""

from arnica.solvers_2d.boundary import *
from arnica.solvers_2d.conduction import *
from arnica.solvers_2d.core_fd import *
