"""
=========
The utils
=========

These utils are helpers around CFD-related problems.

- **showy** is a matplotlib helper for using subplots with re-usable templates.
- **show_mat** is a matplotlib helper for fast matrix plotting with legend and axis naming.
- **cloud2cloud** is an inverse distance interpolator without connectivity.
- **directed_projection** is a projection of vectors clouds along their directions.
- **vector_actions** is a set of vector transformation helpers.
- **plot_density_mesh** is a mesh rendering tool using matplotlib hist2d.
- **axi_shell** is a 2D i-j structured mesh mapping axycylindrical splaine-based surfaces.
- **nparray2xmf** is a 1-2-3D i-j-k structured numpy datastructure dumping facility to XDMF format.

Untested - to be deleted :
--------------------------

- **unstructured_adjacency** *unstested* is the beginning of mesh handling using connectivity.
- **mesh_tools** *unstested* is a 2D mesh generation in numpy for solvers
- **datadict2file** was a dumping facility for dictionnary-like data. To be replaced by *hdfdict* or h5py-wrapper* packages.
- **timer_decorator** is a lightweight timer for functions. Better to use cProfile...

"""

#from arnica.utils.axishell import *
from arnica.utils.cloud2cloud import *
from arnica.utils.directed_projection import *
from arnica.utils.nparray2xmf import *
from arnica.utils.vector_actions import *
from arnica.utils.showy import *
from arnica.utils.sample_curve_from_cloud import *
from arnica.utils.same_nob import *

from arnica.utils.axipointcloud import *
