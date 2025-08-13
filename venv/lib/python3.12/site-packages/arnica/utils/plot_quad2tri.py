""" Module to plot a field on cartesian mesh through a triangular mesh """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_connectivity(i, j, shape):
    """
    *Generate connectivities*

    :param i: Index of the cell on u-axis
    :type i: int
    :param j: Index of the cell on v-axis
    :type j: int
    :param shape: Shape of the shell

    :returns:

        - **connectivity** - Array of shape (t, 3) of triangle point indexes
    """
    connectivity = []
    connectivity.append([i*shape[1] + j,
                         i*shape[1] + (j+1),
                         shape[0] * shape[1] + i*(shape[1]-1) + j])
    connectivity.append([i*shape[1] + (j+1),
                         (i+1)*shape[1] + (j+1),
                         shape[0] * shape[1] + i*(shape[1] - 1) + j])
    connectivity.append([(i+1)*shape[1] + (j+1),
                         (i+1)*shape[1] + j,
                         shape[0] * shape[1] + i*(shape[1] - 1) + j])
    connectivity.append([(i+1)*shape[1] + j,
                         i*shape[1] + j,
                         shape[0] * shape[1]+ i*(shape[1] - 1) + j])

    return np.asarray(connectivity)

def quad2tri(grid, field):
    """
    *Divide a cartesian grid into an extended triangular grid*

    :param grid: Array of shape (n_u, n_v, 2) of cartesian coordinates
    :param field Array of shape (n_u, n_v) of field values

    :returns:

        - **triangulation** - Matplotlib.tri.Triangulation object containing \
                              extended grid coordinates and connectivity array
        - **field_at_node** - Array of extended field values stored at the node
    """

    shape = grid.shape[0:2]
    grid_center = np.ones((shape[0] - 1, shape[1] - 1, 2))
    field_center = np.ones((shape[0] - 1, shape[1] - 1))

    connectivity = np.empty((0, 3), dtype=int)
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            grid_center[i, j] = np.mean(grid[i:i+2, j:j+2].reshape(4, 2), axis=0)
            connectivity = np.append(connectivity, get_connectivity(i, j, shape), axis=0)
            field_center[i, j] = np.mean(field[i:i+2, j:j+2].reshape(4))

    coord = np.stack((np.append(np.ravel(grid[:, :, 0]), np.ravel(grid_center[:, :, 0])),
                      np.append(np.ravel(grid[:, :, 1]), np.ravel(grid_center[:, :, 1])))).T
    field_at_node = np.append(np.ravel(field), np.ravel(field_center))
    #field_at_cell = field_at_node[connectivity].mean(axis=-1)

    triangulation = tri.Triangulation(coord[:, 0], coord[:, 1], connectivity)

    return triangulation, field_at_node

def plot_quad2tri(grid, title, field, shading=True):
    r"""\
    *Generate a plot of a field on a x,y shell cartesian grid*

    The cartesian grid is converted into a triangle grid with connectivities.
    The field is extended by interpolation on the new points created.
    The plot is generated from the points coordinates extended and the connectivity, \
    using tripcolor of matplotlib.

    If shading is True : using 'gouraud' from field value defined at nodes.
    If shading is False : using 'flat' from mean field value defined at the nodes or \
                          field value defined at the cell.

    :param grid: Array of shape (n_u, n_v, 2) of cartesian coordinates
    :param title: Title of the plot
    :type title: str
    :param field: Array of shape (n_u, n_v) of field values
    :param shading: Define if plot is shaded of not
    :type shading: bool

    :returns:

        - **plt** - Matplotlib object containing the graph
    ::
            1 quad              4 tri
          O---------O        O---------O
          |         |        | \     / |
          |         |        |  \   /  |
          |         |        |   \ /   |
          |         |  --->  |    O    |
          |         |        |   / \   |
          |         |        |  /   \  |
          |         |        | /     \ |
          O---------O        O---------O

    """

    triangulation, field_at_node = quad2tri(grid, field)

    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    if shading:
        shading = 'gouraud'
    else:
        shading = 'flat'

    tpc = ax1.tripcolor(triangulation,
                        field_at_node,
                        shading=shading)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    fig1.colorbar(tpc, cax=cax)
    ax1.set_title(title)
    ax1.set_xlabel('z (m)')
    ax1.set_ylabel('y (m)')

    return plt
