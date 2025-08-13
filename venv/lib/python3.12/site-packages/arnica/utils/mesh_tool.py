"""
This module contains function to create and modify meshes
"""

import numpy as np


def gen_cart_grid_2d(gridrange, gridpoints):
    """ Generate cartesian grid.

    Parameters :
    ------------
    gridrange : tuple of floats, dimensions of the grid
    gridpoints : tuple of ints (n,m), sampling on the grid

    Returns :
    ---------

    x_coor, y_coor : numpy arrays (n,m) with coordinates
    """
    x_vec = np.linspace(0., gridrange[0], int(gridpoints[0]), endpoint=True)
    y_vec = np.linspace(0., gridrange[1], int(gridpoints[1]), endpoint=True)
    x_coor, y_coor = np.meshgrid(x_vec, y_vec)

    return x_coor, y_coor


def gen_cyl_grid_2d(r_min, r_max, r_points,
                    theta_min, theta_max, theta_points):
    """
    Generate a cylindrical grid center on x = 0 and y = 0

    Parameters
    ----------
    r_min : inner radius
    r_max : outer radius
    theta_min : lower angle [0, 2 * pi]
    theta_max : upper angle [0, 2 * pi]
    r_points : number of points in the radial direction
    theta_points : number of points in the tangential direction

    Returns
    -------
    x_coor : x coordinates of the mesh
    y_coor : y coordinates of the mesh
    """
    r_vec = np.linspace(r_min, r_max, r_points, endpoint=True)
    theta_vec = np.linspace(theta_min, theta_max, theta_points, endpoint=True)
    r_coor, theta_coor = np.meshgrid(r_vec, theta_vec)

    x_coor = r_coor * np.cos(theta_coor)
    y_coor = r_coor * np.sin(theta_coor)

    return x_coor, y_coor


def dilate_center(x_coor, y_coor, perturbation=0.1):
    """ perturb cartesian mesh dilatation in the center

    Parameters :
    ------------
    x_coor : numpy array (n,m) , x_coordinates
    y_coor : numpy array (n,m) , y_coordinates
    perturbation : float, amplitude of the perturbation perturbation
                   with respect to the grid size

    Returns :
    ---------
    x_coor : numpy array (n,m) , x_coordinates shifted
    y_coor : numpy array (n,m) , y_coordinates shifted

     """
    x_width = x_coor.max() - x_coor.min()
    y_width = y_coor.max() - y_coor.min()
    x_center = x_coor.min() + 0.5 * x_width
    y_center = y_coor.min() + 0.5 * y_width

    adim_rad = np.sqrt(((x_coor - x_center) / (0.5 * x_width)) ** 2
                       + ((y_coor - y_center) / (0.5 * y_width)) ** 2)

    rad_dir_x = (x_coor - x_center) / np.maximum(adim_rad, 1.e-16)
    rad_dir_y = (y_coor - y_center) / np.maximum(adim_rad, 1.e-16)
    dilate = np.maximum((1. - adim_rad), 0)
    dilate = 3 * dilate * dilate * (1 - dilate)

    shift_x = perturbation * x_width * dilate * rad_dir_x
    shift_y = perturbation * y_width * dilate * rad_dir_y

    return x_coor + shift_x, y_coor + shift_y, dilate


def get_mesh(params_mesh):
    """
    Call specific meshing functions from mesh parameters dict

    Parameters
    ----------
    params_mesh: dictionary containing mesh parameters

    Returns
    -------
    x_coor: x coordinates of the mesh
    y_coor: y coordinates of the mesh
    """

    if params_mesh["kind"] == "rect":
        x_max = params_mesh["x_max"]
        y_max = params_mesh["y_max"]
        x_res = params_mesh["x_res"]
        y_res = params_mesh["y_res"]
        x_coor, y_coor = gen_cart_grid_2d((x_max, y_max), (x_res, y_res))

    elif params_mesh["kind"] == "cyl":
        r_min = params_mesh["r_min"]
        r_max = params_mesh["r_max"]
        theta_min = params_mesh["theta_min"]
        theta_max = params_mesh["theta_max"]
        angle = theta_max - theta_min
        n_r_pts = params_mesh["n_pts_rad"]

        try:
            n_tgt_pts = params_mesh["n_pts_tgt"]
        except KeyError:
            n_tgt_pts = int(angle * n_r_pts + 1)

        delta = angle / (int(n_tgt_pts) - 1)
        theta_max -= delta

        x_coor, y_coor = gen_cyl_grid_2d(r_min, r_max, n_r_pts, theta_min,
                                         theta_max, int(angle * n_r_pts + 1))
    else:
        raise NotImplementedError(
            'The only mesh kinds implemented are: "rect" and "cyl"')

    return x_coor, y_coor
