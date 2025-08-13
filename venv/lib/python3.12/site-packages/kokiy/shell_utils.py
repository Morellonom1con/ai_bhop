"""Defines cut plane tools.
"""

import numpy as np

from arnica.utils.vector_actions import yz_to_theta
from cloud2cloud import CloudInterpolator

from kokiy.axishell import AxiShell
from kokiy.cartshell import CartShell

# TODO: need to review how general this functions are


__all__ = ["infer_shell_from_xyz", "axishell_create_structured_base",
           "cartshell_create_structured_base",
           "interpolate_solutions_on_shell",
           "shell_geom_type", "shell_repr"]


def infer_shell_from_xyz(mesh_xyz, geom_type, shape, bnd_uv=None, bnd_angles=None):
    """Build a shell from an AVBP Cut.

    Args:
        geom_type (str): Geometry type. Either "cart" or "axicyl".
        shape (array-like): Discretization of the shell. Shape (nu, nv).
        bnd_uv (array-like): Limits shell umin, umax, vmin, vmax. Shape (4,).
    """
    if bnd_angles is not None:
        raise DeprecationWarning("BNd_angle is not used anymore, Use bnd_uv")

    # Build shell
    if geom_type == "axicyl":
        shell = axishell_create_structured_base(
            shape[0],
            shape[1],
            mesh_xyz,
            bnd_uv)
    elif geom_type == "cart":
        shell = cartshell_create_structured_base(
            shape[0],
            shape[1],
            mesh_xyz)
    return shell


def axishell_create_structured_base(n_longi, n_azi, mesh, bnd_uv=None):
    """Obtain the base in a structured mesh.

    Assumes the mesh is the x_axis rotated extrusion of an (x,r) curve.

    Args:
        n_longi (int): Number of radial points of the shell.
        n_azi (int): Number of azimuthal points of the shell.
        mesh (array-like): Coordinates of the cut mesh of shape (n, 3).
        bnd_uv (array-like): Limits shell umin, umax, vmin, vmax. Shape (4,).

    Returns:
        AxiShell: An axysymmetric computational shell.
    """

    x_pos = mesh[:, 0].mean()
    r_vals = np.hypot(mesh[:, 1], mesh[:, 2])
    theta = np.rad2deg(yz_to_theta(mesh[:, :]))

    if bnd_uv is None:
        bnd_uv = [None, None, None, None]
    if bnd_uv[0] is None:
        bnd_uv[0] = r_vals.min()
    if bnd_uv[1] is None:
        bnd_uv[1] = r_vals.max()
    if bnd_uv[2] is None:
        bnd_uv[2] = theta.min()
    if bnd_uv[3] is None:
        bnd_uv[3] = theta.max()

    x_ctrl_pts = [x_pos, x_pos]
    r_ctrl_pts = [bnd_uv[0], bnd_uv[1]]
    tmin = bnd_uv[2]
    tmax = bnd_uv[3]

    shell = AxiShell(
        n_azi,
        n_longi,
        tmax - tmin,  # angle_range,
        x_ctrl_pts,  # x_vals[corners_idx],
        r_ctrl_pts,  # r_vals[corners_idx],
        tmin  # angle_min,
    )

    return shell


def cartshell_create_structured_base(n_longi, n_trans, mesh):
    """Obtain the base in a structured mesh.

    Assumes the path is square, aligned with y and z, with x = constant.

    Args:
        n_longi (int): Number of longitudinal points of the shell.
        n_trans (int): Number of transversal points of the shell.
        mesh (array-like): Coordinates of the cut mesh of shape (n, 3).

    Returns:
        CartShell: A cartesian computational shell.
    """

    x_vals, y_vals, z_vals = mesh.T

    zero = (x_vals.mean(), y_vals.min(), z_vals.min())
    v_max = (x_vals.mean(), y_vals.max(), z_vals.min())
    u_max = (x_vals.mean(), y_vals.min(), z_vals.max())
    shell = CartShell(
        n_trans,
        n_longi,
        zero,
        u_max,
        v_max,
    )

    return shell


def interpolate_solutions_on_shell(shell, mesh, raw_data,
                                   stencil=4,
                                   function=None,
                                   limitsource=None):
    """Interpolates solutions on a shell.

    Args:
        shell (Shell): A `Shell` object.
        mesh (np.array): Mesh with shape (n_pts, 3).
        data_dict (dict or np.array): Arrays shape = (n_steps, n_pts) or (n_pts,).
            Must contain keys if `np.array` (prefer `dict` for this case).

    Notes:
        Check cloud2cloud.CloudInterpolator for meaning of other arguments.
    """
    if function is None:
        function = lambda x: np.power(x, 4)

    # manipulate input
    is_dict = type(raw_data) == dict
    data_dict = raw_data if is_dict else _from_np_struct_to_dict(raw_data)

    # create interpolator
    interpolator = CloudInterpolator(mesh.reshape(-1, 3).T,
                                     shell.xyz.reshape(-1, 3).T,
                                     stencil=stencil,
                                     function=function,
                                     limitsource=limitsource)

    n = mesh.shape[0]
    results_dict = {}
    for key, array in data_dict.items():
        if array.shape[0] == n:
            results_dict[key] = interpolator.interp(array).reshape(*shell.shape)
        else:
            results_dict[key] = interpolator.interp(array.T).T.reshape(-1, *shell.shape)

    # manipulate output
    results = results_dict if is_dict else _from_dict_to_np_struct(results_dict)

    return results


def interpolate_solution_on_shell(shell, mesh, array, **kwargs):
    """Interpolates a solution on a shell.

    Args:
        shell (Shell): A `Shell` object.
        mesh (np.array): Mesh with shape (n_pts, 3).
        array (np.array): Shape = (n_steps, pts) or (n_pts,).
        kwargs : cloud2cloud.CloudInterpolator parameters.
    """
    return interpolate_solutions_on_shell(shell, mesh, {'data': array},
                                          **kwargs)['data']


def shell_geom_type(shell):
    """Return the shell geom_type.

    .. deprecated:: 0.2.0
        Use `shell.geom_type` instead.
    """
    return shell.geom_type


def shell_repr(shell):
    """Return a shell description.

    .. deprecated:: 0.2.0
        Use `repr(shell)` instead.
    """
    return repr(shell)


def _from_np_struct_to_dict(array):
    return {key: array[key] for key in array.dtype.names}


def _from_dict_to_np_struct(my_dict, type_=np.float32):
    keys = list(my_dict.keys())
    dtype = dict(names=keys, formats=[type_] * len(keys))
    shape = my_dict[keys[0]].shape

    array = np.empty(shape, dtype=dtype)
    for key, value in my_dict.items():
        array[key] = value

    return array
