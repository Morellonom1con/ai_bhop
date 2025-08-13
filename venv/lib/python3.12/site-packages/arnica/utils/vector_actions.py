""" Module concerning some 3D vector manipulations in numpy

OST :Mercy in Darkness, by Two Steps From Hell

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.spatial.transform import Rotation as Rot

SMALL = 1e-12


def renormalize(np_ar_vect):
    """ renormalize a numpy array of vectors considering the last axis
    """
    result = (
        np_ar_vect / np.maximum(
            np.expand_dims(np.linalg.norm(np_ar_vect, axis=-1),axis=-1),
            SMALL
        )
    )

    return result


def angle_btw_vects(np_ar_vect1, np_ar_vect2, convert_to_degree=False):
    """ compute the angle in deg btw two UNIT vectors """
    sdot = np.sum(renormalize(np_ar_vect1)
                  * renormalize(np_ar_vect2), axis=-1)

    # Clipping is needed when pscal.max() > 1 : 1.000000000002
    sdot = np.clip(sdot, -1.0, 1.0)
    angles = np.arccos(sdot)
    if convert_to_degree:
        angles = np.rad2deg(angles)
    return angles


def yz_to_theta(np_ar_vect):
    """ return theta , a radians measure of ange in the yz plane,
    - range -pi/pi
    - 0 on the y+ axis (z=0, y>0)
    spanning -pi to pi
                   0pi=0deg
                  Y
                  ^
                  |
                  |
-0.5pi=-90deg     o------>Z   0.5pi=90deg

    """
    # return np.arctan2(np_ar_vect[:, 2],
    #                   np_ar_vect[:, 1])
    return np.arctan2(np.take(np_ar_vect, 2, axis=-1),
                      np.take(np_ar_vect, 1, axis=-1))


def rtheta2yz(rrr, theta):
    """ return yz fror rtheta ,
    theta in  radians measure of ange in the yz plane,
    - range -pi/pi
    - 0 on the y+ axis (z=0, y>0)
    spanning -pi to pi

::

                        0pi=0deg
                      Y
                      ^
                      |
                      |
    -0.5pi=-90deg     o------>Z   0.5pi=90deg

    """
    yyy = rrr * np.cos(theta)
    zzz = rrr * np.sin(theta)
    return yyy, zzz


def rotate_vect_around_x(np_ar_vect, angle_deg):
    """ rotate vector around axis x in degree """

    arr_r = np.hypot(np.take(np_ar_vect, 1, axis=-1),
                     np.take(np_ar_vect, 2, axis=-1))
    arr_theta = (yz_to_theta(np_ar_vect)
                 + np.deg2rad(angle_deg))

    result = np.stack((np.take(np_ar_vect, 0, axis=-1),
                       arr_r * np.cos(arr_theta),
                       arr_r * np.sin(arr_theta)),
                      axis=-1)

    return result

def rotate_vect_around_axis(xyz, *tuples_rot):
    """
    *Rotate vector around vector or series of vector*

    :param xyz: Array of xyz-coordinates of shape (n,3)
    :param tuples_rot: List of tuple with rotation data : \
                       Axis array of shape (3,) axis, \
                       Float angle in degree

    :returns: Array of rotated xyz-coordinates of shape (n,3)
    """
    quat = Rot.from_rotvec([0, 0, 0])
    for axis, angle in tuples_rot:
        rot_vect = np.multiply(renormalize(axis),
                               angle * np.pi / 180.)
        quat *= Rot.from_rotvec(rot_vect)

    return quat.apply(xyz)

def dilate_vect_around_x(azimuth, np_ar_vect, angle_deg_init=None, angle_deg_targ=360):
    """ dilate vectors around axis x from a specified initial range angle
        to a target range angle.
    Parameters :
    ------------
    np_ar_vect : numpy array of dim (n,3)
    angle_deg_targ : tuple or float

    Returns :
    ---------
    numpy array of dim (n,3)
    """

    if angle_deg_init is None:
        angle_deg_init = np.max(azimuth) - np.min(azimuth)
    dilate_factor = angle_deg_targ / angle_deg_init

    result = rotate_vect_around_x(np_ar_vect,
                                  azimuth * (dilate_factor - 1))
    return result


def cart_to_cyl(vects_xyz):
    """
    *Transform vects from xyz-system to xrtheta-system*

    x -> x     : x = x
    y -> r     : r = sqrt(y^2 + z^2)
    z -> theta : theta = arctan2(z,y)

    :param vects_xyz: Array of dim (n,3) of xyz components
    :type vects_xyz: np.array

    :return:

        - **vects_cyl** - Array of dim (n,3) of xrtheta components
    """

    np_x = np.take(vects_xyz, 0, -1)
    np_r = np.sqrt(np.take(vects_xyz, 1, -1) ** 2
                   + np.take(vects_xyz, 2, -1) ** 2)
    np_theta = yz_to_theta(vects_xyz)

    vects_cyl = np.stack((np_x, np_r, np_theta), axis=-1)

    return vects_cyl

def clip_by_bounds(points_coord, bounds_dict, keep="in", return_mask=False):
    """
    *Clip a cloud by keeping only or removing a bounded region*

    The dict to provide must be filled as follow :
    bounds_dict = {component_1 : (1_min, 1_max),
                   compoment_2 : (2_min, 2_max),
                   ...}

    component_1 = ["x", "y", "z", "r", "theta"]

    The bounded region can either be :

        - A 1D slice if only 1 component is provided ;
        - A 2D box if 2 components are provided ;
        - A 3D box if 3 components are provided.

    If keep="in", returns the point coordinates inside the bounds.
    If keep="out", returns the point coordinates outside the bounds.

    If returns=True, returns the coordinates clipped
    If returns=False, returns the mask of boolean than can be applied\
    on other arrays

    :param point_cloud: Array of dim (n,k) of coordinates
    :type point_cloud: np.array
    :param bounds_dict: Dict of MAX lengh k of tuple of floats
    :param keep: Either keeps what is inside or outside
    :type keep: str

    :returns:

        - **points_coord_clipped** - Array of dim (m,k) with m<=n
        OR
        - **mask** - Array of dim (n,) of booleans
    """
    mask = np.ones(points_coord.shape[0], dtype=bool)
    for dim, bounds in bounds_dict.items():
        mask &= mask_cloud(points_coord, dim, bounds)

    if keep == "out":
        mask = np.invert(mask)

    if return_mask:
        return mask
    return points_coord[mask]

def cyl_to_cart(vects_cyl):
    """
    *Transform vects from xrtheta-system to xyz-system*

    x -> x     : x = x
    r -> y     : y = r * cos(theta)
    theta -> z : z = r * sin(theta)

    :param vects_cyl: Array of dim (n,3) of xrtheta components
    :type vects_cyl: np.array

    :return:

        - **vects_xyz** - Array of dim (n,3) of xyz components
    """

    np_x = np.take(vects_cyl, 0, -1)
    np_y = np.take(vects_cyl, 1, -1) * np.cos(np.take(vects_cyl, 2, -1))
    np_z = np.take(vects_cyl, 1, -1) * np.sin(np.take(vects_cyl, 2, -1))

    vects_xyz = np.stack((np_x, np_y, np_z), axis=-1)

    return vects_xyz

def vect_to_quat(vect_targ, vect_source):
    """
    *Generate a quaternion from two vectors*

    A quaternion is a rotation object. From two vectors,\
    the rotation angle and the rotation axis are computed.
    The rotation vector generates then a quaternion for each
    serie of vectors.

    :param vect_targ: Array of dim (n,3) of vect components
    :type vect_targ: np.array
    :param vect_source: Array of dim (n,3) of vect components
    :type vect_source: np.array

    :return:

        - **quat** - Array of quaternion of dim (n,)
    """
    vect_targ = renormalize(vect_targ)
    vect_source = renormalize(vect_source)

    prod_scal = np.clip(np.sum(vect_targ * vect_source,
                               axis=-1),
                        -1, 1)

    rot_ang = np.arccos(prod_scal)
    if vect_targ.ndim >= 2:
        rot_ang = rot_ang[:, np.newaxis]
    else:
        rot_ang = np.array([rot_ang])

    rot_axe = np.cross(vect_source, vect_targ)
    idx = np.where(rot_ang > 1e-6)[0]
    rot_axe[idx] = renormalize(rot_axe[idx])

    rot_vect = np.multiply(rot_ang, rot_axe)

    quat = Rot.from_rotvec(rot_vect)

    return quat


def make_radial_vect(coord, vects):
    """
    *Recalibrate vectors to make them radial.*

    The vectors are readjusted to cross x-axis. It is mainly done for nodes
    on the limit of the boundary for axi-cylindrical geometries.

    :param coord: Array of dim (n,3) of float coordinates
    :type coord: np.array
    :param vects: Array of dim (n,3) of float components
    :type vects: np.array

    :return:

        - **radial_vect** - Array of dim (n,3) of float components
    """

    tol = 10
    diff_angle = (yz_to_theta(coord) - yz_to_theta(vects)) * 180. / np.pi
    diff_angle = np.where(diff_angle > 90., diff_angle - 180., diff_angle)
    diff_angle = np.where(diff_angle < -90., diff_angle + 180., diff_angle)
    diff_angle = np.where(np.absolute(diff_angle) > tol, 0.0, diff_angle)

    radial_vect = rotate_vect_around_x(vects, diff_angle)
    return radial_vect

def mask_cloud(np_ar_xyz,
               axis,
               support):
    """ mask a cloud of n 3D points in in xyz
    axis among x,y,z,theta,r
    support a 2 value tuple : (0,3), (-12,float(inf))

    x and (0,3) reads as  0 <= x < 3 ( lower bound inclusive)
    z and (-12,float(inf)) reads as  -12 <= z
    theta in degree, cyl. coordinate around x axis
    - range -180,180
    - 0 on the y+ axis (z=0, y>0)
    """

    def true_between(np_arr, low, high):
        """ return a boolean array of the same shape,
        true if value are btw low and high"""
        return (np_arr >= low) * (np_arr <= high)

    possible_axes = ["x", "y", "z", "theta", "r"]
    if axis not in possible_axes:
        raise IOError("axis " + axis + "not among possible axes "
                      + ";".join(possible_axes))

    if axis == "x":
        mask = true_between(
            np.take(np_ar_xyz, 0, axis=-1),
            support[0],
            support[1])
    if axis == "y":
        mask = true_between(
            np.take(np_ar_xyz, 1, axis=-1),
            support[0],
            support[1])
    if axis == "z":
        mask = true_between(
            np.take(np_ar_xyz, 2, axis=-1),
            support[0],
            support[1])
    if axis == "r":
        mask = true_between(
            np.hypot(np.take(np_ar_xyz, 1, axis=-1),
                     np.take(np_ar_xyz, 2, axis=-1)),
            support[0],
            support[1])
    if axis == "theta":
        mask = true_between(
            np.rad2deg(yz_to_theta(np_ar_xyz)),
            support[0],
            support[1])
    return mask
