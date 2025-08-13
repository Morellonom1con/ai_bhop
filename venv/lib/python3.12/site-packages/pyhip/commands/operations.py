"""This module contains hip functionnalities
    intending to perform mesh operations
"""
from pyhip.hipster import pyhip_cmd

__all__ = ["generate_mesh_2d_3d",
           "adapt_with_factor",
           "adapt_with_var",
           "interpolate",
           "duplicate",
           "transform_translate",
           "transform_scale",
           "transform_rotate",
           "transform_reflect",
           "set_bctype",
           "set_bctext",
           "set_checklevel",
           "list_periodic",
           "list_surface",
           "hip_exit"]

def generate_mesh_2d_3d(lower_corner, upper_corner, resolution, convert2tri=False,
                        extru_axis=None, extru_range=None, extru_res=None, **kwargs):
    """
    Generate an unstructured mesh in 2D or 3D
    given bounding box.
    Can be converted from quat to tri or from hexa to tetra

    :param lower_corner: List/tuple (x_min, y_min) or (x_min, y_min, z_min)
    :param upper_corner: List/tuple (x_min, y_min) or (x_min, y_min, z_min)
    """
    cmds = []
    l_c = isinstance(lower_corner, (list, tuple)) and len(lower_corner) == 2
    u_c = isinstance(upper_corner, (list, tuple)) and len(upper_corner) == 2
    res = isinstance(resolution, (list, tuple)) and len(resolution) == 2
    if not (l_c and u_c and res):
        raise ValueError("Corners and resolution must be either lists or tuples"
                         "of 2 elements.")

    cmds += _generate_2d_mesh(lower_corner,
                              upper_corner,
                              resolution,
                              **kwargs)

    if all(var is not None for var in (extru_axis, extru_range, extru_res)):
        if not (isinstance(lower_corner, (list, tuple)) and len(lower_corner) == 2):
            raise ValueError("extru_range must be either list or tuple of 2 elements.")

        cmds += _extrude_2d_mesh(extru_range,
                                 extru_res,
                                 extru_axis,
                                 **kwargs)
        if convert2tri:
            cmds += _convert_hex2tet(**kwargs)
    elif convert2tri:
        cmds += _convert_quad2tri(**kwargs)

    return cmds

def _generate_2d_mesh(lower_corner, upper_corner, resolution, **kwargs):
    """Generate an unstructured rectangular mesh
       of quadrilateral elements given box bounds

       Parameters:
       ==========
       lower_corner: A tuple/list of box lower corner
                     coordinates (x, y)
       upper_corner: A tuple/list of box upper corner
                     coordinates (x, y)
       resolution : a tuple/list of mesh resolution
                    (i.e number of points) (n_x, n_y)
    """
    if lower_corner[0] > upper_corner[0] or lower_corner[1] > upper_corner[1]:
        raise ValueError(f"Lower corner coords {lower_corner} must be "
                         f"lower than upper corner coords {upper_corner}")

    command = (f"{lower_corner[0]} {lower_corner[1]} {upper_corner[0]}"
               f" {upper_corner[1]} {resolution[0]} {resolution[1]}")
    command = f"generate {command}"

    return pyhip_cmd(command, **kwargs)

def _extrude_2d_mesh(extrude_coords, extrude_res, axis, **kwargs):
    """ Extrude 2-dimensional mesh to a 3D mesh.

        Parameters:
        ==========
        extrude_coords: a tuple/list of extrusion extremes
                        (ext_min, ext_max), ext_min and ext_max
                        can refer to positions (if extusion is
                        done around x, y or z axis) or to angles
                        in degrees (if extusion is choosen to be
                        axisymmetric)
        extrude_node_num: Number of elements slices
        axis: axis around which extrusion is performed, possible
              values :
               - x, y, z or axi
    """
    if axis not in ['x', 'y', 'z', 'axi']:
        raise ValueError("extrusion axis must be in 'x', 'y', 'z', 'axi'")
    if extrude_coords[0] > extrude_coords[1]:
        raise ValueError(f"Lower coords {extrude_coords[0]} must be "
                         f"lower than upper coords {extrude_coords[1]}")

    command = f"copy 3D {extrude_coords[0]} {extrude_coords[1]} {extrude_res} {axis}"

    return pyhip_cmd(command, **kwargs)

def _convert_quad2tri(**kwargs):
    """ Convert current quad mesh to triangle elements.

        Parameters:
        ==========
        none
    """
    command = 'copy q2t'

    return pyhip_cmd(command, **kwargs)

def _convert_hex2tet(**kwargs):
    """ Convert current hex mesh to tet elements.

        Parameters:
        ==========
        none
    """
    command = 'copy 2tets'

    return pyhip_cmd(command, **kwargs)

def adapt_with_factor(factor, hgrad=1.4, hmin=0.0, **kwargs):
    """ Adapt mesh using a coefficient.

        Parameters:
        ==========
        isofactor : a double parameters by which every edge length
                    will be multiplied
        hgrad : *optional*, maximum gradient between edge sizes during adaptation
        Hdist : *optional*, Hausdorff maximum distance
        hmin : *optional*, minimal edge size
        hmax : *optional*, maximal edge size
    """
    command = f"mmg3d -f {factor}"
    if hgrad != 1.4:
        command += f" -g {hgrad}"
    if hmin != 0.0:
        command += f" -l {hmin}"

    return pyhip_cmd(command, **kwargs)

def adapt_with_var(variable, hgrad=1.4, hmin=None, **kwargs):
    """ Adapt mesh using a variable.

        Parameters:
        ==========
        variable : a variable name (string) contained in the current memory
        hGrad : *optional*, maximum gradient between edge sizes during adaptation
        Hdist : *optional*, Hausdorff maximum distance
        hmin : *optional*, minimal edge size
        hmax : *optional*, maximal edge size
    """
    command = f"mmg3d -v {variable}"
    if hgrad != 1.4:
        command += f" -g {hgrad}"
    if hmin is not None:
        command += f" -l {hmin}"

    return pyhip_cmd(command, **kwargs)

def interpolate(grid_id=1, **kwargs):
    """ interpolates solution to current mesh in memory
        Parameters:
        ==========
        GridId : id of the grid info to use as source (optional, default 1)
    """
    command = f"interpolate grid {grid_id}"

    return pyhip_cmd(command, **kwargs)

def duplicate(ncopies, value, operation='rotate', axis='x', **kwargs):
    """ copies a mesh (and solution) using the operation type with the values parameters

        Parameters:
        ==========
        ncopies : int number of copies
        axis : 'x', 'y', or 'z'
        angle : double value of the angle
    """
    if operation == 'rotate':
        if axis not in ['x', 'y', 'z']:
            raise ValueError("Duplication by rotation's axis must be in 'x', 'y', 'z'")
        if not isinstance(value, float):
            raise ValueError("Duplication by rotation's value must be a float angle")
    elif operation == 'translate':
        axis = ''
        if isinstance(value, list) and len(value) == 3:
            value = f"{value[0]} {value[1]} {value[2]}"
        else:
            raise ValueError("Duplication by translation value must be a 3-elements list")
    else:
        raise ValueError("Duplication operation must be 'rotate' or 'translate'")

    if isinstance(ncopies, int):
        command = f"copy uns {ncopies} {operation} {axis} {value}"
    else:
        raise ValueError("The number of copies (ncopies) must be an integer")

    return pyhip_cmd(command, **kwargs)

def transform_translate(d_x, d_y, d_z, **kwargs):
    """ translates mesh along a direction
        Parameters:
        ==========
        direction : tuple of 3 direction components (dx, dy, dz)
    """
    command = f"transform translate {d_x} {d_y} {d_z}"

    return pyhip_cmd(command, **kwargs)

def transform_scale(s_x, s_y, s_z, **kwargs):
    """ apply a scaling by factor in each direction
        Parameters:
        ==========
        factor : tuple of 3 scaling factor (sx, sy, sz)
    """
    command = f"transform scale {s_x} {s_y} {s_z}"

    return pyhip_cmd(command, **kwargs)

def transform_rotate(axis, angle, **kwargs):
    """ rotate a mesh around x, y or z-axis by an angle in degree
        The right hand rule is applied.
        Parameters:
        ==========
        axis : str 'x', 'y' or 'z' of the axis
        angle : float value of the angle
    """
    if axis in ['x', 'y', 'z']:
        command = f"transform rotate {axis} {angle}"
    else:
        raise ValueError("rotation axis must be in 'x', 'y', 'z'")

    return pyhip_cmd(command, **kwargs)

def transform_reflect(axis, **kwargs):
    """ reflect a mesh around a plane perpendicular to the axis at the origin
        Parameters:
        ==========
        axis : str 'x', 'y' or 'z' of the axis
    """
    if axis in ['x', 'y', 'z']:
        command = f"transform reflect {axis}"
    else:
        raise ValueError("duplication axis must be in 'x', 'y', 'z'")

    return pyhip_cmd(command, **kwargs)

def set_bctype(patch, bctype, **kwargs):
    """ set boundary condition type on a specific patch
        Parameters:
        ==========
        patch : str patch name
        bctype : BC type of the patch
    """
    command = f"set bc-type {patch} {bctype}"

    return pyhip_cmd(command, **kwargs)

def set_bctext(patch, bctext, **kwargs):
    """ set boundary condition type on a specific patch
        Parameters:
        ==========
        patch : str patch name
        bctype : BC type of the patch
    """
    command = "set bc-text %s %s" % (patch, bctext)

    return pyhip_cmd(command, **kwargs)

def set_checklevel(level, **kwargs):
    """ Set check level
    """
    if not isinstance(level, int):
        raise TypeError("the level given must be an integer between 0 and 5")
    if not -1 < level < 6:
        raise ValueError("Check levels are only between 0 and 5 included")
    command = f"set checklevel {level}"

    return pyhip_cmd(command, **kwargs)

def list_periodic(**kwargs):
    """ List periodicities if exist
    """
    command = "list periodic"

    return pyhip_cmd(command, **kwargs)

def list_surface(**kwargs):
    """ List surfaces if exist
    """
    command = "list surface"

    return pyhip_cmd(command, **kwargs)

def hip_exit(**kwargs):
    """ exit hip
    """
    command = "exit"

    return pyhip_cmd(command, **kwargs)
