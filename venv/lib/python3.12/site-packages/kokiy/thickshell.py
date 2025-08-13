"""Defines a 3D ThickShell object built by extruding 2D shells.
"""

import numpy as np
from scipy import interpolate

import meshio

from kokiy.shell import Shell
from kokiy._mesh_utils import get_bound_nodes
from kokiy._structured import get_uv
from kokiy._structured import interp_on_uv


class ThickShell(Shell):
    """A 3D shell object build by extruding a xyz mesh in the normals direction.

    Args:
        init_xyz (np.array): 2d mesh (e.g. from Shell2D).
        normals (np.array): 2d mesh normals with same shame as `init_xyz`.
        width_points (np.array): Width profile or width matrix.
        n_layers (int): Number of layers.
        shift (float): Additional depth.

    Attributes:
        shape (array-like): Shape of the shell.
    """
    geom_type = 'thick'

    def __init__(self, init_xyz, normals, width_points, n_layers=2, shift=0.0):

        shape = init_xyz.shape[:-1] + (n_layers,)
        super().__init__(shape)

        self.init_xyz = init_xyz
        self.normals = normals
        self.width_points = np.array(width_points)
        self.shift = shift
        # compute variables
        self._xyz, self._dz = self._bake()  # private for the deprecated object
        # to know where bottom is
        self._neg_dir = False if np.sign(width_points[0][-1]) == 1.0 else True

    @property
    def xyz(self):
        """xyz matrix of shape (`*self.shape`, 3).
        """
        return self._xyz

    @property
    def dz(self):
        """z-direction spacing matrix of shape `self.shape`.
        """
        return self._dz

    def _bake(self):
        n_layers = self.shape[-1]
        if self.width_points.shape == self.shape[:2]:
            width_matrix = self.width_points
        else:
            width_matrix = create_width_profile(self.width_points, self.shape[:2])

        xyz = self._compute_xyz(width_matrix, n_layers)
        dz = self._compute_dz(width_matrix, n_layers)

        return xyz, dz

    def _compute_xyz(self, width_matrix, n_layers):
        xyz = np.empty((*self.shape, 3))
        for j in range(3):
            n_j = np.take(self.normals, j, -1)
            for i in range(n_layers):
                xyz[:, :, i, j] = (np.take(self.init_xyz, j, -1)
                                + (1.0 * i / n_layers
                                    * width_matrix + self.shift) * n_j)

        return xyz

    def _compute_dz(self, width_matrix, n_layers):
        dz = np.empty(self.shape)
        for i in range(n_layers):
            dz[:, :, i] = abs(width_matrix / n_layers)
        dz[:, :, 0] /= 2
        dz[:, :, -1] /= 2

        return dz

    @classmethod
    def bake_from_shell(cls, shell, width_points, n_layers=1, shift=0.0):
        return ThickShell(shell.xyz, shell.n_xyz, width_points,
                          n_layers=n_layers, shift=shift)

    def dump(self, filename='shell', fields=None):
        """Dump shell geometrical and physical properties into hdf file.

        Args:
            filename (str): Name of the file to dump.
            fields (dict): Additional fields to dump. (key, `np.array`).
        """
        var_names = ['dz']
        self._dump(filename, var_names, fields=fields)

    def dump_shell(self, name='shell', fields=None):
        """Dump shell geometrical properties into hdf file.

        Args:
            name (str): Name of the file to dump. Default 'shell'.
            fields (dict): Additional fields to dump. (key, `np.array`).

        .. deprecated:: 0.2.0
            Use `dump` instead.
        """
        self.dump(filename=name, fields=fields)

    def _get_bound_nodes(self):
        return get_bound_nodes(*self.shape, reverse=self._neg_dir)

    def _get_bound_nodes_display(self, **kwargs):
        return self._get_bound_nodes()

    def _get_points(self):
        """Returns raveled points.
        """
        points = np.empty((0, 3))
        for k in range(self.shape[2]):
            layer_coords = np.take(self.xyz, k, axis=2)
            coords = np.array([np.take(layer_coords, i, axis=-1).ravel() for i in range(3)]).T
            points = np.r_[points, coords]

        return points

    def _get_cells(self, elem_type, **kwargs):
        return super()._get_cells(elem_type, reverse=self._neg_dir)

    def _reshape_simplified_bnd_conns(self, conns, show_all):
        size = conns.size

        if size == (self.shape[0] * self.shape[1]):  # bottom or top
            new_conns = self._reshape_simplified_bnd_conns_helper(
                conns, self.shape[1], layered=False, show_all=show_all)
        else:
            if size == (self.shape[0] * self.shape[2]):  # laterals
                ref_shape = self.shape[0]
            else:  # back and front
                ref_shape = self.shape[1]
            new_conns = self._reshape_simplified_bnd_conns_helper(
                conns, ref_shape, layered=True, show_all=show_all)

        return [meshio.CellBlock('line', new_conns)]

    def _compute_replicate_info(self, shape):
        # interpolates mesh, normals and width_matrix (if applicable) to new shape
        shape_2d = shape[:2]

        init_shape = self.init_xyz.shape[:2]
        init_uv = get_uv(*init_shape, matrix_form=True).reshape(-1, 2)
        new_uv = get_uv(*shape_2d, matrix_form=True).reshape(-1, 2)

        init_xyz = self.init_xyz.reshape(-1, 3)
        new_xyz = interp_on_uv(init_uv, init_xyz, new_uv).reshape(*shape_2d, 3)

        init_normals = self.normals.reshape(-1, 3)
        new_normals = interp_on_uv(init_uv, init_normals, new_uv).reshape(*shape_2d, 3)

        if self.width_points.shape == init_shape:
            new_width_points = interp_on_uv(init_uv,
                                            self.width_points.reshape(-1),
                                            new_uv).reshape(shape_2d)
        else:
            new_width_points = self.width_points

        return new_xyz, new_normals, new_width_points

    def replicate(self, shape):
        """Creates a similar instance, but possibly with different shape.
        """
        new_xyz, new_normals, new_width_points = self._compute_replicate_info(shape)

        return ThickShell(new_xyz, new_normals, new_width_points,
                          n_layers=shape[-1], shift=self.shift)


class CakeDict(ThickShell, dict):
    """A ThickShell that keeps dict behavior.

    Use ThickShell instead.

    .. deprecated:: 0.2.0
        Use `ThickShell` instead.
    """

    def __init__(self, *args, **kwargs):
        ThickShell.__init__(self, *args, **kwargs)
        dict.__init__(self, xyz=self._xyz, dz=self._dz)
        self.fields = self

    @property
    def xyz(self):
        return self['xyz']

    @property
    def dz(self):
        return self['dz']

    @classmethod
    def bake_from_shell(cls, shell, width_points, n_layers=1, shift=0.0):
        return CakeDict(shell.xyz, shell.n_xyz, width_points,
                        n_layers=n_layers, shift=shift)

    def dump(self, filename='shell', fields=None):
        """Dump shell geometrical and physical properties into hdf file.

        Args:
            filename (str): Name of the file to dump.
            fields (dict): Additional fields to dump. (key, `np.array`).
        """
        var_names = ['dz']
        self._dump(filename, var_names, fields=fields,
                   exclude_internal_fields=['xyz', 'dz'])

    def replicate(self, shape):
        """Creates a similar instance, but possibly with different shape.
        """
        new_xyz, new_normals, new_width_points = self._compute_replicate_info(shape)

        return CakeDict(new_xyz, new_normals, new_width_points,
                        n_layers=shape[-1], shift=self.shift)


def create_width_profile(width_points, shape):

    (x_tuple, y_tuple) = tuple(zip(*width_points))

    # Extend Bounds
    xlist = [0] + list(x_tuple) + [1]
    ylist = [y_tuple[0]] + list(y_tuple) + [y_tuple[-1]]
    # Generate continuous fictive spline
    f_int = interpolate.interp1d(xlist, ylist)
    # Generate discretise spline
    xnew = np.linspace(0, 1, num=shape[1])
    ynew = f_int(xnew)

    return np.tile(ynew, (shape[0], 1))
