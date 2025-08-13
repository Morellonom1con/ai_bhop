"""Defines base class from which other `Shell` objects will inherit.
"""

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from scipy import spatial

import meshio

from arnica.utils import nparray2xmf
import yamio
from yamio.ensight.gold import GeoWriter
from yamio.mesh_utils import get_local_points_and_cells

from kokiy._mesh_utils import get_conns


MAX_SPLINE_ORDER = 3
SPLINE_SMOOTHNESS = 0

AXIS_NAMES = ['time', 'v', 'u']


class Shell(metaclass=ABCMeta):
    """Abstract shell.

    Args:
        shape (np.array): Shape of the shell.

    Attributes:
        shape (array-like): Shape of the shell.
        fields (dict): Nodal data.
    """

    def __init__(self, shape):
        self.shape = shape
        self.fields = {}

    def __repr__(self):
        rep = f'Shell\n-----\n\nshape : {self.shape}\ntype : {self.geom_type}'

        return rep

    def set_mask_on_shell(self, point_cloud, tol):
        """Create a mask on the shell from a point cloud.

        Args:
            point_cloud (np.array): Coordinates with shape (n, 3).
            tol (float): Tolerance.

        Notes:
            The mask value is 0 for shell points located near cloud points.
            Otherwise the mask value is 1.
        """

        # Create a KDTree from cloud points
        kdtree = spatial.KDTree(point_cloud)

        # Compute distances between cloud points and
        # the closest neighbor from the 2D shell points
        dists, _ = kdtree.query(self.xyz, k=1)

        # Mask takes 1 if the distance is above tolerance.
        # TODO: make int (breaks oms tests)
        mask = np.where(dists.reshape(self.shape) > tol, 1., 0.)

        return mask

    def _dump(self, name, var_names, fields=None, exclude_internal_fields=()):
        """Dump shell geometrical properties into hdf file.

        Args:
            name (str): Name of the file to dump. Default 'shell'.
            fields (dict): Additional fields to dump. (key, `np.array`).
        """
        # TODO: rename to just `dump`?
        fields = fields or {}

        # grid
        xdmf = nparray2xmf.NpArray2Xmf(f'{name}.h5')
        xdmf.create_grid(np.take(self.xyz, 0, axis=-1),
                         np.take(self.xyz, 1, axis=-1),
                         np.take(self.xyz, 2, axis=-1))

        # internal vars
        for var_name in var_names:
            xdmf.add_field(getattr(self, var_name), var_name)

        # internal fields
        for field_name, field_value in self.fields.items():
            if field_name in exclude_internal_fields:
                continue
            xdmf.add_field(field_value, field_name)

        # additional fields
        for key, value in fields.items():
            xdmf.add_field(value, key)

        xdmf.dump()

    def export_geo(self, filename, show_all=False, **kwargs):
        """Exports simplified boundary representation for visualization.

        Uses ensight gold format.

        Args:
            show_all (bool): Show all boundary edges?
            kwargs: will be passed to writer.write method.

        Notes:
            If goal is to proper export original mesh use `export_mesh`.
        """

        points = self._get_points()
        bnd_node_sets = self._get_bound_nodes_display(show_all=show_all)

        parts = {}
        for name, conns in bnd_node_sets.items():
            cells = self._reshape_simplified_bnd_conns(conns, show_all=show_all)
            new_points, new_cells = get_local_points_and_cells(points, cells)
            parts[name] = meshio.Mesh(new_points, new_cells)

        geo_writer = GeoWriter()
        geo_writer.write(filename, parts, **kwargs)

    def export_mesh(self, filename, elem_type, **kwargs):
        """Exports mesh.

        Args:
            filename (str): file name.
                Extension guides file type (any acceptable by `yamio`).
            elem_type (str): 'tri', 'quad', 'tetra', 'hexahedron'.
            kwargs: will be passed to writer.write method.

        Notes:
            `hip` does not allow 3d surface meshes.
        """
        mesh = self.get_mesh(elem_type)
        yamio.write(filename, mesh, **kwargs)

    def get_mesh(self, elem_type):
        """Returns a yamio.Mesh.
        """
        points, cells = self._get_points_and_cells(elem_type)
        bnd_node_groups = self._get_bound_nodes()

        return yamio.Mesh(points, cells, bnd_patches=bnd_node_groups)

    def _get_cells(self, elem_type, **kwargs):
        conns = get_conns(*self.shape, elem_type=elem_type, **kwargs)
        return [meshio.CellBlock(elem_type, conns)]

    def _get_points_and_cells(self, elem_type):
        return self._get_points(), self._get_cells(elem_type)

    @abstractmethod
    def _get_bound_nodes(self):
        pass

    @abstractmethod
    def _get_bound_nodes_display(self, **kwargs):
        pass

    @abstractmethod
    def _get_points(self):
        pass

    @abstractmethod
    def _reshape_simplified_bnd_conns(self, conns, show_all):
        pass

    @abstractmethod
    def replicate(self, shape):
        pass

    @staticmethod
    def _reshape_simplified_bnd_conns_helper(conns, ref_size, layered, show_all,
                                             rm_bnd_lines=False):
        """Connects surface nodes by lines.

        Args:
            ref_size (int): Dimension of size of reference (i.e. number of
                nodes by layer).
            layered (bool): If False, inner lines in reference direction are
                ignored. If show_all is True, then it is ignored.
            show_all (bool): If True, all the lines are drawn.
            rm_bnd_lines (bool): If True, removes boundary lines parallel to main
                direction.
        """
        # TODO: make this a standalone function? (may be useful in different contexts)

        longi = conns.copy().reshape(-1, ref_size)
        if not layered and not show_all:
            longi = longi[[0, -1], :]
        repeats_layer = [1] + [2 for _ in range(longi.shape[1] - 2)] + [1]
        repeats = repeats_layer * longi.shape[0]
        new_conns_longi = np.repeat(longi.ravel(), repeats, axis=0).reshape(-1, 2)

        if show_all:
            azi = conns.copy().reshape(-1, ref_size).T
        else:
            azi = conns.copy().reshape(-1, ref_size)[:, [0, -1]].T
        repeats_layer = [1] + [2 for _ in range(azi.shape[1] - 2)] + [1]
        repeats = repeats_layer * azi.shape[0]
        new_conns_azi = np.repeat(azi.ravel(), repeats, axis=0).reshape(-1, 2)

        if rm_bnd_lines:
            new_conns_longi = new_conns_longi[azi.shape[0] - 1:-(azi.shape[0] - 1), :]
            new_conns_azi = new_conns_azi[longi.shape[0] - 1:-(longi.shape[0] - 1), :]

        return np.r_[new_conns_longi, new_conns_azi]
