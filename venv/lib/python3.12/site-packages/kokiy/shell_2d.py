"""Defines base class from which other 2D `Shell` objects will inherit.
"""

from abc import ABCMeta

import numpy as np
import matplotlib.pyplot as plt
import meshio

from kokiy.shell import Shell
from kokiy.thickshell import CakeDict
from kokiy.thickshell import create_width_profile
from kokiy._mesh_utils import get_bound_nodes
from kokiy._structured import get_uv


AXIS_NAMES = ['time', 'v', 'u']


class Shell2D(Shell, metaclass=ABCMeta):
    """Abstract shell.

    Args:
        n_axe_1 (int): Number of x/longitudinal shell points.
        n_axe_2 (int): Number of y/azimuthal shell points.

    Attributes:
        shape (array-like): Shape of the shell: (`n_axe_1`, `n_axe_2`).
        rad (np.array): rad matrix of shape `self.shape`.
        theta (np.array): theta matrix of shape `self.shape`.
        x (np.array): x matrix of shape `self.shape`.
        y (np.array): y matrix of shape `self.shape`.
        z (np.array): z matrix of shape `self.shape`.
        n_x (np.array): x-normal matrix of shape `self.shape`.
        n_r (np.array): r-normal matrix of shape `self.shape`.
        n_y (np.array): y-normal matrix of shape `self.shape`.
        n_z (np.array): z-normal matrix of shape `self.shape`.
        du (np.array): u-direction spacing matrix of shape `self.shape`.
        dv (np.array): v-direction spacing matrix of shape `self.shape`.
        u (np.array): u-direction adimensional matrix of shape `self.shape`.
        v (np.array): v-direction adimensional matrix of shape `self.shape`.
        abs_curv (np.array): Central curvilinear abscissa array of shape (`self.shape[1]`,).
        dwu (np.array): Weighted u-direction spacing matrix of shape `self.shape`.
        dwv (np.array): Weighted v-direction spacing matrix of shape `self.shape`.
        surf (np.array): Weighted surface matrix of shape `self.shape`.
    """

    def __init__(self, n_axe_1, n_axe_2, lims_u=(0., 1.), lims_v=(0., 1.)):

        super().__init__((n_axe_1, n_axe_2))
        
        self.axis_names = AXIS_NAMES
        self.width_matrix = {}
        self.u, self.v = get_uv(n_axe_1, n_axe_2, lims_axe_1=lims_u,
                                lims_axe_2=lims_v)

    @property
    def uv(self):
        """uv matrix of shape `self.shape`.
        """
        return np.stack((self.u, self.v), axis=-1)

    @property
    def xyz(self):
        """xyz matrix of shape (`*self.shape`, 3).
        """
        return np.stack((self.x, self.y, self.z), axis=-1)

    @property
    def n_xyz(self):
        """Normals with same shape as `xyz`.
        """
        return np.stack((self.n_x, self.n_y, self.n_z), axis=-1)

    @property
    def surf(self):
        """Surface element [m2]
        """
        return np.multiply(self.dwu, self.dwv)

    @property
    def rad(self):
        """radius of point [m]
        """
        return np.hypot(self.y , self.z)

    @property
    def theta(self):
        """azimuth of point [rad]
        """
        return np.arctan2(self.z, self.y)

    @property
    def n_r(self):
        """radial component of normal [-]
        """
        return self.n_y * np.cos(self.theta)+self.n_z * np.sin(self.theta)

    @property
    def abs_curv(self):
        """Curvilinear abcissa on first dimension U [m]
        """
        return np.cumsum(np.take(self.du, 0, 0)) - self.du[0, 0]    

    @property
    def u_adim(self):
        """
        .. deprecated:: 0.2.0
            Use `u` instead.
        """
        return self.u

    @property
    def v_adim(self):
        """
        .. deprecated:: 0.2.0
            Use `v` instead.
        """
        return self.v

    def dump(self, filename='shell', fields=None):
        """Dump shell geometrical and physical properties into hdf file.

        Args:
            filename (str): Name of the file to dump.
            fields (dict): Additional fields to dump. (key, `np.array`).
        """
        fields = fields or {}

        var_names = self._get_dump_var_names()

        # due to name inconsistency
        fields.update({'r': self.rad})

        # dump
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

    def add_curviwidth(self, label, points):
        """Add a 2D width matrix of shell shape extruded from points spline.

        Args:
            label (str): Label of the width matrix.
            points (array-like): Coordinates with shape (n, 2).

        Notes:
            This method makes this object stateful. Avoid it if you can.
        """
        width_profile = create_width_profile(points, self.shape)
        if label in self.width_matrix:
            self.width_matrix[label] += width_profile
        else:
            self.width_matrix[label] = width_profile

    def bake_millefeuille(self, width_matrix_label, n_layers, shift=0.0):
        """Create a millefeuille-like shell.

        Extrude a 2D shell in the normal direction up
        pointwise height given by "width_matrix_label" matrix.

        Args:
            width_matrix_label (str): Label of the width matrix.
            n_layers (int): Number of layer for extrusion.
            shift (float): Additional depth.

        Returns:
            ThickShell

        Notes:
            Use `Thickshell.bake_from_shell` when possible.

        "Bon appetit!"
        """
        width_matrix = self.width_matrix[width_matrix_label]
        return CakeDict.bake_from_shell(self, width_matrix, n_layers=n_layers,
                                        shift=shift)

    def average_on_shell_over_dirs(self, variable, directions, scale=True):
        """Performs an integration (average) over one or multiple directions.

        Args:
            variable (np.array): Data to be averaged of shape (n_time, n_v, n_u).
            directions (array-like): A list() of directions on which the average process
                is to be performed. Contains keywords from ['time', 'v', 'u'].
            scale (bool): Keeps nominal averaging (if `False`) or takes into account
                surface element (if `True`).

        Returns:
            np.array: Averaged data on given directions.
        """
        if scale:
            variable = np.multiply(variable, self.surf)

        return self.operate_on_shell_over_dirs(variable, directions,
                                               operator=np.mean)

    def operate_on_shell_over_dirs(self, variable, directions, operator=np.mean):
        """Applies an operator over one or multiple direction.

        Notes:
            Directions are not commutative for non-linear operators.
        """

        # check valid directions
        if not (len(directions) > 0 and all(dir_ in self.axis_names for dir_ in directions)):
            mess = 'Integrating directions do not conform to criteria\n'
            mess += 'It should be a list containing one of or all items\n'
            for axis in self.axis_names:
                mess += axis + ', '
            raise ValueError(mess)

        # operate
        dirs = self.axis_names.copy()
        for dir_ in directions:
            index = dirs.index(dir_)
            variable = operator(variable, axis=index)
            dirs.pop(index)

        return variable

    def plot(self, savefile=None):
        """Plot 2D curve of the `Shell`.

        Args:
            savefile (str): filename.
                If None, the figure is only plotted.
        """

        plt.plot(self.xyz[0, :, 0], self.xyz[0, :, 1])
        plt.title("Shell")
        plt.xlabel("x")
        plt.ylabel("y")

        if savefile is None:
            plt.show()
        else:
            plt.savefig(savefile)

    def _invert_normals(self, normal_vars=('n_x', 'n_y', 'n_z')):
        for var in normal_vars:
            setattr(self, var, -getattr(self, var))

    def invert_normals(self):
        """Invert directions of normal vectors.
        """
        self._invert_normals()

    def rad_theta_components(self, vect_y, vect_z):
        """Computes the radial and azimuthal components of a vect y, z.
        """
        # TODO: example of use case?

        unit_rad_y = self.y / self.rad
        unit_rad_z = self.z / self.rad

        unit_tgt_y = -unit_rad_z
        unit_tgt_z = unit_rad_y

        vect_rad = vect_y * unit_rad_y + vect_z * unit_rad_z
        vect_tgt = vect_y * unit_tgt_y + vect_z * unit_tgt_z

        return vect_rad, vect_tgt

    def _get_points(self):
        return self.xyz.reshape(-1, 3)

    def _reshape_simplified_bnd_conns(self, conns, **kwargs):
        size = conns.size
        if size == self.shape[0] or size == self.shape[1]:  # not inner
            new_conns = conns.copy()
            repeats = [1] + [2 for _ in range(new_conns.shape[0] - 2)] + [1]
            new_conns = np.repeat(new_conns, repeats, axis=0).reshape(-1, 2)
        else:
            new_conns = self._reshape_simplified_bnd_conns_helper(
                conns, self.shape[1], False, True, rm_bnd_lines=True)

        return [meshio.CellBlock('line', new_conns)]

    def _get_bound_nodes_display(self, show_all=False):
        bnd_nodes = self._get_bound_nodes()
        if show_all:
            # all nodes, not only inner (then connections of bnd are ignored)
            bnd_nodes['Inner'] = np.array([i for i in range(np.prod(self.shape))])

        return bnd_nodes

    def _get_bound_nodes(self):
        return get_bound_nodes(*self.shape)

    # def _compute_abs_curv(self, du):
    #     return np.cumsum(np.take(du, 0, 0)) - du[0, 0]

    def _compute_weight_interv_adim(self, du, dv):
        dwu = du.copy()
        dwu[:, (0, -1)] /= 2

        dwv = dv.copy()
        dwv[(0, -1), :] /= 2

        return dwu, dwv

    # def _compute_surf(self, dwu, dwv):
    #     return np.multiply(dwu, dwv)

    def _get_dump_var_names(self):
        return ['theta', 'n_x', 'n_y', 'n_z', 'n_r', 'du', 'dv',
                'u', 'v', 'dwu', 'dwv', 'surf']

def compute_normals_from_xyz(x,y,z, perio_u=False, perio_v=False):
    """Compute normals from coordinates"""
    du_vect=np.stack([
        np.gradient(x,axis=0,edge_order=2),
        np.gradient(y,axis=0,edge_order=2),
        np.gradient(z,axis=0,edge_order=2)
    ], axis=2)
    dv_vect=np.stack([
        np.gradient(x,axis=1,edge_order=2),
        np.gradient(y,axis=1,edge_order=2),
        np.gradient(z,axis=1,edge_order=2)
    ], axis=2)
    
    dn = np.cross(du_vect,dv_vect,axisc=2) 
    
    if perio_u:
        avg = 0.5*(dn[0,:,:]+dn[-1,:,:])
        dn[0,:,:]=avg
        dn[-1,:,:]=avg
       
    if perio_v:
        avg = 0.5*(dn[:,0,:]+dn[:,-1,:])
        dn[:,0,:]=avg
        dn[:,-1,:]=avg
       
    dn /= np.linalg.norm(dn,axis=2)[:,:,np.newaxis]

    #dn= clean_zeros(dn)
    return dn[:,:,0],dn[:,:,1],dn[:,:,2]

def clean_zeros(arr, tol=1e-12):

    return np.where(np.abs(arr)<tol, 0, arr)