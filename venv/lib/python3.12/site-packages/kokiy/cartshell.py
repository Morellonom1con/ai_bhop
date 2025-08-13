"""Defines a cartesian shell object.
"""

import numpy as np

from kokiy.shell_2d import Shell2D, compute_normals_from_xyz


class CartShell(Shell2D):
    r"""Base class for cartesian computational shells.

      Some attributes are based on **u** and **v**, defined as the curvilinear
      longitudinal (u) abscissa and curvilinear transversal (v) abscissa
      respectively.

      The shell is set using three corners (assuming they are orthogonal).


      Args:
          n_trans (int): Number of azimuthal shell points.
          n_longi (int): Number of longitudinal shell points.
          zero (np.array): Lower left corner of shell of shape (3,).
          umax (np.array): Corner of shell for umax of shape (3,).
          vmax (np.array): corner of shell for vmax of shape (3,).


      ::

            X umax
            ________________
            |              |            Cartesian system
            |              |            Example longi/u <=> x  and v <==> z
            |              |            zero. (0, 0, 0)
            |              |            umax. (1, 0, 0)
            |              |            vmax. (0, 0, 1)
            |              |            x - longi/u
            |              |            ^
            |              |            |
            |              |            |
            |              |            O-----> z - transvers/v
            |______________| X vmax     y
            X zero.
    """
    geom_type = 'cart'

    def __init__(self, n_trans, n_longi, zero, umax, vmax):

        super().__init__(n_trans, n_longi)
        self.zero = zero
        self.umax = umax
        self.vmax = vmax
        self._build_shell(np.array(zero), np.array(umax), np.array(vmax))

    def _build_shell(self, zero, umax, vmax):
        """Build shell from geometric features.

        Notes:
            - Construct a spline used as base for extrusion from control\
                points: tck
            - Discretise the spline: shell_crest
            - Compute normal vectors for the 1D shell_crest
            - Compute r, n_x, n_r-components for 2D shell
            - Compute theta-components for 2D shell
            - Compute xyz, n_y, n_z-components for 2D shell
        """

        vec_u = umax - zero
        vec_v = vmax - zero
        self.x, self.y, self.z = _build_xyz(self.shape, zero, vec_u,vec_v)  

        self.n_x,self.n_y,self.n_z= compute_normals_from_xyz(self.x,self.y,self.z)

        # Surface element
        d_u = np.linalg.norm(vec_u) / (self.shape[0] - 1)
        d_v = np.linalg.norm(vec_v) / (self.shape[1] - 1)
        self.du = np.full(self.shape, d_u)
        self.dv = np.full(self.shape, d_v)

        # Compute weight intervals in u and v directions array of shape (n_transvers, n_longi)
        self.dwu, self.dwv = self._compute_weight_interv_adim(self.du, self.dv)
    
    def replicate(self, shape):
        """Creates a similar instance, but possibly with different shape.
        """
        return CartShell(*shape, self.zero, self.umax, self.vmax)


def _build_xyz(shape, zero, vec_u, vec_v):

    u_range = np.linspace(0., 1., num=shape[0])
    v_range = np.linspace(0., 1., num=shape[1])
    
    v_coor, u_coor = np.meshgrid(v_range, u_range)
    # multiply coord matrix by each vec component
    u_coor = vec_u[:, np.newaxis, np.newaxis] * u_coor[np.newaxis, :, :]
    v_coor = vec_v[:, np.newaxis, np.newaxis] * v_coor[np.newaxis, :, :]
    xyz = np.stack([zero[i] + u_coor[i] + v_coor[i] for i in range(3)],
                    axis=-1)
    x,y,z = [np.take(xyz, i, -1) for i in range(3)]
    return x,y,z