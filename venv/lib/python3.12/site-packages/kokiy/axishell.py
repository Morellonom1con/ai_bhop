"""Defines an x-axisymmetric shell object.
"""

import numpy as np
from scipy import interpolate

from kokiy.shell_2d import Shell2D, compute_normals_from_xyz

MAX_SPLINE_ORDER = 3
SPLINE_SMOOTHNESS = 0


class AxiShell(Shell2D):
    r"""An x-axisymmetric computational shell.

    Some attributes are based on **u** and **v**, defined as the curvilinear
    longitudinal (x/r) abscissa and curvilinear azimutal (theta) abscissa
    respectively.


    Args:
        n_azi (int): Number of azimuthal shell points.
        n_longi (int): Number of longitudinal shell points.
        angle (float): Range angle of the axicylindrical geometry.
        ctrl_pts_x (np.array): x-coordinates of the points defining the spline
            of shape (n,).
        ctrl_pts_r (np.array): r-coordinates of the points defining the spline
            of shape (n,).
        angle_min (float): Minimum angle of the shell.


    ::

                      (x_n, r_n)
                _____X_____
         ___----     |     ----___     Cylindrical system
        \            |            /    Example longi/u <=> r
         \           |           /
          \          |          /      r - longi/u
           \         |         /       ^
            \        | (x_0, r_0)      |
             \      _X_      /         |
              \__---   ---__/          X-----> theta - azi/v
                                      x

              ................
          ...                 ..       Cylindrical system
        ..                     .       Example longi/u <=> x/r
        .                      .
        .      ................         r
        .     .                         ^
         .     ...................      |
         ..                             |
            ......................      X-----> x
                          <--      theta - azi/v
                     x/r - longi/u
    """
    geom_type = 'axicyl'

    def __init__(self, n_azi, n_longi, angle, ctrl_pts_x, ctrl_pts_r, angle_min=None):
        super().__init__(n_azi, n_longi)

        self.ctrl_pts_x = ctrl_pts_x
        self.ctrl_pts_r = ctrl_pts_r
        self.angle = angle
        self.angle_min = angle_min

        self._build_shell()

    def _build_shell(self):
        """Builds shell from geometric features.

        Notes:
            - Construct a spline used as base for extrusion
              from control points : tck
            - Discretise the spline : shell_crest
            - Compute normal vectors for the 1D shell_crest
            - Compute r,n_x,n_r-components for 2D shell
            - Compute theta-components for 2D shell
            - Compute xyz,n_y,n_z-components for 2D shell
        """

        # Construct Shell Crest
        shell_crest = _compute_shell_crest(self.ctrl_pts_x, self.ctrl_pts_r,
                                           self.shape[1])

        # Compute radius and theta matrices of shape (n_azi, n_longi)
        rot_angle = 0
        if self.angle_min is not None:
            rot_angle = 0.5 * self.angle + self.angle_min

        min_theta = (rot_angle - 0.5 * self.angle) * np.pi / 180
        max_theta = (rot_angle + 0.5 * self.angle) * np.pi / 180
        theta_vec = np.linspace(min_theta, max_theta, num=self.shape[0])
        rad, theta = np.meshgrid(shell_crest[1], theta_vec)

        # Compute xyz matrix of shape (n_azi, n_longi, 3)
        self.x = np.tile(shell_crest[0], (self.shape[0], 1))
        self.y = rad * np.cos(theta)
        self.z = rad * np.sin(theta)

        perio_u = False
        if self.angle == 360.:
            perio_u = True
        self.n_x,self.n_y,self.n_z= compute_normals_from_xyz(self.x,self.y,self.z, perio_u=perio_u)

        # Compute du, dv matrix of shape (n_azi, n_longi)
        self.du = np.pad(np.sqrt(np.diff(self.x, axis=1) ** 2
                                 + np.diff(rad, axis=1) ** 2),
                         ((0, 0), (1, 0)), 'edge')
        self.dv = rad * np.pad(np.diff(theta, axis=0),
                                    ((1, 0), (0, 0)), 'edge')
        
        # Compute weight intervals in u and v directions array of shape (n_transvers, n_longi)
        self.dwu, self.dwv = self._compute_weight_interv_adim(self.du, self.dv)

    def replicate(self, shape):
        """Creates a similar instance, but possibly with different shape.
        """
        return AxiShell(*shape, self.angle, self.ctrl_pts_x, self.ctrl_pts_r,
                        angle_min=self.angle_min)


def _compute_shell_crest(ctrl_pts_1, ctrl_pts_2, n):
    """Build the shell crest of shape (2, self.shape[1]) and respecting
    controls points.
    """

    # Build crest
    spline_order = min(len(ctrl_pts_1) - 1, MAX_SPLINE_ORDER)
    # Generate continuous spline from control points.
    tck, _ = interpolate.splprep([ctrl_pts_1, ctrl_pts_2],
                                 s=SPLINE_SMOOTHNESS,
                                 k=spline_order)
    unew = np.linspace(0, 1, num=n)
    # Generate discrete spline
    # Numpy Array of dim (2, self.shape[1]) [x,r] / [x,y]
    shell_crest = np.asarray(interpolate.splev(unew, tck))

    return shell_crest