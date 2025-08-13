import numpy as np
from scipy.spatial.transform import Rotation as R

from kokiy.shell_2d import Shell2D, compute_normals_from_xyz
from kokiy.axishell import _compute_shell_crest


ATOL = 1e-8

class TorchShell(Shell2D):
    geom_type = "torch"

    def __init__(
        self, n_azi, n_longi, pt_a, pt_b, radius_a, radius_b, ecc_pt
    ):
        super().__init__(n_azi, n_longi)

        self.pt_a = np.array(pt_a)
        self.pt_b = np.array(pt_b)
        self.ecc_pt = np.array(ecc_pt)
        self.radius_a = radius_a
        self.radius_b = radius_b

        # useful
        _vec_ab = self.pt_b - self.pt_a
        self.len_ab =  np.linalg.norm(_vec_ab)
        self.axis_vect = _vec_ab / np.linalg.norm(_vec_ab)

        self._ecc = self.ecc_pt - self.pt_b
        self._verify_ecc()
        self._ecc_unit = self._ecc / np.linalg.norm(self._ecc)

        self._build_shell()

    def _verify_ecc(self):
        if abs(np.sum(self._ecc)) < ATOL and abs(np.prod(self._ecc)) < ATOL:
            raise Exception("Null eccentricities are not accepted")

        if abs(np.dot(self._ecc, self.axis_vect)) > ATOL:
            raise Exception("Eccentricity must be perpendicular to axis")

    def _build_shell(self):

        # Construct Shell Crest
        ctrl_pts_r = np.array([self._get_radius_sect(v) for v in self.v[0]])
        shell_crest = _compute_shell_crest(self.v[0], ctrl_pts_r, self.shape[1])

        # define directions
        center_vec = np.array([self.axis_vect * v for v in self.v[0]])

        # Compute radius and theta matrices of shape (n_azi, n_longi)
        _rad, _theta, _ = self._get_rad_theta_aux_normals(
            ctrl_pts_r
        )

        # get xyz
        xyz = []
        for i, sect_center in enumerate(center_vec):
            rotations = [
                R.from_rotvec(theta * self.axis_vect) for theta in _theta[:, i]
            ]
            rad_direcs = np.array(
                [rotation.apply(self._ecc_unit) for rotation in rotations]
            )

            xyz.append(
                sect_center
                + np.array(
                    [
                        sect_radius * rad_direc
                        for sect_radius, rad_direc in zip(_rad[:, i], rad_direcs)
                    ]
                )
            )

        self.x, self.y, self.z = [np.take(np.array(xyz), i, -1).T for i in range(3)]

        self.n_x,self.n_y,self.n_z= compute_normals_from_xyz(self.x,self.y,self.z, perio_u=True)

        shell_crest_scaled = shell_crest.copy()
        shell_crest_scaled[0] *= self.len_ab

        # Compute du, dv matrix of shape (n_azi, n_longi)
        longi_vals = np.tile(shell_crest_scaled[0], (self.shape[0], 1))
        self.du = np.pad(
            np.sqrt(np.diff(longi_vals, axis=1) ** 2 + np.diff(self.rad, axis=1) ** 2),
            ((0, 0), (1, 0)),
            "edge",
        )
        self.dv = _rad * np.pad(np.diff(_theta, axis=0), ((1, 0), (0, 0)), "edge")

        # Compute weight intervals in u and v directions array of shape (n_transvers, n_longi)
        self.dwu, self.dwv = self._compute_weight_interv_adim(self.du, self.dv)

    def _get_radius_sect(self, v):
        return (self.radius_b - self.radius_a) * v + self.radius_a

    def _get_eccentricity_sect(self, v):
        return self._ecc * v

    def _get_rad_theta_aux_normals(self, ctrl_pts_r):

        min_theta, max_theta = 0.0, np.deg2rad(360)
        theta_vec = np.linspace(min_theta, max_theta, num=self.shape[0])
        lim_theta_b = np.abs(
            np.arctan2(self.radius_b, np.linalg.norm(self._ecc))
        )

        # get phis (angles in circular region)
        phis = _get_phis(theta_vec, lim_theta_b)

        normal_90, normal_270 = self._get_planar_region_normals()

        rads = []
        thetas = []
        normals = []

        for sect_rad,  v in zip(ctrl_pts_r, self.v[0]):
            sect_ecc = self._get_eccentricity_sect(v)
            sect_ecc_norm = np.linalg.norm(sect_ecc)
            lim_theta = np.abs(np.arctan2(sect_rad, sect_ecc_norm))
            corr_coeff = _compute_corr_coeff(lim_theta)

            rads_ = []
            thetas_ = []
            normals_ = []
            for phi, theta in zip(phis, theta_vec):
                if phi is None:  # rectangular region
                    theta_ = _get_theta_sect_rect(self.radius_b, v, theta, sect_rad)
                    rad_ = np.abs(sect_rad / np.sin(theta_))
                    normal_ = normal_90 if theta_ < np.pi else normal_270

                else:
                    theta_ = _get_theta_sect_via_phi(phi, corr_coeff)
                    rot_angle = phi[0]
                    rotation = R.from_rotvec(rot_angle * self.axis_vect)
                    radius_vec = self._ecc_unit * sect_rad

                    if phi[1] == np.pi:
                        vec = -sect_ecc + rotation.apply(-radius_vec)
                        normal_ = -rotation.apply(self._ecc_unit)
                    else:
                        vec = sect_ecc + rotation.apply(radius_vec)
                        normal_ = rotation.apply(self._ecc_unit)

                    rad_ = np.linalg.norm(vec)

                rads_.append(rad_)
                thetas_.append(theta_)
                normals_.append(normal_)

            rads.append(rads_)
            thetas.append(thetas_)
            normals.append(normals_)

        return (
            np.array(rads).T,
            np.array(thetas).T,
            np.array(normals).transpose(1, 0, 2),
        )

    def _get_planar_region_normals(self):
        rotations = [
            R.from_rotvec(rot_angle * self.axis_vect)
            for rot_angle in [np.pi / 2, 3 / 2 * np.pi]
        ]
        return [rotation.apply(self._ecc_unit) for rotation in rotations]

    def replicate(self, shape):
        return TorchShell(
            *shape,
            self.pt_a,
            self.pt_b,
            self.radius_a,
            self.radius_b,
            self.ecc_pt
        )



def _get_theta_sect_rect(radius_b, v, theta, sect_rad):
    """Gets horizontal distance in planar region."""
    horiz_dist = v * radius_b / np.tan(theta)

    return abs(np.arctan2(sect_rad, horiz_dist)) + _get_theta_add(theta)

def _get_phis(theta_vec, lim_theta_b):
    """Get angles of rotation of right face.

    Within the circular region, these angles are kept among layers. Outside
    the circular region it returns None.
    """
    corr_coeff_b = _compute_corr_coeff(lim_theta_b)

    phis = []
    for theta in theta_vec:
        if theta < lim_theta_b:
            phi = (theta * corr_coeff_b, 0.0)
        elif theta > 2 * np.pi - lim_theta_b:
            phi = ((theta - 2 * np.pi) * corr_coeff_b, 2 * np.pi)
        elif theta > (np.pi - lim_theta_b) and theta < np.pi + lim_theta_b:
            phi = ((theta - np.pi) * corr_coeff_b, np.pi)
        else:
            phi = None
        phis.append(phi)
    return phis

def _compute_corr_coeff(lim_theta):
    return (np.pi / 2) / lim_theta

def _get_theta_sect_via_phi(phi, corr_coeff):
    return phi[0] / corr_coeff + phi[1]

def _get_theta_add(theta):
    add = 0.0 if theta < np.pi else np.pi
    return add