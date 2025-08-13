"""Defines a frustum shell object.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from kokiy.axishell import AxiShell

class FrustumShell(AxiShell):
    """A frustum computational shell.

    Args:
        n_azi (int): Number of azimuthal shell points.
        n_longi (int): Number of longitudinal shell points.
        pt_a (np.array): coordinates of left circle's center.
        pt_b (np.array): coordinates of right circle's center.
        radius_a (float): left circle's radius.
        radius_b (float): right circle's radius.
        direc_pt (np.array): Deprecated
    """
    geom_type = 'frustum'

    def __init__(self, n_azi, n_longi, pt_a, pt_b, radius_a,
                 radius_b, direc_pt=None):
        
        # Create an AxiShell of same Shape, then rotate and translate.
        vect_b = np.array(pt_b)
        vect_a = np.array(pt_a)
        vect_a_b = vect_b-vect_a
        len_ab = np.linalg.norm(vect_a_b)
        vect_a_b /= len_ab
        
        super().__init__(n_azi, n_longi, 360., (0., len_ab), (radius_a, radius_b))

        rotation = _find_rotation(vect_a_b)
        self.x, self.y, self.z = _rotate_xyz(rotation, self.x, self.y, self.z)
        self.n_x, self.n_y, self.n_z = _rotate_xyz(rotation, self.n_x, self.n_y, self.n_z)

        self.x += vect_a[0]
        self.y += vect_a[1]
        self.z += vect_a[2]


def _find_rotation(vect_a_b):
    """Find the scipy rotation from vect X to vect AB"""
    rot_vect = np.cross((1.,0.,0.),vect_a_b)
    parallelogram = np.linalg.norm(rot_vect)
    if parallelogram == 0.:
        rot_vect *=0
    else:
        angle = np.arcsin(parallelogram)
        rot_vect *= angle/parallelogram
    return Rotation.from_rotvec(rot_vect)

        
def _rotate_xyz(rotation, x,y,z):
    """Apply a scipty rotation on vects x,y,z"""
    shape_ = x.shape
    xyz = np.stack([
        x.ravel(),
        y.ravel(),
        z.ravel(),
    ], axis=1)

    new_xyz = rotation.apply(xyz)

    new_x = new_xyz[:,0].reshape(shape_)
    new_y = new_xyz[:,1].reshape(shape_)
    new_z = new_xyz[:,2].reshape(shape_)
    
    return  new_x,new_y,new_z