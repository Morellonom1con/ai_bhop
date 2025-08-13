"""interpolate a cloud from an other cloud 


Alternate version with Rbf for testing purpose.
NOT WORKING YET"""

import numpy as np
from scipy import spatial
from scipy.interpolate import Rbf

def RbfWeights(source, target, power, eps=1.e-16):
    def wrapper(d):
            shp = d.shape
            node_shp = shp[:-1]
            if d.ndim == 1:
                    mode = '1-D'
            else:
                    d = d.reshape(shp[0], np.prod(node_shp))
                    mode = 'N-D'
            function = lambda r: 1./(r+eps)**power
            return Rbf(*source, d, mode=mode, function=function)(*target).reshape(target[0].size, *node_shp)
    return wrapper

def cloud2cloud2(
    source_xyz, 
    source_val, 
    target_xyz, 
    power = 1.0):
    """ Interpolate  form a cloud to an other  NOT WORKING YET

    Parameters :
    ------------
    source_xyz : numpy array shape (:, 3) or (:,:, 3) or  (:,:,:, 3) 
    source_val : numpy array shape (:, 3) or (:,:, k) or  (:,:,:, k) k variables , first dims equal to source mesh.
    target_xyz : numpy array shape (:, 3) or (:,:, 3) or  (:,:,:, 3) 
    stencil (int): nb of neigbors to compute (1 is closest point)

    Optional keyword arguments
    --------------------------
    limitsource (int) : maximum nb of source points allowed (subsample beyond)
    power(float) : Description
    tol(float) : Description
    Returns :
    ----------
    target_val : numpy array shape (:, 3) or (:,:, 3) or  (:,:,:, 3) , first dims equal to target mesh.

    """

    def _extract_compnts(in_xyz):
        x = np.ravel(np.take(in_xyz, 0, axis=-1))
        y = np.ravel(np.take(in_xyz, 1, axis=-1))
        z = np.ravel(np.take(in_xyz, 2, axis=-1))
        return (x, y, z)
        

    interp_ = RbfWeights(
        _extract_compnts(source_xyz), 
        _extract_compnts(target_xyz),
        power = power
    )

    n_sce_pts = int(source_xyz.size/3)
    print("n_sce_pts", n_sce_pts )
    print("sce_val", source_val.shape )
    n_d_data = np.reshape(source_val, (n_sce_pts, -1))
    out_shape = list(target_xyz.shape)
    out_shape[-1] = source_val.shape[-1]
    n_d_interp_out = np.reshape(interp_(n_d_data), out_shape)

    return n_d_interp_out