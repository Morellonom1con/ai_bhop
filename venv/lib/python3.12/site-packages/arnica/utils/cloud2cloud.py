"""interpolate a cloud from an other cloud """

import numpy as np
from scipy import spatial
# from arnica.utils.vector_actions import (yz_to_theta)

SMALL = 1e-16

def _build_ravel_component(in_xyz, limitsize=None):
    """ build source and target raveled components

    Parameters:
    -----------
    in_xyz (dict of nparray): las index are coordinantes 3D. (:,:,3)
    limitsize : maximum size of raveled output

    Returns:
    --------
    out_xyz : raveld version, limited in size
    """
    skipper = 1
    if limitsize is not None:
        input_size = np.ravel(np.take(in_xyz, 0, axis=-1)).shape[0]
        skipper = max(int(input_size * 1.0 / limitsize), 1)
        if skipper > 1:
            print("Warning, source data too big ({0}/{1}),\n "
                  "sub sampling every {2}th element.".format(input_size,
                                                             limitsize,
                                                             skipper))
    out_xyz = np.stack((np.ravel(np.take(in_xyz, 0, axis=-1))[::skipper],
                        np.ravel(np.take(in_xyz, 1, axis=-1))[::skipper],
                        np.ravel(np.take(in_xyz, 2, axis=-1))[::skipper]),
                       axis=1
                      )

    return out_xyz, skipper


def _interpolate_var(in_val, index, inv_dist):
    """interpolate nP_array of source
    intour a np.arry of target"""

    estimate = (np.sum(inv_dist *
                       np.reshape(in_val[np.ravel(index)],
                                  np.shape(inv_dist)),
                       axis=1))

    estimate /= np.sum(inv_dist, axis=1)

    return estimate

def _compute_dists(source_xyz, target_xyz, limitsource, stencil, power):
    sce_ravel, skipper = _build_ravel_component(source_xyz,
                                                limitsize=limitsource)
    tgt_ravel, _ = _build_ravel_component(target_xyz)

    kdtree = spatial.cKDTree(sce_ravel) #pylint: disable=not-callable

    dists, index = kdtree.query(tgt_ravel, k=stencil,)

    if power != 1.0:
        dists = np.power(dists, power)

    inv_dist = np.reciprocal(np.maximum(dists, SMALL))
    return index, dists, inv_dist, skipper

def cloud2cloud(source_xyz, source_val, target_xyz, stencil=3, limitsource= None, power = 1.0, tol =None):
    """ Interpolate  form a cloud to an other

    Parameters :
    ------------
    source_xyz : numpy array shape (n_s, 3) either (1000, 3 )  or (10,10,10, 3)
    source_val : numpy array shape (n_s, k) of k variables
    target_xyz : numpy array shape (n_t, 3)
    stencil (int): nb of neigbors to compute (1 is closest point)

    Optional keyword arguments
    --------------------------
    limitsource (int) : maximum nb of source points allowed (subsample beyond)
    power(float) : Description
    tol(float) : Description
    Returns :
    ----------
    target_val : numpy array shape (n_t, k)

    """


    index, dists, inv_dist, skipper = _compute_dists(source_xyz, target_xyz,
                                                     limitsource,
                                                     stencil,
                                                     power)
    target_val = {}
    for key in source_val:
        source  = source_val[key]

        if stencil > 1:
            estimate = _interpolate_var(np.ravel(source)[::skipper],
                                        index,
                                        inv_dist,
                                        )
        else:
            estimate = np.ravel(source)[::skipper][index]

        if tol is not None:
            estimate = np.where(dists[:, 0] > tol, 0, estimate)

        target_val[key] = np.reshape(estimate,
                                     np.take(target_xyz, 0, axis=-1).shape)


    return target_val
