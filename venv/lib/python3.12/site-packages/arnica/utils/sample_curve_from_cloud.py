""" SORT AND SAMPLE POINTS FROM CURVE """

import time
import numpy as np
from scipy import spatial

def get_neighbor(kdtree, point, list_points):
    """
    *Find the closiest neighbor that has not already been found*

    From the kdtree, find the two closiest points of 'point', the 1st\
    being itself.
    If the second closiest point has already been found, then the\
    number of researched points is increased until finding a point that\
    has not already been found.

    :param kdtree: KDTree of all points.
    :param point: Array of dim (k,) of point float coordinates.
    :type point: np.array
    :param list_points: List of integer indexes

    :return:

        - **index** - Index (int) of the unfound closiest point.
        - **dist** - Distance (float) of the closiest point from point.
    """

    k_neighbors = 2
    dists, indexes = kdtree.query(point, k=k_neighbors)
    subindexes = np.setdiff1d(indexes, list_points, assume_unique=True)

#    while indexes[-1] in list_points:
    while len(subindexes) == 0:
        k_neighbors += 1
        dists, indexes = kdtree.query(point, k=k_neighbors)
        subindexes = np.setdiff1d(indexes, list_points, assume_unique=True)
        #if k_neighbors == 4000:
        #    print("k diverges")
        #    stop = True
        #    break

    index = subindexes[0]
    dist = dists[np.where(indexes == subindexes[0])[0][0]]

    return index, dist

def sort_points_by_dist(points_coor, starting_pt):
    #pylint: disable=too-many-locals
    r"""\
    *Reorder a point cloud by distances*
                                                   .
    From a starting fictive point, the first point 1 of the spline\
    iget_neighbors found, as the closiest one.       .
    From that point, the following points are obtained with get_neighbor()
    until having sorted all points.

    ::
            ................                   ................       r/y
        ...      2     8    ..             ...                 ..    ^
      ..                     .           ..                     .    |
      .                      .           .                      .    |      x
      .      ................     ===>   .      ................     o----->
      .     .                            .     .
       .     ...................          .     ...................
       ..                                 ..
          ......................             ......................
            1         3          X                         8    321
                                  starting_pt

    :param points_coor: Array of dim (n,k) of points coordinates
    :type points_coor: np.array
    :param starting_pt: Array of dim (k,) of starting pt coordinates
    :type starting_pt: np.array

    :returns:

        - **ordered_indexes** - Array of dim (n,) of indexes
        - **ordered_dists** - Array of dim (n,) of floats
    """

    kdtree = spatial.cKDTree(points_coor) #pylint: disable=not-callable

    list_points = []
    list_dists = []

    _, idx = kdtree.query(starting_pt, k=1)
    list_points.append(idx)
    list_dists.append(0)

    progress = -1
    time_i = time.time()
    while len(list_points) < len(points_coor):

        index, dist = get_neighbor(kdtree,
                                   points_coor[list_points[-1]],
                                   list_points)

        list_dists.append(dist)
        list_points.append(index)

        new_progress = int(len(list_points)/len(points_coor) * 100)
        if int(new_progress / 10) > int(progress / 10):
            progress = new_progress
            delta_time = time.time() - time_i
            print("Sorting process... {:d}% : {:.3f}s".format(progress, delta_time), end="\r")

    print()
    ordered_indexes = np.asarray(list_points)
    ordered_dists = np.asarray(list_dists)

    return ordered_indexes, ordered_dists

def sample_arrays_by_dist_interval(dists, samples_res, *args):
    r"""\
    *Sample data and optional args at each sample_res along data*

    From an array containing the distance between sorted points, \
    and from a sample resolution defined by the distance between\
    two samples, the function picks up the indexes corresponding\
    to a sample, and returns an array of sampled distances and\
    arrays of sampled arguments from those indexes.

    __ = samples_res

    ::
            ................                   .__.__.__.__.__.
        ...                 ..             .__                 ..
      ..                     .           ..                     |
      .                      .           |                      .
      .      ................     ===>   |      .__.__.__.__.__.
      .     .                            .     .
       .     ...................          .     .__.__.__.__.__.__.
       ..                                 ..
          ......................           9 .__.__.__.__.__.__.__.
                      9      321                            3  2  1


    :param dists: Array of dim (n,) of floats
    :type dists: np.array
    :param samples_res: Sample resolution for sampling
    :type samples_res: float
    :param args: Tuple de data of dim (n,)
    """

    cum_dists = np.cumsum(dists, axis=0)
    sum_dists = cum_dists[-1]
    indexes = [0]

    for dist in np.arange(samples_res, sum_dists, samples_res):
        next_index = np.where((cum_dists >= dist) & (cum_dists < dist + samples_res))[0]
        if next_index.size > 0:
            indexes.append(next_index[0])

    if indexes[-1] != len(dists) - 1:
        indexes[-1] = len(dists) - 1

    dists = dists[indexes]
    args_out = []
    for arg in args:
        args_out.append(arg[indexes])

    return dists, tuple(args_out)

def sample_points_from_cloud(points_coor,
                             starting_pt,
                             n_samples=None,
                             samples_res=None):
    """
    *Sort and sample unsorted curve*

    First the point coordinates are sorted by distances.
    Secondly compute sample resolution if not provided.
    Finally sample the ordered indexes points by distance.
    Returns the array of coordinates ordered and sampled.

    :param points_coor: Array of dim (n,k) of points coordinates
    :type skin_coor: np.array
    :param starting_pt: Array of dim (k,) of starting point coordinates
    :type starting_pt: np.array
    :param n_samples: Number of samples to extract
    :type n_samples: int
    :param sample_res: Resolution of the sampling
    :type sample_res: float

    :returns:

        - **skin_cyl** - Array of dim (n_samples, 3) of coordinates
    """

    # Sort the points and get samples of them.
    ordered_indexes, ordered_dists = sort_points_by_dist(points_coor, starting_pt)

    # Calculate optional variable from specified one.
    if samples_res is None:
        # Sample resolution : distance between two point samples.
        if n_samples is None:
            n_samples = 80
        samples_res = np.sum(ordered_dists) / n_samples

    _, (sampled_ordered_indexes,) = sample_arrays_by_dist_interval(ordered_dists,
                                                                   samples_res,
                                                                   ordered_indexes)

    return points_coor[sampled_ordered_indexes]
