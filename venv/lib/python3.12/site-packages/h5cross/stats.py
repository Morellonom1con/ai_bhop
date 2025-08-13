
from collections import namedtuple

import numpy as np

from h5cross.utils import unwrap_dict
from h5cross.utils import wrap_dict


STATS = {
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
    'median': np.median,
    'std': np.std
}


def compute_stat(array, stat):
    try:
        out = float(STATS[stat](array))
    except ValueError:
        out = STATS[stat](array)
    return out
    


def compute_stats(array, stats):
    StatsCollection = namedtuple('StatsCollection', stats)
    return StatsCollection(*[compute_stat(array, stat) for stat in stats])


def compute_dict_stats(dict_, skip_list=(),
                       stats_list=('min', 'max', 'mean', 'median', 'std')):

    unwrapped_dict = unwrap_dict(dict_)  # to avoid nested

    stats = {}
    for key, array in unwrapped_dict.items():
        if key.split('/')[-1] in skip_list:
            continue

        try:
            stats[key] = compute_stats(array, stats_list)
        except TypeError:
            pass

    # wrap again
    return wrap_dict(stats)
