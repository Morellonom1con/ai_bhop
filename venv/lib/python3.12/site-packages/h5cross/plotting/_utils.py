
import importlib

import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt


def _load_backend(backend):
    return importlib.import_module(f'h5cross.plotting._{backend}')


def _get_density(x_array, y_array, bins=24):
    data, xedge, yedge = np.histogram2d(x_array, y_array, bins=bins, density=True)

    density_array = interpn(
        (0.5 * (xedge[1:] + xedge[:-1]), 0.5 * (yedge[1:] + yedge[:-1])),
        data, np.vstack([x_array, y_array]).T, method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    # TODO: review? use case? probably undesired?
    density_array[np.where(np.isnan(density_array))] = 0.0

    return density_array


def _sort_by_density(x_array, y_array, density_array):
    idx = density_array.argsort()
    return x_array[idx], y_array[idx], density_array[idx]


def _get_error_norm(array1, array2, order=2):
    error_norm = np.linalg.norm(array1 - array2, ord=order)
    error_norm /= np.linalg.norm((array1 + array2) / 2.0, ord=order)

    return error_norm


def _get_scatter_title(array1, array2):
    error_norm = _get_error_norm(array1, array2, order=2)
    return f"Similarity: {1-error_norm:.4f}"


def _get_bin_edges(array1, array2):
    bin_edges, _ = np.lib.histograms._get_bin_edges(np.r_[array1, array2], 'auto',
                                                    None, None)

    return bin_edges


def _save_plot(filename, axes_labels=None):
    if axes_labels:
        filename = f'{filename}_{axes_labels[0]}_{axes_labels[1]}'

    plt.savefig(f'{filename}.png')
