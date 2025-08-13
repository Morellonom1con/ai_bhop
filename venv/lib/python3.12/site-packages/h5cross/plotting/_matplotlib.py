
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from h5cross.plotting._utils import _get_density
from h5cross.plotting._utils import _sort_by_density
from h5cross.plotting._utils import _get_scatter_title
from h5cross.plotting._utils import _get_bin_edges


def _plot_scatter(array1, array2, flag_xy_line=False):
    """Function to generate a scatter plot of two data arrays of same length with matplotlib.
    If scipy is available a density scatter plot will be generated.

    Input:
        array1: numpy array of first data set
        array2: numpy array of second data set
        axes_labels: list of string of length 2, used to specify the labels of output axis
        Optional Arguments:
            flag_save_: boolean controlling if the plots get saved (default = False)
            save_name_: (string), base name to be used in saving the plots (default = None)
            flag_show_: boolean setting if the plot gets showed or not (default = True)

    Output:
        Default = None, interactive view
        If flag_save = True then outputs a png image
    """
    # TODO: add histograms

    # Sort the points by density, so that the densest points are plotted last
    density_array = _get_density(array1, array2)
    array1, array2, density_array = _sort_by_density(array1, array2,
                                                     density_array)

    fig, ax = plt.subplots()
    ax.scatter(array1, array2, c=density_array)

    # creates colorbar
    norm = Normalize(vmin=np.min(density_array), vmax=np.max(density_array))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel("Density")

    # add line
    if flag_xy_line:
        xmin = np.min(np.append(array1, array2))
        xmax = np.max(np.append(array1, array2))
        ax.plot([xmin, xmax], [xmin, xmax], "k-", linewidth=2)

    textstr = _get_scatter_title(array1, array2)
    ax.set_title(label=textstr, loc="center")

    return ax


def _plot_distributions(array1, array2):
    """Function to generate an histogram of two data arrays of different length with matplotlib

    Input:
        array1: numpy array of first data set
        array2: numpy array of second data set
        Optional Arguments:
            flag_save_: boolean controlling if the plots get saved (default = False)
            save_name_: (string), base name to be used in saving the plots (default = None)
            flag_show_: boolean setting if the plot gets showed or not (default = True)

    Output:
        Default = None, interactive view
        If flag_save = True then outputs a png image
    """

    _, ax = plt.subplots()

    bin_edges = _get_bin_edges(array1, array2)

    for i, (array, color, alpha) in enumerate(zip(
            [array1, array2], ["cornflowerblue", "lightgreen"], [0.8, 0.6])):
        ax.hist(array, bins=bin_edges, density=False, facecolor=color, alpha=alpha,
                label=f"field {i}")

    ax.legend()

    return ax
