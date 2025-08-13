
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from h5cross.plotting._utils import _get_density
from h5cross.plotting._utils import _sort_by_density
from h5cross.plotting._utils import _get_scatter_title
from h5cross.plotting._utils import _get_bin_edges


def _plot_scatter(array1, array2, flag_xy_line=False):
    """Function to generate a scatter plot of two data arrays of same length with seaborn

    Input:
        array1: numpy array of first data set
        array2: numpy array of second data set
        axes_labels: list of string of length 2, used to specify the labels of output axis
        Optional Arguments:
            flag_save_: boolean controlling if the plots get saved (default = False)
            save_name_: (string), base name to be used in saving the plots (default = None)

    Output:
        Default = None, interactive view
        If flag_save = True then outputs a png image
    """

    sns.set_theme()

    density_array = _get_density(array1, array2)
    array1, array2, density_array = _sort_by_density(array1, array2,
                                                     density_array)

    data_plot = {"field1": array1, "field2": array2}
    joint_grid = sns.jointplot(data=data_plot, x="field1", y="field2",
                               kind="scatter", marginal_ticks=True,
                               marginal_kws=dict(stat="density"))

    joint_grid.plot_joint(plt.scatter, c=density_array, cmap="viridis")

    # draw line
    if flag_xy_line:
        xmin = np.min(np.append(array1, array2))
        xmax = np.max(np.append(array1, array2))
        joint_grid.ax_joint.plot([xmin, xmax], [xmin, xmax], "k-", linewidth=2)

    textstr = _get_scatter_title(array1, array2)
    joint_grid.fig.suptitle(textstr)

    return joint_grid.ax_joint


def _plot_distributions(array1, array2):
    """Function to generate histograms of two data arrays of different length with seaborn

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

    sns.set_theme()
    # Will be deprecated soon, must update data structure
    # https://seaborn.pydata.org/tutorial/data_structure.html
    data_plot = {"field1": array1, "field2": array2}

    bin_edges = _get_bin_edges(array1, array2)

    ax = sns.histplot(data_plot, bins=bin_edges,
                      palette=["cornflowerblue", "lightgreen"])

    return ax
