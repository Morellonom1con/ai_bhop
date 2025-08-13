

import matplotlib.pyplot as plt
from nob import Nob

from h5cross.io import load_hdf_as_dict
from h5cross.plotting._utils import _save_plot
from h5cross.plotting._utils import _load_backend

DEFAULT_BACKEND = 'matplotlib'


# TODO: rotate axis labels in right histogram


def compare_scatter_h5(file1, file2, variable_list, save_name=None,
                       flag_show=True, flag_seaborn=True, flag_xy_line=False):
    """Function that controls the generation of scatterplots comparing the same
    variable from two hdf5 files

    Input:
        :file1: (string), name of first file for comparison
        :file2: (string), name of second file for comparison
        :variable_list: list of type string containing keywords of the
                        variables to select for comparison. The same variable
                        will be selected from each file.
        Optional Arguments:
            :flag_save: boolean controlling if the plots get saved (default = False)
            :save_name: (string), base name to be used in saving the plots (default = None)
            :flag_show: boolean setting if the plot gets showed or not (default = True)
            :flag_seaborn: boolean controlling whether seaborn is used or not (default = True)
    Output:
        :None: interactive view. If flag_save = True, a png image of the scatter plots gets saved.
    """
    # TODO: keep flag?
    backend = 'seaborn' if flag_seaborn else 'matplotlib'

    nob1 = Nob(load_hdf_as_dict(file1))
    nob2 = Nob(load_hdf_as_dict(file2))
    for var_name in variable_list:
        array1 = nob1[var_name][:]
        array2 = nob2[var_name][:]
        if (array1 is not None) and (array2 is not None):
            if len(array1) == len(array2):
                plot_scatter(array1, array2, axes_labels=[var_name, var_name],
                             flag_show=False, save_name=save_name,
                             flag_xy_line=flag_xy_line, backend=backend)
            else:
                print("Warning: data arrays have different length, plotting distribution instead")

                plot_distributions(array1, array2, axes_labels=[var_name, var_name],
                                   flag_show=False,
                                   save_name=save_name, backend=backend)

    if flag_show:
        plt.show()


def plot_scatter(*args, axes_labels=None, save_name=None, flag_show=True,
                 backend=DEFAULT_BACKEND, **kwargs):
    module = _load_backend(backend)

    ax = module._plot_scatter(*args, **kwargs)

    if axes_labels is not None:
        x_label = f'{axes_labels[0]} field 1'
        y_label = f'{axes_labels[1]} field 2'
    else:
        x_label = 'field 1'
        y_label = 'field 2'
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()

    if save_name:
        _save_plot(save_name, axes_labels)

    if flag_show:
        plt.show()

    return ax


def plot_distributions(*args, axes_labels=None, save_name=None, flag_show=True,
                       backend=DEFAULT_BACKEND, **kwargs):
    module = _load_backend(backend)

    ax = module._plot_distributions(*args, **kwargs)

    x_label = 'field 1' if axes_labels is None else f'{axes_labels[0]} field 1'
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    plt.tight_layout()

    if save_name:
        _save_plot(save_name, axes_labels)

    if flag_show:
        plt.show()

    return ax
