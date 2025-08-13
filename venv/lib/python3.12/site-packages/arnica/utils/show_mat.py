"""
This script contains function to properly visualize matrices
"""

import numpy as np

__ALL__ = ["show_mat"]

def show_mat(matrix, title, show=True, save=False):
    """
    Show and/or save a matrix visualization.

    Parameters
    ----------
    matrix: 2d matrix
    title: Title of the plot
    show: Boolean to show the plot or not
    save: Boolean to save the plot or nor (automatic name from title)

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    try:
        # If matrix is a csr then take it back to numpy array
        data = matrix.toarray()
    except AttributeError:
        data = np.array(matrix)

    #fig = plt.figure()
    #plt.title(title)
    fig, (axe0) = plt.subplots(1, 1)
    axe0.set_xlabel(r"$axis_1 j_{\rm WE}$")
    axe0.set_ylabel(r"$axis_0 i_{\rm NS}$")
    cax = axe0.matshow(data)
    fig.colorbar(cax)
    plt.tight_layout()

    if save:
        name = filter_stupid_characters(title)
        plt.savefig('%s' % name)

    if show:
        plt.show()


def filter_stupid_characters(string):
    """
    Delete and replace stupid characters to save the figure

    Parameters
    ----------
    string: title of the plot to be changed into the filename

    Returns
    -------
    cleaned string
    """
    for char in [",", "$", "\\", " ", "__", "___"]:
        string = string.replace(char, "")

    if string.startswith("_"):
        string = string[1:-1]

    return string
