"""Plot density mesh module"""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def filter_stupid_characters(string):
    """filter_stupid_characters"""
    for char in ["$", "\\", " ", "__", "___"]:
        string = string.replace(char, "_")

    if string.startswith("_"):
        string = string[1:-1]

    return string


def heat_map_mesh(x_crd, y_crd, z_crd, show=False, save=False, view_axes='xr'):
    """ heat map plot of skin """

    plt.rcParams['image.cmap'] = 'Blues'

    def get_bins(x_crd, y_crd):
        """Computes equal bins from aspect ratio"""
        l_y = np.abs(np.max(y_crd) - np.min(y_crd))
        l_x = np.abs(np.max(x_crd) - np.min(x_crd))
        ratio = l_y / l_x
        return 500., 500. * ratio

    def plot_skin(tuple_data, tuple_labels):
        """Plot skin as hist2d"""
        x_crd, y_crd = tuple_data
        x_label, y_label = tuple_labels

        fig = plt.figure()
        ax_ = fig.add_subplot(111)

        title = "%s, %s" % tuple_labels
        ax_.set_title(title)
        ax_.hist2d(x_crd, y_crd,
                   bins=get_bins(x_crd, y_crd),
                   norm=LogNorm())
        ax_.set_xlabel("%s" % x_label)
        ax_.set_ylabel("%s" % y_label)
        ax_.set_aspect(aspect=1)
        if save:
            name = filter_stupid_characters(title)
            plt.savefig('%s' % name)

    radius = np.hypot(y_crd, z_crd)


    if view_axes == 'xr':
        plot_skin((x_crd, radius), ('$x$', '$r$'))
    elif view_axes == 'xz':
        plot_skin((x_crd, z_crd), ('$y$', '$z$'))
    elif view_axes == 'xy':
        plot_skin((x_crd, y_crd), ('$x$', '$y$'))
    elif view_axes == 'zy':
        plot_skin((z_crd, y_crd), ('$z$', '$y$'))
    else:
        plot_skin((x_crd, radius), ('$x$', '$r$'))
        plot_skin((x_crd, z_crd), ('$y$', '$z$'))
        plot_skin((x_crd, y_crd), ('$x$', '$y$'))
        plot_skin((z_crd, y_crd), ('$z$', '$y$'))

    if show:
        plt.show()


def scatter_plot_mesh(x_crd, y_crd, z_crd, axisym=False, show=False):
    """ scatter plot of skin """

    if axisym:
        radius = np.sqrt(np.square(x_crd) + np.square(y_crd))
        theta = np.arctan2(z_crd, y_crd)

        plt.figure()
        plt.title(r"x, theta")
        plt.scatter(x_crd, theta, edgecolors='none', alpha=0.25)
        plt.xlabel("x")
        plt.xlabel("theta")

        plt.figure()
        plt.title("x, r")
        plt.scatter(x_crd, radius, edgecolors='none', alpha=0.25)
        plt.xlabel("x")
        plt.xlabel("x")

        plt.figure()
        plt.title("r, theta")
        plt.scatter(radius, theta, edgecolors='none', alpha=0.25)
        plt.xlabel("r")
        plt.xlabel("theta")
    else:
        plt.figure()
        plt.title("z, y")
        plt.scatter(z_crd, y_crd, edgecolors='none', alpha=0.25)
        plt.xlabel("z")
        plt.xlabel("y")

        plt.figure()
        plt.title("x, y")
        plt.scatter(x_crd, y_crd, edgecolors='none', alpha=0.25)
        plt.xlabel("x")
        plt.xlabel("y")

        plt.figure()
        plt.title("x, z")
        plt.scatter(x_crd, z_crd, edgecolors='none', alpha=0.25)
        plt.xlabel("x")
        plt.xlabel("z")

    if show:
        plt.show()
