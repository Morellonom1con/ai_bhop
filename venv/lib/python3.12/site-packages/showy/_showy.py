

import math
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import floor

from showy.preprocess_utils import preprocess_layout
from showy.validation import validate_layout


# TODO: add export of data
# TODO: layout merging
# TODO: add schema validation


def showy(layout, dataframe, show=True, preprocess_fnc=preprocess_layout):
    """It displays the desired graphs described by the provided layout
    thanks to the provided key-value object which contains
    the required data
    """
    validate_layout(layout)

    dataframes = dataframe
    if type(dataframes) not in [tuple, list]:
        dataframes = [dataframes]

    comparison_mode = len(dataframes)>1

    available_varnames = _get_available_varnames(dataframes)

    layout = preprocess_fnc(layout, available_varnames)

    display_params = _initialize_display(layout)
    max_graphs = display_params['max_graphs']
    fig_layout = display_params['layout']

    figure_number = 0
    graph_position = 1
    for graph_layout in layout['graphs']:
        if (graph_position > max_graphs or figure_number == 0):
            graph_position = 1
            figure_number += 1
            figure = _build_new_figure(fig_number=figure_number,comparison_mode=comparison_mode, **display_params)

        subplot = figure.add_subplot(*fig_layout, graph_position)
        _build_new_graph(subplot, graph_layout, dataframes)
        figure.tight_layout()
        graph_position += 1

    if show:
        plt.show()


def _get_available_varnames(dataframes):
    var_names = set(dataframes[0].keys())
    for dataframe in dataframes[1:]:
        var_names = var_names.intersection(set(dataframe.keys()))

    return list(var_names)


def _initialize_display(layout):
    """Sets some parameters needed by the function "display", thanks
    to the provided layout, and returns them in a dictionary.
    """

    n_graphs = len(layout['graphs'])

    fig_layout = layout.get('figure_structure', None)
    if fig_layout is None:
        fig_layout = [1, n_graphs] if n_graphs < 4 else [2, 3]

    n_max_graphs_fig = fig_layout[0] * fig_layout[1]
    n_figs = math.ceil(n_graphs / n_max_graphs_fig)

    display_params = {
        'layout': fig_layout,
        'max_graphs': n_max_graphs_fig,
        'n_figs': n_figs,
        'dpi': layout.get('figure_dpi', None),
        'inches_size': layout.get('figure_size', (14, 8)),
        'title': layout.get('title', '')}

    return display_params


def _build_new_figure(dpi=None, inches_size=(14, 8), fig_number=None,
                      n_figs=None, title="",comparison_mode=False, **kwargs):
    figure = plt.figure(dpi=dpi)
    figure.set_size_inches(*inches_size)

    figure_name = title
    if comparison_mode:
        figure_name+= " (comparison)"
    if fig_number is not None and n_figs is not None and n_figs > 1:
        figure_name += f' ({fig_number}/{n_figs})'

    figure.canvas.manager.set_window_title(figure_name.strip())

    figure.subplots_adjust(left=0.11, bottom=0.12,
                           right=0.9, top=0.9,
                           wspace=0.38, hspace=0.32)

    return figure


def _build_new_graph(ax, graph_layout, dataframes):
    """Builds a new matplotlib graph in the provided subplot.
    """
    # TODO: handle tick label overlap
    # TODO: handle no legend space

    ax.set_xlabel(graph_layout.get('x_label', None))
    ax.set_ylabel(graph_layout.get('y_label', None))
    ax.set_title(graph_layout.get('title', None), fontweight='bold')

    for data_number, dataframe in enumerate(dataframes):
        linewidth=2
        if data_number > 0:
            linewidth=4
        x = dataframe[graph_layout["x_var"]]
        for curve_number, curve_layout in enumerate(graph_layout["curves"]):
            symbols_markevery = _symbols_markevery(x)
            custom_style = _custom_cycler_iterate(curve_number, data_number, symbols_markevery)
            try :
                ax.plot(x, dataframe[curve_layout["var"]], **custom_style, linewidth=linewidth)
            except KeyError:
                print(f"Could not find the key {curve_layout['var']} in current dataframe")
            

    # improve legends
    handles = []
        
    for curve_number, curve_layout in enumerate(graph_layout['curves']):
        label = curve_layout.get('legend', None)
        custom_style = _custom_cycler_iterate(curve_number, 0)
        if label is not None:
            handles.append(Line2D([0], [0], label=label, **custom_style))

    if handles:
        ax.legend(handles=handles, loc=0)


def _custom_cycler_iterate(index, family_index=0, markevery=None):
    """Returns a dict with keys and values in accordance to the Matplotlib
    syntax.

    The graph style includes a color from a list defined for colour-blind people
    by https://personal.sron.nl/~pault/, depending if the data is compared with
    other data.
    """

    custom_cycler =     ['#4477AA','#66CCEE','#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB']
    custom_linestyles = ['solid',  'dashed' , 'dotted', 'dash', 'dash', 'dash', 'dash']
    custom_symbols = ["d", "o", "v", "^", ">", "<", "s"]

    index_ = index % len(custom_cycler)
    family_index_ = family_index % len(custom_linestyles)

    custom_style = {
        'color': custom_cycler[index_],
        'linestyle': custom_linestyles[family_index_],
        'marker': custom_symbols[index_],
        'markevery': markevery,
    }

    if index > len(custom_cycler):
        warnings.warn(f"More than {len(custom_cycler)} plots on one graph!")

    if family_index > len(custom_linestyles):
        warnings.warn(f"More than {len(custom_linestyles)} types of curves on one graph")

    return custom_style

def _symbols_markevery(x):
    """compute indices where to show symbols

    Args:
        x (numpy array): x vector
    """
    
    if len(x) < 10: # Failsafe for too short signals
        return list(range(len(x)))

    nb_markers = 10
    # set step between two symbols
    step = len(x) / (nb_markers-1)
    # list of symbols indexes (first symbol at first index)
    symbols_indexes = [int(k * step) for k in range(nb_markers)]

    symbols_indexes[-1] = min(symbols_indexes[-1],len(x)-1)


    return symbols_indexes