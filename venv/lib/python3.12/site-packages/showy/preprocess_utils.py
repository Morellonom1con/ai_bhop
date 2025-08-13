
import re


def preprocess_layout(layout, available_varnames):
    new_layout = layout.copy()
    new_layout['graphs'] = _preprocess_stars(layout['graphs'],
                                             available_varnames)

    return new_layout


def _preprocess_stars(graphs_layouts, available_varnames):
    new_graphs_layouts = []

    for graph_layout in graphs_layouts:
        if not _has_stars(graph_layout):
            new_graphs_layouts_ = [graph_layout]
        else:
            new_graphs_layouts_ = _replace_stars(graph_layout,
                                                 available_varnames)

        new_graphs_layouts.extend(new_graphs_layouts_)

    return new_graphs_layouts


def _has_stars(graph_layout):
    """Checks if the variables of a curve template contain stars.
    """
    for name in _get_curve_var_names(graph_layout):
        if '*' in name:
            return True

    return False


def _get_curve_var_names(graph_layout):
    return [curve['var'] for curve in graph_layout['curves']]


def _replace_stars(graph_layout, available_varnames):
    curve_var_names = _get_curve_var_names(graph_layout)
    stars_values = _collect_stars_values(curve_var_names, available_varnames)

    new_graph_layouts = [_replace_stars_by_values(graph_layout, stars_values_)
                         for stars_values_ in stars_values]

    return new_graph_layouts


def _replace_stars_by_values(graph_layout, stars_values):
    new_graph_layout = {}

    # handle curves
    curves = []
    for curve in graph_layout['curves']:
        new_curve = {}
        for key, value in curve.items():
            new_curve[key] = _replace_stars_in_string(value, stars_values)
        curves.append(new_curve)
    new_graph_layout['curves'] = curves

    # handle other
    for key, value in graph_layout.items():
        if key == 'curves':
            continue
        new_graph_layout[key] = _replace_stars_in_string(
            value, stars_values)

    return new_graph_layout


def _collect_stars_values(var_names, available_varnames, sort=True):
    stars_values_groups = []
    for var_name in var_names:
        if var_name.count('*') == 0:
            continue

        stars_values_groups.append(_find_stars_values_in_string(var_name, available_varnames))

    # set intersection
    stars_values = set(stars_values_groups[0])
    for stars_values_group in stars_values_groups[1:]:
        stars_values = stars_values.intersection(stars_values_group)

    stars_values = list(stars_values)
    if sort:
        stars_values.sort(key=lambda x: x[0])

    return stars_values


def _find_stars_values_in_string(var_name, available_varnames):
    n = var_name.count('*')
    regex_var = var_name.replace('*', '(.*)')

    stars_values = re.findall(regex_var, '\n'.join(available_varnames))
    if n == 1:
        stars_values = [(star_value,) for star_value in stars_values]

    return stars_values


def _replace_stars_in_string(string, stars_values):
    if string.count('*') == 0:
        return string

    string_ls = string.split('*')

    new_string = ''
    for string_, star_value in zip(string_ls, stars_values):
        new_string += f'{string_}{star_value}'
    new_string += string_ls[-1]

    return new_string
