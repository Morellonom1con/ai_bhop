import numpy as np


def get_conns(*args, elem_type='quad', **kwargs):
    """Get connectivities given element type.

    Args:
        args: number of columns, rows and (if applicable) layers.
        elem_type (str): 'tri', 'quad', 'tetra', 'hexahedron'.
    """
    get_conns_elem = {'triangle': get_tri_conns,
                      'quad': get_quad_conns,
                      'tetra': get_tetra_conns_layered,
                      'hexahedron': get_hexa_conns_layered}

    return get_conns_elem[elem_type](*args, **kwargs)


def get_quad_conns(c, r, circular=False):
    dofs = [i for i in range(c * r)]
    if circular:
        dofs += dofs[:c]

    conns = []
    for j in range(c - 1 + circular):
        for i in range(r - 1):
            k = j * r
            conns.append((dofs[i + k], dofs[1 + i + k], dofs[1 + i + k + r], dofs[i + k + r]))

    return np.array(conns)


def get_tri_conns(c, r, circular=False):
    dofs = [i for i in range(c * r)]
    if circular:
        dofs += dofs[:c]

    conns = []
    for j in range(c - 1 + circular):
        for i in range(r - 1):
            k = j * r
            conns.append((dofs[i + k], dofs[1 + i + k], dofs[1 + i + k + r]))
            conns.append((dofs[i + k], dofs[1 + i + k + r], dofs[i + k + r]))

    return np.array(conns)


def get_hexa_conns(c, r, reverse=False):
    """
    Args:
        reverse (bool): If False, assumes 4 first nodes are below last nodes.
            Affects sign of volumes (which must be positive).

    """
    n_elem_layer = r * c

    # roll due to connectivity order
    conns_square = np.roll(get_quad_conns(c, r), 1, axis=1)
    if not reverse:
        conns_square = conns_square[:, ::-1]
    conns = np.c_[conns_square, conns_square + (n_elem_layer)]

    return conns


def get_hexa_conns_layered(c, r, n, reverse=False):
    conns = get_hexa_conns(c, r, reverse=reverse)
    return _repeat_node_group_layer(c, r, n - 1, conns)


def get_tetra_conns(c, r, reverse=False):
    """
    Notes:
        Following Figure 9 of [1] to define elements.

    References:
        [1] https://www.researchgate.net/publication/271707352_Human_Facial_Soft_Tissue_Modeling_Using_Finite_Element_Method/figures?lo=1
    """
    conns_quad = get_quad_conns(c, r)
    opposite_face = conns_quad + (c * r)

    conns = []
    for i in range(conns_quad.shape[0]):
        if reverse:
            v0, v1, v3, v2 = conns_quad[i]
            v4, v5, v7, v6 = opposite_face[i]
        else:
            v4, v5, v7, v6 = conns_quad[i]
            v0, v1, v3, v2 = opposite_face[i]

        # just choosing the right dofs for each element (think about a cube)
        elem0 = [v4, v7, v5, v0]
        elem1 = [v0, v3, v2, v7]
        elem2 = [v4, v6, v7, v0]
        elem3 = [v1, v5, v7, v0]
        elem4 = [v0, v1, v3, v7]
        elem5 = [v2, v7, v6, v0]

        conns.extend([elem0, elem1, elem2, elem3, elem4, elem5])

    return np.array(conns)


def get_tetra_conns_layered(c, r, n, reverse=False):
    conns = get_tetra_conns(c, r, reverse=reverse)
    return _repeat_node_group_layer(c, r, n - 1, conns)


def _repeat_node_group_layer(c, r, repeats, group):
    """Returns corresponding nodes in successive layers.
    """
    group_layered = group.copy()
    for i in range(1, repeats):
        group_layered = np.r_[group_layered, group + i * c * r]

    return group_layered


def get_bound_nodes(*args, **kwargs):
    if len(args) == 2:
        return get_bound_nodes_2d(*args, **kwargs)
    elif len(args) == 3:
        return get_bounds_nodes_3d(*args, **kwargs)


def get_bound_nodes_2d(c, r, labels=('Bottom', 'Right', 'Top', 'Left')):
    n = r * c
    bottom = np.array([i for i in range(r)])
    right = np.array([j * r + r - 1 for j in range(c)])
    top = np.array([i for i in range(n - r, n)])
    left = np.array([j * r for j in range(c)])

    node_sets = [bottom, right, top, left]
    return {label: node_set for label, node_set in zip(labels, node_sets)}


def get_bounds_nodes_3d(c, r, n,
                        labels=('Bottom', 'Top', 'Left', 'Right', 'Back', 'Front'),
                        reverse=False):
    n_bottom = c * r
    bottom = np.array([i for i in range(n_bottom)])
    top = bottom + (n - 1) * n_bottom
    left = _repeat_node_group_layer(c, r, n, np.array([i for i in range(r)]))
    right = _repeat_node_group_layer(c, r, n, np.array([i + n_bottom - r for i in range(r)]))
    back = _repeat_node_group_layer(c, r, n, np.array([j * r for j in range(c)]))
    front = _repeat_node_group_layer(c, r, n, np.array([j * r + r - 1 for j in range(c)]))

    node_sets = [bottom, top, left, right, back, front]
    if reverse:
        node_sets[0], node_sets[1] = node_sets[1], node_sets[0]

    return {label: node_set for label, node_set in zip(labels, node_sets)}
