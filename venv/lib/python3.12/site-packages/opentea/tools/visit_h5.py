"""Visit h5py file

This should be removed because it is deprecated
"""
import h5py
import numpy as np
import yaml

__all__ = ["get_h5_structure", "print_h5_structure"]


def get_h5_structure(h5_filename):
    """ Show hdf5 file components

        Parameters:
        ----------
        h5_filename : path to hdf5 file to inspect
    """
    with h5py.File(h5_filename, "r") as node:
        out = log_hdf_node(node)
    return out


def print_h5_structure(h5_filename):
    """ Show hdf5 file components

        Parameters:
        ----------
        h5_filename : path to hdf5 file to inspect
    """
    dict_ = get_h5_structure(h5_filename)

    print(yaml.dump(dict_, default_flow_style=False))


def log_hdf_node(node):
    """
    Build a dictionnary with the structure of a HDF5 node

    Parameters:
    -----------
    node : hdf5 node

    Returns:
    --------
    a dictionnary
    """
    out = dict()

    print("Warning: log_hdf_node is deprecated, use hdfdict instead...")

    def extend_dict(dict_, address, attr):
        tmp = dict_
        for key in address[:]:
            if key not in tmp:
                #tmp[key] = dict()
                tmp[key] = attr.copy()
            tmp = tmp[key]

    def visitor_func(name, node):
        key_list = [item.strip() for item in name.split('/')]
        if isinstance(node, h5py.Dataset):
            attr = dict()
            attr["dtype"] = str(node.dtype)
            attr["value"] = _get_node_description(node)
            extend_dict(out, key_list, attr)
        else:
            pass

    node.visititems(visitor_func)

    return out

def _ascii2string(ascii_list):
    """ Ascii to string conversion

        Parameters:
        -----------
        ascii_list : a list of string to be converted

        Returns:
        --------
        a string joining the list elements

    """
    return ''.join(chr(i) for i in ascii_list[:-1])


def _get_node_description(node):
    """Get number of elements in an array or
      value of a single-valued node.

    Parameters:
    -----------
    node : hdf5 node

    Returns:
    --------
    a value with a Python format
    None if data is not a singlevalued quantity
    """
    out = None
    value = node[()]
    shape = node.shape
    if np.prod(shape) == 1:
        # this is a strong assumption because if you find int8
        # your are probably looking at an hdf5 file applying the cgns standard
        if node.dtype in ["int8"]:
            out = np.char.array(_ascii2string(value))[0]
        elif shape in [(1), (1,)]:
            if node.dtype in ["int32", "int64"]:
                out = int(value[0])
            elif node.dtype in ["float32", "float64"]:
                out = float(value[0])
            #else:
            #    raise RuntimeError()
        # else:
        #     raise RuntimeError()
    else:
        out = "array of %s elements" %(" x ".join([str(k) for k in shape]))
    return out
