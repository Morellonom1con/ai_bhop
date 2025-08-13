""" Module container for class ProxyH5
"""

# pylint: disable=duplicate-code
import warnings
import h5py
import numpy as np
from opentea.noob.noob import nob_get

__all__ = ["ProxyH5"]



class ProxyH5:
    """Class container for hdf5 file inspector

       Parameters :
       ===========
       h5_filename : Path to hdf5 file
    """

    def __init__(self, h5_filename):
        warnings.simplefilter("ignore")
        msgwarn = "ProxyH5 is deprecated, move to hdfdict for h5 quick access"
        warnings.warn(msgwarn)
        print(msgwarn)
        with h5py.File(h5_filename, "r") as node:
            self.nob = h5_node_to_dict(node)
            self._structure = _log_hdf_node(node)
            self.datatsets_names = h5_datasets_names(node)

    def show(self, style=None):
        """ Pretty print of the hdf5 content

            Parameters :
            ==========
            style : style of printing, possible options
                    - yaml : for a yaml formatting
                    - json : for a yaml formatting
                    - None : for a default printing
        """
        pprint_dict(self._structure, style=style)

    def get_field(self, identifier):
        """ Get a value of a field given its identifier

            Parameters:
            ==========
            identifier : the adress of the content to retrieve.

                    Possible options:
                        - String : key of the value to retrieve
                         (e.g 'content')

                        - String : A posix-like full address
                          (e.g 'full/address/to/content')

                        - List : A list of adress stages, complete or not
                          (e.g ['full', 'address', 'to', 'content'] or,
                           ['content'])

                        - A list of lists : a list holding address stages,
                          as it is done for instance in h5py.
                          (e.g [['full'], ['address'], ['to'], ['content']]])
            Returns:
            =======
            field : array or value of the query field
        """
        return hdf5_query_field(self.nob, identifier)


class H5LookupError(Exception):
    """ Exception class for h5 lookup
    """

    def __init__(self, message):
        self.message = message
        super().__init__(message)


def h5_datasets_names(node):
    """Get hdf5 node datasets names

        Parameters:
        ==========
        node : hdf5 node

        Returns:
        =======
        ds_names : a list of datasets names
    """

    ds_names = []
    for path, _ in _rec_h5_dataset_iterator(node):
        ds_names.append(path.split("/")[-1].strip())
    return ds_names


def hdf5_query_field(node, address):
    """Get the content of the adress

    Parameters:
    -----------
    node : hdf5 node
    address : the adress of the content to retrieve. Possible options:
                - String : key of the value to retrieve (e.g 'content')
                - String : A posix-like full address
                  (e.g 'full/address/to/content')
                - List : A list of adress stages, complete or not
                  (e.g ['full', 'address', 'to', 'content'] or, ['content'])
                - A list of lists : a list holding address stages, as it is
                  done for instance in h5py.
                  (e.g [['full'], ['address'], ['to'], ['content']]])

    Returns:
    --------
    content of the address
    """
    if isinstance(address, str):
        keys = address.split("/")
    elif isinstance(address, list):
        if all([isinstance(item, str) for item in address]):
            keys = address
        else:
            keys = [item for sublist in address for item in sublist]
    else:
        msg = "Unknown address %s !\n" % str(address)
        msg = msg + "address should be a list of keys or a string"
        raise H5LookupError(msg)

    return nob_get(node, *keys, failsafe=False)


def _log_hdf_node(node):
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

    def extend_dict(dict_, address, attr):
        tmp = dict_
        for key in address[:]:
            if key not in tmp:
                # tmp[key] = dict()
                tmp[key] = attr.copy()
            tmp = tmp[key]

    def visitor_func(name, node):
        key_list = [item.strip() for item in name.split("/")]
        if isinstance(node, h5py.Dataset):
            attr = dict()
            attr["dtype"] = str(node.dtype)
            attr["value"] = _get_node_description(node)
            extend_dict(out, key_list, attr)
        else:
            pass

    node.visititems(visitor_func)

    return out


def h5_node_to_dict(node):
    """Read hdf5 node values and structure into a dictionnary

        Parameters:
        ==========
        node : hdf5 node

        Returns:
        =======
        data_dict : a dictionnary holding the data
    """

    def _get_datasets(buf=None, data_dict=None, path_=""):
        if data_dict is None:
            data_dict = dict()

        for gname, group in buf.items():
            data_dict[gname] = dict()
            path = path_ + "/" + gname
            if not isinstance(group, h5py.Dataset):
                _get_datasets(buf[gname], data_dict[gname], path_=path)
            else:
                if buf[path][()].size >= 1:
                    data_dict[gname] = buf[path][()]
        return data_dict

    data_dict = _get_datasets(node)

    return data_dict


def _rec_h5_dataset_iterator(node, prefix=""):
    for key in node.keys():
        item = node[key]
        path = "{}/{}".format(prefix, key)
        if isinstance(item, h5py.Dataset):  # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group):  # test for group (go down)
            yield from _rec_h5_dataset_iterator(item, path)


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
            # else:
            #    raise RuntimeError()
        # else:
        #     raise RuntimeError()
    else:
        out = "array of %s elements" % (" x ".join([str(k) for k in shape]))
    return out


def pprint_dict(dict_, style="yaml"):
    """Pretty pring a dictionnary using yaml or json formatting"""
    if style == "json":
        lines = json.dumps(dict_, indent=4)
    elif style == "yaml":
        lines = yaml.dump(dict_, default_flow_style=False)
    else:
        raise RuntimeError(f"style {style} not found")
    print(lines)


def test(filename):
    """ example of usage
    """
    prox = ProxyH5(filename)
    prox.show(style="yaml")
    print(prox.get_field(["GaseousPhase", "rhou"]))
    print(prox.get_field("GaseousPhase/rhou"))
    print(prox.get_field([["GaseousPhase"], ["rhou"]]))
    print(prox.get_field("rhou"))


if __name__ == "__main__":
    test("/Users/erraiya/Desktop/HULK/hulk/hulk/Data/solut_00027500.h5")
