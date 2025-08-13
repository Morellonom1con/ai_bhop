"""Module controlling YAML and hdf5 output files.
"""


import yaml
import h5py

from h5cross import hdfdict


def load_hdf_as_dict(h5_filename, lazy=False):
    """Return a dictionary of an hdf5 file.
    """
    with h5py.File(h5_filename, "r") as h5pf:
        data_dict = hdfdict.load(h5pf, lazy=False)
    return data_dict


def write_dict_to_yaml(dict_, filename, setindent=2):
    """ Generic function writin a dictionary as a yaml file

        Input:
            :dict_: dictionary to write out as yaml
            :filename: output name of the .yml file
            :setindent: indentation setting of yaml file
        Output:
            a YAML file
    """
    with open(f'{filename}.yml', 'w') as yaml_out:
        yaml.dump(dict_, yaml_out, indent=setindent)


def write_h5(dict_, filename='dump'):
    with h5py.File(f'{filename}.h5', "w") as fout:
        hdfdict.dump(dict_, fout, lazy=False)
