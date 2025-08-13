
import numpy as np
from pymech.dataset import open_dataset

from h5cross.convert._utils import fill_dict_empty_nested


def _add_values_to_dict_from_datasetnek5000(dict_, dataset, skip_list=None):
    """ Function adding nek5000 formatted field variables in a given dictionary
        containing variable names.

        Input:
            :dict_ : empty nested dictionary to fill
            :dataset: nek5000 dataset object
            :skip_list: optional list of type string with dict_ inputs for which
                        to not add the fields
        Output:
            :None: adds fields to the input dict_
    """
    if skip_list is None:
        skip_list = []
    total_len = dataset.dims['x'] * dataset.dims['y'] * dataset.dims['z']

    for key in list(dict_.keys()):
        if key not in skip_list:
            dict_[key] = np.reshape(dataset[key].data, total_len)


def convert_nek5000(file, skip_vars=None):
    """ Main function controlling the conversion of a nek5000 file format
        to dictionary for hdf5 output.

        Input:
            :file: path to nek5000 file
            :skip_vars: optional list of type string containing fields we wish to
                not consider transferring to the hdf5 output
                (default = None)
        Output:
            :new_dict: python dictionary format structured for hdf5 write output
    """

    if skip_vars is None:
        skip_vars = []

    dataset = open_dataset(file)
    var_names = []
    for item in list(dataset.data_vars):
        if item not in skip_vars:
            var_names.append(item)

    new_dict = dict()
    fill_dict_empty_nested(new_dict, var_names)
    _add_values_to_dict_from_datasetnek5000(new_dict, dataset)

    # Add some additional info
    new_dict['Parameters'] = {}
    # print(type(dataset.dims['x']))
    new_dict['Parameters']['Dimensions'] = {'Nx': np.array([dataset.dims['x']], dtype="int32"),
                                            'Ny': np.array([dataset.dims['y']], dtype="int32"),
                                            'Nz': np.array([dataset.dims['z']], dtype="int32")}
    new_dict['Parameters']['time'] = np.array([dataset.coords['time'].data], dtype="float64")

    return new_dict


READER_MAP = {
    'nek5000': convert_nek5000
}
