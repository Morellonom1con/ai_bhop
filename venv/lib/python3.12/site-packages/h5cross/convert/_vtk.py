
import numpy as np
import vtk

from h5cross.convert._utils import fill_dict_empty_nested


def _add_values_to_dict_from_datasetvtu(dict_, dataset, skip_list=None,
                                        add_mesh=True):
    """ Function adding vtk unstructured formatted field variables in a given dictionary
        containing variable names.

        Input:
            :dict_ : empty nested dictionary to fill
            :dataset: vtk file reader output object
            :skip_list: optional list of type string with dict_ inputs for which
                        to not add the fields
        Output:
            :None: adds fields to the input dict_
    """

    if skip_list is None:
        skip_list = []

    # Field variables part
    pointdata = dataset.GetPointData()
    for variter, varname in enumerate(dict_.keys()):
        if varname not in skip_list:
            array = pointdata.GetArray(variter)
            # Check for multicomponent 1D arrays
            if array.GetNumberOfComponents() > 1:
                # structure in (X0,Y0,Z0) (X1,Y1,Z1) ...
                num_comp = array.GetNumberOfComponents()
                num_values = array.GetNumberOfValues()
                tmp_data = np.zeros((int(num_values / num_comp), num_comp))

                for ii in range(int(num_values / num_comp)):
                    for jj in range(num_comp):
                        tmp_data[ii][jj] = array.GetValue(ii * num_comp + jj)

                comp_names = []
                for jj in range(num_comp):
                    comp_names.append('Component' + str(jj))
                # TODO: review
                fill_dict_empty_nested(dict_, comp_names, nested_key=varname)

                for kk, component in enumerate(comp_names):
                    dict_[varname][component] = np.array(tmp_data[:, kk])

            else:
                tmp_data = []
                for ii in range(array.GetNumberOfValues()):
                    tmp_data.append(array.GetValue(ii))
                dict_[varname] = np.array(tmp_data)

    if add_mesh:
        _add_mesh_values_to_dict_from_datasetvtu(dict_, dataset)


def _add_mesh_values_to_dict_from_datasetvtu(dict_, dataset=None):
    """ Function to add mesh points info from pvtu or vtu file to the given dictionary

        Input:
            :dict_ : empty nested dictionary to fill
            :dataset: vtk file reader output object
        Output:
            :None: adds fields to the input dict_
    """
    # Mesh points
    meshpoints = dataset.GetPoints().GetData()
    num_comp = meshpoints.GetNumberOfComponents()
    num_values = meshpoints.GetNumberOfValues()
    tmp_data = np.zeros((int(num_values / num_comp), num_comp))

    for ii in range(int(num_values / num_comp)):
        for jj in range(num_comp):
            tmp_data[ii][jj] = meshpoints.GetValue(ii * num_comp + jj)
    comp_names = []
    for jj in range(num_comp):
        comp_names.append('Component' + str(jj))

    dict_['Mesh'] = {}
    fill_dict_empty_nested(dict_, comp_names, nested_key='Mesh')
    for kk, component in enumerate(comp_names):
        dict_['Mesh'][component] = np.array(tmp_data[:, kk])


def _get_vtkxml_filetype(filename, readmax_=5):
    """ Function returning the filetype of a vtk xml file.

        Input:
            :filename: input vtk xml file to check
            :readmax_: int, max number of lines of file to search for the type
        Output:
            :myline: str, file type found
            :Exception raised: in case file type not found
    """

    with open(filename, 'r') as fin:
        for _ in range(readmax_):
            myline = fin.readline()
            if "type=" in myline:
                return myline.split("type=")[1].split(' ')[0].split('"')[1]

        raise ValueError("VTK xml File type not found, did you specify the correct file type?")


def convert_vtu(file, parallel=False, skip_vars=None):
    """ Main function controlling the conversion of a vtk xml unstructured file format
        to dictionary for hdf5 output.

        NOTE:   Is currently only based on point data and not cell data.
                No connectivity info is added at the moment.

        Input:
            :file: path to vtk unstructured file
            :skip_vars: optional list of type string containing fields we wish to
                not consider transferring to the hdf5 output
                (default = None)
            :parallel: boolean, specification if we are dealing with pvtu of normal vtu file

        Output:
            :new_dict: python dictionary format structured for hdf5 write output
    """
    # TODO: cell data is ignored for now

    if skip_vars is None:
        skip_vars = []

    # Create reader object
    filetype = _get_vtkxml_filetype(file)
    if parallel:
        assert filetype == 'PUnstructuredGrid', "VTK file type does not coincide with user input"
        reader = vtk.vtkXMLPUnstructuredGridReader()
    else:
        assert filetype == 'UnstructuredGrid', "VTK file type does not coincide with user input"
        reader = vtk.vtkXMLUnstructuredGridReader()

    # Read vtk file
    reader.SetFileName(file)
    reader.Update()

    # Obtain variable names
    vtkdata = reader.GetOutput()
    pointdata = vtkdata.GetPointData()
    vars_num = reader.GetNumberOfPointArrays()
    vars_name = []
    for variter in range(vars_num):
        vars_name.append(pointdata.GetArrayName(variter).split()[0])

    # Create dictionary with structure
    new_dict = dict()
    fill_dict_empty_nested(new_dict, vars_name)

    # Fill dictionary with data
    _add_values_to_dict_from_datasetvtu(new_dict, vtkdata)

    return new_dict


def convert_vtk(file, skip_vars=None):
    """ Main function controlling the conversion of a vtk file format
        to dictionary for hdf5 output.

        NOTE:   Is currently only based on point data and not cell data.
                No connectivity info is added at the moment.
                Currently only for vtk unstructured.

        Input:
            :file: path to vtk unstructured file
            :skip_vars: optional list of type string containing fields we wish to
                not consider transferring to the hdf5 output
                (default = None)

        Output:
            :new_dict: python dictionary format structured for hdf5 write output
    """

    if skip_vars is None:
        skip_vars = []

    # Select reader
    reader = vtk.vtkUnstructuredGridReader()

    # Read vtk file
    reader.SetFileName(file)
    reader.Update()

    # Obtain variable names
    vtkdata = reader.GetOutput()
    pointdata = vtkdata.GetPointData()
    vars_num = pointdata.GetNumberOfArrays()

    vars_name = []
    for variter in range(vars_num):
        vars_name.append(pointdata.GetArrayName(variter).split()[0])

    # Create dictionary with structure
    new_dict = dict()
    fill_dict_empty_nested(new_dict, vars_name)

    # Fill dictionary with data
    _add_values_to_dict_from_datasetvtu(new_dict, vtkdata)

    return new_dict


READER_MAP = {
    'vtk': convert_vtk,
    'vtu': lambda *args, p=False, **kwargs: convert_vtu(*args, parallel=p, **kwargs),
    'pvtu': lambda *args, p=True, **kwargs: convert_vtu(*args, parallel=p, **kwargs)
}
