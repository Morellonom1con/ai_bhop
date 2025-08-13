""" module to data array-like dictionnary to files
    for visulaisation or storage pupopses
"""
import os
import numpy as np
from arnica.utils.nparray2xmf import (create_time_collection_xmf,
                                      NpArray2Xmf)

def dump_dico_2d_nparrays(data_dict, filename,
                          x_coords, y_coords,
                          z_coords, **kw_args):
    """

    Parameters:
    ===========
    data_dict : dictionnary holding the  2d arrays, on the format
                data_dict[key] = array(n1, n2)
                where (n1, n_2) is a subset of (n_x, n_y, n_z)

    filename : the xmf filename

    <x|y|z>_coords : 2d numpy arrays for coordiantes over each axis
                    must be of shape (n_1, n_2)

    keyword args (optional):
    ========================
    time  :  physical time corresponding to the array
             used in the xmf file as <Time Value="time"....
    grid_name : the name of the grid to be used in the xmf file
                as <Grid Name="grid_name"....
    domain_name : the name of the domain to be used in the xmf file
                as <Domain Name=domain_name....

    Returns:
    ========
    None
    """
    grid_name = kw_args.get('grid_name')
    time = kw_args.get('time')
    domain_name = kw_args.get('domain_name')

    fields = list(data_dict.keys())

    visu_ = NpArray2Xmf(filename,
                        time=time,
                        mesh_name=grid_name,
                        domain_name=domain_name)
    visu_.create_grid(x_coords, y_coords, z_coords)
    for field in fields:
        if field != 'xyz':
            visu_.add_field(data_dict[field], field)
    visu_.dump()

def dump_dico_2d_time_nparrays(data_dict, root_path, prefix,
                               x_coords, y_coords, z_coords,
                               **kw_args):
    #pylint: disable=too-many-arguments, too-many-locals
    """ Dumps a dictionnary of time series 2d arrays
       to xmf files

    Parameters:
    ===========
    data_dict : dictionnary holding the times series 2d
                arrays, on the format
                data_dict[key] = array(n_time, n1, n2)
                where:
                    - n_time is the number of time steps
                    - (n1, n_2) is a subset of (n_x, n_y, n_z)

    prefix : the prefix to be used to generate xmf filenames

    <x|y|z>_coords : 2d numpy arrays for coordiantes over each axis
                    must be of shape (n_1, n_2)

    keyword args (optional):
    ========================

    steps : a list of integer time series steps
            that will be used to generate xmf files
            on the format : <prefix>_<step>.xmf
            if None steps will be generated as the
            range of time dimension of data arrays

    times : a list of float physical times that will
            be used in xmf files to describe the
            time of each step.
            if None will be generated as the
            range of time dimension of data arrays
    """
    path = os.path.abspath(root_path)
    if not os.path.exists(path):
        os.makedirs(path)

    n_time = data_dict[list(data_dict.keys())[0]].shape[0]

    times = kw_args.get('times')
    steps = kw_args.get('steps')

    if times is None:
        times = np.linspace(0.0, n_time - 1., num=n_time, dtype=float)
    if steps is None:
        steps = np.linspace(0, n_time - 1, num=n_time, dtype=int)

    collection = []
    for step in range(n_time):
        step_dict = dict()
        for field in data_dict:
            step_dict[field] = data_dict[field][step, :, :]
        filename = "%s/%s_%08d.xmf" %(path, prefix, steps[step])
        grid_name = "%s_%08d" %(prefix, steps[step])
        dump_dico_2d_nparrays(step_dict, filename, x_coords, y_coords,
                              z_coords, time=times[step], domain_name=prefix,
                              grid_name=grid_name)

        collection.append("%s_%08d.xmf" %(prefix, steps[step]))

    filename = "%s/%s_collection.xmf" %(path, prefix)
    create_time_collection_xmf(collection, filename)


def dump_dico_1d_nparrays(filename, data_dict):
    #pylint: disable=too-many-locals
    """Write statistics to file

        Parameters:
        ===========
        filename : the file name to which array dictionnary are dumped
                    possible extensions : - .xlsx (if pandas is found)
                                          - .csv (default format)

        data_dict : a dictionnary holding the data arrays

        Returns:
        =======
        None
    """

    file_out = os.path.abspath(filename)
    if not os.path.exists(os.path.dirname(file_out)):
        os.makedirs(os.path.dirname(file_out))
    file_out, format_ = os.path.splitext(os.path.abspath(file_out))
    format_ = format_.replace('.', '').lower()

    if format_ not in ['csv', 'xlsx']:
        msg = ("Extension", format_, "is not supported. Supported extensions : "
               "'csv', 'xlsx'. Default extension 'csv' will be used.")
        print("WARNING [dump_dico_1d_nparrays] : ", msg)
        format_ = 'csv'

    elif format_ == "xlsx":
        try:
            #pylint: disable=import-outside-toplevel, import-error, unused-import
            import pandas as pd
            import openpyxl
        except ImportError as import_error:
            msg = ("dump_dico_1d_nparrays cannot find 'pandas' neither/nor "
                   "'openpyxl' packages. Import those packages or use '.csv' "
                   "extension to dump classic csv files instead.")
            raise ImportError(msg) from import_error

    if format_ == 'xlsx':
        file_out = "%s.%s" %(file_out, format_)
        df_ = pd.DataFrame.from_dict(data_dict)
        df_.to_excel(file_out, index=False)
    else:
        format_ = 'csv'
        file_out = "%s.%s" %(file_out, format_)
        header = " ; ".join(list(data_dict.keys()))
        n_columns = len(data_dict.keys())
        n_lines = data_dict[next(iter(data_dict.keys()))].shape[0]
        out_array = np.zeros((n_lines, n_columns))
        try:
            for col, key in enumerate(data_dict, 0):
                out_array[:, col] = data_dict[key][:]
        except ValueError as err:
            msg_err = f"Shape error in key {key}\n"
            msg_err += str(err)
            raise ValueError(msg_err)
        np.savetxt(file_out, out_array, delimiter=' ; ', header=header)

def dump_dict2xmdf(filename, grid, data_dict):
    """
    *Dump 2D matrices into hdf5 file*

    :param filename: Name of the hdf file
    :type filename: str
    :param grid: Array of xyz-coordinates of shape (n_v, n_u, 3)
    :param data_dict: Dict of field arrays of shape (n_v, n_u)
    """

    print("DEPRECATED TOOL. USE dump_dico_2d_nparrays() INSTEAD.")

    if '.h5' not in filename:
        filename += '.h5'

    xmdf = NpArray2Xmf(filename)
    xmdf.create_grid(np.take(grid, 0, -1),
                     np.take(grid, 1, -1),
                     np.take(grid, 2, -1))

    for var in data_dict:
        if var not in ['xyz']:
            xmdf.add_field(data_dict[var], var)

    xmdf.dump()

def dump_dico_0d(filename, data_dict):
    """Write statistics to file

        Parameters:
        ===========
        filename : the file name to which array dictionnary are dumped
                    possible extensions : - .xlsx (if pandas is found)
                                          - .csv (default format)

        data_dict : a dictionnary holding the data arrays

        Returns:
        ========
        None
    """

    file_out = os.path.abspath(filename)
    if not os.path.exists(os.path.dirname(file_out)):
        os.makedirs(os.path.dirname(file_out))
    file_out, format_ = os.path.splitext(os.path.abspath(file_out))
    format_ = format_.replace('.', '').lower()

    if format_ not in ['csv', 'xlsx']:
        msg = ("Extension", format_, "is not supported. Supported extensions : "
               "'csv', 'xlsx'. Default extension 'csv' will be used.")
        print("WARNING [dump_dico_0d] : ", msg)

    elif format_ == "xlsx":
        try:
            #pylint: disable=import-outside-toplevel, import-error, unused-import
            import pandas as pd
            import openpyxl
        except ImportError as import_error:
            msg = ("dump_dico_1d_nparrays cannot find 'pandas' neither/nor "
                   "'openpyxl' packages. Import those packages or use '.csv' "
                   "extension to dump classic csv files instead.")
            raise ImportError(msg) from import_error

    if format_ == 'xlsx':
        file_out = "%s.%s" %(file_out, format_)
        df_ = pd.DataFrame(data_dict, index=[0])
        df_.to_excel(file_out, index=False)
    else:
        format_ = 'csv'
        file_out = "%s.%s" %(file_out, format_)
        header = " , ".join(list(data_dict.keys()))
        n_columns = len(data_dict.keys())
        out_array = np.zeros(n_columns)
        for col, key in enumerate(data_dict, 0):
            out_array[col] = data_dict[key]
        np.savetxt(file_out, out_array[np.newaxis, :], delimiter=' , ', header=header)

def plot_dict_data_as_file(data_dict, filename, x_data, y_data, **kw_args):
    """ Generates and write XY-plot to file

    Parameters:
    ===========
    filename : the file to which the plot is written it contains
                the extension that defines the format e.g: 'plot_toto.png'
                Supported formats/extensions : png, pdf, ps, eps and svg
                If not provided, by default "pdf" extension is used.
    data_dict : a dictionnary holding the data arrays
    x_data : the key to the array holding the abscissa data
    y_data : the key to the array holding the y data

    keyword args (optional):
    ========================
    x_label : label of the x axis, by default x_data is used (supports latex)
    y_label : label of the y axis, by default y_data is used (supports latex)
    """
    #pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt

    x_label = kw_args.get('x_label')
    if x_label is None:
        x_label = x_data

    y_label = kw_args.get('y_label')
    if y_label is None:
        y_label = y_data

    path = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(path):
        os.makedirs(path)
    file_out, format_ = os.path.splitext(os.path.abspath(filename))
    format_ = format_.replace('.', '').lower()

    if format_ not in ["png", "pdf", "ps", "eps", "svg"]:
        format_ = 'pdf'


    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(5.5, 5))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15,
                        top=0.9, wspace=0.27)

    axe = fig.add_subplot(111)
    axe.plot(data_dict[x_data], data_dict[y_data], c='k', lw=2, marker='s')
    axe.set_ylabel(y_label, fontsize=20, fontweight='bold')
    axe.set_xlabel(x_label, fontsize=20, fontweight='bold')
    axe.grid(which='major', color='gray', linestyle='--',
             dashes=(8, 12), linewidth=0.5)

    axe.minorticks_on()

    axe.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    for axis in [axe.xaxis, axe.yaxis]:
        #for tick in axis.get_major_ticks():
        #    tick.label.set_fontsize(16)
        tick = axis.get_offset_text()
        tick.set_size(16)

    file_out = "%s.%s" %(file_out, format_)
    plt.savefig(file_out)
