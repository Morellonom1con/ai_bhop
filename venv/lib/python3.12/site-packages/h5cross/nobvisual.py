"""Module with functionalities related to the visual aspects of h5cross.
"""

from nobvisual.nob2nstruct import visual_treenob
from nobvisual.nobcompare import nob_compare_tkinter

from h5cross.visit_h5 import get_h5_structure
from h5cross.io import load_hdf_as_dict
from h5cross.stats import compute_dict_stats
from h5cross.utils import merge_dicts


def compare_h5(file1, file2, add_stats=False, start_mainloop=True):
    """Function allowing the comparison of two hdf5 files

    Input:
        :file1: (string), name of first file for comparison
        :file2: (string), name of second file for comparison
        :add_stats: (boolean), option to compute statistics on both input files
    Output:
        :None: interactive comparative view
    """

    # TODO: write separate functionalities for this?? (Jimmy)
    nobs = []
    for file in [file1, file2]:
        dict_ = get_h5_structure(file, as_dict=not add_stats)

        if add_stats:
            data_dict = load_hdf_as_dict(file)
            stats = compute_dict_stats(data_dict)
            dict_ = merge_dicts(dict_, stats, as_dict=True)

        nobs.append(dict_)

    title = "Showing file differences"
    title += f"\nLeft: {file1}"
    title += f"\nRight: {file2}"
    nob_compare_tkinter(*nobs, title=title, start_mainloop=start_mainloop)


def visual_h5(filename, start_mainloop=True):
    """Function calling nobvisual for visual view of hdf5 file

    Input:
        :filename: name of hdf5 file
    Output:
        :None: interactive view
    """

    nob = get_h5_structure(filename)
    visual_treenob(nob, title=f'Showing {filename}',
                   start_mainloop=start_mainloop)
