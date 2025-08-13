"""
Python tool to test if h5 files are identical.
"""
import h5py
import filecmp
import difflib
import pytest
import os
import numpy as np

# TODO: improve logging
# TODO: shouldn't this be based on hdfdict?


def _read_attributes(hval):
    """Return the attribute."""
    attr = {}
    for k in hval.attrs:
        attr[k] = type(hval.attrs[k])
    return attr


def _read_group(hval):
    """ returns summary of group.
    the only element for comparison here is the group's attributes
    """
    desc = {}
    desc["attr"] = _read_attributes(hval)
    desc["htype"] = "group"
    return desc


def _read_data(hval):
    """ returns summary of dataset
    the only elements for comparison here are the dataset's attributes,
    and the dataset dtype"""
    desc = {}
    desc["attr"] = _read_attributes(hval)
    desc["htype"] = "dataset"
    desc["dtype"] = type(hval[()])
    desc["data"] = hval[()]
    return desc


def _evaluate_group(path, grp):
    """creates and returns a summary description
    for every element in a group
    """
    desc = {}
    for key, val in grp.items():
        if isinstance(val, h5py.Dataset):
            desc[key] = _read_data(val)
        elif isinstance(val, h5py.Group):
            desc[key] = _read_group(val)
        else:
            raise NotImplementedError(
                "Unknown h5py type: %s (%s -- %s)" %
                (type(val), path, key))
    return desc


def _same_groups(file1, grp1, file2, grp2, path):
    """Compare groups idientified as same."""
    log = str()
    log += ("------------------------------\n")
    log += ("Examining " + path + "\n")
    output = True

    desc1 = _evaluate_group(path, grp1)
    desc2 = _evaluate_group(path, grp2)
    common = []
    for k in desc1:
        if k in desc2:
            common.append(k)
        else:
            output = False
            log += ("** Element '%s' only in '%s' (DIFF_UNIQUE_A)**\n" %
                    (k, file1))
    for k in desc2:
        if k not in desc1:
            output = False
            log += ("** Element '%s' only in '%s' (DIFF_UNIQUE_B)**\n" %
                    (k, file2))
    for i, _ in enumerate(common):
        name = common[i]
        log += ("\t" + name + "\n")
        # compare types
        htype_1 = desc1[name]["htype"]
        htype_2 = desc2[name]["htype"]
        if htype_1 != htype_2:
            output = False
            log += (
                "**  Different element types: "
                + "'%s' and '%s' (DIFF_OBJECTS)\n" % (htype_1, htype_2))
            continue    # different hdf5 types -- don't try to compare further
        if htype_1 not in ("dataset", "group"):
            log += (
                "WARNING: element is not a recognized type"
                + " (%s) and isn't being evaluated\n" % htype_1)
            continue
        # handle datasets first
        if desc1[name]["htype"] != "dataset":
            continue
        # compare data
        if desc1[name]["dtype"] != desc2[name]["dtype"]:
            dtype_1 = desc1[name]["dtype"]
            dtype_2 = desc2[name]["dtype"]
            output = False
            log += ("** Different dtypes: '%s' and '%s' (DIFF_DTYPE)**\n" % (dtype_1, dtype_2))
        # compare attributes
        for k in desc1[name]["attr"]:
            if k not in desc2[name]["attr"]:
                output = False
                log += ("** Attribute '%s' only in '%s' (DIFF_UNIQ_ATTR_A)**\n" %
                        (k, file1))
        for k in desc2[name]["attr"]:
            if k not in desc1[name]["attr"]:
                output = False
                log += ("** Attribute '%s' only in '%s' (DIFF_UNIQ_ATTR_B)**\n" %
                        (k, file2))
        for k in desc1[name]["attr"]:
            if k in desc2[name]["attr"]:
                val = desc1[name]["attr"][k]
                val2 = desc2[name]["attr"][k]
                if val != val2:
                    output = False
                    log += (
                        "** Attribute '%s' has different type: '%s' and '%s' (DIFF_ATTR_DTYPE)\n" %
                        (k, val, val2))

    for i, _ in enumerate(common):
        name = common[i]
        # compare types
        if desc1[name]["htype"] != desc2[name]["htype"]:
            continue    # problem already reported
        if desc1[name]["htype"] != "group":
            if desc1[name]["htype"] == "dataset":
                try:
                    if desc1[name]["data"].shape != desc2[name]["data"].shape:
                        output = False
                        log += (f'datasets  {name} shape mismatch: A {desc1[name]["data"].shape}, B {desc2[name]["data"].shape} \n')
                        log += (f'\t A= {desc1[name]["data"]}\n')
                        log += (f'\t B= {desc2[name]["data"]}\n')
                        continue
                except AttributeError:
                    continue
                try:    
                    aresame= np.allclose(desc1[name]["data"],desc2[name]["data"])
                except TypeError: # Stricter if not dataset
                    aresame = desc1[name]["data"]==desc2[name]["data"]

                if not aresame:
                    output = False
                    log += (f"datasets {name} A and B failed Numpy Allclose test\n")
                    log += (f'\t A= {desc1[name]["data"]}\n')
                    log += (f'\t B= {desc2[name]["data"]}\n')
                    
                    
                continue
            else:
                log += (f'Skipping htype {desc1[name]["htype"]}\n')
                continue
        # compare attributes
        for k in desc1[name]["attr"]:
            if k not in desc2[name]["attr"]:
                output = False
                log += ("** Attribute '%s' only in '%s' (DIFF_UNIQ_ATTR_A)**\n" %
                        (k, file1))
        for k in desc2[name]["attr"]:
            if k not in desc1[name]["attr"]:
                output = False
                log += ("** Attribute '%s' only in '%s' (DIFF_UNIQ_ATTR_B)**\n" %
                        (k, file2))
        # recurse into subgroup
        sub_log, sub_out = _same_groups(
            file1, grp1[name], file2, grp2[name], path + name + "/")
        log += sub_log
        if sub_out is False:
            output = False

    return log, output


def h5same_files(file1: str, file2: str)-> bool:
    """
    *Main call function to test two h5 files.*

    Use as:
        >assert h5same_files(fileA, fileB)

    :param file1: Path of the first file to compare
    :type file1: str
    :param file2: Path of the second file to compare
    :type file2: str
    :returns: **True** if files are identical, **False** otherwise
    """
    fin_1 = h5py.File(file1, 'r')
    fin_2 = h5py.File(file2, 'r')

    log, sameh5 = _same_groups(file1, fin_1["/"], file2, fin_2["/"], "/")
    print(log)
    return sameh5



def compare_textfile(txtfile1: str, txtfile2: str)-> None:
    """Small utility to better format diff lib
    
    Use as:
        >compare_textfile(txtfile1, txtfile2)  #No assert needed (included inside)

    raises:
        pytest exception of file differs
    """
    try:
        assert filecmp.cmp(txtfile1, txtfile2)
    except AssertionError:
        with open(txtfile1, "r") as fin:
            str1 =fin.readlines()
        with open(txtfile2, "r") as fin:
            str2 = fin.readlines()
        diff_ = list()
        for line in difflib.ndiff(str1, str2):
            for mark in ["+ ", "- ", "? "]:
                if line.startswith(mark):
                    diff_.append(line)
                    print(">>>>", line)
        pytest.fail(
            "Comparing text files :\n"
            + "..."+ os.path.split(txtfile1)[1] + "\n"
            + "..."+ os.path.split(txtfile2)[1] + "\n"
            + '\n'.join(diff_))
