"""Module with tools to recursively compare nested objects
BEWARE of stack overflows in recursions"""

import numpy as np
import h5py

from h5cross import hdfdict

# Handle numpy 2.0+ deprecations
optional_types = []
if hasattr(np, 'float_'):
    optional_types.append(np.float_)
if hasattr(np, 'complex_'):
    optional_types.append(np.complex_)

DATA_TYPES = (
    np.short,
    np.ushort,
    np.int8,
    np.uint8,
    np.intc,
    np.uintc,
    np.int16,
    np.uint16,
    np.uint,
    np.int_,
    np.int32,
    np.uint32,
    np.longlong,
    np.ulonglong,
    np.uint64,
    np.int64,
    np.float16,
    np.half,
    np.float32,
    np.single,
    np.float64,
    np.double,
    np.longdouble,
    *optional_types,
    float,
    int,
    complex
)

DESC_TYPES = (np.bytes_, np.string_, str, bytes, np.bool_, bool)


def h5dict_safe(file: str) -> dict:
    """
    h5 files safe loader with hdfdict.
    :param file: str ou os.Path menant au h5 à lire
    """
    with h5py.File(file, 'r') as fin:
        try:
            return hdfdict.load(fin, lazy=False)
        except TypeError:
            return hdfdict.load(fin)


def h5_same(source_file: str, target_file: str, **kwargs):
    """
    *Main call function to test two h5 files.*

    :param source: Path to the source file to compare
    :param target: Path to the target file to compare
    :returns: **True** if files are identical, **False** otherwise
    """
    source_dict = h5dict_safe(source_file)
    target_dict = h5dict_safe(target_file)
    try:
        return dict_same(source_dict, target_dict, **kwargs)
    except ValueError as err:
        raise ValueError(
            f"\nSource file is {source_file}\nTarget file is {target_file}\n" + str(err)
        )


def dict_same(source_dict: dict, target_dict: dict, **kwargs):
    """
    *Main call function to test two dictionaries.*

    :param source_dict: first dict to compare
    :param target_dict: second dict to compare
    :returns: **True** if files are identical, raises an error otherwise
    """
    is_identical, log = _same_nob(source_dict, target_dict, **kwargs)
    if not is_identical:
        raise ValueError(log)
    return True


def _same_nob(src: dict, tgt: dict, path="/") -> bool:
    """Compare NOb identified as same, numpy arrays included, recursively"""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    log = str()
    log += "------------------------------\n"
    log += "Examining " + path + "\n"
    output = True

    common = dict()

    # compare keys
    for key in src:
        if not key in tgt:
            output = False
            log += f"Attribute {key} only in source h5 file.\n"
        else:
            common[key] = src[key]
    for key in tgt:
        if not key in src:
            output = False
            log += f"Attribute {key} only in target h5 file.\n"

    # compare whats inside
    for key in common:
        log += "\t" + key + "\n"
        tgt_t = type(tgt[key])
        src_t = type(src[key])
        if not src_t == tgt_t:
            output = False
            log += (
                f"**  Different attribute types: {src_t} and {tgt_t} (DIFF_OBJECTS)\n"
            )
            continue  # different hdf5 types -- don't try to compare further

        # handle datasets first
        if isinstance(tgt[key], (DESC_TYPES, DATA_TYPES)):
            if not src[key] == tgt[key]:
                output = False
                log += f"** Different values for elements {key}: {src[key]} and {tgt[key]}\n"

        elif isinstance(tgt[key], (list, tuple, np.ndarray)):
            if np.shape(tgt[key]) != np.shape(src[key]):
                output = False
                log += f" ** Different shapes for arrays {key}"
            elif isinstance(tgt[key][0], (np.ndarray, DATA_TYPES)):
                if not np.allclose(src[key], tgt[key]):
                    output = False
                    log += f"** Different values in arrays {key}\n"
            elif isinstance(tgt[key][0], DESC_TYPES):
                if not np.all(src[key] == tgt[key]):
                    output = False
                    log += f"** Different values in arrays {key}: {src[key]} and {tgt[key]}\n"
            else:
                raise NotImplementedError(
                    f"Attribute's values type: {type(tgt[key][0])}"
                )

        # handle nob recursively
        elif isinstance(tgt[key], dict):
            subout, sublog = _same_nob(src[key], tgt[key], path=path + key + "/")
            log += sublog
            if subout is False:
                output = subout
        else:
            raise NotImplementedError(f"Attribute type: {type(tgt[key])}")

    return output, log
