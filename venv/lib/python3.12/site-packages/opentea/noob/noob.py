"""Nested object services.

A nested object is here:
 -nested dicts
 -nested lists
 -a mix of nested lists and nested dicts

An address is a list of strings and/or integers
giving a position in the nested object

Hereafter, the -address, complete or not- statement
refer to an address with potentially missing elements.

EXAMPLE:@
for a dict such as :
d["a"]["b"]["c"]["d"]["e"]["f"]
this the full address
[ ["a","b","c","d","e"] ]
can be found  with either:
nob_find(d, "a","b","c","d","e") (full path)
nob_find(d, "b","d","e") (partial path)
nob_find(d, "e") (only one hint )


"""

import copy
import yaml
from typing import List

class NobReferenceError(Exception):
    """TO BE ADDED"""
    # print('NobReferenceError')


def str_address(addr)->str:
    """Return an address -key list- into a path-like string."""
    return '/'.join([str(i) for i in addr])


def nob_find(obj_, *keys)->List[ List[str]]:
    """Find all occurences matching a serie of keys in a nested object.

    Parameters
    ----------
    obj_ : nested object
    keys : address, complete or not

    Returns :
    ---------
    list of addresses matching the input address
    """
    if not keys:
        raise RuntimeError("No key provided..")
    target_key = keys[-1]
    matching_addresses = []

    def rec_findkey(obj_, target_key, path):
        if isinstance(obj_, dict):
            for key in obj_:
                if key == target_key:
                    matching_addresses.append(path + [key])
                rec_findkey(obj_[key], target_key, path + [key])
        if isinstance(obj_, list):
            for key, item in enumerate(obj_):
                if isinstance(item, dict):
                    if "name" in item:
                        if key == item["name"]:
                            matching_addresses.append(path + [key])
                rec_findkey(item, target_key, path + [key])

    rec_findkey(obj_, target_key, [])

    out = []
    for addr in matching_addresses:
        if all([clue in addr for clue in keys[:-1]]):
            out.append(addr)
    return out


def nob_find_unique(obj_, *keys)-> List[str]:
    """Find a unique occurences of a key in a nested object.
    Raise exceptions if problems

    Parameters
    ----------
    obj_ : nested object
    keys : address, complete or not

    Returns :
    ---------
    one single address matching the input address
    """
    out = None
    error_found = False
    error_msg = str()

    matchlist = nob_find(obj_, *keys)
    keys_str = " ".join([str(key) for key in [*keys]])

    if not matchlist:
        error_found = True
        error_msg = ("No match for keys -" + keys_str + "-")
    elif len(matchlist) > 1:
        error_found = True
        match_err = [str_address(match) for match in matchlist]
        error_msg = ("Multiple match for key -"
                     + keys_str + "-\n"
                     + "\n".join(match_err))
    else:
        out = matchlist[0]

    if error_found:
        raise NobReferenceError(error_msg)

    return out


def nob_node_exist(obj_, *keys)-> bool:
    """ Test if one node exist in a nested object

    Parameters
    ----------
    obj_ : nested object
    keys : address, complete or not

    Returns :
    ---------
    boolean
    """
    try:
        nob_find_unique(obj_, *keys)
    except NobReferenceError:
        return False
    return True


def nob_get(obj_, *keys, failsafe=False) :
    """Access a nested object by keys.

    Parameters
    ----------
    obj_ : nested object
    keys : address, complete or not
    failsafe : what to do if the node is missing
        - False : raise an exception
        - True : return None is missing
                 return the shortes path is several matches

    Returns
    -------
    if points to a leaf:
        immutable, the value stored in the leaf
    if points to a node:
        mutable : the object (dict or list) found at this address
    """

    if failsafe:
        match = nob_find(obj_, *keys)
        if match:
            address = match[0]
        else:
            return None
    else:
        address = nob_find_unique(obj_, *keys)

    tmp = obj_.copy()
    for key in address:
        tmp = tmp[key]
    return tmp


def unique_dict_key(dict_) -> str:
    """Return the only key of a single child dict."""
    keys = list(dict_.keys())
    if len(keys) != 1:
        raise ValueError(f'No unique key: {keys}')
    if not keys:
        raise ValueError('No child found')
    return keys[0]

def nob_get_only_child(obj_, *keys) -> str:
    """Return the only key of a single child dict."""
    dict_ = nob_get(obj_, *keys)
    if not isinstance(dict_, dict):
        raise ValueError("This is not a single child dictionnary")
    l_keys = list(dict_.keys())
    if not l_keys:
        raise ValueError("No child found")
    if len(l_keys) > 1:
        raise ValueError("Several child found")
    return l_keys[0]


def nob_set(obj_, value, *keys):
    """Assign a value to an object from a nested object.

    Parameters
    ----------
    obj_ : nested object
    keys : address, complete or not

    Returns
    -------
    change the object in argument (NOT STATELESS)
    """
    address = nob_find_unique(obj_, *keys)
    if address[:-1]:
        father = nob_get(obj_, *address[:-1])
    else:
        father = obj_
    father[address[-1]] = value


def nob_del(obj_, *keys, verbose=False):
    """ Delete all matchinf addresses in the nested object.
    Not a deletion in place, only the output argument is cropped.

    Parameters
    ----------
    obj_ : nested object
    keys : address, complete or not

    Returns
    -------
    obj_ : nested object without the matching keys
    """
    address_do_del = nob_find(obj_, *keys)
    if verbose:
        log = "Will delete :\n"
        for addr in address_do_del:
            log += "    " + str_address(addr) + "\n"
        print(log)

    def rec_delete_items(nob_in, curr_addr=None):

        nob_out = None
        if curr_addr is not None:
            pass
        else:
            curr_addr = list()

        if isinstance(nob_in, dict):
            nob_out = dict()
            for key in nob_in:
                next_addr = [*curr_addr, key]
                if next_addr not in address_do_del:
                    nob_out[key] = rec_delete_items(nob_in[key], next_addr)
                else:
                    if verbose:
                        print("removing", str_address(next_addr))
        elif isinstance(nob_in, list):
            nob_out = list()
            for i, list_item in enumerate(nob_in):
                next_addr = [*curr_addr, i]
                if next_addr not in address_do_del:
                    nob_out.append(rec_delete_items(list_item, next_addr))
                else:
                    if verbose:
                        print("removing", str_address(next_addr))
        else:
            nob_out = nob_in
        return nob_out
    nob_out = rec_delete_items(obj_)
    return nob_out


def nob_pprint(obj_, max_lvl=None)-> str:
    """ return a pretty print of a nested object.
    yaml.dump() in use for the display

    Parameters :
    ------------
    obj_ : nested object
    max_lvel : optional :  maximum nber of levels to show

    Output :
    --------
    out : string showing the nested_object structure
    """
    yamlstr = None
    if max_lvl is None:
        yamlstr = str(yaml.dump(obj_, default_flow_style=False))
    else:
        def rec_copy(obj_, lvl):
            out = None
            if lvl == 0:
                out = "(...)"
            else:
                if isinstance(obj_, dict):
                    out = {}
                    for key in obj_:
                        out[key] = rec_copy(obj_[key], lvl-1)
                elif isinstance(obj_, list):
                    out = []
                    for elmt in obj_:
                        out.append(rec_copy(elmt, lvl-1))
                else:
                    out = obj_
            return out

        out = rec_copy(obj_, max_lvl)
        yamlstr = str(yaml.dump(out, default_flow_style=False))
    return yamlstr


def nob_merge_agressive(base_obj, obj_to_add):
    """Merge two nested objects.

    In case of conflict, the object to add is prevalent

    Input :
    -------
    base_obj : the initial object
    obj_to_add : the object to add

    Output :
    --------
    merged_obj : the merged dictionnaries
    """
    merged_obj = copy.deepcopy(base_obj)

    if isinstance(obj_to_add, dict):
        if isinstance(merged_obj, dict):
            for key in obj_to_add:
                if key not in merged_obj:
                    merged_obj[key] = obj_to_add[key]
                else:
                    merged_obj[key] = nob_merge_agressive(merged_obj[key],
                                                          obj_to_add[key])
        else:
            merged_obj = obj_to_add
    else:
        merged_obj = obj_to_add

    return merged_obj
