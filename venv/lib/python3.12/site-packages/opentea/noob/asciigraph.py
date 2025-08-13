"""String represeantion of a nested object"""

import re

TAB_LENGTH = 2
LINE_LENGTH = 256

def nob_asciigraph(nob):
    """Pretty printing of a nested object
    Inputs:
    ----------
    nob : The nested object

    Returns :
    ---------
    None
    """

    lout = asciigraph_rec(nob, level=0)
    if not lout:
        out = "WARNING from nob_asciigraph: no data to display!!!)"
    else:
        out = "\n".join(lout)
    return out


def asciigraph_rec(nob, level):
    """Pretty printing of a nested object
    Inputs:
    ----------
    nob : The nested object
    level : The initial level to start with

    Returns :
    ---------
    None
    """

    output_ls = list()
    children = []
    if isinstance(nob, dict):
        children.extend(_get_keys(nob))
    elif isinstance(nob, list):
        children = range(len(nob))
    chars = _nob_print_params(TAB_LENGTH, LINE_LENGTH)
    prefix = _nob_print_prefix(level, TAB_LENGTH, LINE_LENGTH)

    for child_id in children:
        item = "%s%s " % (prefix, chars['standard_child'])
        if child_id == list(children)[-1]:
            item = "%s%s " % (prefix, chars['last_child'])

        if isinstance(child_id, int):
            item = item + 'List item #%d' % child_id
        else:
            item = item + child_id
        child = nob[child_id]
        item = item + ' (%s)' % _get_type_name(child)
        if not isinstance(child, (dict, list)):
            suffix = _nob_print_value(child)
            item = item + ' : ' + suffix
        output_ls.append(item)
        output_ls.extend(asciigraph_rec(child, level=level+1))

    return output_ls


def _nob_print_params(tab_length=2, line_length=40):
    """Defining parameters and special characters
       for nested object printing
    Inputs:
    ----------
    tab_length : Number of spaces describing a tab
    line_length : the maximum allowed number of characters in a line

    Returns :
    ---------
    chars : a dictionnary gathering special characters and max line length
    """
    chars = dict()
    chars['vertical_bar'] = u'\u2503'
    chars['horizontal_bar'] = u'\u2501'
    chars['child_bar'] = u'\u2513'
    chars['standard_child'] = u'\u2523'
    chars['last_child'] = u'\u2517'
    chars['tab'] = ' '*tab_length
    chars['max_length'] = line_length
    return chars


def _nob_print_prefix(level, tab_length=2, line_length=40):
    """Formatting prefix for a line showing a nested object
       item
    Inputs:
    ----------
    level : level of the item in the nested object tree
    tab_length : Number of spaces describing a tab
    line_length : the maximum allowed number of characters in a line

    Returns :
    ---------
    prefix : prefix for a line showing a nested object item
    """
    chars = _nob_print_params(tab_length, line_length)
    prefix = ''
    for i in range(level):
        prefix = prefix + chars['vertical_bar']+(i+1)*chars['tab']
    return prefix


def _nob_print_value(value, tab_length=2, line_length=40):
    """Formatting a nested object item value
    Inputs:
    ----------
    value : The value of the nested object item
    tab_length : Number of spaces describing a tab
    line_length : the maximum allowed number of characters in a line

    Returns :
    ---------
    str_value : A formatted string corresponding to the value
    """
    chars = _nob_print_params(tab_length, line_length)
    str_value = re.sub(' \n', ' ', str(value))
    str_value = re.sub(' +', ' ', str_value).strip()
    if len(str_value) > chars['max_length']:
        str_value = str_value[:chars['max_length']+1]+'...'
    return str_value


def _get_keys(nob):
    """Get nested object items

    Inputs :
    --------
    nob :  a nested object (could be a dict or a list of dicts)

    Outputs :
    --------
    keys_list : a list of the nested object items
    """
    if isinstance(nob, dict):
        keys_list = nob.keys()
    elif isinstance(nob, list):
        keys_list = _get_list_keys(nob)
    else:
        keys_list = []
    return keys_list


def _get_list_keys(nob):
    """Get the keys of the dictionnaries contained
    by a list

    Inputs :
    --------
    nob :  a list of dictionnaries

    Outputs :
    --------
    keys_list : a list of the dictionnaries keys
    """
    keys_list = []
    for item in nob:
        keys_list.extend(list(item.keys()))
    return keys_list


def _get_type_name(variable):
    """ get the type name of a variable

    Inputs:
    -----------
    variable : a python variable/class instance

    Return:
    -------
    type_ : A string describing the type or the class name of the variable
    """
    type_ = variable.__class__.__name__
    return type_
