
def fill_dict_empty_nested(dict_, keys_to_add, nested_key=None):
    """Function to add nested levels in a dictionary.

    Empty nested dictionaries are added for given keywords to be addded.

    Input:
        dict_: dictionary to which nested levels will be added
        nested_key_: (optional) type string, sets the dictionary level
                    to which to add keys (default = None)
       keys_to_add_ = keyword entries to add to dictionary.
    Output:
        None, modifies input dict_
    """
    # TODO: delete?

    if nested_key is None:
        for item in keys_to_add:
            if not dict_ == '':
                dict_[item] = {}
            else:
                dict_ = {item: ""}
    else:
        for item in keys_to_add:
            if not dict_[nested_key] == '':
                dict_[nested_key][item] = {}
            else:
                dict_[nested_key] = {item: ""}
