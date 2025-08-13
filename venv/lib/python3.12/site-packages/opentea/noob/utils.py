
def remove_keys_from_nested_dict(nested_dict_, keys):
    """Remove keys from nested dict.
    """

    # remove at current level
    for key in keys:
        if key in nested_dict_:
            del nested_dict_[key]

    # remove at nested levels
    for dict_ in nested_dict_.values():
        if type(dict_) is not dict:
            continue

        remove_keys_from_nested_dict(dict_, keys)
