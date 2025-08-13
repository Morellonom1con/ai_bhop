
from collections import namedtuple

import yaml


def print_dict(dict_):
    print(yaml.dump(dict_, default_flow_style=False))


def unwrap_dict(dict_, sep='/'):
    new_dict = {}
    _unwrap_item(dict_, '', new_dict, sep=sep)

    return new_dict


def wrap_dict(unwrapped_dict, sep='/', operate_on_value=None):
    new_keys = _group_keys(unwrapped_dict.keys(), sep=sep)
    return _wrap_item(unwrapped_dict, new_keys, sep=sep,
                      operate_on_value=operate_on_value)


def _unwrap_item(dict_, parent_key, new_dict, sep='/'):

    for key, value in dict_.items():
        if parent_key:
            new_key = f'{parent_key}{sep}{key}'
        else:
            new_key = key

        if type(value) is dict:
            _unwrap_item(value, new_key, new_dict,
                         sep=sep)
        else:
            new_dict[new_key] = value


def _wrap_item(unwrapped_dict, new_keys, sep='/', operate_on_value=None):
    new_dict = {}

    for key, key_value in new_keys.items():
        if type(key_value) is dict:
            value = _wrap_item(unwrapped_dict, key_value, sep=sep,
                               operate_on_value=operate_on_value)
        else:
            value = unwrapped_dict[key_value]

            if callable(operate_on_value):
                value = operate_on_value(value)

        new_dict[key] = value

    return new_dict


def _group_keys_one_level(group, sep='/'):
    groups = {}

    for key, value in group.items():
        key_split = key.split(sep)

        if len(key_split) == 1:
            groups[key] = value
            continue

        start_key = f'{sep}'.join(key_split[:1])
        end_key = f'{sep}'.join(key_split[1:])

        if start_key not in groups:
            groups[start_key] = {}

        groups[start_key][end_key] = value

    return groups


def _group_keys_recursively(group, sep='/'):
    nested_group = _group_keys_one_level(group, sep=sep)

    new_group = {}
    for key, group_ in nested_group.items():
        if type(group_) is dict:
            new_group[key] = _group_keys_recursively(group_, sep=sep)
        else:
            new_group[key] = group_

    return new_group


def _group_keys(keys, sep='/'):

    group = {key: key for key in keys}
    groups = _group_keys_recursively(group, sep=sep)

    return groups


def _merge_item(dict_, stats):
    # TODO: pass as_dict
    # TODO: if not stats, no merging
    new_dict = {}

    for key, value in dict_.items():
        if type(value) is dict:
            value_ = _merge_item(value, stats)
        else:

            if isinstance(value, tuple):
                value_ = value._asdict()

        new_dict[key] = value_

    return new_dict


def merge_dicts(dict_left, dict_right, name_left='left_value',
                name_right='right_value', as_dict=False):
    """Merges structure dict with stats dict.

    Assumes last element is namedtuple or value. If namedtuple, `name_left` or
    `name_right` are ignored.

    If `as_dict`, then merge is `dict` instead of `namedtuple`.
    """
    unwrapped_left = unwrap_dict(dict_left)
    unwrapped_right = unwrap_dict(dict_right)

    new_dict = {}

    # act on left
    for key, left_value in unwrapped_left.items():

        if isinstance(left_value, tuple):
            value = left_value._asdict()
        else:
            value = {name_left: left_value}

        right_value = unwrapped_right.get(key, None)
        if right_value is not None:
            if isinstance(right_value, tuple):
                value.update(right_value._asdict())
            else:
                value.update({name_right: right_value})

        new_dict[key] = get_namedtuple_from_dict(value)

    # act on right
    for key, right_value in unwrapped_right.items():
        if key in unwrapped_left:
            continue

        if isinstance(right_value, tuple):
            value = right_value._asdict()
        else:
            value = {name_right: right_value}

        new_dict[key] = get_namedtuple_from_dict(value)

    # wrap again
    operate_on_value = get_dict_from_namedtuple if as_dict else None
    return wrap_dict(new_dict, operate_on_value=operate_on_value)


def get_namedtuple_from_dict(dict_, name='Container'):
    NamedTuple = namedtuple(name, dict_)
    return NamedTuple(**dict_)


def get_dict_from_namedtuple(namedtuple_):
    return namedtuple_._asdict()
