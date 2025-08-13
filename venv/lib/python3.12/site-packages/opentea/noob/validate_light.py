"""Lightweight validate from jsonschema"""

import jsonschema
import numpy as np
from opentea.noob.noob import nob_pprint
from loguru import logger

class ValidationErrorShort(Exception):
    """Valodation error"""


__all__ = ["validate_light"]


def validate_light(data, schema):
    """Schema validation procedure.

    Parameters:
    -----------
    data : a nested dict to validate
    schema : the schema to validate against (jsonschema grammar)

    Return:
    -------
    THIS IS NOT A BOOLEAN
    Only exceptions are returned if any problem, else none."""

    # too verbose to be of use
    # jsonschema.validate(data, schema)
    err_msg = validate_base(data, schema)
    if err_msg:
        raise ValidationErrorShort(err_msg)


def build_err_msg(data, schema, path, custom_msg=None, tag=None):
    """Build the error mesagges"""
    err_msg = ""
    err_msg += "\n" + "=====ValidateLightError======="
    if custom_msg is None:
        err_msg += "\n" + nob_pprint(data)
        err_msg += "\n" + " does not validate against "
        err_msg += "\n" + nob_pprint(schema, max_lvl=4)
    else:
        err_msg += "\n" + custom_msg

    err_msg += "\n" + " at " + path

    if tag is not None:
        err_msg += "(" + tag + ")"

    return err_msg


def validate_base(data, schema, path=""):
    """Validate in the default case ."""
    # validator = jsonschema.Draft4Validator(schema)
    # errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    # print("VB"+path)
    # out = None
    # if errors:
    #     for error in errors:

    if "oneOf" in schema:
        out = test_oneof(data, schema, path=path)
    elif "items" in schema:
        out = test_array(data, schema, path=path)

    elif "properties" in schema:
        out = test_properties(data, schema, path=path)

    else:
        # Default case
        out = test_default(data, schema, path=path)
    return out


def test_default(data, schema, path=""):
    """Remove all unsignifican errors"""
    out = None

    if isinstance(data, np.ndarray):
        data = data[0]

    validator = jsonschema.Draft4Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)

    if errors:
        msg = str(errors[0])
        out = build_err_msg(data, schema, path, custom_msg=msg, tag="default")
    return out


def test_array(data, schema, path=""):
    """Remove all unsignifican errors"""
    out = None
    for i, item in enumerate(data):
        out = validate_base(item, schema["items"], path=path + str(i))
        if out is not None:
            return out
    return out


def test_properties(data, schema, path=""):
    """Remove all unsignificant errors"""
    out = None
    if data is None:
        if "void" in schema["properties"]:
            pass
        else:
            out = path, ": None values only allowed for void elements"
    elif data == {"void": None}:
        pass
    else:
        list_keys = list(data.keys())
        allowed_props = list(schema["properties"].keys())
        for key in list_keys:
            if key in allowed_props:
                out = validate_base(
                    data[key], schema["properties"][key], path=path + "." + key
                )
                if out is not None:
                    logger.warning(f"Props not validated {path}, {(path + '.' + key)}")
                    return out

        if "additionalProperties" in schema:
            if schema["additionalProperties"] is False:
                list_keys = list(data.keys())
                allowed_props = list(schema["properties"].keys())
                for key in allowed_props:
                    if key in list_keys:
                        list_keys.remove(key)
                if list_keys:
                    custom_msg = "\n" + nob_pprint(data)
                    custom_msg += (
                        "\n"
                        + " does not validate because of the "
                        + "following forbidden additionnal properties:"
                    )
                    custom_msg += "\n" + "\n".join(list_keys)
                    out = build_err_msg(
                        data,
                        schema,
                        path,
                        custom_msg=custom_msg,
                        tag="add_prop",
                    )

    return out


def test_oneof(data, schema, path=""):
    """Validate in the case on an opentea oneOf schema."""
    out = None
    if data is None:
        msg = "oneOf : instance is void"
        out = build_err_msg(
            data, schema, path, custom_msg=msg, tag="oneOf: void"
        )
    else:
        l_keys = list(data.keys())
        if len(l_keys) > 1:
            msg = "oneOf : Several child found"
            out = build_err_msg(
                data, schema, path, custom_msg=msg, tag="oneOf: toomany"
            )
        else:
            item = l_keys[0]
            oneof_list = list()
            npath = path + "." + item
            for option in schema["oneOf"]:
                oneof_list.append(option["required"][0])
            if item not in oneof_list:
                srtl = " ".join(oneof_list)
                msg = f"oneOf : {item} not in {srtl}"
                out = build_err_msg(
                    data, schema, npath, custom_msg=msg, tag="oneOf: unknown"
                )
            else:
                idx = oneof_list.index(item)

                out = validate_base(
                    data[item],
                    schema["oneOf"][idx]["properties"][item],
                    path=npath,
                )

    return out
