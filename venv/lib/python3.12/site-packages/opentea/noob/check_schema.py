"""Test  the structure of a schema.

The first one is for generic schema

The second is for the use of gui_forms"""
import sys
import os
import json
import yaml
from opentea.noob.noob import str_address

class NobSchemaError(Exception):
    """Error due to schema structure"""


def boolean_check_schema(schema):
    """Same as check schema with a boolean output."""
    out = True

    try:
        nob_check_schema(schema)
    except NobSchemaError:
        out = False

    return out


def nob_check_schema(schema):
    """Check if a schema is valid agains opentea requirements

    Input :
    -------
    schema :  a schema object
    """

    list_errors = rec_check_schema(schema, path=None)
    if list_errors:
        raise NobSchemaError("\n".join(list_errors))


def rec_check_schema(schema, path):
    """Recursive inference.

    Input :
    -------
    schema :  a schema object

    Output :
    --------
    list_err : a list of errors encountered
    """
    if path is None:
        path = list()
    out = list()
    if isinstance(schema, dict):
        if "type" not in schema:
            out.append(
                "ERROR :  node "
                + str_address(path)
                + ": -type- is missing")

        elif schema["type"] == "object":
            if "oneOf" in schema:
                out.extend(rec_check_xor(schema, path))
                if "properties" in schema:
                    out.append(
                        "ERROR :  object "
                        + str_address(path)
                        + ": -properties-  cannot be "
                        + "at the same leval as a oneOf")

            else:
                if "properties" not in schema:
                    out.append(
                        "ERROR :  object "
                        + str_address(path)
                        + ": -properties- is missing")
                else:
                    out.extend(rec_check_properties(
                        schema["properties"],
                        path))
        elif schema["type"] in ["string",
                                "integer",
                                "number",
                                "boolean"]:
            out.extend(rec_check_leafs(schema, path))
        elif schema["type"] == "array":
            out.extend(rec_check_array(schema, path))
        else:
            out.append(
                "ERROR :  node "
                + str_address(path)
                + ": dead en in check of dict")
    else:
        out.append(
            "ERROR :  node "
            + str_address(path)
            + ": dead en in check; Non dict input ")

    return out


def rec_check_leafs(schema, path):
    """Recursive check specific to leafs."""
    out = list()
    if "default" not in schema:
        out.append(
            "WARNING :  leaf "
            + str_address(path)
            + ": -default- is missing")
    return out


def rec_check_properties(schema, path):
    """Recursive check specific to properties."""
    out = list()
    for key in schema:
        out.extend(rec_check_schema(schema[key], [*path, key]))
    return out


def rec_check_array(schema, path):
    """Recursive check specific to arrays."""
    out = list()
    if "default" in schema:
        out.append(
            "ERROR :  array "
            + str_address(path)
            + ": -default- not supported yet")
    if "items" not in schema:
        out.append(
            "ERROR :  array "
            + str_address(path)
            + ": -items- is missing")
    if "type" not in schema["items"]:
        out.append(
            "ERROR :  array "
            + str_address(path)
            + ": -type- of items is missing")

    key = "items"
    out.extend(rec_check_schema(schema[key], [*path, key]))
    return out


def rec_check_xor(schema, path):
    """Recursive check specific to exlusive or."""
    out = list()
    if "default" in schema:
        out.append(
            "WARNING :  oneof "
            + str_address(path)
            + ": -default- is not used."
            + " The first item of oneOf is the default")

    for idx, option_schema in enumerate(schema["oneOf"]):
        if "required" not in option_schema:
            out.append(
                "ERROR :  oneof "
                + str_address(path)
                + " item " + str(idx)
                + ": -required- is needed to enable json validation")
        if "properties" not in option_schema:
            out.append(
                "ERROR :  oneof "
                + str_address(path)
                + " item " + str(idx)
                + ": -properties- is needed to enable json validation")
        list_prop = list(option_schema["properties"].keys())

        if list_prop != option_schema["required"]:
            out.append(
                "ERROR :  oneof "
                + str_address(path)
                + " item " + str(idx)
                + " -properties- single item must match -required- list:"
                + str(list_prop) + ".vs." + str(option_schema["required"]))
        if len(list_prop) != 1:
            out.append(
                "ERROR :  oneof "
                + str_address(path)
                + " item " + str(idx)
                + "-properties- must have a single item")
        out.extend(rec_check_schema(option_schema, [*path, idx]))
    return out


def read_serialized_data(fname):
    """read any serialized data file"""
    ext = os.path.splitext(fname)[1]
    with open(fname, "r") as fin:
        if ext in [".yml", ".yaml"]:
            data = yaml.load(fin, Loader=yaml.FullLoader)
        elif ext in [".json"]:
            data = json.load(fin)
        else:
            raise NotImplementedError(
                "Cannot read file with extension :" + ext)
    return data


if __name__ == "__main__":
    nob_check_schema(read_serialized_data(sys.argv[1]))
