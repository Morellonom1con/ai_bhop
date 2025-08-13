"""Module to operate a trplie layer of validation"""


import json
from loguru import logger
from opentea.noob.noob import (nob_find,
                               # nob_find_unique,
                               nob_node_exist,
                               nob_get,
                               nob_set,
                               nob_pprint)
from opentea.noob.schema import clean_schema_addresses
from opentea.noob.inferdefault import nob_complete
from opentea.noob.validate_light import validate_light


class OpenteaSchemaError(Exception):
    """Error in OptenTea schema structure"""



def opentea_clean_data(nobj):
    """Check if data is opentea proof"""
    #return nobj

    return rec_validate_opentea_data(nobj)


def rec_validate_opentea_data(nobj):
    """Recursive validation for opeatea structured dict"""
    if isinstance(nobj, dict):
        out = dict()
        for dict_key in nobj:
            out[dict_key] = rec_validate_opentea_data(nobj[dict_key])
    elif isinstance(nobj, list):
        if isinstance(nobj[0], dict):
            out = list()
            existing_names = list()
            for list_item in nobj:
                out_name, clean_item = clean_opentea_list_item(list_item,
                                                               existing_names)
                existing_names.append(out_name)
                out.append(rec_validate_opentea_data(clean_item))
        else:
            out = nobj
    else:
        out = nobj
    return out


def clean_opentea_list_item(item_in, existing_names):
    """Add # to list elements to avoid duplication."""
    out_name = item_in["name"]
    item_out = item_in.copy()
    if out_name == "":
        out_name = "#"

    if out_name in existing_names:
        while out_name in existing_names:
            out_name += "#"

    item_out["name"] = out_name
    return out_name, item_out


def validate_opentea_schema(schema):
    """Check if schema is OpenTEA-proof.

    - named items in arrays of dicts,
    - required optin in xors
    - existif and require defined
    """
    rec_validate_schema(schema)


def rec_validate_schema(schema):
    """Validate if schema is compatible with opetea structure."""
    if isinstance(schema, dict):

        if "type" in schema:
            if schema["type"] == "array":
                validate_array(schema)

        if "oneOf" in schema:
            validate_oneof(schema)

        for child_name in schema:
            rec_validate_schema(schema[child_name])

    elif isinstance(schema, list):
        for child in schema:
            rec_validate_schema(child)
    else:
        pass


def validate_array(schema):
    """Validate an opentea multiple structure."""
    context = "\n\nMULTIPLE ERROR\n"
    context += nob_pprint(schema, max_lvl=4)

    if "items" not in schema:
        context += "\n -items- is missing... "
        raise OpenteaSchemaError(context)

    if "type" not in schema["items"]:
        context += "\n -items/type- is missing... "
        raise OpenteaSchemaError(context)

    if schema["items"]["type"] == "object":
        if "properties" not in schema["items"]:
            context += "\n -items/type- is missing..."
            raise OpenteaSchemaError(context)

        if "name" not in schema["items"]["properties"]:
            context += "\n -items/properties/name- is missing..."
            raise OpenteaSchemaError(context)
    else:
        if schema["items"]["type"] not in ["integer",
                                           "number",
                                           "string"]:
            context += ("Type "
                        + schema["items"]["type"]
                        + " unsupported in arrays")
            raise OpenteaSchemaError(context)


def validate_oneof(schema):
    """Validate an opentea multiple structure."""
    for oneof_item in schema["oneOf"]:
        context = "\n\nXOR ERROR\n"
        context += nob_pprint(oneof_item, max_lvl=4)
        if not isinstance(oneof_item, dict):
            context += "\n oneOf item must be dicts "
            raise OpenteaSchemaError(context)
        if "properties" not in oneof_item:
            context += "\n oneOf item need a -properties- item "
            raise OpenteaSchemaError(context)
        if "required" not in oneof_item:
            context += "\n oneOf item need a -required- list item"
            raise OpenteaSchemaError(context)
        if len(oneof_item["required"]) != 1:
            context += ("\n oneOf item need a -required- "
                        + "list item with a single element ")
            raise OpenteaSchemaError(context)
        if not isinstance(oneof_item["properties"], dict):
            context += "\n oneOf item need a dict in their properties"
            raise OpenteaSchemaError(context)
        if oneof_item["required"] != list(oneof_item["properties"].keys()):
            context += ("\n all oneOf item need matching "
                        + "-required- and -properties keys- ")
            context += ("\n " + str(oneof_item["required"])
                        + " <> "
                        + str(list(oneof_item["properties"].keys())))

            raise OpenteaSchemaError(context)

        real_shape = oneof_item["properties"][oneof_item["required"][0]]
        if not isinstance(real_shape, dict):
            context += "\n all oneOf item must have a dict in the -properties-"
            raise OpenteaSchemaError(context)


class ErrorExistIf(Exception):
    """Errors on exist if elements."""


def opentea_resolve_existif(data, schema):
    """Validate existif dependencies.

    if an item existence depends on the value of one other item

    Parameters :
    -----------
    data : a nested dict to validate
    schema : the schema to validate against (jsonschema grammar)

    Return:
    -------
    data_out : a nested dict to synchronize
    data with require updated"""

    for sch_address in nob_find(schema, "existif"):
        condition = nob_get(schema, *sch_address)
        dat_address = clean_schema_addresses(sch_address[:-1])
        if nob_node_exist(data, *dat_address):
            if not nob_node_exist(data, condition["node"]):
                msgerr = ("node -" + "/".join(dat_address)
                          + "- Existif cannot be solved if a condition node -"
                          + condition["node"]
                          + "- is missing.")
                raise ErrorExistIf(msgerr)

            outcome = None
            value = nob_get(data, condition["node"])
            if isinstance(value, (int, float)):
                if condition["operator"] == "==":
                    outcome = value == condition["value"]
                elif condition["operator"] == ">=":
                    outcome = value >= condition["value"]
                elif condition["operator"] == "<=":
                    outcome = value <= condition["value"]
                elif condition["operator"] == ">":
                    outcome = value > condition["value"]
                elif condition["operator"] == "<":
                    outcome = value < condition["value"]
                elif condition["operator"] != "!=":
                    outcome = value != condition["value"]
                else:
                    raise NotImplementedError(
                        "operator :" + condition["operator"])

            if isinstance(value, (bool)):
                if condition["operator"] == "==":
                    outcome = value is condition["value"]
                elif condition["operator"] == "!=":
                    outcome = value is not condition["value"]
                else:
                    raise NotImplementedError(
                        "operator :" + condition["operator"])

            if outcome is False:
                msgerr = ("node " + "/".join(dat_address)
                          + "do no pass test :"
                          + condition["node"]
                          + str(condition["operator"])
                          + str(condition["value"]))
                raise ErrorExistIf(msgerr)


class ErrorRequire(Exception):
    """Errors on require elements."""


def opentea_resolve_require(data, schema, verbose=False):
    """Validate require dependencies.

    if  children of an item depends of the value of one other item.
    -tgt- is the node to update
    -src- is the information used to update

    Parameters :
    -----------
    data : a nested dict to synchronize
    schema : the schema to validate against (jsonschema grammar)

    Return:
    -------
    data_out : a nested dict to synchronize
    data with require updated"""
    log = "## Resolve require log"
    data_out = data.copy()
    for tgt_schema_address in nob_find(schema, "ot_require"):
        tgt_schema = nob_get(schema, *tgt_schema_address[:-1])
        src_nodename = nob_get(schema, *tgt_schema_address)
        multiple_fill = False
        if tgt_schema["type"] == "array":
            if tgt_schema["items"]["type"] == "object":
                multiple_fill = True

        tgt_data_address = clean_schema_addresses(tgt_schema_address[:-1])
        log += "ot_require : " + "/".join(tgt_schema_address)
        log += "  depending from " + src_nodename + ")\n"
        log += "( Multiple mode :" + str(multiple_fill) + ")\n"
        if nob_node_exist(data, *tgt_data_address):
            log += _require_update_data(data,
                                        src_nodename,
                                        tgt_data_address,
                                        tgt_schema,
                                        data_out,
                                        multiple_fill)

    logger.debug(log)

    return data_out

#pylint: disable=too-many-arguments
def _require_update_data(data,
                         src_nodename,
                         tgt_data_address,
                         tgt_schema,
                         data_out,
                         multiple_fill):
    """Update data due to opentea require.


    Input :
    -------
    data : the nested dict to synchronize
    src_nodename : the name of the source
    tgt_data_address : the address of the target
    tgt_schema : the schema description of the targed
    InOut :
    --------
    data_out : the nested dict synchronized
    NOT STATELESS!
    """
    log = ''

    if not nob_node_exist(data, src_nodename):
        msgerr = ("node children -" + "/".join(tgt_data_address)
                  + "- ot_require cannot be solved if "
                  + "the required node -" + src_nodename
                  + "- is missing.")
        raise ErrorRequire(msgerr)

    data_to_paste = nob_get(data, src_nodename)
    current_names = nob_get(data_out, *tgt_data_address)
    if multiple_fill:
        current_names = list()
        current_data = dict()
        for items in nob_get(data_out, *tgt_data_address):
            current_names.append(items["name"])
            current_data[items["name"]] = items
    else:
        current_names = nob_get(data_out, *tgt_data_address)

    if data_to_paste != current_names:
        log = "Update is needed:\n"
        log += "former data : " + str(current_names) + "\n"
        log += "new data.   : " + str(data_to_paste) + "\n"
        if not multiple_fill:
            nob_set(data_out, data_to_paste, *tgt_data_address)
        else:
            multiple_paste = list()
            for name in data_to_paste:
                if name in current_names:
                    item_content = current_data[name]
                else:
                    item_content = nob_complete(
                        tgt_schema["items"], {"name": name})
                multiple_paste.append(item_content)
            log += "Setting to:\n" + nob_pprint(multiple_paste,
                                                max_lvl=3)
            nob_set(data_out, multiple_paste, *tgt_data_address)

    return log


def main_validate(data, schema):
    """Main validation procedure.

    Parameters:
    -----------
    data : a nested dict to validate
    schema : the schema to validate agains (jsonschema grammar)

    Return:
    -------
    Only exceptions are returned if any problem"""

    validate_light(data, schema)
    #validate_existifs(data, schema)
    #validate_require(data, schema)

if __name__ == "__main__":
    with open("./schema.json", "r") as fin:
        SCH = json.load(fin)
    with open("./test.json", "r") as fin:
        DAT = json.load(fin)

    main_validate(DAT, SCH)
