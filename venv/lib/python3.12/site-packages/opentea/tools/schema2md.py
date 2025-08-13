"""Translate a schema nested object into Markdown table."""


import yaml



def md_table_header():
    """The regular table header"""
    out = str()
    out += "| Key | Types | description | default|\n"
    out += "| --- | ----- | ----------- | -------|\n"
    return out


def html_list(list_):
    """Represent a list with html markup."""
    out = "<ul>"
    for item in list_:
        out += "<li>" + str(item) + "</li>"
    out += "</ul>"
    return out


def type_color(type_str):
    """Redirect according to types."""
    out = str(type_str)
    if type_str in ["string"]:
        out = html_colored(type_str, color="orange")
    if type_str in ["integer"]:
        out = html_colored(type_str, color="green")
    if type_str in ["boolean"]:
        out = html_colored(type_str, color="purple")
    if type_str in ["number"]:
        out = html_colored(type_str, color="blue")
    if type_str in ["array"]:
        out = html_colored(type_str, color="cyan")
    return out

def html_colored(str_, color="black"):
    """Represent a string with html color markup."""
    out = '<span style="color:' + color + '">' + str_ + '</span>'
    return out


def to_table_line(nob, path):
    """Reccursive cfn to create tables."""
    print(".    Parsing -", path, "-")
    out = "|" + path.replace("_", r"\_") + " | "

    if "type" in nob.keys():
        out += type_color(nob["type"])

        if nob["type"] == "object":
            out += " --- | --- | --- |\n"
            out = ""
            for key in nob["properties"].keys():
                out += to_table_line(
                    nob["properties"][key],
                    path + "/" + key)
            if "oneOf" in nob:
                out += xor2md(nob, path)
            else:
                raise RuntimeWarning("Unexpected branch object at" + path)

        if nob["type"] == "array":
            out += array2md(nob, path)

        if "enum" in nob.keys():
            out += html_list(nob["enum"])

        # explanation
        out += " | "
        if "title" in nob.keys():
            out += nob["title"] + " "

        if "description" in nob.keys():
            out += nob["description"]
        out += " | "

        # default
        if "default" in nob.keys():
            out += str(nob["default"])
        out += " | \n"
    else:
        print("Skipping at", path)
#        else:
#            raise RuntimeWarning("Unexpected branch without type at" + path)
    return out


def array2md(nob, path):
    """Print an array item into markdown"""
    out = str()
    array_type = nob["items"]["type"]
    if array_type == "object":
        out += (" array | list of items with a structure"
                + " detailed in the following table |")

        out += "\n"
        out += "## List of objects for " + path
        out += "\n"
        out += md_table_header()

        for key in nob["items"]["properties"].keys():
            out += to_table_line(
                nob["items"]["properties"][key],
                path + "/" + key)
    else:
        out += " of " + type_color(array_type)
    return out


def xor2md(nob, path):
    """Print a eXclusive Or item into markdown"""
    out = str()
    out += (" oneOf (XOR) | must match one of the"
            + " options detailed in the following table |")

    out += "\n"
    out += "## eXclusive Or for " + path
    out += "\n"

    for option in nob["oneOf"]:
        opt_name = option["required"][0]
        out += "\n"
        out += "### option " + opt_name
        out += "\n"
        out += md_table_header()

        opt_path = "./" + opt_name

        for key in option["properties"][opt_name]["properties"]:
            out += to_table_line(
                option["properties"][opt_name]["properties"][key],
                opt_path + "/" + key)
    return out


def schema2md(schema):
    """Convert schma type nest objet into Markdown"""

    out = str()

    for key in schema.keys():
        if key.startswith("$"):
            print("skipping ", key)
        elif key == "description":
            out += schema[key] + "\n\n"
        elif key in ["type"]:
            pass
        elif key in ["properties"]:
            for p_key in schema[key]:
                out += "\n# " + p_key + "\n\n"
                out += md_table_header()
                out += to_table_line(schema[key][p_key], p_key)

    return out

if __name__ == "__main__":
    with open("test.yml", "r") as fin:
        SCHEMA = yaml.load(fin, Loader=yaml.FullLoader)

    MD_ = schema2md(SCHEMA["properties"])
    with open("test.md", "w") as fout:
        fout.write(MD_)
    