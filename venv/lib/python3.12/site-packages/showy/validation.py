

from pkg_resources import resource_string

from yaml import safe_load
from jsonschema import validate


def load_schema():
    schema = resource_string('showy',
                             'schema.yaml')

    return safe_load(schema)


def validate_layout(layout):
    return validate(instance=layout, schema=load_schema())
