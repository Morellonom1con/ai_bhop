"""Startup script to call calculator gui."""
import os
import inspect
import yaml
from opentea.gui_forms.otinker import main_otinker


def main(data_file=None):
    """Call the otinker gui."""
    schema_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "schema_simple.yaml")
    with open(schema_file, 'r') as fin:
        schema = yaml.load(fin, Loader=yaml.SafeLoader)
    base_dir = inspect.getfile(inspect.currentframe())
    base_dir = os.path.dirname(os.path.abspath(base_dir))
    main_otinker(schema, data_file=data_file, calling_dir=base_dir)

if __name__ == "__main__":
    main()
