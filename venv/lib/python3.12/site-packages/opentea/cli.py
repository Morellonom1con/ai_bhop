#!/usr/bin/env python
"""
cli.py

Command line interface for tools in pyavbp
"""

import click

from opentea.noob.check_schema import (
    nob_check_schema,
    read_serialized_data)



@click.group()

def main_cli():
    """---------------    O P E N T E A  III  --------------------

    You are now using the Command line interface of Opentea 3,
    a Python3 Tkinter GUI engine based on SCHEMA specifications,
    created at CERFACS (https://cerfacs.fr).

    This is a python package currently installed in your python environement.
    See the full documentation at : https://opentea.readthedocs.io/en/latest/.
    """
    pass #pylint: disable=unnecessary-pass

# Testing guis
@click.command()
@click.argument('guiname',
                type=click.Choice(
                    ['trivial', 'simple', 'calculator'],
                    case_sensitive=False))
@click.option("--file", "-f", type=str, default=None, help="Project to open. (.yml)")
def gui(guiname, file):
    """Examples of OpenTEA GUIs"""
    from opentea.examples.trivial.startup import main as guitrivial
    from opentea.examples.simple.startup import main as guisimple
    from opentea.examples.calculator.startup import main as guicomplex


    if guiname == "trivial":
        guitrivial(data_file=file)
    elif guiname == "simple":
        guisimple(data_file=file)
    else:   # guicomplex
        guicomplex(data_file=file)
main_cli.add_command(gui)

#===========================================================

@click.command()
@click.argument("schema_file", type=click.File('r'))
def test_schema(schema_file):
    """Test if a yaml SCHEMA_FILE is valid for an opentea GUI."""
    schema = read_serialized_data(schema_file)
    nob_check_schema(schema)
    click.echo("** Congratulations! **")
    click.echo(
        schema_file
        + " SCHEMA structure is valid\nfor opentea requirements.")


main_cli.add_command(test_schema)

if __name__ == '__main__':
    main_cli()
