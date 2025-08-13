#!/usr/bin/env python
"""
cli.py

Command line interface for tools in Arnica
"""


import click
import arnica

# --------------------------------------


def add_version(f):
    """
    Add the version of the tool to the help heading.
    :param f: function to decorate
    :return: decorated function
    """
    doc = f.__doc__
    f.__doc__ = "Package " + arnica.__name__ + " v" + arnica.__version__ + "\n\n" + doc

    return f


@click.group()
@add_version
def main_cli():
    """---------------    ARNICA  --------------------

    You are now using the Command line interface of Arnica,
    a Python3 helper for reactive multispecies computation, created at CERFACS (https://cerfacs.fr).

    This is a python package currently installed in your python environement."""
    pass


@click.command()
@click.argument("filename", nargs=1)
@click.argument("list_vars", nargs=1)
def plotcsv(filename, list_vars):
    """Plot the content of a CSV file

    FILENAME is the path to the csv file
    LIST_VARS is the names of the column to plot separated by ":"
    the first will the the x_axis
    all the others will be the curves on y_axis

    ---

    Example:

    The CSV FILE should look like
    \n
    aa,bb,cc,dd\n
    1,2,3,4,5\n
    1,2,3,4,5\n
    1,2,3,4,5\n
    1,2,3,4,5\n

    The command
    >arnica plotcsv toto.csv aa:bb:cc:dd
    will plot

    bb .vs. aa, cc .vs. aa, dd .vs. aa,


    """
    from arnica.utils.plotcsv import plot_csv_data

    print(plot_csv_data(filename, list_vars.split(":")))


main_cli.add_command(plotcsv)
