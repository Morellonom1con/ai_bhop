"""Module for the first tab."""

from opentea.process_utils import process_tab


def custom_fun(nob):
    """Update the result."""

    operation = nob["first_tab"]["first_block"]["operand"]
    nb1 = nob["first_tab"]["first_block"]["number_1"]
    nb2 = nob["first_tab"]["first_block"]["number_2"]

    res = None
    if operation == "+":
        res = nb1 + nb2
    elif operation == "-":
        res = nb1 - nb2
    elif operation == "*":
        res = nb1 * nb2
    elif operation == "/":
        res = nb1 / nb2
    else:
        res = None

    nob["first_tab"]["first_block"]["result"] = res
    return nob


if __name__ == "__main__":
    process_tab(custom_fun)
