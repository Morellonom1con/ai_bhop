"""
Convert a Nested object to a circlify object
============================================

"""

from ctypes import c_int64  # a mutable int
import tkinter as tk

from nobvisual.tkinter_circlify import tkcirclify
from nobvisual.utils import load_file,val_as_str,path_as_str,mv_to_dict
from nobvisual.helpers import from_nested_struct_to_nobvisual


__all__ = ["visual_treefile", "visual_treenob"]


class UnknownType:
    pass


TYPE2COLOR = {
    bool: "#00ffff",
    int: "#00ccff",
    float: "#0066aa",
    str: "#00aa00",
    type(None): "#dddddd",
    UnknownType: "#ffcc00",
}

def visual_treefile(path: str, start_mainloop: bool=True, fast:bool=False):
    """Show the circular nested packing of a serialization file.

    The circular packing is computed using the circlify package.
    The graphical output is done using tkinter.
    Area of circles is proportional to lthe log10 of file sizes.
    """
    nob = load_file(path)
    visual_treenob(nob, title=f"Showing {str(path)}", start_mainloop=start_mainloop, fast=fast)


def visual_treenob(nob, title: str="", start_mainloop: bool=True, fast:bool=False):
    """Show the circular nested packing of a nested object.

    The circular packing is computed using the circlify package.
    The graphical output is done using tkinter.
    Area of circles is proportional to lthe log10 of file sizes.
    """
    nstruct = build_nstruct(nob)
    circles = from_nested_struct_to_nobvisual(nstruct, fast=fast)
    draw_canvas = tkcirclify(
        circles,
        color="#eeeeee",
        legend=_get_legend(),
        title=title,
    )
    draw_canvas.show_names(level=2)
    if start_mainloop:
        tk.mainloop()


def _get_legend()-> list:
    """Return the legend of types"""
    types = [
        ('int', int),
        ('float', float),
        ('string', str),
        ('boolean', bool),
        ('None', type(None)),
        ('other', UnknownType)]

    return [(label, TYPE2COLOR[type_]) for label, type_ in types]


def build_nstruct(nob)-> list:
    """Build the nested_structure of a nested object.
    :param nob: a nested object
        for exampke the content of a YAML or JSON file.

    :returns:

    nested dicts of type

    ::

        {
            "id": name,
            "datum": 1.,      # <- tot nb of children
            "children: [ ],   # <- if children list of nesting here
        }

    in a list. This is compatible with the circlify package.
    """
    nobd = mv_to_dict(nob)
    out = [_rec_nstruct(nobd, item_id=c_int64(-1))]
    return out


def _rec_nstruct(in_, item_id=c_int64(-1), path=None)-> dict:
    """[RECURSIVE] building of nstruct"""
    if path is None:
        path = list()
    text = path_as_str(path)

    text_ls = text.split()
    name = text_ls[-1].strip() if len(text_ls) else text

    item_id.value += 1
    out = {
        "id": item_id.value,
        "datum": 1.0,
        "name": name,
        "text": text,
    }

    if isinstance(in_, dict):
        out["datum"] = float(dict_childs(in_))
        out["children"] = list()
        for key in in_:
            out["children"].append(_rec_nstruct(in_[key], item_id=item_id,
                                                path=path + [key]),)
    else:
        value_str = val_as_str(in_)
        out["text"] += f"\n:{value_str}"
        out["name"] += f'\n:{value_str}'
        out['color'] = TYPE2COLOR.get(type(in_), TYPE2COLOR[UnknownType])

    return out

#TODO AD: is this counting very necessary?
def dict_childs(data, childs:int=1)-> int:
    """ compute dictionnary population
    Warning - recursive

    Parameters:
    -----------
    data : the init-dictionnary /or one of its subitems
    childs : integer, the level to return

    Returns:
    --------
    childs : integer, the total number of childs
    """
    if not isinstance(data, dict) or not data:
        return childs

    childs = 0
    for key in data:
        childs += dict_childs(data[key])
    return childs
