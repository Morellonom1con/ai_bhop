"""Compare two nested objects, and store the result in a circlufy nested structure"""

from ctypes import c_int64  # a mutable int
import tkinter as tk

from nobvisual.utils import load_file,val_as_str,path_as_str,mv_to_dict
from nobvisual.tkinter_circlify import tkcirclify
from nobvisual.helpers import from_nested_struct_to_nobvisual


__all__ = ["nob_compare", "nob_compare_tkinter", "visual_comparefile"]

LEFT_COLOR = "#ffd700"
RIGHT_COLOR = "#005b96"
DIFFER_COLOR = "#d62d20"


def visual_comparefile(path_left:str, path_right:str, start_mainloop:bool=True):
    """Show visually the differences between two serialization file.

    The circular packing is computed using the circlify package.
    The graphical output is done using tkinter.

    """
    noba = load_file(path_left)
    nobb = load_file(path_right)

    title = "Showing file differences"
    title += "\nLeft: " + path_left
    title += "\nRight: " + path_right
    nob_compare_tkinter(noba, nobb, title=title, start_mainloop=start_mainloop)


def nob_compare_tkinter(noba, nobb, title:str=None, start_mainloop:bool=True):
    """Compare two nested objects.

    :params noba: left node
    :params nobb: right node

    :returns nothing:
    Open a tkinter object to show the comparison
    """
    nstruct = nob_compare(noba, nobb)

    circles = from_nested_struct_to_nobvisual(nstruct)

    draw_canvas = tkcirclify(
        circles,
        color="#eeeeee",
        shade=-0.1,
        legend=[
            ("Only in left", LEFT_COLOR),
            ("Only in right", RIGHT_COLOR),
            ("Differ", DIFFER_COLOR),
        ],
        title=title,
    )
    draw_canvas.show_names(level=2)

    if start_mainloop:
        tk.mainloop()


def nob_compare(noba, nobb)-> list:
    """Compare two nested objects.

    :params noba: left node
    :params nobb: right node

    :returns cirlify_nob:

    """
    out = [
        _rec_compare(
            mv_to_dict(noba), mv_to_dict(nobb),
        )
    ]
    return out


def _rec_compare(left, right, item_id=c_int64(-1), path:str=None, ptype="both"):
    """Recursive build"""

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

    type_ = ptype
    void = dict()
    size = 1
    out["children"] = list()

    if isinstance(left, dict) and isinstance(right, dict):

        for key in left:
            if key in right:
                out["children"].append(
                    _rec_compare(left[key], right[key], path=path + [key], ptype=type_,
                                 item_id=item_id),
                )
            else:
                out["children"].append(
                    _rec_compare(left[key], void, path=path + [key], ptype="only_left",
                                 item_id=item_id),
                )
            size += out["children"][-1]["datum"]

        for key in right:
            if key not in left:
                out["children"].append(
                    _rec_compare(
                        void, right[key], path=path + [key], ptype="only_right",
                        item_id=item_id),
                )
                size += out["children"][-1]["datum"]

    else:

        if left != right and left != void and right != void:
            type_ = "differ"
            val_left = val_as_str(left)
            val_right = val_as_str(right)

            out["text"] += f"\n<- {val_left}\n-> {val_right}"
            out["name"] += f"\n<- {val_left}\n-> {val_right}"

        elif left != void:
            val_left = val_as_str(left)
            out["text"] += f'\n:{val_left}'
            out["name"] += f'\n:{val_left}'
        elif right != void:
            val_right = val_as_str(right)
            out["text"] += f'\n:{val_right}'
            out["name"] += f'\n:{val_right}'

    out["datum"] = size

    if type_ == "differ":
        color = DIFFER_COLOR
    elif type_ == "only_left":
        color = LEFT_COLOR
    elif type_ == "only_right":
        color = RIGHT_COLOR
    else:
        color = "default"

    out['color'] = color

    return out
