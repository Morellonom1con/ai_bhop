
import os
import json
import yaml
import f90nml
from typing import Tuple
from matplotlib import colors 
# from http://personal.sron.nl/~pault/ (bright color scheme))
PAULT_BRIGHT = [
    "#4477AA",
    "#66CCEE",
    "#228833",
    "#CCBB44",
    "#EE6677",
    "#AA3377",
]


def load_file(path: str):
    """ Load a nob from a serialization file
    :params path: path to a file
    """

    ext = os.path.splitext(path)[-1]
    if ext in [".yaml", ".yml"]:
        with open(path, "r") as fin:
            nob = yaml.load(fin, Loader=yaml.SafeLoader)
    elif ext in [".json"]:
        with open(path, "r") as fin:
            nob = json.load(fin)
    elif ext in [".nml"]:
        nob = f90nml.read(path)
    else:
        raise RuntimeError("Format not supported")
    return nob


def mv_to_dict(data)-> dict:
    """ Convert a multiple value data into a dict

    Parameters:
    -----------
    data : a multiple valued data , usually list or dict

    Returns:
    --------
    dictdata :  same data as a disctionnary
    """
    if isinstance(data, list):
        n_data = dict()
        for i, item in enumerate(data):
            n_data[str(i)] = item
        data = n_data
    if isinstance(data, dict):
        for key in data:
            data[key] = mv_to_dict(data[key])
    return data


def val_as_str(data, max_=30)-> str:
    """return a short_string for any value"""
    if isinstance(data, dict):
        out = ""
    else:
        val_ = str(data)
        if len(val_) > max_:
            val_ = val_[:10] + " (...) " + val_[-max_ + 10:]
        out = val_
    return out


def path_as_str(path)-> str:
    """represent path as string"""
    if len(path) == 0:
        return ""

    sep = "  "
    out = f'- {path[0]}'
    for i, item in enumerate(path[1:]):
        level = i + 1
        out += f"\n{sep*level}- {item}"

    return out

# COLORS


def shade_color(color:str, adjust: float)-> str:
    """ alter a color to ligther or darker tone
    Parameters :
    ------------
    color : tuple (r,g,b) from (0,0,0) to (255,255,255)
    adjust : float from -1 (darkest) to 1 (brightest)

    Returns:
    --------
    shaded : tuple of the color, shaded
    """
    colorrgb = colors.to_rgb(color)
    shaded = []
    for col in colorrgb:
        if adjust > 0:
            out = 1 * adjust + float(col) * (1 - adjust)
        else:
            out = float(col) * (1 - abs(adjust))
        shaded.append(out)
    colorout = colors.to_hex(shaded)
    return colorout


# def rgb2hex(list_rgb:Tuple[int,int,int] )-> str:
#     """Convert rgb list to hex"""
#     return "#{:02x}{:02x}{:02x}".format(
#         int(list_rgb[0]), int(list_rgb[1]), int(list_rgb[2])
#     )


# def hex2rgb(str_rgb:str )-> Tuple[int,int,int]:
#     """Convert hexadecimal color to rgb"""
#     try:
#         rgb = str_rgb[1:]

#         if len(rgb) == 6:
#             red, grn, blu = rgb[0:2], rgb[2:4], rgb[4:6]
#         elif len(rgb) == 9:
#             red, grn, blu = rgb[0:3], rgb[3:6], rgb[6:9]
#         elif len(rgb) == 3:
#             red, grn, blu = rgb[0] * 2, rgb[1] * 2, rgb[2] * 2
#         else:
#             raise ValueError()
#     except:  # TODO: what kind of error can be raised?
#         raise ValueError("Invalid value %r provided for rgb color." % str_rgb)

#     return tuple(int(val, 16) for val in (red, grn, blu))


def get_hex_from_color_scale(alpha: float)-> str:
    """return hex color acording to color scale
    [0 green to 1 red]
    """
    alpha = min(alpha, 1.)
    alpha = max(alpha, 0.)

    ncolors = len(PAULT_BRIGHT)
    delta = 1. / (ncolors - 1.)
    color = PAULT_BRIGHT[int(alpha / delta)]
    return color
