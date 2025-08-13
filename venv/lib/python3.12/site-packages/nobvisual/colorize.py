"""Module to ease the colorization of circles"""

from tol_colors import tol_cmap
from math import log10
import fnmatch
import matplotlib


EPS = 1.0e-10
BIG= 1.0e+10

def reset_colors(nstruc:list, default=None)-> list:
    """[RECURSIVE, INPLACE] Reset all colors to default color, which can be overriden"""
    for child in nstruc:
        try:
            child["color"]=default
            reset_colors(child["children"],default=default)
        except KeyError:
            pass
        
        
def color_by_name_pattern(nstruc:list, patterns:list)-> list:
    """[RECURSIVE, INPLACE] Change colors according to a list of patterns
    against the field "path" of the nested structure
    
    typically, a list of patterns looks like:

    [
       ("*phasefield*", "green"),
       ("*moment*", "red"),
       ("*__init__*", "yellow"),
    ]
    """
    
    for child in nstruc:
        if "path" in child:
            for pattern,color in patterns:
                str_ = "/".join(child["path"])
                if fnmatch.fnmatch( str_, pattern):
                    child["color"]=color
        if "children" in child:
            color_by_name_pattern(child["children"],patterns)



def read_val(data: dict, variable:str, logarithmic:bool=False):
    out = None
    if variable in data:
        out = data[variable]
        if logarithmic:
            out = log10(out)
    return out



def rec_miner(data:list, variable:str, min_val=BIG, logarithmic:bool=False):
    """Get the mi value in a nested object
    
    list of items:
             
    """
    out = min_val
    for item in data:
        try:
            out = min(read_val(item, variable, logarithmic=logarithmic),out )
        except TypeError:
            pass
        if "children" in item:
            out = min(rec_miner(item["children"],variable,  min_val=out), out)
    
    return out

def rec_maxer(data:list, variable:str, max_val=-BIG, logarithmic:bool=False):
    out = max_val
    for item in data:
        try:
            out = max(read_val(item,variable, logarithmic=logarithmic),out )
        except TypeError:
            pass
        if "children" in item:
            out = max(rec_maxer(item["children"], variable, max_val=out), out)
    return out


def color_by_value(nstruc:list, variable:str, invert:bool=False, logarithmic:bool=False, tolcmap='YlOrBr', min_val:float=None, max_val:float=None, leg_levels:int=6)-> list:
    """ Color a nest object by pattern
    
    nstruct starts with a list of items, like being the "children" attribute of root
    Items are as : 

        item : {
                "color" : "white"  | if "None" color is computed, else color is kept
                "children:  [ __ sub_items __]
                 ___ other keys ___
        }
    """
    
    cmap = tol_cmap(tolcmap)

    


    ## Find min value
    if min_val is None:
        min_val = rec_miner(nstruc,variable,logarithmic=logarithmic)
    else:
        if logarithmic:
            max_val=log10(max_val)
    
    # Find max value
    if max_val is None:
        max_val = rec_maxer(nstruc,variable,logarithmic=logarithmic)
    else : 
        if logarithmic:
            max_val=log10(max_val)
    


    def _color_child(child:dict):
        """Assign color to nodes"""
        
        if variable not in child:
            child["color"]=None
        else:
            # val = child[variable]
            # if logarithmic:
            #     val = log10(val)

            try:    
                val = (read_val(child, variable)-min_val)/(max_val-min_val+EPS)
            except TypeError:
                raise RuntimeError(f"Field {variable} not found")
            val = min(max(val,0.), 1.)
            if invert:
                val=1.-val
            child["color"]=matplotlib.colors.to_hex(cmap(val))
        if "children" in child:
            for granson in child["children"]:
                _color_child(granson)
    
    for child in nstruc:
        _color_child(child)

    #Build legend
    legend=[]
    for lvl in range(leg_levels):
        ratio = 1-lvl /(leg_levels-1)
        val = min_val+ratio*(max_val-min_val)
        if logarithmic:
            print(">val<", val)
            val=10**val
        str = f"{val:.2f}"

        if len(str)>4:
            str = str.split(".")[0]
        if invert:
            ratio=1.-ratio
        
        legend.append((str, matplotlib.colors.to_hex(cmap(ratio)) ))
    return legend


