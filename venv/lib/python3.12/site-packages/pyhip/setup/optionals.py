"""this module is for optional scenarii routines"""
from pyhip.commands.operations import (adapt_with_factor,
                                       set_bctext,
                                       interpolate)

__all__ = ["remove_duplicated_bctext",
           "interpolation",
           "adapt"]


def remove_duplicated_bctext(nob_in, **kwargs):
    """method to remove all boundary condition and set type to 'none'"""
    cmds = []
    for patch in nob_in.read.patch_list[:]:
        if patch in nob_in.interpolate.patch_list[:]:
            cmds += set_bctext(patch, patch + '_src', **kwargs)
    return cmds

def interpolation(**kwargs):
    """interpolate solution on current mesh"""
    return interpolate(grid_id=1, **kwargs)

def adapt(nob_in, **kwargs):
    """method to refine or coarse a mesh"""
    if 'hGrad' not in nob_in.factor:
        print("Default value of hGrad is used: 1.4")
        nob_in.factor['hGrad'] = 1.4
    if 'hmin' not in nob_in.factor:
        print("No minimal distance is targeted")
        nob_in.factor['hmin'] = 0.0

    cmds = adapt_with_factor(nob_in.factor.coefficient[:],
                             hgrad=nob_in.factor.hGrad[:],
                             hmin=nob_in.factor.hmin[:],
                             **kwargs)
    return cmds
