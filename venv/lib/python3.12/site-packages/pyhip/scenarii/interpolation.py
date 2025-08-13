""" This module contains scripts for full basic scenario with pyhip"""
from pyhip.setup import (runner_setup,
                         read_mesh,
                         transform,
                         set_none_bctype,
                         periodize,
                         write_files,
                         process_end,
                         interpolation,
                         remove_duplicated_bctext)

__all__ = ["interpolation_setup"]


def interpolation_setup(nob_in, fallback, checklevel=True):
    """generic hip interpolation setup scenario"""
    hip_cmds = [runner_setup(fallback, checklevel=checklevel)]

    #meshes reading
    if 'interpolate' in nob_in:
        hip_cmds += read_mesh(nob_in.interpolate, fallback=fallback)
        hip_cmds += remove_duplicated_bctext(nob_in, fallback=fallback)
    else:
        raise RuntimeError("an 'interpolate' block must be specified.")

    if 'read' in nob_in:
        hip_cmds += read_mesh(nob_in.read, fallback=fallback)
    else:
        raise RuntimeError("a 'read' block must be specified")

    #mesh operations
    if 'transform' in nob_in:
        hip_cmds += transform(nob_in.transform, fallback=fallback)

    #mesh periodization
    if 'periodize' in nob_in:
        if 'interpolate' in nob_in:
            hip_cmds += set_none_bctype(nob_in.interpolate.patch_list, fallback=fallback)
        hip_cmds += set_none_bctype(nob_in.read.patch_list, fallback=fallback)
        hip_cmds += periodize(nob_in, nob_in.read.patch_list, fallback=fallback)

    #solution interpolation
    if 'interpolate' in nob_in:
        hip_cmds += interpolation(fallback=fallback)

    #mesh writing
    if 'write' in nob_in:
        hip_cmds += write_files(nob_in.write, fallback=fallback)
    else:
        raise RuntimeError("a 'write' block must be specified")

    hip_cmds += process_end(fallback)

    return hip_cmds
 