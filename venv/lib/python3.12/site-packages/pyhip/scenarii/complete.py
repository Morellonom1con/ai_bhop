""" This module contains scripts for full basic scenario with pyhip"""
from pyhip.setup import (runner_setup,
                         read_mesh,
                         generate_mesh,
                         transform,
                         set_none_bctype,
                         periodize,
                         write_files,
                         process_end,
                         adapt,
                         interpolation,
                         remove_duplicated_bctext)
from pyhip.commands import generate_mesh_idcard
from opentea.noob.noob import unique_dict_key
#from reportlab.lib.pagesizes import A4
LANDSCAPE = (960., 480.)

__all__ = ["complete_setup"]

def complete_setup(nob_in, fallback=True, checklevel=True):
    """generic hip complete setup scenario, the nested object in input must be of the Nob class"""
    hip_cmds = runner_setup(fallback, checklevel=checklevel)

    #meshes reading
    if 'interpolate' in nob_in:
        hip_cmds += read_mesh(nob_in.interpolate, fallback=fallback)
        hip_cmds += remove_duplicated_bctext(nob_in, fallback=fallback)

    if 'read' in nob_in:
        if 'generate' in nob_in:
            raise RuntimeError("Can't 'read' and 'generate'"
                               "blocks be both specified")
        hip_cmds += read_mesh(nob_in.read, fallback=fallback)
        patch_list = nob_in.read.patch_list[:]
    elif 'generate' in nob_in:
        hip_cmds += generate_mesh(nob_in.generate, fallback=fallback)
        patch_list = nob_in.generate.patch_list[:]
    else:
        raise RuntimeError("A 'read' or 'generate' block must be specified")

    #mesh operations
    if 'transform' in nob_in:
        hip_cmds += transform(nob_in.transform, fallback=fallback)

    #mesh periodization
    if 'periodize' in nob_in:
        if 'interpolate' in nob_in:
            hip_cmds += set_none_bctype(nob_in.interpolate.patch_list[:], fallback=fallback)
        hip_cmds += set_none_bctype(patch_list, fallback=fallback)
        hip_cmds += periodize(nob_in, patch_list, fallback=fallback)

    #solution interpolation
    if 'interpolate' in nob_in:
        hip_cmds += interpolation(fallback=fallback)

    #mesh adaptations
    if 'adapt' in nob_in:
        hip_cmds += adapt(nob_in.adapt, fallback=fallback)

    #mesh writing
    if 'write' in nob_in:
        hip_cmds += write_files(nob_in.write, fallback=fallback)
    else:
        raise RuntimeError("a 'write' block must be specified")

    if unique_dict_key(nob_in.hip_exec[:]) == "hipexec":
        hip_exec = nob_in.hipexec_path[:]
        hip_cmds += process_end(fallback, hipexec=hip_exec)
    else:
        hip_cmds += process_end(fallback)

    if nob_in.write.casetype[:] == 'hdf5':
        generate_mesh_idcard(
            nob_in.write.casename[:] + '.mesh.h5',
            LANDSCAPE)

    return hip_cmds
