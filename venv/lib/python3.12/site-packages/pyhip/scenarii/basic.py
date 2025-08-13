""" This module contains scripts for full basic scenario with pyhip"""
from pyhip.setup import (runner_setup,
                         read_mesh,
                         transform,
                         set_none_bctype,
                         periodize,
                         write_files,
                         process_end)

__all__ = ["basic_setup"]


def basic_setup(nob_in, fallback, checklevel=True):
    """generic hip basic setup scenario"""
    #pylint: disable=duplicate-code
    hip_cmds = [runner_setup(fallback, checklevel=checklevel)]

    #mesh reading
    if 'read' in nob_in:
        hip_cmds += read_mesh(nob_in.read, fallback=fallback)
    else:
        raise RuntimeError("a 'read' block must be specified")

    #mesh operations
    if 'transform' in nob_in:
        hip_cmds += transform(nob_in.transform, fallback=fallback)

    #mesh periodization
    if 'periodize' in nob_in:
        hip_cmds += set_none_bctype(nob_in.patch_list, fallback=fallback)
        hip_cmds += periodize(nob_in, nob_in.patch_list, fallback=fallback)

    #mesh writing
    if 'write' in nob_in:
        hip_cmds += write_files(nob_in.write, fallback=fallback)
    else:
        raise RuntimeError("a 'write' block must be specified")

    hip_cmds += process_end(fallback)

    return hip_cmds
