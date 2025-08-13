""" this module contains scripts for basic scenario usage"""
import warnings
from pyhip.commands.readers import read_mesh_files
from pyhip.commands.writers import (write_hdf5,
                                    write_avbp,
                                    write_ensight,
                                    write_cgns,
                                    write_gmsh,
                                    write_fieldview,
                                    dump_wired)
from pyhip.commands.operations import (transform_scale,
                                       transform_rotate,
                                       transform_translate,
                                       duplicate,
                                       generate_mesh_2d_3d,
                                       set_bctype,
                                       set_bctext,
                                       set_checklevel,
                                       hip_exit)

__all__ = ["runner_setup",
           "read_mesh",
           "generate_mesh",
           "transform",
           "set_none_bctype",
           "periodize",
           "write_files",
           "process_end"]


warnings.filterwarnings('always', category=UserWarning)

def runner_setup(fallback, checklevel=True):
    """runnner setup routine"""
    if fallback:
        warnings.warn_explicit("The fallback version of pyhip is being used.",
                               UserWarning,
                               'setup/core.py',
                               34)

    if checklevel:
        return set_checklevel(5, fallback=fallback)

    return set_checklevel(0, fallback=fallback)


def read_mesh(nob_in, **kwargs):
    """method to read different type of meshes in function of the nested object input"""
    return read_mesh_files(nob_in.files[:], nob_in.meshtype[:], **kwargs)

def transform(nob_in, **kwargs):
    """basic transformations"""
    cmds = []
    if nob_in.duplicate.nbr_sector_tgt[:] > 1:
        angle = 360. / nob_in.duplicate.nbr_sector_tot[:]
        cmds.extend(duplicate(nob_in.duplicate.nbr_sector_tgt[:] - 1, angle, **kwargs))

    if nob_in.scale.scaling_factor[:] != [1., 1., 1.]:
        cmds.extend(transform_scale(nob_in.scale.scaling_factor[:][0],
                                    nob_in.scale.scaling_factor[:][1],
                                    nob_in.scale.scaling_factor[:][2],
                                    **kwargs))

    for axis, angle in nob_in.rotate.angles[:].items():
        if angle != 0.:
            cmds.extend(transform_rotate(axis, angle, **kwargs))

    if nob_in.translate.direction[:] != [0., 0., 0.]:
        cmds.extend(transform_translate(nob_in.translate.direction[:][0],
                                        nob_in.translate.direction[:][1],
                                        nob_in.translate.direction[:][2],
                                        **kwargs))

    return cmds

def set_none_bctype(patch_list, **kwargs):
    """method to remove all boundary condition and set type to 'none'"""
    cmds = []
    for patch in patch_list:
        cmds += set_bctype(patch, 'n', **kwargs)

    return cmds

def set_bcname(new_patch_list, patch_list=None, patch_nbr=None, **kwargs):
    """method to set boundary condition names from number or name"""
    cmds = []
    if patch_list is None:
        if patch_nbr is None:
            raise RuntimeError("Either bc name or number must be provided"
                               "to set new patch names")
        for nbr, new_patch in zip(patch_nbr, new_patch_list):
            cmds += set_bctext(nbr, new_patch, **kwargs)
    else:
        for patch, new_patch in zip(patch_list, new_patch_list):
            cmds += set_bctext(patch, new_patch, **kwargs)

    return cmds

def generate_mesh(nob_in, **kwargs):
    """generate 2D or 3D mesh"""

    val = lambda n, v: n[n.find(v)[0]][:] if n.find(v) else None

    cmds = generate_mesh_2d_3d(
        nob_in.lower_corner[:],
        nob_in.upper_corner[:],
        nob_in.resolution[:],
        nob_in.convert2tri[:],
        extru_axis=val(nob_in, 'extru_axis'),
        extru_range=val(nob_in, 'extru_range'),
        extru_res=val(nob_in, 'extru_resolution'),
        **kwargs)

    return cmds

def periodize(nob_in, patch_list, **kwargs):
    """periodization of patch pairs for interpolated mesh with solution"""
    cmds = []
    if len(nob_in.periodize) > 0:
        for idx, pair in enumerate(nob_in.periodize[:]):
            if 'leader' in pair and 'follower' in pair:
                if pair['leader'] not in patch_list:
                    msg = f"Leader patch '{pair['leader']}' not in given patch list :"
                    msg += "\n - "
                    msg += "\n - ".join(patch_list)
                    raise ValueError(msg)
                if pair['follower'] not in patch_list:
                    msg = f"Follower patch '{pair['follower']}' not in given patch list :"
                    msg += "\n - "
                    msg += "\n - ".join(patch_list)
                    raise ValueError(msg)
                cmds += set_bctype(pair['leader'], f'u{idx:02}', **kwargs)
                cmds += set_bctype(pair['follower'], f'l{idx:02}', **kwargs)
            else:
                raise KeyError("A 'leader' and a 'follower' patch must be specified.")
    else:
        warnings.warn_explicit(('All patches will see their kind set to none,'
                                'all periodicities removed.'),
                               UserWarning,
                               'setup/core.py',
                               105)

    return cmds

def write_files(nob_in, **kwargs):
    """method to read different type of meshes in function of the nested object input"""
    # cgns, ensight, fluent, gmsh, hdf5
    msg = ("The only writing file types implemented are 'hdf5', 'avbp', 'ensight', ",
           "'cgns', 'gmsh' and 'fieldview'")
    if nob_in.casetype[:] == 'hdf5':
        cmds = write_hdf5(nob_in.casename[:], **nob_in.options[:], **kwargs)
    elif nob_in.casetype[:] == 'avbp':
        cmds = write_avbp(nob_in.casename[:], **nob_in.options[:], **kwargs)
    elif nob_in.casetype[:] == 'ensight':
        cmds = write_ensight(nob_in.casename[:], **nob_in.options[:], **kwargs)
    elif nob_in.casetype[:] == 'cgns':
        cmds = write_cgns(nob_in.casename[:], **kwargs)
    elif nob_in.casetype[:] == 'gmsh':
        cmds = write_gmsh(nob_in.casename[:], **kwargs)
    elif nob_in.casetype[:] == 'fieldview':
        cmds = write_fieldview(nob_in.casename[:], **kwargs)
    else:
        raise NotImplementedError(msg)

    cmds.extend(dump_wired(nob_in.casename[:], **kwargs))

    return cmds

def process_end(fallback,hipexec=None):
    """simple function for exiting pyhip in scenarii"""
    hip_exit(fallback=fallback,hipexec=hipexec)
    print("\nPyhip successfuly exited.\n")

    return ['exit']
