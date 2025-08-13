"""Module for the first tab."""
from nob import Nob
import os
from glob import glob
import h5py
from opentea.process_utils import process_tab
from opentea.noob.noob import unique_dict_key
from pyhip.commands.readers import read_mesh_files, infer_mesh_format
from pyhip.commands.writers import (
    write_hdf5,
    dump_wired,
    )
from pyhip.commands.operations import (
    list_periodic,
    set_bctype,
    hip_exit,
    )
from pyhip.gui.parser_utils import (
    get_patch_list_AVBP,
    parse_meshinfo,
    parse_versioninfo,
    )

def process_interpolate(nob_in):
    """
    *Update mesh infos and show them in the IHM like this :*

    :param nob_in: A dict( ) with IHM's parameters

    :returns: A dict( ) with some update infos:

                - *patches_list*
    """
    nob_out = Nob(nob_in.copy())
    fallback = True # nob_out.fallback[:]
    msgs = []

    if unique_dict_key(nob_out.interpolation[:]) == 'No interpolation':
        return nob_out[:]

    # Reading of the mesh
    _ = read_mesh_files([nob_out.src_mesh[:]], "auto", fallback=fallback)
    
    # Récupération du filaire -> moteur 3D
    geo = nob_out.src_mesh[:].split('/')[-1].split('.')[0]
    _, _ = dump_wired(geo, fallback=fallback)
    nob_out.interpolate.geofile.set("./" + geo + ".geo")

    # Exit/Execute hip
    log, _ = hip_exit(fallback=fallback)

    # Setting of patch list
    patch_list = get_patch_list_AVBP(nob_out.src_mesh[:])
    nob_out.interpolate.patch_list_src.set(patch_list)

    if fallback:
        version_info = parse_versioninfo(log)
        mesh_info = parse_meshinfo(log)
        nob_out.interpolate.meshinfo.set(version_info + mesh_info)
    else:
        nob_out.interpolate.meshinfo.set("Mesh info parsing cython not implemented yet.")

    return nob_out[:]

if __name__ == "__main__":
    process_tab(process_interpolate)

