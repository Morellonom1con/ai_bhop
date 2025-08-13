#!/usr/bin/env python
"""
cli.py

Command line interface for tools in pyhip
"""

import click

FORMATS = ('hdf5', 'fluent', 'centaur', 'gmsh', 'cgns', 'ensight')

@click.group()
def main_cli():
    """---------------  PYHIP  --------------------

You are now using the Command line interface of PYHIP,
a Python3 helper to interact with the mesh management tool HIP
Pyhip was created at CERFACS (https://cerfacs.fr).

This is a python package currently installed in your python environement.
"""
    pass

@click.command()
@click.option("--file", "-f", type=str, default=None, help="Project to open. (.yml)")
def gui(file):
    """ Launch PyHIP IHM """

    import pyhip.gui.startup as gui_startup
    gui_startup.main(data_file=file)
main_cli.add_command(gui)

@click.command()
@click.argument('meshfile', type=click.Path(exists=True), nargs=1)
@click.option(
    '-f', '--format',
    type=click.Choice(['portrait', 'landscape'], case_sensitive=False),
    default='landscape')
def meshid(meshfile, format):
    """ Generate mesh ID Card """

    from pyhip.commands import generate_mesh_idcard
    format_dict = {
        'portrait': (595.2755905511812, 841.8897637795277),
        'landscape': (960., 480.)
    }
    pagesize = format_dict[format.lower()]
    generate_mesh_idcard(meshfile, pagesize)

main_cli.add_command(meshid)

@click.command()
@click.argument('meshfile', type=click.Path(exists=True), nargs=1)
def info(meshfile):
    """ Display mesh informations """

    from pyhip.commands.mesh_idcard import extract_hdf_meshinfo, arrange_meshinfo
    meshinfo_dict, _, _, patch_labels = extract_hdf_meshinfo(meshfile)
    meshinfo_dict['Patch labels'] = {str(i+1): patch for i, patch in enumerate(patch_labels)}
    meshinfo_as_lines = arrange_meshinfo(meshinfo_dict)
    
    print('\n'.join(meshinfo_as_lines))
main_cli.add_command(info)

@click.command()
@click.argument('mesh_source', type=click.Path(exists=True), nargs=1)
@click.argument('solution_source', type=click.Path(exists=True), nargs=1)
@click.argument('mesh_target', type=click.Path(exists=True), nargs=1)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def interpolate(mesh_source, solution_source, mesh_target, checklevel, fallback):
    """ Interpolate a source solution from 
        a source mesh to a target_mesh """
    from pathlib import Path
    from pyhip.commands.readers import read_mesh_files
    from pyhip.commands.operations import interpolate, set_checklevel
    from pyhip.commands.writers import write_hdf5
    from pyhip.setup import process_end

    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([mesh_source, solution_source], 'hdf5', fallback=fallback)
    read_mesh_files([mesh_target], 'hdf5', fallback=fallback)
    interpolate(grid_id=1, fallback=fallback)
    write_hdf5(
       Path(mesh_target).name.split('.')[0],
       only_solution=True,
       fallback=fallback,
    )

    process_end(fallback=fallback)
main_cli.add_command(interpolate)

@click.command()
@click.argument('mesh_source', type=click.Path(exists=True), nargs=1)
@click.argument('solution_source', type=click.Path(exists=True), nargs=1)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def check_perio(mesh_source, solution_source, checklevel, fallback):
    """
    perform a check perio if solution provided and mesh is periodc """
    from pathlib import Path
    from pyhip.commands.readers import read_mesh_files
    from pyhip.commands.operations import set_checklevel
    from pyhip.commands.writers import write_hdf5
    from pyhip.setup import process_end

    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([mesh_source, solution_source], 'hdf5', fallback=fallback)
    write_hdf5(
       Path(mesh_source).name.split('.')[0],
       only_solution=True,
       fallback=fallback,
    )
    process_end(fallback=fallback)
main_cli.add_command(check_perio)



@click.command()
@click.argument('mesh_source', type=click.Path(exists=True), nargs=1)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def refresh(mesh_source, checklevel, fallback):
    """
    read and rewrite mesh to refresh avbp hdf mesh format """
    from pathlib import Path
    from pyhip.commands.readers import read_mesh_files
    from pyhip.commands.operations import set_checklevel
    from pyhip.commands.writers import write_hdf5
    from pyhip.setup import process_end

    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([mesh_source], 'hdf5', fallback=fallback)
    write_hdf5('refreshed',fallback=fallback)
    process_end(fallback=fallback)
main_cli.add_command(refresh)

@click.command()
@click.argument('mesh_source', type=click.Path(exists=True), nargs=1)
@click.argument('upper', type=str, nargs=1)
@click.argument('lower', type=str, nargs=1)
@click.argument('pair_nb', type=int, nargs=1)
@click.option("--remove", is_flag=True, help="Remove the periodic pair")
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def periodic_pair(mesh_source, upper, lower, pair_nb, checklevel, fallback, remove):
    """
    Set up a periodic pair with  patch UPPER as 'u' and patch name LOWER as 'l'.
    Axi periodicities are automatically detected.
    The PAIR_NB identify the periodic pair, and should start with 0.
    
    If REMOVE flag is used, this will remove the periodic pair
    """
    from pathlib import Path
    from pyhip.commands.readers import read_mesh_files
    from pyhip.commands.operations import set_checklevel, set_bctype
    from pyhip.commands.writers import write_hdf5
    from pyhip.setup import process_end

    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([mesh_source], 'hdf5', fallback=fallback)
    if remove:
        set_bctype(upper, 'n')
        set_bctype(lower, 'n')
    else:    
        set_bctype(upper, 'u '+str(pair_nb))
        set_bctype(lower, 'l '+str(pair_nb))
    write_hdf5(
       Path(mesh_source).name.split('.')[0],
       only_solution=False,
       fallback=fallback,
    )
    process_end(fallback=fallback)
main_cli.add_command(periodic_pair)



@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('lower_corner', type=float, nargs=2)
@click.argument('upper_corner', type=float, nargs=2)
@click.argument('resolution', type=int, nargs=2)
@click.option('--tri', is_flag=True)
@click.option('--fallback/--nofallback', default=True)
def generate2d(lower_corner, upper_corner, resolution, tri, fallback):
    """ Generate 2d mesh """

    from pyhip.commands import (
        generate_mesh_2d_3d,
        write_hdf5,
    )
    from pyhip.setup import process_end

    generate_mesh_2d_3d(
        lower_corner,
        upper_corner,
        resolution,
        convert2tri=tri,
        fallback=fallback,
    )
    write_hdf5('mesh_2d', fallback=fallback)

    process_end(fallback=fallback)
main_cli.add_command(generate2d)

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('lower_corner', type=float, nargs=2)
@click.argument('upper_corner', type=float, nargs=2)
@click.argument('resolution', type=int, nargs=2)
@click.argument('extru_axis', type=click.Choice(['x', 'y', 'z', 'axi']), nargs=1)
@click.argument('extru_range', type=float, nargs=2)
@click.argument('extru_res', type=int, nargs=1)
@click.option('--tetra', is_flag=True)
@click.option('--fallback/--nofallback', default=True)
def generate3d(lower_corner, upper_corner, resolution, extru_axis,
               extru_range, extru_res, tetra, fallback):
    """ Generate 3d mesh """

    from pyhip.commands import (
        generate_mesh_2d_3d,
        write_hdf5,
    )
    from pyhip.setup import process_end

    generate_mesh_2d_3d(
        lower_corner,
        upper_corner,
        resolution,
        extru_axis=extru_axis,
        extru_range=extru_range,
        extru_res=extru_res,
        convert2tri=tetra,
        fallback=fallback,
    )
    write_hdf5('mesh_3d', fallback=fallback)

    process_end(fallback=fallback)
main_cli.add_command(generate3d)

@click.command()
@click.argument('meshfile', type=click.Path(exists=True), nargs=1)
@click.argument('factor', type=float, nargs=3)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def scale(meshfile, factor, checklevel, fallback):
    """ Scale geometry by scaling coefficients vector """

    from pathlib import Path
    from pyhip.commands import (
        set_checklevel,
        read_mesh_files,
        transform_scale,
        write_hdf5,
    )
    from pyhip.setup import process_end

    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([meshfile], 'hdf5', fallback=fallback)
    transform_scale(*factor, fallback=fallback)
    write_hdf5(Path(meshfile).name.split('.')[0] + '_scaled', fallback=fallback)

    process_end(fallback=fallback)
main_cli.add_command(scale)

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('meshfile', type=click.Path(exists=True), nargs=1)
@click.argument('axis', type=click.Choice(('x', 'y', 'z'), case_sensitive=False), nargs=1)
@click.argument('angle', type=float, nargs=1)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def rotate(meshfile, axis, angle, checklevel, fallback):
    """ Rotate geometry of angle [deg] around axis """

    from pathlib import Path
    from pyhip.commands import (
        set_checklevel,
        read_mesh_files,
        transform_rotate,
        write_hdf5,
    )
    from pyhip.setup import process_end

    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([meshfile], 'hdf5', fallback=fallback)
    transform_rotate(axis, angle, fallback=fallback)
    write_hdf5(Path(meshfile).name.split('.')[0] + '_rotated', fallback=fallback)

    process_end(fallback=fallback)
main_cli.add_command(rotate)

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('meshfile', type=click.Path(exists=True), nargs=1)
@click.argument('direction', type=float, nargs=3)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def translate(meshfile, direction, checklevel, fallback):
    """ Translate geometry by direction vector """

    from pathlib import Path
    from pyhip.commands import (
        set_checklevel,
        read_mesh_files,
        transform_translate,
        write_hdf5,
    )
    from pyhip.setup import process_end

    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([meshfile], 'hdf5', fallback=fallback)
    transform_translate(*direction, fallback=fallback)
    write_hdf5(Path(meshfile).name.split('.')[0] + '_translated', fallback=fallback)

    process_end(fallback=fallback)
main_cli.add_command(translate)

@click.command()
@click.argument('meshfile', type=click.Path(exists=True), nargs=1)
@click.argument('format_in', type=click.Choice(FORMATS, case_sensitive=False), nargs=1)
@click.argument('format_out', type=click.Choice(FORMATS, case_sensitive=False), nargs=1)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def convert(meshfile, format_in, format_out, checklevel, fallback):
    """ Convert a meshfile into specific format mesh """

    from pathlib import Path
    from pyhip.commands import (
        set_checklevel,
        read_mesh_files,
        write_hdf5,
    )
    from pyhip.setup import process_end

    if format_out.lower() != 'hdf5':
        raise NotImplementedError(f"Format '{format_out} not implemented yet.")

    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([meshfile], format_in.lower(), fallback=fallback)
    if format_out.lower() == 'hdf5':
        write_hdf5(Path(meshfile).name.split('.')[0], fallback=fallback)

    process_end(fallback=fallback)
main_cli.add_command(convert)

@click.command()
def interactive():
    """ Launch bash hip.exe """
    import os
    import platform
    import pkg_resources

    hip_cmd = pkg_resources.resource_filename("pyhip", "hip_%s.exe" % platform.system())
    os.system(hip_cmd)
main_cli.add_command(interactive)

@click.command()
@click.argument('script', type=click.Path(exists=True), nargs=1)
def script(script):
    """ Excecute hip script in batch mode """

    import os
    import platform
    import pkg_resources

    with open(script, 'a') as fscript:
        fscript.write('exit\n')

    hip_cmd = pkg_resources.resource_filename("pyhip", "hip_%s.exe" % platform.system())
    os.system(f'{hip_cmd} {script}')
main_cli.add_command(script)

@click.command()
@click.argument('meshfile', type=click.Path(exists=True), nargs=1)
@click.argument('factor', type=float, nargs=1)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def coarsen(meshfile, factor, checklevel, fallback):
    """ Coarsen a mesh with isoFactor > 1 """

    from pathlib import Path
    from pathlib import Path
    from pyhip.commands import (
        set_checklevel,
        read_mesh_files,
        adapt_with_factor,
        write_hdf5,
    )
    from pyhip.setup import process_end

    if factor <= 1.:
        msg = "Coarsening factor must be greater than 1."
        raise IOError(msg)

    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([meshfile], 'hdf5', fallback=fallback)
    adapt_with_factor(factor, fallback=fallback)
    write_hdf5(Path(meshfile).name.split('.')[0] + '_coarsened', fallback=fallback)

    process_end(fallback=fallback)
main_cli.add_command(coarsen)

@click.command()
@click.argument('meshfile', type=click.Path(exists=True), nargs=1)
@click.argument('factor', type=float, nargs=1)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def refine(meshfile, factor, checklevel, fallback):
    """ Refine a mesh with isoFactor < 1 """

    from pathlib import Path
    from pyhip.commands import (
        set_checklevel,
        read_mesh_files,
        adapt_with_factor,
        write_hdf5,
    )
    from pyhip.setup import process_end

    if factor >= 1.:
        msg = "Refining factor must be lower than 1."
        raise IOError(msg)
    
    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([meshfile], 'hdf5', fallback=fallback)
    adapt_with_factor(factor, fallback=fallback)
    write_hdf5(Path(meshfile).name.split('.')[0] + '_refined', fallback=fallback)

    process_end(fallback=fallback) 
main_cli.add_command(refine)

@click.command()
@click.argument('meshfile', type=click.Path(exists=True), nargs=1)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def improve(meshfile, checklevel, fallback):
    """ Improve mesh using HIP algorithm """

    from pathlib import Path
    from pyhip.commands import (
        set_checklevel,
        read_mesh_files,
        adapt_with_factor,
        write_hdf5,
    )
    from pyhip.setup import process_end
    
    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([meshfile], 'hdf5', fallback=fallback)
    adapt_with_factor(1., fallback=fallback)
    write_hdf5(Path(meshfile).name.split('.')[0] + '_improved', fallback=fallback)

    process_end(fallback=fallback)
main_cli.add_command(improve)

@click.command()
@click.argument('meshfile', type=click.Path(exists=True), nargs=1)
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
def geo(meshfile, checklevel, fallback):
    """ Generate .geo and .case file """

    from pathlib import Path
    from pyhip.commands import(
        set_checklevel,
        read_mesh_files,
        dump_wired,
    )
    from pyhip.setup import process_end

    set_checklevel(checklevel, fallback=fallback)
    read_mesh_files([meshfile], 'hdf5', fallback=fallback) 
    root = Path(meshfile).parent / Path(meshfile).name.split('.')[0] 
    dump_wired(root, fallback=fallback)
    
    process_end(fallback=fallback)
main_cli.add_command(geo)

@click.command()
@click.argument('meshfiles', type=click.Path(exists=True), nargs=-1)
@click.option(
    "--shading",
    type=click.Choice(['none', 'linear', 'radial', 'flat' ], case_sensitive=False),
    default="flat",
    )
@click.option('--checklevel', type=click.IntRange(0, 5, clamp=True), default=0)
@click.option('--fallback/--nofallback', default=True)
@click.pass_context
def view(ctx, meshfiles, shading, checklevel, fallback):
    """ Quick mesh graphical view """

    import os
    from pathlib import Path
    from tiny_3d_engine import Engine3D, load_file_as_scene

    # White, Red, Blue, Green, Yellow
    colors = ('#ffffff', '#FF2D00', '#0013FF', '#00BF0C', '#FFF000')
    if len(meshfiles) > 5:
        msg = "View can not accept more than 5 meshes in a row"
        raise IOError(msg)

    scene = None
    for meshfile, color in zip(meshfiles, colors):
        prefix = Path(meshfile).name.split('.')[0] 
        geofile = Path(meshfile).parent / f'{prefix}.geo'
        if not geofile.exists():
            ctx.invoke(
                geo,
                meshfile=meshfile,
                checklevel=checklevel,
                fallback=fallback,
            )
        scene = load_file_as_scene(
            geofile,
            prefix=prefix,
            scene=scene,
            color=color,
        )
        scene.add_axes()

    engine = Engine3D(scene, shading=shading)
    engine.clear()
    engine.mainloop()
main_cli.add_command(view)


