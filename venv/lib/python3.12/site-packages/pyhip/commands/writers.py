"""Module containing hip writers functions"""
import os
from pyhip.hipster import pyhip_cmd


__all__ = ["write_fieldview",
           "write_avbp",
           "write_gmsh",
           "write_ensight",
           "write_cgns",
           "write_hdf5",
           "dump_wired"]

def _write_file(rootfile, output_format, writer_options=None, **kwargs):
    parent_dir = os.path.dirname(os.path.abspath(rootfile))
    basename, _ = os.path.splitext(os.path.basename(os.path.abspath(rootfile)))
    options = ['write', output_format]
    os.makedirs(parent_dir, exist_ok=True)
    if writer_options is not None:
        options.extend(writer_options)
    options.append(basename)

    commands = ["set path %s"%parent_dir]
    commands.append(" ".join(options))
    for command in commands:
        _ = pyhip_cmd(command, **kwargs)

    return commands

def write_fieldview(rootfile, **kwargs):
    """ Write mesh into fieldview format

        Parameters:
        ==========
        rootfile : output file base name path (without extension)
    """
    return _write_file(rootfile, "fieldview", **kwargs)

def write_avbp(rootfile, flavour=None, level=None, **kwargs):
    """ Write mesh and optionnaly solution to avbp format

        Parameters:
        ==========
        rootfile : output file base name path (without extension)
        flavour: should be one of the avbp flavours :
                 "avad", "avbp4.2", "avbp4.7", "avbp5.1", "avbp5.3eg",
                 "avbp5.3", "avh"
        level : level of coarsed mesh to be written, default is finest
                mesh
    """
    flavours = ["avad", "avbp4.2", "avbp4.7", "avbp5.1", "avbp5.3eg",
                "avbp5.3", "avh"]
    options = []
    if flavour is not None and flavour in flavours:
        options.append(flavour)
    if level is not None:
        try:
            options.append("%d" %int(level))
        except:
            raise ValueError("write_avbp : level should be an integer")

    return _write_file(rootfile, "avbp", options, **kwargs)

def write_gmsh(rootfile, **kwargs):
    """ Write mesh into gmsh format

        Parameters:
        ==========
        rootfile : output file base name path (without extension)
    """
    return _write_file(rootfile, "gmsh", **kwargs)

def write_ensight(rootfile,
                  writing_ascii=False,
                  write_node_ids=False,
                  segments=False,
                  extrude_2d=True,
                  **kwargs):
    """ Write mesh int ensight gold format

        Parameters:
        ==========
        rootfile : output file base name path (without extension)
        writing_ascii : if True, ascii format is adopted
        write_node_ids : if True, store node ids
        segments : if true, writes only 1D elements in geofile instead of 2D/3D ones
        this ignores the extrude_2d
        extrude_2d : if True, 2D mesh is extruded to 3D
                     in order to be readable by paraview
    """
    options = []
    if writing_ascii:
        options.append('-a')
    if write_node_ids:
        options.append('-n on')
    if segments:
        options.append('-s0')
    elif extrude_2d:
        options.append('-3')
    else:
        options.append('-2')

    return _write_file(rootfile, "ensight", options, **kwargs)

def dump_wired(rootfile, **kwargs):
    """ dump a wired geometry into rootfile.geo and rootfile.case
        Parameters:
        ==========
        rootfile : str root of the files to dump
    """
    command1 = "decimate"
    cmd1 = pyhip_cmd(command1, **kwargs)

    cmd23 = write_ensight(rootfile, writing_ascii=True, segments=True, **kwargs)

    return cmd1 + cmd23

def write_cgns(rootfile, **kwargs):
    """ Write mesh into fieldview format

        Parameters:
        ==========
        rootfile : output file base name path (without extension)
    """
    return _write_file(rootfile, "cgns", **kwargs)

def write_hdf5(rootfile,
               write_all=True,
               only_solution=False,
               separate_boundary_shell=False,
               add_metis_graph=False,
               write_faces_list=False,
               compression_level=None,
               **kwargs):
    """ Write mesh and optionnaly solution to file

        Parameters:
        ==========
        rootfile : output file base name path (without extension)
        write_all : if True all the variables that are stored,
                    otherwize, only the standard set is stored
        separate_boundary_shell : if True, the boundary shell (skin)
                                  is written as a separately numbered
                                  mesh entity for each boundary patch.
        compression_level: sets the zip compression level.
                            Note that due to overhead, compression with
                            the hdf native routines does not always result
                            in relevant file size reduction, or at times
                            any reduction at all.
        add_metis_graph: If True adds the METIS style elGraph.
        write_faces_list : if True, a complete list of faces between
                            elements is written to file.
                            Warning: Writing the complete list of faces
                            will enlarge the file size considerably.
        only_solution: if True, write a solution only but not the mesh
    """
    #pylint: disable=unused-argument, too-many-arguments
    usr_opts = locals().copy()
    opt_values = {"write_all":["-a", "-a0"],
                  "only_solution":["-s", ""],
                  "separate_boundary_shell":["-b", ""],
                  "add_metis_graph":['-e', ''],
                  "write_faces_list":['-f', '']}
    options = ['-7']
    for opt in opt_values:
        value = opt_values[opt][1]
        if usr_opts[opt]:
            value = opt_values[opt][0]
        if value:
            options.append(value)

    if usr_opts["compression_level"] is not None:
        options.append("-c %d" % compression_level)
    return _write_file(rootfile, "hdf5", options, **kwargs)
