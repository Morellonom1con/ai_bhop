"""Module containing hip readers functions """
import os
from pyhip.hipster import pyhip_cmd


__all__ = ["read_mesh_files",
           "read_centaur_mesh",
           "read_cgns_mesh",
           "read_ensight_mesh",
           "read_fluent_mesh",
           "read_gmsh_mesh",
           "read_hdf5_mesh"]

def _split_file_path(file_path):
    """ Returns the file basename and the parent
        directory given file path

        Parameters:
        ==========
        file_path : path to file (absolute or relative)

        Returns:
        =======
        parent : file parent directory
        filename : file name
    """
    abs_path = os.path.abspath(file_path)
    parent = os.path.dirname(abs_path)
    filename = os.path.basename(abs_path)
    return parent, filename

def _rel_path(files, start_path):

    abspaths = [os.path.abspath(file) for file in files]
    rel_paths = [os.path.relpath(abspath, start=start_path)
                 for abspath in abspaths]
    return rel_paths

def read_mesh_files(mesh_files, mesh_format, **kwargs):
    """ read a mesh file
        Parameters:
        ==========
        mesh_file : A list of mesh files paths (absol or rel)
        mesh_format : Format of the mesh file (e.g hdf5)

        Returns:
        =======
        None
    """
    files = [file for file in mesh_files if file is not None]
    parent_dir, _ = _split_file_path(files[0])
    files = _rel_path(files, parent_dir)
    if mesh_format.strip().lower() == 'hdf5' and len(files) == 2:
        files = "-a %s -s %s" %(files[0], files[1])
    else:
        files = ' '.join(files)

    commands = ["set path %s"%parent_dir]
    commands.append("read %s %s" %(mesh_format, files))
    commands.append("var")
    for command in commands:
        _ = pyhip_cmd(command, **kwargs)
    return commands

# def read_bae_mesh(fro_file, gri_file, bco_file):
#     """Read Britsh Aerospace and Swansea University
#         tetrahedral meshes.
#def _rel_path(files, start_path):

#         Parameters:
#         ==========
#         fro_file : boundary faces and singular lines file (binary)
#         gri_file : connectivity and coordinates file (binary)
#         bco_file : file containing boundary condition to the
#                   various surfaces mapping. (ascii)

#     """
#     return read_mesh_files([fro_file, gri_file, bco_file], "bae")

# def read_cedre_mesh(cdre_mesh_file):
#     """Read unstructured face-based mesh format used Cedre code.

#        Parameters:
#        ==========
#        cdre_mesh_file : Cedre unstructured face-based mesh file
#     """
#     return read_mesh_files([cdre_mesh_file], "cdre")

def read_centaur_mesh(centaur_mesh_file, **kwargs):
    """Read hybrid grid in Centaursofts format.

       Parameters:
       ==========
       centaur_mesh_file : Centaursofts format mesh_file. Supported file
                           format are versions:
                           - 4 (single record)
                           - 5 (multiple record).
    """
    return read_mesh_files([centaur_mesh_file], "centaur", **kwargs)

def read_cgns_mesh(grid_file, sol_file=None, abnd_file=None, **kwargs):
    """Read an unstructured CGNS database.

       Parameters:
       ==========
       grid_file: CGNS grid file
       sol_file: CGNS solution file
       abnd_file: if bnd info in an AVBP-style .asciiBound file is given,
                  this supersedes boundary condition definition in
                  the CGNS grid_file.

    """
    return read_mesh_files([grid_file, sol_file, abnd_file], "cgns", **kwargs)

def read_ensight_mesh(ensight_case_file, **kwargs):
    """Read an unstructured grid (no Solution) in
       Ensight Gold format

       Parameters:
       ==========
       ensight_case_file: Ensight Gold format file

    """
    return read_mesh_files([ensight_case_file], "ensight", **kwargs)

def read_fluent_mesh(mesh_file, solution_file=None, **kwargs):
    """Read an unstructured grid and optionally
       a solution in Fluent’s v5 and v6 ascii and binary
       formats

       Parameters:
       ==========
       mesh_file: Fluent/Gambit mesh file (.msh/.cas extensions)
       solution_file : Fluent solution file (.dat extension)
    """
    return read_mesh_files([mesh_file, solution_file], "fluent", **kwargs)

def read_gmsh_mesh(gmsh_mesh_file, list_variables=None, **kwargs):
    """Read an unstructured grid in gmsh ASCII format

       Parameters:
       ==========
       gmsh_mesh_file: gmsh mesh file
       list_variables : a list of variable files
                        (Up to MAX UNKNOWNS = 256)
    """
    if list_variables is None:
        cmds = read_mesh_files([gmsh_mesh_file], "gmsh", **kwargs)
    else:
        cmds = read_mesh_files([gmsh_mesh_file, *list_variables], "gmsh", **kwargs)
    return cmds

def read_hdf5_mesh(grid_file, sol_file=None, **kwargs):
    """Read an unstructured grid in hdf5 format

       Parameters:
       ==========
       grid_file: hdf5 mesh file
       sol_file : Solution_file
    """
    return read_mesh_files([grid_file, sol_file], "hdf5", **kwargs)


def infer_mesh_format(filename):
    """Try to guess the mesh fromat from its extension"""
    _,ext = os.path.splitext(filename)
    if ext in [".h5", ".hdf5"]:
        return "hdf5"
    if ext in [".cgns"]:
        return "cgns"
    if ext in [".hyb"]:
        return "centaur"
    if ext in [".geo", ".cas"]:
        return "ensight"
    if ext in [".msh"]:
        print("Warning, found .msh! Assuming its Fluent, not Gmsh")
        return "fluent"
    # if ext in [".msh"]:
    #     return "gmsh"
    
    raise RuntimeError(f"Could not infer the type of format {ext} for {filename}")
        
    