""" Functionalities to convert a certain format to hdf5.
    Currently supported:
        - nek5000 (requires "pymech" package)
        - vtk: vtu, pvtu from ALYA solver, vtk unstructured from OpenFOAM (requires "vtk" package)
"""


from h5cross.convert._convert import convert
