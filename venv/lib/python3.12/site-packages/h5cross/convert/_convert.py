
# TODO: can be simplified by using meshio/yamio?
# TODO: move nek5000 to yamio?

import importlib

EXT_MODULE_MAP = {
    'vtk': 'vtk',
    'vtu': 'vtk',
    'pvtu': 'vtk',
    'nek5000': 'nek5000'

}


def _get_reader(file_format):
    if file_format not in EXT_MODULE_MAP:
        return None

    module_name = EXT_MODULE_MAP[file_format]
    module = importlib.import_module(f'h5cross.convert._{module_name}')

    return module.READER_MAP[file_format]


def convert(filename, file_format, **kwargs):
    convert_ = _get_reader(file_format)

    if convert_ is None:
        raise Exception("Format not available.")

    return convert_(filename, **kwargs)
