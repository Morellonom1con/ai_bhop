"""
*Module of generation of a mesh ID Card*

This module generate a mesh ID Card summing up main mesh
informations and displaying geometry.

"""
from io import BytesIO
from collections import namedtuple
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm, inch
from reportlab.lib.utils import ImageReader
from scipy.spatial.transform import Rotation as Rot
from arnica.utils.vector_actions import (
    renormalize,
    rotate_vect_around_axis,
)

__all__ = ['generate_mesh_idcard']

MARGIN = 1.*cm
THOUSAND_SEP = "'"
LINE_HEIGHT = 0.35*cm
Bounds = namedtuple('Bounds', ['min', 'max'])
mid_ = lambda arr: (arr.max(axis=0) + arr.min(axis=0)) / 2.
range_ = lambda arr: arr.max(axis=0) - arr.min(axis=0)


def _get_datasets(fin, ds_names, data_dict=None, path_='', gr_skip=None):
    """ Get list of dataset from hdf file """
    if data_dict is None:
        data_dict = dict()
    if gr_skip is None:
        gr_skip = []
    for gname, group in fin.items():
        path = path_ + "/" + gname
        if not isinstance(group, h5py.Dataset) and gname not in gr_skip:
            _get_datasets(fin[gname], ds_names, data_dict, path_=path, gr_skip=gr_skip)
        else:
            if gname in ds_names:
                data_dict[gname] = fin[path][()]
    return data_dict

def extract_hdf_meshinfo(meshfile):
    #pylint: disable=too-many-locals
    """
    *Get mesh informations from hdf file and format them*

    :param meshfile: Path to hdf mesh file
    :type meshfile: str

    :returns:

        - **meshinfo** - Dict of mesh informations
        - **idx_patches** - List of patch index (starting by 1) without periodic patches
    """
    ds_names = (
        'hipversion',
        'vol_domain',
        'h_min',
        'h_max',
        'r_min',
        'r_max',
        'vol_elem_max',
        'vol_elem_min',
        'x_min',
        'x_max',
        'bnode_lidx',
        'periodic_patch',
        'periodic_angle',
        'PatchLabels')
    connectivities = {
        'tri->node': {'name': 'Triangles', 'nnode': 3},
        'qua->node': {'name': 'Quadrilateral', 'nnode': 4},
        'tet->node': {'name': 'Tetrahedra', 'nnode': 4},
        'pyr->node': {'name': 'Pyramids', 'nnode': 5},
        'pri->node': {'name': 'Prisms', 'nnode': 6},
        'hex->node': {'name': 'Hexahedra', 'nnode': 8}}

    with h5py.File(meshfile, 'r') as fin:
        data_dict = _get_datasets(fin, ds_names)
        connectivities_dict = _get_datasets(fin, connectivities.keys())

    if 'periodic_angle' in data_dict:
        is_axi = True
    else:
        is_axi = False
    ndim = len(data_dict['x_min'])

    meshinfo = {}
    meshinfo['Mesh name'] = meshfile.split('/')[-1]
    if isinstance(data_dict['hipversion'], np.ndarray):
        data_dict['hipversion'] = data_dict['hipversion'][0]
    meshinfo['Hip version'] = data_dict['hipversion'].decode('utf-8')
    meshinfo['Number of nodes'] = 0
    meshinfo['Number of boundary nodes'] = int(data_dict.get('bnode_lidx', [0])[-1])
    meshinfo['Domain volume [m3]'] = data_dict['vol_domain'][0]
    meshinfo['Number of cells'] = {}

    for c_type, dataset in connectivities_dict.items():
        name = connectivities[c_type]['name']
        nnode = connectivities[c_type]['nnode']
        ncells = dataset.size / nnode
        meshinfo['Number of cells'][name] = int(ncells)
        meshinfo['Number of nodes'] += int(dataset.max())

    meshinfo['Metric'] = {}
    meshinfo['Metric']['Edge length [m]'] = Bounds(
        data_dict['h_min'][0],
        data_dict['h_max'][0])
    meshinfo['Metric']['Element volume [m3]'] = Bounds(
        data_dict['vol_elem_min'][0],
        data_dict['vol_elem_max'][0])

    meshinfo['Bounding box'] = {}
    meshinfo['Bounding box']['x [m]'] = Bounds(
        data_dict['x_min'][0],
        data_dict['x_max'][0])
    if ndim == 2:
        meshinfo['Bounding box']['y [m]'] = Bounds(
            data_dict['x_min'][1],
            data_dict['x_max'][1])
    else:
        if is_axi:
            meshinfo['Bounding box']['r [m]'] = Bounds(
                data_dict['r_min'][0],
                data_dict['r_max'][0])
            meshinfo['Bounding box']['theta [deg]'] = Bounds(
                data_dict['r_min'][1],
                data_dict['r_max'][1])
        else:
            meshinfo['Bounding box']['y [m]'] = Bounds(
                data_dict['x_min'][1],
                data_dict['x_max'][1])
            meshinfo['Bounding box']['z [m]'] = Bounds(
                data_dict['x_min'][2],
                data_dict['x_max'][2])

    patch_labels = np.array([name.decode('utf-8').strip() for name in data_dict['PatchLabels']])
    if 'periodic_angle' in data_dict:
        meshinfo['Periodic angle [deg]'] = round(data_dict['periodic_angle'][0], 5)
    if 'periodic_patch' in data_dict:
        idx_perio = data_dict['periodic_patch'] - 1
        idx_perio = idx_perio.reshape((int(idx_perio.size/2), 2))
        meshinfo['Periodic pairs'] = {f"{i*' '}": f"{pair[0]} - {pair[1]}"
                                      for i, pair in enumerate(patch_labels[idx_perio])}

    return meshinfo, is_axi, ndim, patch_labels

def arrange_meshinfo(meshinfo_dict, col_width=None, lines=None, shift=0):
    """
    *Format mesh informations as lines to be drawn on pdf*

    :param meshinfo_dict: Dict of mesh informations

    :optional arguments:

    :param col_width: Width in digit of label column
    :type col_width: int
    :param lines: List of lines to be drawn
    :param shift: Tabulation shift in digit for sub elements
    :type shift: int
    """
    if lines is None:
        lines = []
    if col_width is None:
        col_width = max([len(label) for label in meshinfo_dict.keys()]) + 3

    for label, info in meshinfo_dict.items():
        if isinstance(info, dict):
            lines.append(f"{' '*shift}{label:{col_width}}")
            arrange_meshinfo(info, col_width=col_width, lines=lines, shift=shift+2)
        else:
            if isinstance(info, Bounds):
                if abs(info.min) > 1.e-4 and abs(info.max) > 1.e-4:
                    value = f"min={f'{info.min:6f}':17} max={info.max:6f}"
                else:
                    value = f"min={f'{info.min:6e}':17} max={info.max:6e}"
            elif isinstance(info, int):
                value = f"{info:,}".replace(',', THOUSAND_SEP)
            elif isinstance(info, str):
                value = info
            else:
                if abs(info) > 1.e-4:
                    value = f"{info:6f}"
                else:
                    value = f"{info:6e}"
            lines.append(f"{' '*shift}{label:{col_width-shift}}" + value)

    return lines

def _create_perspective_mark(xyz_rough, *rot_tuples, coef=100.):
    """
    *Generate a perspective xyz array*

    :param xyz_rough: Array of dim (n,3) of xyz-coordinates
    :param rot_tuples: List of tuple of rotation data (Axis, angle)
    :param coef: Coefficient of perspective
    :type coef: float

    :returns: Array of dim (n,3) of perspectived xyz-coordinates
    """
    center = (xyz_rough.min(axis=0) + xyz_rough.max(axis=0)) / 2.
    xyz_rough -= center
    xyz_persp = np.multiply(xyz_rough,
                            np.power(coef, (-xyz_rough[:, 0][:, np.newaxis])))
    xyz_persp = rotate_vect_around_axis(xyz_persp, *rot_tuples)
    return xyz_persp

def _get_boundary_nodes(meshfile, ndim):
    """ Return boundary nodes array """
    axis = ['x', 'y', 'z'][:ndim]
    coords = [[] for _ in range(ndim)]

    with h5py.File(meshfile, 'r') as fin:
        for idx in fin[f'Patch/'].keys():
            for i, axes in enumerate(axis):
                coords[i].extend(fin[f'Patch/{idx}/Coordinates/{axes}'][()])
    return np.asarray(coords)

def _plot_skin(axis, coords, labels):
    """Plot hist2d from coords"""

    def get_bins(x_crd, y_crd):
        """Computes equal bins from aspect ratio"""
        ratio = range_(y_crd) / range_(x_crd)
        return int(250.), int(250. * ratio)

    plt.rcParams['image.cmap'] = 'Blues'
    axis.hist2d(*coords,
                bins=get_bins(*coords),
                norm=LogNorm())
    axis.set_aspect('equal')
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.xaxis.set_ticks([coords[0].min(), coords[0].max()])
    axis.yaxis.set_ticks([coords[1].min(), coords[1].max()])
    axis.yaxis.set_label_coords(-0.1, 0.5)
    for edge in ['top', 'bottom', 'right', 'left']:
        axis.spines[edge].set_visible(False)

def plot_2d_density_view(meshfile): 
    """
    *Plot front 2D mesh view as hist2d*

    :param meshfile: Path to hdf5 mesh file
    :type meshfile: str
    """
    x_arr, y_arr = _get_boundary_nodes(meshfile, 2)

    fig_width = 10.
    aspect_ratio = range_(y_arr) / range_(x_arr)
    fig, axis = plt.subplots(figsize=(fig_width, fig_width * aspect_ratio))
    _plot_skin(axis, (x_arr, y_arr), ('x', 'y'))
    fig.subplots_adjust(left=0.125, right=0.95, bottom=0.1, top=0.95)

    buffer_hist2d = BytesIO()
    fig.savefig(buffer_hist2d, format='png', transparent=True, dpi=600)
    buffer_hist2d.seek(0)
    return buffer_hist2d

def plot_3d_density_view(meshfile, is_axi):
    #pylint: disable=too-many-locals
    """
    *Plot 3 sides and 1 perspective 3D mesh views as hist2d*

    :param meshfile: Path to hdf5 mesh file
    :type meshfile: str
    :param is_axi: Determine if geometry is axi-periodic.
    :type is_axi: bool
    """
    x_arr, y_arr, z_arr = _get_boundary_nodes(meshfile, 3)
    x_per, y_per, _ = _create_perspective_mark(
        np.stack((x_arr, y_arr, z_arr), axis=-1),
        ([0, 1, 0], 60.),
        ([0, 0, 1], 10.)).T
    aspect_ratio = min(range_(z_arr) / range_(x_per),
                       range_(z_arr) / range_(y_per))
    x_per *= aspect_ratio
    x_per += mid_(z_arr) - mid_(x_per)
    y_per *= aspect_ratio
    y_per += mid_(z_arr) - mid_(y_per)

    Views = namedtuple('View', ['labels', 'coords'])
    if is_axi:
        r_arr = np.hypot(y_arr, z_arr)
        view_axes = [Views(('x', 'r'), (x_arr, r_arr)),
                     Views(('z', 'y'), (z_arr, y_arr)),
                     Views(('x', 'z'), (x_arr, z_arr))]
    else:
        view_axes = [Views(('x', 'y'), (x_arr, y_arr)),
                     Views(('z', 'y'), (z_arr, y_arr)),
                     Views(('x', 'z'), (x_arr, z_arr))]

    plt.rcParams['image.cmap'] = 'Blues'
    fig_width = 10.
    aspect_ratio = (range_(y_arr) + range_(z_arr)) / (range_(x_arr) + range_(z_arr))
    fig, axes = plt.subplots(
        figsize=(fig_width, fig_width * aspect_ratio),
        nrows=2, ncols=2,
        sharex='col', sharey='row',
        gridspec_kw={'height_ratios': (range_(y_arr), range_(z_arr)),
                     'width_ratios': (range_(x_arr), range_(z_arr)),
                     'hspace': 0.12,
                     'wspace': 0.12})

    _plot_skin(axes[1, 1], (x_per, y_per), ('', ''))
    for i, view in enumerate(view_axes):
        _plot_skin(axes[i//2, i%2], view.coords, view.labels)
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(left=0.125, right=0.95, bottom=0.1, top=0.95)

    buffer_hist2d = BytesIO()
    fig.savefig(buffer_hist2d, format='png', transparent=True, dpi=600)
    buffer_hist2d.seek(0)
    return buffer_hist2d

def plot_hist_from_meshfile(meshfile, width, height):
    """
    *Plot histogram of dataset occurence*

    :param meshfile: Path to hdf5 mesh file
    :type meshfile: str

    :returns: **buffer_hist** - BytesIO object of the plot
    """
    with h5py.File(meshfile, 'r') as fin:
        vol_node = _get_datasets(fin, ['volume'])['volume']

    vol_cell = vol_node / 4.5
    len_edge = np.power(vol_cell * 12. / np.sqrt(2.), 1. / 3.)

    bins = np.logspace(np.log10(len_edge.min()),
                       np.log10(len_edge.max()),
                       100)
    plt.figure(figsize=(width / inch, height / inch))
    plt.hist(len_edge,
             bins=bins,
             weights=np.ones_like(len_edge)/len_edge.size)
    plt.axvline(len_edge.min(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(len_edge.max(), color='k', linestyle='dashed', linewidth=1)
    _, max_ylim = plt.ylim()
    plt.text(
        len_edge.min(),
        0.85*max_ylim,
        f' Min = {len_edge.min():.3e} ',
        horizontalalignment='left')
    plt.text(
        len_edge.max(),
        0.85*max_ylim,
        f' Max = {len_edge.max():.3e} ',
        horizontalalignment='right')
    #plt.grid()
    plt.xlabel('Edge length [m]')
    plt.xscale('log')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=2))
    plt.title('Histogram of Edge length')
    plt.tight_layout()

    buffer_hist = BytesIO()
    plt.savefig(buffer_hist, format='png', transparent=True, dpi=400)
    buffer_hist.seek(0)
    return buffer_hist

def _get_plot_size(width_max, height_max, width_, height_):
    """ Compute fig size to fit in dimensions max available """
    ratio_width = width_max / width_
    ratio_height = height_max / height_
    ratio = min(ratio_width, ratio_height) * 0.99
    return (width_*ratio, height_*ratio)

def _make_title(pdf, mesh_name, coords, width, height):
    """ Draw title section """
    offset = 0.1*cm
    section_height = 1.5*cm
    title = mesh_name.upper() + ' MESH CARD'
    coords[1] -= section_height

    pdf.rect(
        coords[0] + offset,
        coords[1] + offset,
        width - 2 * (MARGIN + offset),
        section_height - 2 * offset)
    pdf.setFont('Courier', 20)
    pdf.drawCentredString(
        width / 2.,
        height - MARGIN - section_height / 2. - 6.,
        title)

def _make_meshinfo(pdf, meshinfo_dict, coords):
    """ Draw meshinfo section """
    pdf.setFont('Courier', 8)
    coords[0] += 0.3*cm
    coords[1] -= LINE_HEIGHT * 1.3
    meshinfo_lines = arrange_meshinfo(meshinfo_dict)

    for line in meshinfo_lines:
        pdf.drawString(*coords, line)
        coords[1] -= LINE_HEIGHT

    coords[0] -= 0.3*cm

def _make_hist(pdf, meshfile, coords, size_max):
    """ Draw histogram section """
    size_max = [side - 0.2*cm for side in size_max]
    buffer = plot_hist_from_meshfile(meshfile, *size_max)
    hist = ImageReader(buffer)
    size = _get_plot_size(*size_max, *hist.getSize())
    coords[1] -= size[1] + 0.1*cm
    pdf.drawImage(hist, *coords, *size)


def _make_meshview(pdf, meshfile, is_axi, ndim, coords, size_max, mid=False):
    #pylint: disable=too-many-arguments
    """ Draw meshview section """
    size_max = [side - 0.2*cm for side in size_max]

    if ndim == 2:
        buffer_plots = plot_2d_density_view(meshfile)
    else:
        buffer_plots = plot_3d_density_view(meshfile, is_axi)
    plot = ImageReader(buffer_plots)
    size = _get_plot_size(*size_max, *plot.getSize())
    coords[1] -= size[1] + 0.1*cm
    if mid is True:
        coords[0] += (size_max[0] - size[0]) / 2.
        coords[1] -= (size_max[1] - size[1]) / 2.
    pdf.drawImage(plot, *coords, *size)
    coords[1] -= 0.1*cm

def generate_landscape(meshfile, pagesize):
    """
    *Generate mesh ID cart from mesh in landscape format*

        - Get mesh informations
        - Build title
        - Build mesh info section
        - Build plots section
        - Build volume vertex histogram

    :param meshfile: Path to hdf5 mesh file
    :type meshfile: str
    :param pagesize: Tuple (width, height) of the page size. \
                     1 pt = 1/72 inch ~ 0.035277 cm
    """
    meshinfo_dict, is_axi, ndim, _ = extract_hdf_meshinfo(meshfile)

    pdf_file = meshinfo_dict['Mesh name'].split('.')[0] + '.mesh.pdf'
    pdf = canvas.Canvas(pdf_file, pagesize=pagesize, verbosity=0)
    width, height = pagesize

    # Title section
    coords = [MARGIN, height - MARGIN]
    mesh_name = meshinfo_dict['Mesh name'].split('.')[0]
    _make_title(pdf, mesh_name, coords, *pagesize)

    pdf.line(*coords, width - MARGIN, coords[1])

    # Meshinfo section
    meshinfo_coords = coords.copy()
    _make_meshinfo(pdf, meshinfo_dict, coords)

    pdf.line(*coords, 0.4*width, coords[1])

    # Histogram section
    size_max = [0.4*width - MARGIN, coords[1] - MARGIN]
    _make_hist(pdf, meshfile, coords, size_max)

    # MeshView section
    coords = [0.4 * width, meshinfo_coords[1]]
    size_max = [0.6 * width - MARGIN, meshinfo_coords[1] - MARGIN]
    _make_meshview(pdf, meshfile, is_axi, ndim, coords, size_max, mid=True)

    pdf.line(0.4*width, meshinfo_coords[1], 0.4*width, MARGIN)

    # Main rectangle
    pdf.rect(MARGIN, MARGIN, width - 2*MARGIN, height - 2*MARGIN)

    print(f'{pdf_file} file saved.')
    pdf.save()


def generate_portrait(meshfile, pagesize):
    """
    *Generate mesh ID cart from mesh in portrait format*

        - Get mesh informations
        - Build title
        - Build mesh info section
        - Build plots section
        - Build volume vertex histogram

    :param meshfile: Path to hdf5 mesh file
    :type meshfile: str
    :param pagesize: Tuple (width, height) of the page size. \
                     1 pt = 1/72 inch ~ 0.035277 cm
    """
    meshinfo_dict, is_axi, ndim, _ = extract_hdf_meshinfo(meshfile)

    pdf_file = meshinfo_dict['Mesh name'].split('.')[0] + '.mesh.pdf'
    pdf = canvas.Canvas(pdf_file, pagesize=pagesize, verbosity=0)
    width, height = pagesize

    # Title section
    coords = [MARGIN, height - MARGIN]
    mesh_name = meshinfo_dict['Mesh name'].split('.')[0]
    _make_title(pdf, mesh_name, coords, *pagesize)

    pdf.line(*coords, width - MARGIN, coords[1])

    # Meshinfo section
    _make_meshinfo(pdf, meshinfo_dict, coords)

    pdf.line(*coords, width - MARGIN, coords[1])

    # MeshView section
    size_max = [width - 2*MARGIN, coords[1] - 6*cm]
    _make_meshview(pdf, meshfile, is_axi, ndim, coords, size_max)

    pdf.line(*coords, width - MARGIN, coords[1])

    # Histogram section
    size_max = [width - 2*MARGIN, coords[1] - MARGIN]
    _make_hist(pdf, meshfile, coords, size_max)

    # Main rectangle
    pdf.rect(MARGIN, MARGIN, width - 2*MARGIN, height - 2*MARGIN)

    print(f'{pdf_file} file saved.')
    pdf.save()

def generate_mesh_idcard(meshfile, pagesize):
    """
    *Main function distributing to landscape or portrait in function of the size*

    :param meshfile: Path to hdf5 mesh file
    :type meshfile: str
    :param pagesize: Tuple of page size (width, height) in pt (1 pt = 1/72 inch)
    """
    if pagesize[0] > pagesize[1]:
        generate_landscape(meshfile, pagesize)
    else:
        generate_portrait(meshfile, pagesize)
