"""Startup script to call calculator gui."""
import os
import inspect
import yaml
from loguru import logger
from opentea.gui_forms.otinker import main_otinker
import matplotlib.pyplot as plt
from tiny_3d_engine.scene3d import Scene3D
from opentea.gui_forms.viewer2d import HoverItems


def main(data_file: str = None):
    """Example on how to call the otinker gui"""

    # First, load the blueprint of the GUI, using the SCHEMA standard
    schema_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "schema_calculator.yaml"
    )
    with open(schema_file, "r") as fin:
        schema = yaml.load(fin, Loader=yaml.FullLoader)

    # Keep the base dir where you call the GUI
    base_dir = inspect.getfile(inspect.currentframe())
    base_dir = os.path.dirname(os.path.abspath(base_dir))

    # Invoke the GUI
    main_otinker(
        schema,
        data_file=data_file,
        calling_dir=base_dir,
        tab_3d=callback_3d,
        tab_2d=callback_2d,
        acq_2d=callback_2d_acq,
        theme="aqua",
    )


def callback_2d(axes: plt.axes, hover: HoverItems, nob: dict):
    """Example of a 2D callback.

    Simply use the plt.axes object and the nested object data given as parameters
    """
    import numpy as np
    from nob import Nob

    # Here we use Nob for fast retrieval of data without spefying the full path
    snob = Nob(nob)
    sampling = snob.sampling[:]
    frequency = snob.frequency[:]

    # Classical numpy-based plot
    t = np.linspace(0.0, 1.0, int(sampling))
    y = np.sin(t * (2 * np.pi) * frequency)
    axes.plot(t, y)
    hover.add_line(t,y+1, "Yp1")
    axes.set_title("Frequency plot")


def callback_2d_acq(nob: dict,  acq_data: dict):
    """Example of a 2D acquisition callback."""
    from nob import Nob
    snob = Nob(nob)
    # here we use the acquisition data so set a list in memory
    snob.list_acq_lines = acq_data["acq_names"]  
    # in the schema, a multiple depends on this list.
    return snob[:]


def callback_3d(nob: dict) -> Scene3D:
    """Example of a 2D callback.

    Create a Scene3D object and return it, the widget will take care of the plotting.
    """
    from nob import Nob
    from tiny_3d_engine.part3d import Part3D
    from tiny_3d_engine.geo_loader import load_geo
    snob = Nob(nob)

    part1 = Part3D()
    part1.add_cartbox(
        snob.coque1.point0[:],
        snob.coque1.point1[:],
        snob.coque1.ngrid[:],
        snob.coque1.ngrid[:],
        snob.coque1.ngrid[:],
    )

    part2 = Part3D()
    part2.add_frustum(
        snob.coque2.point0[:],
        snob.coque2.point1[:],
        snob.coque2.radius0[:],
        snob.coque2.radius1[:],
        snob.coque2.ngrid[:],
        snob.coque2.ngrid[:],
    )

    scene = Scene3D()
    scene.update("cube", part1.points, part1.conn, color="#ff0000")
    scene.update("cone", part2.points, part2.conn, color="#00ff00")
    try:
        out = load_geo(snob.geo_file[:])
        for part in out:
            name = part
            coor = out[part]["coor"]
            el_type = out[part]["el_type"]
            conn = out[part][el_type]
            scene.update(name, coor, conn, color="#0000ff")
    except FileNotFoundError:
        logger.warning(f"GEO File '{snob.geo_file[:]}' not found for 3d rendering")
    return scene


if __name__ == "__main__":
    main()
