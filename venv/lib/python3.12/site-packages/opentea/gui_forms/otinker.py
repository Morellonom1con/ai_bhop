"""Generate a Tk from upon a Gui schema.

A GUI schema is a JSON-Schema dictionnary,
with tags require and existifs added to declare explicit cyclic depenencies
"""
import os
from tkinter import (
    Tk,
    ttk,
)
import time
from loguru import logger
import opentea
from opentea.gui_forms.root_widget import OTRoot
from opentea.gui_forms.constants import (
    set_constants,
)
from opentea.gui_forms.utils import quit_dialog
from opentea.noob.validation import validate_opentea_schema


def create_voidfile(path):
    """Create a dummy file for otinker"""
    void_ct = """
# Opentea Project
"""
# meta:
#   validate: {}
# """
    with open(path, "w") as fout:
        fout.write(void_ct)


# pylint: disable=too-many-arguments
def main_otinker(
    schema,
    calling_dir=None,
    start_mainloop=True,
    theme="clam",
    # -----------------
    data_file=None,
    tab_3d: callable = None,
    tab_2d: callable = None,
    acq_2d: callable = None,
    paraview_exec: str = None,
):
    """Startup the gui generation.

    Inputs :
    --------
    schema : dictionary compatible with json-schema
    calling_dir : directory from which otinker was called
    test_only : only for testing

    Outputs :
    ---------
    a tkinter GUI
    """
    start = time.time()
    # global CALLING_DIR
    # CALLING_DIR = calling_dir
    logger.debug(f" (**,) Staring up OpenTEA GUI engine v.{opentea.__version__}...")

    if isinstance(tab_3d, bool):
        msgwrn = """Your gui is a bit behind this opentea3 version. The 3d viewport is probably disconnected"""
        logger.warning(msgwrn)
    if isinstance(tab_2d, bool):
        msgwrn = """Your gui is a bit behind this opentea3 version. The 2d viewport is probably disconnected"""
        logger.warning(msgwrn)

    validate_opentea_schema(schema)
    tksession = Tk()
    tksession.protocol("WM_DELETE_WINDOW", quit_dialog)

    sty = ttk.Style()
    sty.theme_use(theme)

    set_constants(tksession, calling_dir, theme)

    if start_mainloop is False or data_file is None:
        create_voidfile("dummy.yml")
        logger.warning(" (oO,) |No file provided, using dummy.yml for now.")
        data_file = "dummy.yml"

    data_file = os.path.abspath(data_file)

    logger.success(f" (^^,) |Opening {data_file}...")
    otroot = OTRoot(
        schema,
        tksession,
        sty,
        data_file=data_file,
        tab_3d=tab_3d,
        tab_2d=tab_2d,
        acq_2d=acq_2d,
        paraview_exec=paraview_exec,
    )
    end = time.time()
    logger.info(f" (,==) |Opening in {end - start:.2f}s")
    
    if start_mainloop:
        otroot.mainloop()

    return otroot
