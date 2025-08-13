import os
from abc import ABCMeta
from abc import abstractmethod
import tkinter as tk
from tkinter import filedialog
from loguru import logger
from nobvisual.nob2nstruct import visual_treenob
from nobvisual.nobvisual import nobvisual
import subprocess

from opentea.noob.asciigraph import nob_asciigraph
from opentea.gui_forms.utils import quit_dialog
from opentea.gui_forms.constants import toggle_verbose
from opentea.gui_forms.generic_widgets import TextConsole
from opentea.gui_forms.monitors import show_monitor


# TODO: about in md instead?
ABOUT = """
This is GUI FORMS, front ends provided by OpenTEA.

OpenTEA is an open source python package to help
the setup of complex softwares.
OpenTEA is currently developed at Cerfacs by the COOP team.
Meet us at coop@cerfacs.fr.
"""


class DefaultMenubar:
    """The main munubar on the top of the screen"""

    def __init__(self, otroot):
        self.otroot = otroot
        self.menus = []
        self._add_menus()

    @property
    def menubar(self):
        if len(self.menus) == 0:
            return None
        return self.menus[-1].master

    def _add_menus(self):
        self.add_menu(FileMenu(self.otroot))
        self.add_menu(DebugMenu(self.otroot, menubar=self.menubar))
        self.add_menu(HelpMenu(self.otroot, menubar=self.menubar))

    def add_menu(self, menu):
        self.menus.append(menu)

    def activate(self):
        self.otroot.tksession.configure(menu=self.menubar)


class _Menu(tk.Menu, metaclass=ABCMeta):
    def __init__(self, otroot, label, menubar=None, **kwargs):
        if menubar is None:
            menubar = tk.Menu()

        super().__init__(menubar, tearoff=0, **kwargs)
        menubar.add_cascade(label=label, menu=self)

        self.otroot = otroot
        self._add_items()
        self._bind_items()

    @property
    def root(self):
        return self.otroot.tksession

    @abstractmethod
    def _add_items(self):
        pass

    def _bind_items(self):
        pass


class FileMenu(_Menu):
    def __init__(self, otroot, label="File", **kwargs):
        super().__init__(otroot, label, **kwargs)

    def _add_items(self):
        self.add_command(
            label="Load  (Ctrl+O)",
            image=self.otroot.icons["load"],
            compound="left",
            command=self.on_load,
        )

        self.add_command(
            label="Save (Ctrl+S)",
            image=self.otroot.icons["save"],
            compound="left",
            command=self.on_save,
        )

        self.add_command(
            label="Save as (Shift+Ctrl+S)",
            image=self.otroot.icons["save"],
            compound="left",
            command=self.on_save_as,
        )

        self.add_separator()

        self.add_command(
            label="Quit   (Ctrl+W)",
            image=self.otroot.icons["quit"],
            compound="left",
            command=self.on_quit,
        )

    def _bind_items(self):
        self.root.bind("<Control-o>", self.on_load)
        self.root.bind("<Control-s>", self.on_save)
        self.root.bind("<Shift-Control-s>", self.on_save_as)
        self.root.bind("<Control-w>", self.on_quit)

    def on_load(self, event=None):
        """Load data in current application."""
        file = filedialog.askopenfilename(
            title="Select file",
            filetypes=(
                ("YAML files", "*.yml"),
                ("YAML files", "*.yaml"),
                ("all files", "*.*"),
            ),
        )
        if file != "":
            # First time to update the dependents lists
            self.otroot.load_project(file)
            # Second time to update all data according to dependents lists
            self.otroot.load_project(file)

            # This is less ugly than introducing a Setting order in datasets
            # Note : Double loading is not necessary at startup because the GUI building order prevails
            # Note2 :  its logical , for multiples, to load only data that matches the currently loaded patches
             
        else:
            logger.warning("No file selected -> Load aborted.")
            

    def on_save_as(self, event=None):
        """Save data in current application to an other project"""
        filename = filedialog.asksaveasfilename(
            title="Select a new location for your project",
            defaultextension=".yml",
            filetypes=(("YAML files", "*.yml"), ("all files", "*.*")),
        )
        if filename == "":
            logger.warning("No file selected -> Save aborted.")
            return
        
        self.otroot.data_file = os.path.abspath(filename)
        self.otroot.save_project()
        #  to stay in this directory for the next filemenu
        os.chdir(os.path.dirname(filename))

    def on_save(self, event=None):
        """Save data in current application."""
        self.otroot.save_project()

    def on_quit(self, event=None):
        """Quit full application from the menu."""

        quit_dialog()


class DebugMenu(_Menu):
    def __init__(self, otroot, label="Debug", **kwargs):
        super().__init__(otroot, label, **kwargs)

    def _add_items(self):
        self.add_command(
            label="Show tree",
            image=self.otroot.icons["tree"],
            compound="left",
            command=self.on_show_tree,
        )

        self.add_command(
            label="Show circular map",
            image=self.otroot.icons["tree"],
            compound="left",
            command=self.on_show_circular,
        )
        self.add_command(
            label="Show status map",
            image=self.otroot.icons["tree"],
            compound="left",
            command=self.on_show_status,
        )
        self.add_command(
            label="Toggle verbose log",
            compound="left",
            command=self.on_toggle_verbose,
            image=self.otroot.icons["plus"],
        )
        self.add_command(
            label="Show project datafile",
            compound="left",
            command=self.open_dataset_project,
            image=self.otroot.icons["plus"],
        )
        self.add_command(
            label="Show dataset_to_gui",
            compound="left",
            command=self.open_dataset_to_gui,
            image=self.otroot.icons["plus"],
        )
        self.add_command(
            label="Show gui_to-dataset",
            compound="left",
            command=self.open_dataset_from_gui,
            image=self.otroot.icons["plus"],
        )
        self.add_command(
            label="Show evaluate_status",
            compound="left",
            command=self.show_monitor_evaluate_status,
            image=self.otroot.icons["plus"],
        )
    def _bind_items(self):
        self.root.bind("<Control-h>", self.on_show_tree)

    def on_toggle_verbose(self, event=None):
        """Toggle verbose mode in terminal"""
        toggle_verbose()

    def open_dataset_project(self, event=None):
        """Open current project"""
        try_to_open_file(self.otroot.data_file)

    def open_dataset_from_gui(self, event=None):
        """Open current data send to processes"""
        try_to_open_file(".dataset_from_gui.yml")

    def open_dataset_to_gui(self, event=None):
        """Open current data sent by to processes"""
        try_to_open_file(".dataset_to_gui.yml")

    def show_monitor_evaluate_status(self, event=None):
        """Monitoring option"""
        show_monitor(["evaluate_local_status_changing","evaluate_local_status_validate"])


    def on_show_tree(self, event=None):
        toplevel = tk.Toplevel(self.root)
        toplevel.title("Tree View")
        toplevel.transient(self.root)

        memory = tk.StringVar(value=nob_asciigraph(self.otroot.get()))

        TextConsole(toplevel, memory, search=True)

    def on_show_circular(self, event=None):
        """Show memory with nobvisual."""
        # TODO: data or project_file?

        title = f"Current memory of {self.otroot.data_file}"

        visual_treenob(self.otroot.get(), title=title)

    def on_show_status(self, event=None):
        """Show memory with nobvisual."""
        # TODO: data or project_file?

        def color_status(status):
            if status == None:
                return "grey"
            elif status == -1:
                return "red"
            elif status == 0:
                return "gold"
            elif status == 1:
                return "forestgreen"
            else:
                return "pink"

        title = f"Current memory of {self.otroot.data_file}"

        dict_status = [
            {
                "id": "root",
                "text": f"root \n{type(self.otroot)}",
                "color": color_status(self.otroot._status),
                "children": [],
                "datum": 1.0,
            }
        ]

        def rec_status(node, dict_status_holder):
            if len(node.children.values()) == 1:
                dict_status_holder.append(
                    {
                        "id": f"{node.name}.self",
                        "text": f"{node.name}|{node.title}.self\n{type(node)}\nstatus: {node._status}",
                        "color": color_status(node._status),
                        "children": [],
                        "datum": 0.02,
                    }
                )

            for child in node.children.values():
                dict_status_holder.append(
                    {
                        "id": child.name,
                        "text": f"{child.name}|{child.title}\n{child.ottype}({child.kind})\nstatus: {child._status}",
                        "color": color_status(child._status),
                        "children": [],
                        "datum": 1.0,
                    }
                )
                rec_status(child, dict_status_holder[-1]["children"])

            if node.ottype == "OTDynamicList":
                for i, child in enumerate(node.list_variables):
                    dict_status_holder.append(
                        {
                            "id": child.name,
                            "text": f"dynlist item {i}\n{child.ottype}({child.kind})\nstatus: {child._status}",
                            "color": color_status(child._status),
                            "children": [],
                            "datum": 1.0,
                        }
                    )

        legend = [
            ("None (unset)", color_status(None)),
            ("-1 (invalid)", color_status(-1)),
            ("0 (unknown)", color_status(0)),
            ("1 (valid)", color_status(1)),
            ("? (unexpected)", color_status(666)),
        ]
        rec_status(self.otroot.root_tab, dict_status[0]["children"])
        nobvisual(dict_status, title=title, legend=legend)


class HelpMenu(_Menu):
    def __init__(self, otroot, label="Help", **kwargs):
        super().__init__(otroot, label, **kwargs)

    def _add_items(self):
        self.add_command(
            label="About",
            image=self.otroot.icons["about"],
            compound="left",
            command=self.on_about,
        )

    def on_about(self):
        toplevel = tk.Toplevel(self.root)
        toplevel.title("About")
        toplevel.transient(self.root)

        memory = tk.StringVar(value=ABOUT)

        TextConsole(toplevel, memory)


def try_to_open_file(file):
    commands = ["code", file]
    try:
        subprocess.run(commands)
    except FileNotFoundError:
        logger.warning(f"Could not execute shell command {commands} ")
