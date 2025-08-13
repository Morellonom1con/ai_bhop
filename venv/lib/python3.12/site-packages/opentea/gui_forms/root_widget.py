"""
The root widget
===============

The root widget is the fisrt called by otinker.
It corresponds to the firts node ot the SCHEMA.

At the root widget, the whole window is created.
The root host the main Tab-notebook,
and if necessary the wiew 3D tab.

Tabs callbacks:
===============

As tabs callbacks can change any part of the memory,
These callbacks are passed down to the root widget,
trigerring two subprocesses.

Execute:
--------

Execute is about memory modification.
The callback associated shows the following singature:

nested object > callback > nested object

Update_3d_view:
---------------

Update_3d_view is about 3D feedback.
The callback associated shows the following singature:

nested object, 3D scene > callback > 3D scene

"""

from __future__ import annotations
import abc
import sys, io
from copy import deepcopy
from tkinter import ttk, Tk
from tkinter.scrolledtext import ScrolledText

from loguru import logger
import yaml, json

from tiny_3d_engine.engine import Engine3D

from opentea.gui_forms.constants import (
    PARAMS,
    load_icons,
    set_system,
    config_style,
)


from opentea.gui_forms.node_widgets import (
    OTNodeWidget,
    OTTabWidget,
)
from opentea.gui_forms._exceptions import SetException
from opentea.gui_forms.menus import DefaultMenubar
#from opentea.gui_forms.generic_widgets import TextRedirector

from opentea.gui_forms.soundboard import play_door
from opentea.gui_forms.acquisition2d import InteractivelineDrawer
from opentea.gui_forms.viewer2d import Viewer2D


class OTRoot:
    def __init__(
        self,
        schema: dict,
        tksession: Tk,
        style: str,
        data_file: str,
        tab_3d: callable = None,
        tab_2d: callable = None,
        acq_2d: dict = None,
        paraview_exec: str = None,
    ):
        # TODO: clear tmp_dir and delete at the end (.tmp?)
        # Compatibility with OOTTreeWidget
        self.name = "root"
        self.title = "root"

        self.my_root_tab_widget = None  # See OTTreeElement to understand this one (ADN)
        
        self.processes_callback = {}

        remove_props = []

        tab2d_generator = None
        for prop_key, data in schema["properties"].items():
            if "process" in data:
                self.processes_callback[prop_key] = data["process"]
            if "ot_type" in data:
                remove_props.append(prop_key)
                if data["ot_type"] == "view2d":
                    tab2d_generator = data
        
        for prop_key in remove_props:
            schema["properties"].pop(prop_key)
        
        self.schema = schema
        self.tksession = tksession
        self.data_file = None  # The current data file to store the project

        #########

        # ADN : Todo, remove this horror!
        self._status_temp = 0
        self._status_invalid = 0  # Due to _update_parent_status()
        self._status = 0

        # Configuration of appearance
        self.global_config(style)
        self._menubar = DefaultMenubar(self)
        self._menubar.activate()
        play_door()
        # ===========================

        self.root_tab = RootTabWidget(self.schema, self)

        if tab_3d not in [None, False, True]:
            self.root_tab.view3d = add_viewer_3d(
                self, callback_3d=tab_3d, paraview_exec=paraview_exec
            )
        #if tab_2d not in [None, False, True]:
        if tab2d_generator is not None:
            self.root_tab.view2d = add_viewer_2d(
                self, 
                callback_2d=tab2d_generator["process"],
                title=tab2d_generator.get("title","View2d"),
                controls=tab2d_generator.get("ot_2d_controls",None)
            )
        self.acq_2d = False
        if acq_2d not in [
            None,
        ]:
            self.acq_2d = True
            self.root_tab.acq2d = add_acquisition_2d(self, acq2d_options=acq_2d)
            

        self.load_project(data_file)

    @property
    def ottype(self) -> int:
        """Return Opentea  Object type

        Used for debugging or fancy viz.
        """
        return str(type(self)).split(".")[-1].split("'")[0]

    @property
    def properties(self):
        # TODO: check if this is required
        return self.schema.get("properties", {})

    def global_config(self, style):
        """Main configurations for root widget"""
        self.icons = load_icons()
        set_system()
        config_style(style)
        self.tksession.columnconfigure(0, weight=1)
        self.tksession.rowconfigure(0, weight=1)

        self.paned_window = ttk.PanedWindow(self.tksession, orient="vertical")
        self.paned_window.pack(fill="both", expand=True)

        self.frame = ttk.Frame(self.tksession)
        self.console_frame = ttk.Frame(self.tksession)
        self.paned_window.add(self.frame , weight=6)
        self.paned_window.add(self.console_frame, weight=1)

        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill="both", side="top", padx=2, pady=3, expand=True)
        self._setup_console()

    def _setup_console(self):
        """Setup the console widget and redirect stdout/stderr"""
        # Create console frame
        #self.console_frame = ttk.Frame(self.frame)
        #self.console_frame.pack(fill="both", padx=2, pady=3, expand=True)
        
        # Create and configure the Text widget
        self.console = ScrolledText(self.console_frame, height=6, wrap='word')
        self.console.pack(fill="both", expand=True, padx=2, pady=2)
        self.console.configure(state="disabled", background="#222222", foreground="#CCCCCC")
        
        # Configure tags for different types of output
        self.console.tag_configure("stdout", foreground="white")
        self.console.tag_configure("stderr", foreground="#FF3333")
        self.console.tag_configure("grey", foreground="#AAAAAA")
        self.console.tag_configure("green", foreground="#55FF55")
        self.console.tag_configure("redwhite", foreground="white", background="red")
        self.console.tag_configure("orange", foreground="orange")
        self.console.tag_configure("blue", foreground="#4444FF")
        self.console.tag_configure("purple", foreground="#F87BFC")
        
        
        # Copy std_output and error in window
        sys.stdout = self.TextRedirector(self.console, "stdout",original_stream=sys.stdout)
        sys.stderr =  self.TextRedirector(self.console, "stderr",original_stream=sys.stderr)

        #Redirect stdout and stderr
        # self.loguru_redirector_info = self.TextRedirector(self.console, "loguru",original_stream=sys.stdout)        
        logger.remove()  # Supprimer les gestionnaires par défaut
        log_format = "{level} {message}"
        #logger.add(self.loguru_redirector_info.write, level="INFO", format=log_format)
        logger.add(sys.stdout, level="INFO", format=log_format)
        


    class TextRedirector(io.TextIOBase):
        def __init__(self, widget, tag, original_stream=None):
            self.widget = widget
            self.tag = tag
            self.msg_nb = 0
            self.msg_nb_max = 1000
            self.msg_max_size = 30
            
            self.original_stream = original_stream
            self._buffer = ""
        
        def write(self, msg):


            # The terminal takes always the full lenght of the message
            self.original_stream.write(msg)  
            self.original_stream.flush()

            if msg.count('\n') > self.msg_max_size:
                self._buffer += "-Shortened, check terminal for complete version-\n(...)\n"+ "\n".join( msg.split("\n")[-self.msg_max_size:])
            else:
                self._buffer += msg

            while '\n' in self._buffer:
                line, self._buffer = self._buffer.split('\n', 1)
                self._write_line(line + '\n')  # Include the newline back for formatting
        
        def _write_line(self, line):
            tags =  [self.tag,]
            #               012345
            if line[:5] ==  "DEBUG":
                tags.append("blue")
            # if msg[:4] ==  "INFO":
            #     pass
            #               01234567
            elif line[:7] ==  "WARNING":
                tags.append("orange")
            #               012345678
            elif line[:8] ==  "CRITICAL":
                tags.append("redwhite")
            #               012345678
            elif line[:7] ==  "SUCCESS":
                tags.append("green")
            
            #               012345678
            elif line[:5] ==  "ERROR":
                tags.append("purple")

            self.widget.configure(state="normal")   
            
            # purge begining if limit reached
            if self.msg_nb < self.msg_nb_max:
                self.msg_nb  +=1
            else: 
                self.widget.delete("1.0", "2.0")
            
            self.widget.insert("end", line, tags)
            self.widget.see("end")  # Auto-scroll to the bottom
            self.widget.configure(state="disabled")
            self.widget.update() # Necessaire pour une MAJ du texte au fil de l'eau
        
        def flush(self):
            if self._buffer:
                self._write_line(self._buffer)
                self._buffer = ""
                self.original_stream.flush()
            
    def mainloop(self):
        """Start the mainloop

        usefull when testing to NOT start the mainloop
        """
        self.tksession.mainloop()

    def get(self):
        """How Opentea Save the project"""
        state = self.root_tab.get()
        if self.acq_2d:
            data = self.root_tab.acq2d.get()
            state.update({"acquisition": data})
            logger.warning("Saving acquisition")
            # print(json.dumps(data, indent=4))
        return state

    def set(self, data):
        """How Opentea Update the project""" 

        if self.acq_2d:
            if data is not None:
                if "acquisition" in data:
                    self.root_tab.acq2d.set(data["acquisition"])
                    del data["acquisition"]
                self.root_tab.acq2d.allow_recalibration()
                
        return self.root_tab.set(data)

    def save_project(self):
        """How Opentea Save the project"""

        logger.info(f"Saving project in {self.data_file}")
        # Ensure the title correspond to the last saved file
        self.tksession.title(self.data_file)
       
        data = self.get()
        with open(self.data_file, "w") as fout:
            yaml.safe_dump(data, fout)

    def load_project(self, data_file):
        """How Opentea load the project"""
        logger.info(f"Loading project {data_file}")

        if data_file is None:
            logger.warning("Datafile is none")
            return

        self.data_file = data_file
        with open(data_file, "r") as fin:
            state = yaml.safe_load(fin)
        self.tksession.title(data_file)
        try:
            self.set(state)
        except SetException:
            logger.exception("SetException occured. Project was not loaded properly!")


    # To behave like an ottree elements
    def ping(self, stack=False):
        logger.warning("PING **** root ****")


    def add_child(self, child: RootTabWidget):
        """Necessary to behave like an OTTreeElement

        Called when creating the child RootTabWidget

        Because "
        When you create the element, it adds itself to its parent familly
        self.children[child.name] = child"

        """
        pass
        # indeed no need to update this
        # ADN : really I hate when OO forces you to add void methods
        #  to make it work
        # self.root_tab = child


class RootTabWidget(OTNodeWidget, metaclass=abc.ABCMeta):
    def __init__(self, schema: dict, parent: OTRoot):
        self.title = "RootTabWidget"
        self.view3d = None
        self.view2d = None

        super().__init__(schema, parent, "RootTabWidget")

        self.my_root_tab_widget = self
        self._config_frame()

        # specific attributes to handle dependents
        self._global_dependents = dict()
        self._dependents = self._global_dependents
        self._xor_dependents = dict()
        self._xor_level = 0
        # self._dependent_names = set()

        self._initialize_tabs()

    #########################################
    # Dependencies with nested XOR
    # Lots to unpack and comment
    def prepare_to_receive_xor_dependents(self):
        self._xor_level += 1
        self._dependents = self._xor_dependents

    def assign_xor_dependents(self):
        self._xor_level -= 1

        if self._xor_level == 0:
            self.assign_dependents()
            self._dependents = self._global_dependents

    def add_dependency(self, master_name, slave):
        """Include a reactive dependency of one widget to the other

        If node1 have an ot_require for node2,
        node1 slave is added to node2 slave list, and node2 is the master.
        """
        try:
            self._dependents[master_name].append(slave)
        except KeyError:
            self._dependents[master_name] = [slave]

    def assign_dependents(self):
        # find by name and add dependency
        for master_name, slaves in self._dependents.items():
            master = self.get_child_by_name(master_name)
            if master is None:
                msg = f"Dependency error, -{master_name}- was not found in your Schema"
                raise RuntimeError(msg)
            master.add_dependents(slaves)

        # reset dependents
        self._dependents.clear()

    ############################################

    ############################
    # ADN NEEDED TO REDEFINE
    # @property
    # def status(self):
    #     return self._get_status()

    # @status.setter
    # def status(self, status):
    #     if status == self._status:
    #         return
    #     self._status = status
    ###################################

    def _config_frame(self):
        """Configuration of the present widget"""
        self.frame = ttk.Frame(self.parent.frame)
        self.parent.notebook.add(self.frame, text=self.title)
        self.notebook = ttk.Notebook(self.frame, name="tab_nb")
        self.notebook.pack(fill="both", padx=2, pady=3, expand=True)

   
    def _initialize_tabs(self):
        """Addition of child tabs"""
        for tab_name, tab_obj in self.properties.items():
            OTTabWidget(tab_obj, self, tab_name)  # goes to children when creating tab
        self.assign_dependents()
        # self.validate()

    def _get_validated(self):
        return {tab.name: tab.status for tab in self.children.values()}

    def get(self) -> dict:
        """Add the metavidget setter to the basic get"""
        data_ = super().get()
        return data_

    def set(self, data):
        """Add the metavidget setter to the basic set"""

        data_ = deepcopy(data)
        # self.validate()
        #self.update_status_successors()
        if data_ is  None or data_ == {}:
            logger.warning("No data loaded")
            self.evaluate_status_descending(changing=True)
        else:
            super().set(data_, first_time=True)
            self.evaluate_status_descending()

        self.refresh_status_display_descending()
        


# ====================================================================
# Viewers
# ====================================================================


def add_viewer_2d(otroot: OTRoot, callback_2d: callable, title:str = "View2d", controls:list=None):
    """Injection of a viewer 2D to opentea"""
    wid_name = title.lower().replace(" ","_")
    view2d_fr = ttk.Frame(otroot.notebook, name=wid_name)
    otroot.notebook.add(view2d_fr, text=title)
    viewer = Viewer2D(view2d_fr, otroot, callback_2d=callback_2d, controls=controls)
    return viewer


def add_viewer_3d(otroot: OTRoot, callback_3d: callable, paraview_exec: str):
    title = "3D view"
    view3d_fr = ttk.Frame(otroot.notebook, name=title)
    otroot.notebook.add(view3d_fr, text=title)
    viewer = Viewer3D(
        view3d_fr, otroot, callback_3d=callback_3d, paraview_exec=paraview_exec
    )
    return viewer


def add_acquisition_2d(otroot: OTRoot, acq2d_options: dict):
    title = "2D acq"
    view2da_fr = ttk.Frame(otroot.notebook, name=title)
    otroot.notebook.add(view2da_fr, text=title)
    viewer = InteractivelineDrawer(view2da_fr, otroot, acq2d_options=acq2d_options)
    return viewer


class Viewer3D(Engine3D):
    def __init__(
        self,
        master: ttk.Frame,
        otroot: OTRoot,
        callback_3d: callable,
        paraview_exec: str = None,
    ):
        super().__init__(
            root=master, width=1000, height=550, background=PARAMS["bg_dark"]
        )
        self.otroot = otroot
        self.callback_3d = callback_3d
        self.paraview_exec = paraview_exec
        # _header_frame = ttk.Frame(master)
        # _header_frame.pack(side="top", fill="both", padx=2, pady=3)
        
        refresh3d = ttk.Button(
            self.screen.control_panel, text="Refresh", command=self.refresh_3d_view, width=7
        )
        refresh3d.pack(side="top")
        if self.paraview_exec is not None:
            open_para = ttk.Button(
                self.screen.control_panel, text="Paraview", command=self.open_in_paraview, width=7
            )
            open_para.pack(side="top")

    def refresh_3d_view(self):
        new_scene = self.callback_3d(self.otroot.get())
        self.clear()
        self.update(new_scene)
        self.render()

    def open_in_paraview(self):
        import subprocess

        scene = self.callback_3d(self.otroot.get())
        scene.del_part("axis")  # No need to show axes in paraview
        scene.dump("scene")
        ensight_case_file = "./scene.case"
        try:
            # Build the command to run ParaView
            command = [self.paraview_exec, ensight_case_file]
            # Use subprocess to open ParaView
            subprocess.Popen(command)
            print(f"Opened ParaView with file: {ensight_case_file}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to open ParaView: {e}")
        except FileNotFoundError:
            print("ParaView executable not found. Check the path.")
        except Exception as e:
            print(f"An error occurred: {e}")
