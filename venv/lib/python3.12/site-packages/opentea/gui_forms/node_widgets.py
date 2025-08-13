"""
Recursive edicrection according to SCHEMA type
==============================================
This module staets with the recursive redirection
according to SCHEMA types.

Module for containers widgets
=============================

This module take care of all the node elements of the graph,
which correspond to containers in the Form.
At least the three first level of the SCHEMA must be objects,
 and are treated as containers.
 Oll the containers derive from the generic Node-Widget.

The root level:
---------------

The inital node, Not treated here, see root_widget.

The Tab level:
--------------

The second node level. Treated here, see Tabs_widgets.
This one can support two types of callbacks,
either for editing the memory,
or for updating the 3D view.

The Block level :
-----------------

The Thirl level gather parameters into families.
This one can support descriptions, images and documentation.


Special container : The Multiple widget :
-----------------------------------------

This container correspont to an SCHEMA array of objet items.
See it as a list (free or dependent) of similar containers.

Mutliple can be use freely or with a dependency.
In the last case, the items of the multiple is linked to the value
of a string array somewhere else in the SCHEMA.
For example, in CFD, This is useful for a
set of boundary conditions, found by reading a MESH.


Special container : the XOR widget :
------------------------------------

This is merely a selector between several blocks.
This marks a bifurcation in your graph.
For example in CFD, this is usefull for the selection between
different models taking different parameters.

:warning exception here: XOR is the only node element storing a Value.
The value is implicit : it is *the name of the only child
of the XOR in the memory*.

It could have been designed otherwise, keeping all the children inthe memory,
and a real leaf value to know which one must be taken.
However, thos is impractical here. For example in CFD,
you can have a Multiple of 100 boundary conditions,
with a XOR for each selecting between 30 models of BCs.
Such a graph would be a hassle to read and hack for humans.

"""

import os, sys
import warnings
import copy
import time
import subprocess
from typing import List
from loguru import logger
import colorama as clrm
import tkinter as tk
from tkinter import (
    ttk,
    Menu,
    Entry,
    Frame,
    StringVar,
    LEFT,
    filedialog,
    messagebox,
)

#import line_profiler

import yaml

from nob import Nob

from opentea.noob.noob import nob_pprint

from opentea.gui_forms._base import OTTreeElement, call_stack_str
from opentea.gui_forms._exceptions import (
    GetException,
    SetException,
)
from opentea.gui_forms.constants import IMAGE_DICT, WIDTH_UNIT, PARAMS, get_status_icon, load_and_run_process
from opentea.gui_forms.generic_widgets import (
    SwitchForm,
    MouseScrollableFrame,
)
from opentea.gui_forms.utils import (
    create_description,
    create_image,
    create_documentation,
    is_forgotten_frame,
)
from opentea.gui_forms.leaf_widgets import (
    OTNumericEntry,
    OTEntry,
    OTEmpty,
    OTDynamicList,
    OTStaticList,
    OTChoice,
    OTComment,
    OTBoolean,
    OTFileBrowser,
    OTDocu,
    OTDescription,
    _DeadLeafWidget,
)

from opentea.gui_forms.soundboard import (
    play_door,
    play_pain,
    play_door2,
    play_door2cls,
    play_item,
    play_oof,
)

# TODO: try to verify setting of unused fields
# TODO: read documentation from file?


def redirect_widgets(schema, parent, name, root_frame):
    """Redirect to widgets.

    The schema attributes trigger which widget will be in use.

    Inputs :
    --------
    schema :  a schema object
    root_frame :  a Tk object were the widget will be grafted
    name : name of the element

    Outputs :
    --------
    none
    """
    schema = schema or {}
    out = None

    if "properties" in schema:
        out = OTContainerWidget(schema, parent, name, root_frame)

    elif "oneOf" in schema:
        out = OTXorWidget(schema, parent, name, root_frame)

    elif "enum" in schema or "ot_dyn_choice" in schema:
        out = OTChoice(schema, parent, name, root_frame)

    elif "type" in schema:
        if schema["type"] == "array":
            if "properties" in schema["items"]:
                out = OTMultipleWidget(schema, parent, name, root_frame)
            else:
                state = schema.get("state", "normal")

                if state == "disabled" or "ot_require" in schema:
                    out = OTStaticList(schema, parent, name, root_frame)
                else:
                    out = OTDynamicList(schema, parent, name, root_frame)

        elif schema["type"] == "integer":
            out = OTNumericEntry(schema, parent, name, root_frame)

        elif schema["type"] == "number":
            out = OTNumericEntry(schema, parent, name, root_frame)

        elif schema["type"] == "boolean":
            out = OTBoolean(schema, parent, name, root_frame)

        elif schema["type"] == "string":
            out = redirect_string(schema, parent, name, root_frame)

    if out is None:
        # it will fail later with meaningful error
        out = OTEmpty(schema, parent, name, root_frame)

    return out


def redirect_string(schema, parent, name, root_frame):
    """Redirect to string widgets.

    The schema attributes trigger which string widget will be in use.

    Inputs :
    --------
    schema :  a schema object
    root_frame :  a Tk object were the widget will be grafted
    name : name of the element

    Outputs :
    --------
    none
    """
    ot_type = schema.get("ot_type", "string")

    str2obj = {
        "string": OTEntry,
        "void": OTEmpty,
        "comment": OTComment,
        "file": OTFileBrowser,
        "folder": OTFileBrowser,
        "hidden": _DeadLeafWidget,
        "desc": OTDescription,
        "docu": OTDocu,
    }

    # deal with deprecations
    deprecated = ["desc", "docu"]
    if ot_type in deprecated:
        alternatives = ["description", "documentation"]

        wng = f" at: {name}> attribute"
        wng += f"\n ot_type : {ot_type} is deprecated"
        wng += (
            f"\n prefer {alternatives[deprecated.index(ot_type)]} attribute on blocks"
        )
        warnings.warn(wng, DeprecationWarning)

    constructor = str2obj.get(ot_type, None)

    if constructor is None:
        raise NotImplementedError(f"At node {name} cannot resolve ot_type={ot_type}")

    return constructor(schema, parent, name, root_frame)


####################################
#  Base node  : No Tk !
class OTNodeWidget(OTTreeElement):
    """Main base class for node widgets"""

    def __init__(self, schema, parent, name):
        super().__init__(schema, parent, name)
        self.kind = "node"

    def get(self):
        """Get the data of children widgets.

        Returns :
        ---------
        a dictionary with the get result of childrens
        """
        out = {}
        for child in self.children.values():
            try:
                out[child.name] = child.get()
            except GetException:
                pass

        return out

    def set(self, dict_, first_time: bool = False):
        """Get the data of children widgets.

        Input :
        -------
        a dictionary with the value of the childrens
        """
        # !!! Must not skip if equal, because of dependent widgets (dynamic choices)
        # if dict_ == self.get():
        #     return
        for child in self.properties:
            try:
                if child in dict_:
                    # logger.warning(f"   - Try on child {child}")
                    try:
                        self.children[child].set(dict_[child], first_time=first_time)
                    except SetException as e:
                        logger.exception("SetException encountered")
                        self.children[child].ping()
            except TypeError as e:
                logger.exception("TypeError encountered")
                self.children[child].ping()

    def destroy(self):
        """How to remove a widget"""
        # TODO: need to homogenize holder, verify destroy works with all
        for child in self.children.values():
            child.destroy()
        self._holder.destroy()

    #  Base node
    ####################################


class OTContainerWidget(OTNodeWidget):
    """OT container widget."""

    def __init__(
        self,
        schema,
        parent,
        name,
        root_frame,
        n_width=1,
        relief="ridge",
        show_title=True,
    ):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested object
        root_frame :  a Tk object were the widget will be grafted
        name: string naming the widget
        n_width : float
             relative size of the widget

        """
        super().__init__(schema, parent, name)
        self._tab = None
        self.n_width = n_width
        self.relief = relief
        self.show_title = show_title
        self._create_widgets(root_frame)

        # create children
        for name_child in self.properties:
            schm_child = self.properties[name_child]
            redirect_widgets(schm_child, self, name_child, self.body)

        self._status = 1

    @property
    def tab(self):
        if self._tab is None:
            self._tab = self.parent
            while not isinstance(self._tab, OTTabWidget):
                self._tab = self._tab.parent

        return self._tab

    def _create_widgets(self, root_frame):
        title = self.schema.get("title", "")

        self._holder = ttk.Frame(
            root_frame, relief=self.relief, width=self.n_width * WIDTH_UNIT
        )
        self._holder.pack(side="top", padx=0, pady=10)

        if self.show_title and title:
            self.head = ttk.Label(self._holder, text=title)
            self.head.pack(side="top", fill="x", padx=2, pady=5)
        self.body = ttk.Frame(self._holder, width=self.n_width * WIDTH_UNIT)

        """Forcing the widget size"""
        self._forceps = ttk.Frame(
            self._holder, width=self.n_width * WIDTH_UNIT, height=1
        )
        self._forceps = ttk.Frame(self._holder, width=WIDTH_UNIT, height=1)
        self._forceps.pack(side="top", padx=2, pady=2)

        self.expert = self.schema.get("expert", False)
        self.packed = True

        if self.expert:
            self.packed = False
            self.head.configure(compound=LEFT, image=IMAGE_DICT["plus"])
            self.head.bind("<Button-1>", self.pack_unpack_body)

        if self.packed:
            self.body.pack(side="top", fill="x", expand=False, padx=2, pady=2)

        self._img, self._docu, self._desc = _create_extra_widgets(self, self.body)

    def pack_unpack_body(self, event):
        """switch on or off the packing of the body"""
        # TODO: review
        if self.packed:
            self.packed = False
            self.body.pack_forget()
            self.head.configure(compound=LEFT, image=IMAGE_DICT["plus"])
            play_door2cls()
        else:
            self.packed = True
            self.body.pack(
                side="top",
                fill="x",
                expand=False,
                padx=2,
                pady=2,
            )
            self.head.configure(compound=LEFT, image=IMAGE_DICT["minus"])
            play_door2()

        PARAMS["top"].update_idletasks()
        self.tab.smartpacker()

    def sleep(self):
        self.kind = "unpacked_node"
        self._holder.forget()

    def awake(self):
        self.kind = "node"
        self._holder.pack()
        return self


class OTTabWidget(OTNodeWidget):
    """OT Tab widget container.

    Called for the 1st layer of nodes in the global schema
    """

    def __init__(self, schema, parent, name):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested object
        parent :  the parent
        name: string naming the widget
        """
        super().__init__(schema, parent, name)
        self.title = self.schema.get("title", f"#{self.name}")
        # TODO: need to have get status icon
        self._status = 0
        self._status_icon = None
        self._create_widgets()
        self._config_button()

        # create children
        # TODO: is this general?
        for name_ in self.schema["properties"]:
            redirect_widgets(self.schema["properties"][name_], self, name_, self.holder)

        self.holder.bind("<Configure>", self.smartpacker)

    def _create_widgets(self):
        self.frame = ttk.Frame(self.parent.notebook, name=self.name)
        self.parent.notebook.add(self.frame, text=self.title)
        # scroll form
        sframe = MouseScrollableFrame(self.frame)
        self.scan = sframe.canvas
        self.holder = ttk.Frame(self.scan)
        self.scan.create_window((0, 0), window=self.holder, anchor="nw")

        # footer
        _footer_f = ttk.Frame(self.frame)
        _footer_f.pack(side="top", fill="both", padx=2, pady=3)
        self.footer_text = StringVar()
        self.footer_lb = ttk.Label(
            _footer_f, textvariable=self.footer_text, wraplength=2 * WIDTH_UNIT
        )

        self.parent.notebook.tab(
            self.tab_id, image=IMAGE_DICT["unknown"], compound="left"
        )

    def _config_button(self):
        """Configure the button according to callback presence"""
        self.process = self.schema.get("process", None)
        self._process_success = True

        txt_btn = "Validate"
        if self.process is not None:
            txt_btn = "Process"

        footer_frame = self.footer_lb.master
        _button_bt = ttk.Button(footer_frame, text=txt_btn, command=self.process_button)
        _button_bt.pack(side="right", padx=2, pady=2)
        self.footer_lb.pack(side="right", padx=2, pady=2)
        if "description" in self.schema:
            self._desc = create_description(
                footer_frame, self.schema["description"], size=1.0, side="left"
            )

    def refresh_status_display(self):
        """What to do when the status is changing"""
        #logger.critical(f"Here update Tab {self.title} {self._status}")
        new_status_icon = get_status_icon(self._status)
        if self._status_icon != new_status_icon:
            self.parent.notebook.tab(
                self.tab_id, image=IMAGE_DICT[new_status_icon], compound="left"
            )
            self._status_icon = new_status_icon

    @property
    def tab_id(self):
        for i, frame in enumerate(self.parent.notebook.winfo_children()):
            if frame.winfo_name() == self.name:
                return i

        return None

    def smartpacker(self, event=None):
        """Smart grid upon widget size.

        Regrid the object according to the width of the window
        from the inside
        """
        # TODO: needs to be reviewed
        self.scan.configure(scrollregion=self.scan.bbox("all"))
        ncols = max(int(self.parent.notebook.winfo_width() / WIDTH_UNIT + 0.5), 1)

        large_children = list()
        normal_children = list()
        for child in self.holder.winfo_children():
            # do not consider hidden children
            if is_forgotten_frame(child):
                continue

            if child.winfo_width() > 1.1 * WIDTH_UNIT:
                large_children.append(child)
            else:
                normal_children.append(child)

        height = 0
        x_pos = 10
        y_pos = 10

        max_depth = y_pos

        # Normal children
        max_width = WIDTH_UNIT
        for child in normal_children:
            height += child.winfo_height() + 2
            max_width = max(max_width, child.winfo_width())

        limit_depth = height / ncols

        for child in normal_children:
            child.place(x=x_pos, y=y_pos, anchor="nw")
            y_pos += child.winfo_height() + 10

            max_depth = max(y_pos, max_depth)
            # jump to next column il multicolumn
            if ncols > 1 and y_pos > limit_depth:
                x_pos += max_width + 20
                y_pos = 10

        # Large children
        x_pos = 0
        y_pos = max_depth
        for child in large_children:
            height = child.winfo_height()
            child.place(x=x_pos, y=y_pos, anchor="nw")
            y_pos += height + 2
        max_depth = y_pos

        self.holder.configure(
            height=max_depth + 200, width=ncols * (max_width + 20) + 20
        )

    def process_button(self):
        """Process the main tab button."""
        start = time.time()
        
        #self.update_status(changing=True)
        self._process_status = None # needed to make sure evaluation status is taking process into account
        # todo, maybe move the addtitional evaluation process here
        self.evaluate_status_descending(changing=False)
        self.refresh_status_display_descending()

        if self._status == -1:
            self.footer_text.set("Cannot process with errors in tabs")
            play_oof()
            return
        
        self.footer_text.set(f"Processing...")
        footer_text=""
        try:
            PARAMS["top"].config(cursor="wait")
            PARAMS["top"].update()
        except tk.TclError:
            pass
        
        self.parent.parent.save_project()  # save is in otroot

        if self.process:
            success, duration, returnstr = self.execute(self.process)
            if success:
                footer_text = f"Process done in {duration}"
                play_door()
                self._process_status = 1
                #self.update_status_successors()
                self.evaluate_status_descending()
                self.refresh_status_display_descending()
                # self.evaluate_status_ascending()
                # self.refresh_status_display_ascending()
            else:
                footer_text = f"Process failed -{returnstr}- {duration}"
                play_pain()
                # self.update_status()
                self._process_status = -1
                # self.evaluate_status_ascending()
                # self.refresh_status_display_ascending()
        else:
            pass
            play_item()
            #self.update_status_successors()
        self.evaluate_status_ascending()
        self.refresh_status_display_ascending()

        try:
            PARAMS["top"].config(cursor="")
            PARAMS["top"].update()
        except tk.TclError:
            pass
        end = time.time()
        duration = f"{end - start:.2f}s"
        footer_text += f"(All {duration})"
        self.footer_text.set(footer_text)

    def execute(self, script):
        """execute a script"""
        # TODO: execute via import instead of script?

        full_script = os.path.join(PARAMS["calling_dir"], script)
        logger.info("Executing :"+full_script)
        start = time.time()
        
        if PARAMS["verbose_log"]:
            dump = yaml.dump(self.parent.parent.get(), default_flow_style=False)
            with open(".dataset_from_gui.yml", "w") as fout:
                fout.writelines(dump)
        returnstr=""
        try:
            out_data  = load_and_run_process(script, self.parent.parent.get())
            self.parent.parent.set(out_data)

            if PARAMS["verbose_log"]:
                dump = yaml.dump(out_data, default_flow_style=False)
                with open(".dataset_to_gui.yml", "w") as fout:
                    fout.writelines(dump)

            returnstr="Process successful"
            success=True
        except Exception as e:
            logger.exception(e)
            success=False
            
        end = time.time()
        duration = f"{end - start:.2f}s"
        returnstr += " in  "+duration
        
        if success:
            logger.success(returnstr)
        else:
            logger.warning(returnstr)            

        return success, duration, returnstr


# def logwcolor(text: list) -> str:
#     returnstr = ""
#     clrm.init()

#     for line in text:
#         if "Error" in line:
#             returnstr = line

#         if "DEBUG" in line[25:35]:
#             if PARAMS["verbose_log"]:
#                 print(clrm.Back.BLACK + clrm.Fore.BLUE + clrm.Style.NORMAL + line[25:])
#         elif "INFO" in line[25:35]:
#             print(clrm.Back.BLACK + clrm.Fore.WHITE + clrm.Style.NORMAL + line[25:])
#         elif "SUCCES" in line[25:35]:
#             print(clrm.Back.BLACK + clrm.Fore.GREEN + clrm.Style.BRIGHT + line[25:])
#         elif "WARNING" in line[25:35]:
#             print(clrm.Back.BLACK + clrm.Fore.YELLOW + clrm.Style.BRIGHT + line[25:])
#         elif "CRITICAL" in line[25:35]:
#             print(clrm.Back.BLACK + clrm.Fore.RED + clrm.Style.BRIGHT + line[25:])
#         else:
#             print(clrm.Back.BLACK + clrm.Fore.CYAN + clrm.Style.NORMAL + line)
#     clrm.reinit()
#     return returnstr


# 2023-07-13 06:47:34.825 | WARNING  |
# 01234567890123456789012345678901234567


class OTMultipleWidget(OTNodeWidget):
    """OT multiple widget."""

    # TODO: break in dependent/non-dependent?
    # TODO: add min and max?
    def __init__(self, schema, parent, name, root_frame):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested object
        root_frame :  a Tk object were the widget will be grafted
        name: string naming the widget
        """
        super().__init__(schema, parent, name)
        self.item_schema = self.schema["items"]
        # force name to be in mode "hidden", in fact its more dynamic title
        self.item_schema["properties"]["name"]["ot_type"] = "hidden"
        self._clipboard = None

        self._create_widgets(root_frame)
        self._previous_order = []  # status when only ordering changed/number changed
        self.master_list = (
            None  # If not None, provided by ot_require, and limit widgets
        )
        self._status = 0

    def _create_widgets(self, root_frame):
        title = self.schema.get("title", f"#{self.name}")

        self.holder = ttk.LabelFrame(
            root_frame,
            text=title,
            name=self.name,
            relief="sunken",
            width=2 * WIDTH_UNIT,
        )
        self.holder.pack(side="top", fill="x", padx=2, pady=2, expand=False)
        forceps = ttk.Frame(self.holder, width=2.0 * WIDTH_UNIT, height=1)
        self.tvw = MultipleTreeview(self, self.holder, selectmode="extended", height=15)

        self.switchform = SwitchForm(self.holder, width=WIDTH_UNIT, name="tab_holder")

        self._config_ctrl_panel()

        # grid the main layout
        forceps.grid(column=0, row=1, columnspan=3)
        self.tvw.scrollbar_y.grid(column=1, row=1, sticky="news")
        self.tvw.grid(column=0, row=1, sticky="news")
        self.ctrls.grid(column=0, row=2, sticky="nw")
        self.switchform.grid(column=2, row=1, rowspan=2, sticky="nw")
        self.switchform.grid_propagate(0)

    def _config_ctrl_panel(self):
        self.ctrls = ttk.Frame(self.holder)
        self.ctrls.butt_load = ttk.Button(
            self.ctrls, text="load", command=self.load_from_file
        )
        self.ctrls.butt_load.pack(side="left")
        if not self.dependent:
            self.ctrls.butt_add = ttk.Button(
                self.ctrls, text="add", command=self.add_item_on_cursel
            )
            self.ctrls.butt_del = ttk.Button(
                self.ctrls, text="del", command=self.del_item_on_cursel
            )
            self.ctrls.butt_add.pack(side="left")
            self.ctrls.butt_del.pack(side="left")

            self.ctrls.butt_up = ttk.Button(
                self.ctrls, text="up", command=self.tvw.on_move_up
            )
            self.ctrls.butt_up.pack(side="left")
            self.ctrls.butt_down = ttk.Button(
                self.ctrls, text="down", command=self.tvw.on_move_down
            )
            self.ctrls.butt_down.pack(side="left")

    def check_clipboard(self):
        if self._clipboard is None:
            messagebox.showwarning(message="Nothing to paste")
            self.tvw.focus_set()
            return False

        return True

    def get(self) -> list:
        """Get the data of children widgets.

        Returns :
        ---------
        a list with the get result of childrens
        """
        # TODO: review
        out = list()
        for key in self.get_ordered_keys():
            child = self.children.get(key, None)
            if child is None:
                # happens when treeview widget exists, but not in the tree yet
                # so basically during object creation when child ask get.
                raise GetException()
            try:
                data = child.get()
                if data is not None:
                    out.append(data)
            except GetException:
                pass
        return out

    def set(self, data: list, first_time: bool = False):
        """Set the data of children widgets.

        Input :
        -------
        a list with the value of the children

        Notes:
            All the children should be passed (otherwise they'll be deleted).

            Order of existing children is not dependent on list order, new
            children are added orderly after existing.
        """
        # !!! Must not skip if equal, because of dependent widgets (dynamic choices)
        # if data == self.get():
        #     return   

        if len(data) == 0:
            return

        # limit data to master list if dependent
        if self.master_list is not None:
            pre_children = {child["name"]: child for child in data}
            children = {}
            for name in self.master_list:
                children[name] = {"name": name}
                if name in pre_children:
                    children[name] = pre_children[name]
        else:
            children = {child["name"]: child for child in data}

        # children to delete
        map_key_to_name = self.get_map_key_to_name()
        #logger.warning(f'    Mutiple Widget set data ({children.keys()}')
        for item_id in list(self.children.keys()):
            if map_key_to_name[item_id] not in children:
                # if "mul_bnd_stator" == self.name:
                #     logger.warning(f'    ---> Delete set {item_id}|{name}')
                self.del_item_by_id(item_id)

        # update existing objects and create new ones
        map_name_to_key = self.get_map_name_to_key()
        for child_name, child_data in children.items():
            if child_name in map_name_to_key:  # update existing
                # if "mul_bnd_stator" == self.name:
                #     logger.warning(f'    ---> Update {child_name}|{child_data}')
                
                self.children[map_name_to_key[child_name]].set(
                    child_data, first_time=first_time
                )
            else:  # create new item
                # if "mul_bnd_stator" == self.name:
                #     logger.warning(f'    ---> create {child_name}|{child_data}')
                item_id = self.add_new_item(child_name, child_data)

    #@line_profiler.profile
    def slave_update(self, master_list: List):
        """What to do if this multiple is the slave

        Happen whe there is an ot_require in the muliple declaration

        For the already existing children it updates the name, for the new ones
        it creates default children.

        Args:
            master_list (list of str)
        """
        self.master_list = master_list

        initial_list_names = self.get_ordered_names()
        # logger.info(f"{self.name} is slave-updated with {master_list} (before: {initial_list_names})")

        # if nothing to be done, skip
        if initial_list_names == master_list:
            # logger.info(f"            No changes, skip set slaves")
            return
        
        # save data
        keepsake = {}
        for former_id, former_name in self.get_map_key_to_name().items():
            if former_name in master_list:
                keepsake[former_name] = self.children[former_id].get()
        # we must clean an rebuild everything to make sure we apply the new order
        # clean everything
        for item_id in list(self.children.keys()):
            self.del_item_by_id(item_id)

        # rebuilt everything
        for new_name in master_list:
            item_id = self.add_new_item(new_name)
            if new_name in keepsake:
                self.children[item_id].set(keepsake[new_name], first_time=True)
        del keepsake
        #self.update_status_predecessors()
        self.evaluate_status_ascending()
        self.refresh_status_display_ascending()
        
        # self.debug_show_memory()

    def rename_item(self, item_id, new_name):
        """Rename one element of the multiple.

        Notes:
            It is not allowed to repeat names.
        """
        if new_name == self.children[item_id].name:
            pass

        elif new_name in self.get_item_names():
            messagebox.showwarning(message=f"Name {new_name} already in use")
            self.tvw.focus_set()
        else:
            self.children[item_id].rename(new_name)

    def load_from_file(self):
        """load multiple content from an other file

        For multiple without dependencies, updates common items, deletes
        items absent in new file and add unexisting items. Order of new file
        is kept.

        For multiple with dependencies, only items with the same name are
        updated.
        """
        path = filedialog.askopenfilename(
            title="Partial load from file",
            filetypes=[("YAML Files", ["*.yml", "*.yaml"])],
        )
        if path == "":
            return

        with open(path, "r") as fin:
            data_in = yaml.load(fin, Loader=yaml.FullLoader)
        nob = Nob(data_in)

        try:
            new_content = nob[self.name][:]
        except KeyError:
            messagebox.showwarning(message="Data not found in file")
            return

        # simple validation of content data
        if not isinstance(new_content, list):
            messagebox.showwarning(message="Invalid data format")
            self.tvw.focus_set()
            return

        if "ot_require" not in self.schema:
            self._load_from_file_no_deps(new_content)
        else:
            self._load_from_file_deps(new_content)

        self.tvw.focus_set()

    def _load_from_file_no_deps(self, new_content):
        new_names = [item["name"] for item in new_content]
        new_names_order = new_names.copy()

        for item in self.children.copy().values():  # avoid size change error
            if item.name not in new_names:  # delete items
                item.delete()
            else:  # update items
                index = new_names.index(item.name)
                item.set(new_content[index])
                # to make it easier to create new items
                new_names.pop(index)
                new_content.pop(index)

        # create new items
        for item_new_content in new_content:
            self.add_new_item(item_new_content["name"], item_new_content)

        # update order
        same_order = self.is_sorted_equally(new_names_order)
        if not same_order:
            self.reorder_by_names(new_names_order)

    def _load_from_file_deps(self, new_content):
        new_names = [item["name"] for item in new_content]
        for item in self.children.values():
            if item.name in new_names:
                index = new_names.index(item.name)
                #logger.warning(f"Setting {item.name } with {new_content[index]}")
                item.set(new_content[index])

    def add_item_on_cursel(self):
        """Add an item in the multiple.

        Item will be added after the current selection, otherwise end
        """
        id_cursel = (
            self.tvw.index(self.tvw.selection()[-1]) + 1
            if self.tvw.selection()
            else "end"
        )

        # create a new item with default value
        name = self._get_item_default_name()
        item_id = self.add_new_item(name, pos=id_cursel)

        self.tvw.select_item(item_id)

    def del_item_on_cursel(self):
        """Delete a Multiple item from tv selection."""
        selection = self.tvw.selection()
        if not selection:
            messagebox.showwarning(message="No item selected...")
        else:
            for id_cursel in selection:
                self.del_item_by_id(id_cursel)

    def paste_data(self, item_id):
        item = self.children[item_id]
        data = copy.deepcopy(self._clipboard)
        data["name"] = item.name
        return item.set(data)

    def _get_item_default_name(self):
        default_name = self.item_schema["properties"]["name"].get("default", "item_#")
        return self._get_unique_name(default_name)

    def _get_unique_name(self, new_name):
        while new_name in self.get_item_names():
            new_name += "#"

        return new_name

    def add_new_item(self, name, data=None, pos="end"):
        
        
        #logger.info(f"??? MultipleWidget add item {name}{data}")
        multiple_item = OTMultipleItem(self, name, pos=pos)
        if data is not None:
           multiple_item.set(data, first_time=True)

        if pos != "end":
            self.tvw.update_index_row_all()

        return multiple_item.id

    def add_child(self, multiple_item):
        self.children[multiple_item.id] = multiple_item

    def del_item_by_id(self, item_id: str):
        """Remove item from multiple.

        Args:
            item_id : internal reference like I003
        """
        # logger.info(f"-   MultipleWidget {self.name} deletes {item_id}|{self.children[item_id].name}")
        self.children[item_id].delete()
        self.tvw.update_index_row_all()
        # self.debug_show_memory()

    def debug_show_memory(self):
        logger.info(f"All elements must match:")
        logger.info(f"-  Current Treeview {self.get_item_names()}")
        logger.info(f"-  Current Children {self.get_item_names2()}")
        logger.info(f"-  Current    Get() {[i['name'] for i in self.get()]}")

    def reorder_by_names(self, new_names_order):
        name_to_key = self.get_map_name_to_key()
        for index, name in enumerate(new_names_order):
            item_id = name_to_key[name]
            self.tvw.move(item_id, self.tvw.parent(item_id), index)
        self.tvw.update_index_row_all()

    def is_sorted_equally(self, new_names):
        names = self.get_item_names()
        for name, new_name in zip(names, new_names):
            if name != new_name:
                return False
        return True

    def get_item_names(self):
        """Get item names in treeview."""
        return [self.children[item_id].name for item_id in self.tvw.get_children()]

    def get_item_names2(self):
        """Get item names in children."""
        return [self.children[item_id].name for item_id in self.children]

    def get_map_name_to_key(self):
        return {
            self.children[item_id].name: item_id for item_id in self.tvw.get_children()
        }

    def get_map_key_to_name(self):
        return {
            item_id: self.children[item_id].name for item_id in self.tvw.get_children()
        }

    def get_ordered_keys(self):
        return [item_id for item_id in self.tvw.get_children()]

    def get_ordered_names(self):
        return [self.children[item_id].name for item_id in self.tvw.get_children()]

    # def update_status(self, changing:bool=False):
    #     super().update_status()
    #     if self._status == 1:  # verify order
    #         if not self.same_item_order():
    #             self._status = 0

    def same_item_order(self):
        cur_order = self.get_item_names()

        if len(cur_order) != len(self._previous_order):
            return False

        for cur_name, previous_name in zip(cur_order, self._previous_order):
            if cur_name != previous_name:
                return False

        return True

    # def validate(self):
    #     """What to do when the element is validated

    #     Update the status according to:
    #     - the content
    #     - the children status

    #     MUST update current inner status
    #     MUST return the current inner status as an integer
    #     """

    #     self._previous_order = self.get_item_names()
    #     status = super().validate()
    #     return status

    def once_validated(self):
        self._previous_order = self.get_item_names()


class OTMultipleItem(OTContainerWidget):
    """OT  multiple widget."""

    def __init__(self, multiple, name, pos="end"):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested object
        multiple :  a Tk object were the widget will be grafted
        """
        self.id = multiple.tvw.insert("", pos, text=name)
        self.label_frame = multiple.switchform.add(self.id, title=name)
        super().__init__(multiple.item_schema, multiple, name, self.label_frame)
        self.rename(name)  # for the case it uses default values

    def update_item_in_treeview(self):
        """Correct the appearance of an item in the treevies"""
        data = self.get()
        if data is not None:
            self.parent.tvw.update_row(self.id, data, self._status)

    def rename(self, new_name):
        data = self.get()
        data["name"] = new_name
        return self.set(data, first_time=True)

    def delete(self):
        self.parent.tvw.delete(self.id)
        self.parent.switchform.sf_del(self.id)

        # self._update_parent_status(1)
        del self.parent.children[self.id]

    def set(self, new_data, first_time: bool = False):
        # !!! Must not skip if equal, because of dependent widgets (dynamic choices)
        # if new_data == self.get():
        #     return

        # logger.warning(f" ---> MultipleItem  Set {self.id}|{self.name} (first:{str(first_time)})")
        # logger.warning(call_stack_str())
        self.name = new_data["name"]
        self.label_frame.config(text=self.name)
        super().set(new_data, first_time=first_time)
        self.update_item_in_treeview()

    # def on_update_status(self):
    #     self.update_item_in_treeview()

    def refresh_status_display(self):
        """ """
        self.update_item_in_treeview()


class MultipleTreeview(ttk.Treeview):
    def __init__(self, multiple, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiple = multiple

        self._create_widgets()
        self._config_tags()
        self._config_columns()
        self._config_bindings()

    def _create_widgets(self):
        self.scrollbar_y = ttk.Scrollbar(
            self.master, orient="vertical", command=self.yview
        )
        self.configure(yscrollcommand=self.scrollbar_y.set)

        self.popup_menu = Menu(self, tearoff=False)  # binding in tv bindings

    def _config_tags(self):
        self._highlighted_tag = "highlighted"
        self.tag_configure(self._highlighted_tag, background=PARAMS["hl_bg"])

    def _config_columns(self):
        """Configure the columns apearance"""
        item_props = self.multiple.item_schema["properties"]
        all_columns = list(item_props.keys())
        show_columns = list(item_props.keys())
        self["columns"] = tuple(all_columns)
        self["displaycolumns"] = tuple(show_columns)
        self["show"] = "tree headings"
        tree_col_width = 60
        col_width = int(WIDTH_UNIT / (len(self["columns"]))) + tree_col_width
        for key, item in item_props.items():
            title = key if "title" not in item else item["title"]
            self.column(key, width=col_width)
            self.heading(key, text=title)
        self.column("#0", width=tree_col_width, anchor="c")

    def _config_bindings(self):
        self.bind("<<TreeviewSelect>>", self.on_item_selected, add="+")
        
        self.bind("<Escape>", self.on_deselect)
        self.bind("<Button-2>", self.on_popup_menu_trigger)
        self.bind("<Enter>", self._unbind_global_scroll)
        self.bind("<Leave>", self._bind_global_scroll)
        
        if sys.platform == "darwin":
            self.bind("<Command-c>", self.on_copy)
            self.bind("<Command-v>", self.on_paste_sel)
            self.bind("<Option-Button-1>", self.on_paste_click)
            self.bind("<Option-B1-Motion>", self.on_paste_click)
            self.popup_menu.add_command(label="Copy (Cmd+C)", command=self.on_copy)
            self.popup_menu.add_command(label="Paste (Cmd+V)", command=self.on_paste_sel)

        else:
            self.bind("<Control-c>", self.on_copy)
            self.bind("<Control-v>", self.on_paste_sel)
            self.bind("<Alt-Button-1>", self.on_paste_click)
            self.bind("<Alt-B1-Motion>", self.on_paste_click)
            self.popup_menu.add_command(label="Copy (Ctrl+C)", command=self.on_copy)
            self.popup_menu.add_command(label="Paste (Ctrl+V)", command=self.on_paste_sel)

        # popup bindings
       
        if "ot_require" not in self.multiple.schema:
            self.bind("<Double-1>", self.on_double_click)
            if sys.platform == "darwin":
                self.bind("<Command-u>", self.on_move_up)
                self.bind("<Command-d>", self.on_move_down)
                self.popup_menu.add_command(
                    label="Move up (Cmd+U)", command=self.on_move_up
                )
                self.popup_menu.add_command(
                    label="Move down (Cmd+D)", command=self.on_move_down
                )
            else:    
                self.bind("<Control-u>", self.on_move_up)
                self.bind("<Control-d>", self.on_move_down)
                self.popup_menu.add_command(
                    label="Move up (Ctrl+U)", command=self.on_move_up
                )
                self.popup_menu.add_command(
                    label="Move down (Ctrl+D)", command=self.on_move_down
                )
        
            

       

    def on_item_selected(self, event):
        if self.focus() and len(self.selection()) == 1:
            self.multiple.switchform.sf_raise(self.focus())
            play_item()
        else:
            self.multiple.switchform.sf_raise(None)

    def on_double_click(self, event):
        """Handle a simple click on treeview."""
        col = self.identify_column(event.x)
        if col == "#1":
            item_id = self.identify_row(event.y)
            if item_id:
                self.on_rename_first_column(item_id)

    def on_copy(self, *args):
        item_id = self.focus()
        if not item_id:
            logger.info(f"No item selected, Select with Button1")
            #messagebox.showwarning(message="No item selected")
            return

        if len(self.selection()) > 1:
            logger.info(f"Copy allowed only for single item selection. Clean selection with Esc")
            
            # messagebox.showwarning(
            #     message="Copy only allowed for one element selection"
            # )
            # self.focus_set()
            return

        data = self.multiple.children[item_id].get()
        logger.info(f"Into clipboard: \n{nob_pprint(data)}")
        self.multiple._clipboard = data

    def on_paste_sel(self, *args):
        if not self.multiple.check_clipboard():
            return
        selection = self.selection()
        if not selection:
            if sys.platform == "darwin":
                logger.info(f"No items selected. Select using Button1, Shift-Button1  or Command-Button1")
            else:
                logger.info(f"No items selected. Select using Button1, Shift-Button1  or Ctrl-Button1 ")
            #messagebox.showwarning(message="No items selected")
            #self.focus_set()
            return

        logger.info(f"Paste clipboard into {len(selection)} selected items")
        for item_id in selection:
            self.multiple.paste_data(item_id)
        if sys.platform == "darwin":
            logger.info(f"Use Option-Button1 to speedpaste without selecting first.")
        else:
            logger.info(f"Use Alt-Button1 to speedpaste without selecting first.")

    def on_paste_click(self, event):
        if not self.multiple.check_clipboard():
            return
        item_id = self.identify_row(event.y)
        if not item_id:
            return
        self.multiple.paste_data(item_id)

    def on_deselect(self, *args):
        self.selection_set("")
        self.focus("")

    def on_popup_menu_trigger(self, event):
        self.popup_menu.tk_popup(event.x_root, event.y_root)

    def on_move_up(self, *args):
        item_ids = self.selection()

        if not item_ids:
            return

        for item_id in item_ids:
            current_index = self.index(item_id)
            self.move(item_id, self.parent(item_id), current_index - 1)

        self.update_index_row_all()
        self.focus_set()

    def on_move_down(self, *args):
        item_ids = self.selection()

        if not item_ids:
            return

        for item_id in reversed(item_ids):
            current_index = self.index(item_id)
            self.move(item_id, self.parent(item_id), current_index + 1)

        self.update_index_row_all()
        self.focus_set()

    def on_rename_first_column(self, item_id):
        """Trigger renaming if dialog conditions are met."""

        def _withdraw(args):
            trans_frame.destroy()

        def _tryupdate(args):
            self.multiple.rename_item(item_id, custom_name.get())
            trans_frame.destroy()

        trans_frame = Frame(self, background="red", borderwidth=2)
        bbox = self.bbox(item_id, "#1")
        trans_frame.place(
            x=bbox[0] - 1,
            y=bbox[1] - 1,
            width=bbox[2] + 2,
            height=bbox[3] + 2,
        )

        item_name = self.item(item_id)["values"][0]
        custom_name = StringVar()
        custom_name.set(item_name)
        trans_entry = Entry(trans_frame, textvariable=custom_name)
        trans_entry.pack(fill="both")
        trans_entry.icursor("end")
        trans_entry.focus()

        trans_entry.bind("<Return>", _tryupdate)
        trans_entry.bind("<FocusOut>", _withdraw)
        trans_entry.bind("<Escape>", _withdraw)

    def _bind_global_scroll(self, *args):
        self.event_generate("<<bind_global_scroll>>")

    def _unbind_global_scroll(self, *args):
        self.event_generate("<<unbind_global_scroll>>")

    def update_index_row_all(self):
        for key in self.multiple.get_ordered_keys():
            index = self.index(key)
            self.item(key, text=index + 1)

        # self.multiple.status = self.multiple.status

    def update_row(self, item_id, data, item_status):
        values = self._get_values_from_dict(data)
        index = self.index(item_id) + 1
        self.item(item_id, values=values, text=index)

        if item_status == 1:
            self.reset_row_background(item_id)
        else:
            self.highlight_row_background(item_id)

    def highlight_row_background(self, item_id):
        self.item(item_id, tags=(self._highlighted_tag,))

    def reset_row_background(self, item_id):
        self.item(item_id, tags=())

    def _get_values_from_dict(self, data):
        values = []
        timstr=20
        for key in self["columns"]:
            value = data.get(key, "")
            
            if isinstance(value, dict):
                keys= list(value.keys())
                if len(keys)==1: # usually a XOR
                    value = keys[0]
                else:  # Usually a bundled amound of similar info
                    value = ",".join([str(value[k]) for k in keys])
            else:
                value=str(value)

            if len(value)>timstr:
                value=value[:timstr-2]+"..."
            values.append(value)

        return values

    def select_item(self, item_id):
        self.selection_set(item_id)
        self.focus(item_id)
        self.focus_set()


class OTXorWidget(OTNodeWidget):
    """OT  Or-exclusive / oneOf widget."""

    # TODO: rethink schema (to not break backwards compatibility, we can "convert" the schema)
    # TODO: check if lone changes status

    def __init__(self, schema, parent, name, root_frame, n_width=1):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested object
        root_frame :  a Tk object were the widget will be grafted
        name: string naming the widget
        n_width : float
             relative size of the widget
        """
        super().__init__(schema, parent, name)
        self.n_width = n_width

        self._child = None
        self.previous_child = None

        self._create_widgets(root_frame)

        # TODO: review
        self._hold_previous_update = False  # to avoid creating child when empty
        child_name = self.schema["oneOf"][0]["required"][0]
        self.child = self._create_child_from_schema(child_name)

    @property
    def children(self):
        if self.child is None:
            return {}

        return {self.child.name: self.child}

    @children.setter
    def children(self, value):
        pass

    @property
    def child(self):
        return self._child

    @child.setter
    def child(self, child):
        self._child = child

        title = child.schema.get("title", child.name)
        self._menu_bt.configure(text=title)

    def _create_widgets(self, root_frame):
        self._holder = ttk.Frame(
            root_frame,
            name=self.name,
            relief="sunken",
        )

        title = self.schema.get("title", f"#{self.name}")
        self.title_lbl = ttk.Label(self._holder, text=title)

        self._forceps = ttk.Frame(
            self._holder, width=self.n_width * WIDTH_UNIT, height=1
        )
        self._menu_bt = ttk.Menubutton(self._holder, text="None")

        self._xor_holder = ttk.Frame(self._holder)

        self._holder.pack(side="top", expand=True)
        self.title_lbl.pack(side="top", fill="x", padx=1, pady=1)
        self._forceps.pack(side="top")
        self._menu_bt.pack(side="top")
        self._xor_holder.pack(side="top", padx=1, pady=1)

        self._menu_bt.menu = Menu(self._menu_bt, tearoff=False)
        self._menu_bt["menu"] = self._menu_bt.menu

        def _hierachical_menu():
            list_names = []
            list_titles = []

            for oneof_item in self.schema["oneOf"]:
                name = oneof_item["required"][0]
                ch_s = oneof_item["properties"][name]
                title = ch_s.get("title", name)
                list_names.append(name)
                list_titles.append(title)
            return hierachical_call(list_names, list_names)
            
        for menu_optn, value in _hierachical_menu().items():
            self._menu_bt.menu.add_separator()
            if isinstance(value, str):
                self._menu_bt.menu.add_command(
                    label=value, command=lambda cbck=menu_optn: self.xor_callback(cbck)
                )
            elif isinstance(value, dict):
                self._menu_bt.menu.add_separator()
                for name, title in value.items():
                    self._menu_bt.menu.add_command(
                    label=title, command=lambda cbck=name: self.xor_callback(cbck)
                )
            else:
                raise RuntimeError
        self._img, self._docu, self._desc = _create_extra_widgets(self, self._holder)

    def xor_callback(self, name_child):
        """Event on XOR menu selection."""
        if name_child != self.child.name:
            self.update_xor_content(name_child, data_in=None)

            #self.update_status_predecessors(changing=True)

            self.evaluate_status_ascending(changing=True)
            self.refresh_status_display_ascending()
            play_door2()

    def refresh_status_display(self):
        self._update_xor_style()

        if self._status == 1 and not self._hold_previous_update:
            if (
                self.previous_child is not None
                and self.previous_child is not self.child
            ):
                self.previous_child.destroy()

            self.previous_child = self.child

    def _update_xor_style(self):
        style = "TMenubutton" if self._status == 1 else "Highlighted.TMenubutton"
        self._menu_bt.configure(style=style)

    def update_xor_content(self, child_name, data_in=None, first_time: bool = False):
        """Reconfigure XOR button.

        Inputs :
        --------
        child_name : string, naming the child object
        data_in : dictionary used to pre-fill the data
        """
        # TODO: can this be simplified?

        if (
            self.child is not None
            and self.child.name == child_name
            and data_in is not None
        ):
            self.child.set(data_in, first_time=first_time)
            return

        if self.child is not None and self.previous_child is not self.child:
            self._hold_previous_update = True
            self.child.destroy()

        if self.previous_child is not None and self.previous_child.name == child_name:
            self._hold_previous_update = True
            self.child = self.previous_child.awake()

        else:
            if self.previous_child is not None:
                self.previous_child.sleep()

            # LPA : trick to allow nested xor with dependents
            # AD : for goddam safe LPA, even VScode intellisence cannot follow your distorted mind here!!!
            form = self.my_root_tab_widget
            form.prepare_to_receive_xor_dependents()

            self.child = self._create_child_from_schema(child_name, data_in)
            form.assign_xor_dependents()

        self._hold_previous_update = False

    def _create_child_from_schema(self, child_name, data_in=None):
        for possible_childs in self.schema["oneOf"]:
            if possible_childs["required"][0] == child_name:
                child_schema = possible_childs["properties"][child_name]
        child = OTContainerWidget(
            child_schema,
            self,
            child_name,
            self._xor_holder,
            relief="flat",
            show_title=False,
        )

        if data_in is not None:
            child.set(data_in)

        return child

    def get(self):
        """Get the data of children widgets.

        Returns :
        ---------
        a dictionary with the get result of current children
        """
        if self.child is None:
            return None

        try:
            return {self.child.name: self.child.get()}
        except GetException:
            return None

    def set(self, dict_, first_time: bool = False):
        """Set the data of children widgets.

        Input :
        -------
        a dictionary with the value of the childrens
        """
        # !!! Must not skip if equal, because of dependent widgets (dynamic choices)
        # if dict_ == self.get():
        #     return
        if not isinstance(dict_,dict):
            raise SetException(f"Wrong data type at {self.name}")

        if len(dict_) > 1:
            # TODO: when does this happen?
            raise SetException("Multiple matching option, skipping...")
        
        # TODO: is there a better way to do it?
        for one_of in self.schema["oneOf"]:
            child = next(iter(one_of["properties"]))
            if child in dict_:
                try:
                    self.update_xor_content(child, dict_[child], first_time=first_time)
                    break
                except SetException:
                    # logger.warning(f"SetException on {child.name}")
                    pass


def _create_extra_widgets(ot_widget, frame):
    img, docu, desc = None, None, None

    if "image" in ot_widget.schema:
        img = create_image(ot_widget.schema, frame)

    if "documentation" in ot_widget.schema:
        docu = create_documentation(ot_widget.schema, frame)

    if "description" in ot_widget.schema:
        desc = create_description(frame, ot_widget.schema["description"], side="top")

    return img, docu, desc



def flat_call(list_items: list, list_names: list) -> dict:
    """Flat grouping of items in XOr menus

    If a prefix is repeated, it becomes a subMenu
    """

    out = {}
    # dtect prefixed items
    for item, name in zip(list_items,list_names):
        out[item]=name
    return out

def hierachical_call(list_items: list, list_names: list) -> dict:
    """Smart grouping of items in XOr menus

    If a prefix is repeated, it becomes a subMenu
    """

    out = {}
    # dtect prefixed items
    for i, item in enumerate(list_items):
        if "_" in item:
            prefix = item.split("_")[0] + "_"

            if prefix not in out:
                out[prefix] = {}

            # print(">>>", out,"/",prefix,"/",item,"/", list_names )
            out[prefix][item] = list_names[i]

        else:
            out[item] = list_names[i]

    # remove submenus with a single one
    filter_out = {}
    for key, value in out.items():
        if not isinstance(value, dict):
            filter_out[key] = value
        else:
            sub_keys = list(value.keys())
            if len(sub_keys) > 1:
                filter_out[key] = value
            else:
                lone_key = sub_keys[0]
                filter_out[lone_key] = value[lone_key]

    return filter_out
