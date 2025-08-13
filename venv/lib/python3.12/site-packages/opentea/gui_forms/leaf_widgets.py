"""
Leaf widgets :
==============

Leaves are the lower part to the graph,
at the tip of the branches.
Most of these widget are entries.

Entries:
--------

The generic node EntryNode is the basis for all
single line inputs:

 - numbers
 - integers
 - strings

Additional content can be shown to give info
in the case of validation rules.

Booleans:
---------

This is tranlated to a single Checkbox

Choices:
--------

The choice widget corresponds to the Schema enum property.
This is tranlated to radiobuttons in the form.

FleBrowser:
-----------

This special entry check that the content is a file,
and add a browsing dialog to select the file.

Documentation and Description (DEPRECATED):
-------------------------------------------

Kept fo backward compatibility,
docuementation and descriptions are wats to display strings
in the forms.

Prefer now the documentation and description attributes
in the blocks.

Comments:
---------

Comments are multiline textfields
They can also be usefull for feedbach in disabled mode.

Lists:
------

List corresponds to arrays of parameters,
shown aslist of entries.
These list cannot take dependency links for the moments.

Without a fixed dimension,
specified with "MaxItemns" and "MinItems" attributes in the SCHEMA,
a +/- dialog is shown to extend or crop the list.

"""

import os
import operator
import abc

import tkinter
from tkinter import (
    ttk,
    Variable,
    BooleanVar,
    Toplevel,
    Text,
    Listbox,
    filedialog,
    Menu,
    messagebox,
)

from nob import Nob

from opentea.gui_forms._base import OTTreeElement
from opentea.gui_forms._exceptions import (
    GetException,
    SetException,
)
from opentea.gui_forms.constants import (
    PARAMS,
    WIDTH_UNIT,
    LINE_HEIGHT,
    IMAGE_DICT,
    configure_on_status,
)

from opentea.gui_forms.generic_widgets import TextConsole
from opentea.gui_forms.utils import (
    create_description,
    get_tkroot,
)
from opentea.gui_forms.soundboard import play_switch

from loguru import logger

# TODO: ot_require in widgets should make them disable
# TODO: does it make sense to use a tkvar everywhere? or an equivalent
# TODO: root_frame -> master_frame/holder
# TODO: implement ctrl-Z for validation?

# TODO: bring create_widget and set default to _LeafWidget?
# TODO: create opentea variables


class _LeafWidget(OTTreeElement, metaclass=abc.ABCMeta):
    """All widgets behaving as leaves

    Hre the only GUI aspects are:
    root_frame :  where to pack the widget
    self.tkvar :  the variable monitored for changes (can be redefined)
    """

    def __init__(self, schema, parent, name, root_frame):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested dict object
        root_frame :  a Tk object were the widget will be grafted
        holder_nlines : integer
            custom number of lines for holder
        """
        super().__init__(schema, parent, name)
        self.kind = "leaf"
        self.previous_value = None

        self.leaf_default = self.schema.get("default", infer_default(self.item_type))
        self.leaf_create_common_widgets(root_frame)
        self.leaf_define_var()
        self.leaf_create_specific_widgets()
        self.leaf_set_default()

    def leaf_create_common_widgets(self, root_frame):
        r"""declare The base layout for our widgets

        root_frame
        ┌────────────────────────┐
        │                        │
        │ self._desc───────────┐ │
        │ │                    │ │
        │ └────────────────────┘ │
        │                        │
        │ self._holder─────────┐ │....(rely=0)
        │ │.label ne│nw        │ │
        │ │         │          │ │
        │ │         │          │ │
        │ └─────────┴──────────┘ │....(rely=1)
        │           .            │
        └────────────────────────┘
                    .(relx=0.5)
        """
        self._holder = ttk.Frame(
            root_frame, name=self.name, width=WIDTH_UNIT, height=LINE_HEIGHT
        )

        if self.state != "hidden":
            if "description" in self.schema:
                self._desc = create_description(
                    root_frame, self.schema["description"], size=1, side="top"
                )
            self._holder.pack(side="top", fill="x")

        self._label = ttk.Label(
            self._holder, text=self.title, wraplength=int(0.5 * WIDTH_UNIT)
        )  #  image=IMAGE_DICT["unknown"], compound="left"

        self._label.place(relx=0.5, rely=0.0, anchor="ne")

    def leaf_define_var(self):
        """Creation of the Tk variable to track changes"""
        self.tkvar = Variable()
        self.tkvar.trace_add("write", self.leaf_on_value_change)

    def leaf_on_value_change(self, *args):
        """Callback if value change"""
        self.evaluate_status_ascending(changing=True)
        self.refresh_status_display_ascending()
        #self.update_status(changing=True)
        

    def leaf_create_specific_widgets(self):
        pass

    def leaf_set_default(self):
        if self.dependent:
            return
        self.set(self.leaf_default)

    def once_validated(self):
        self.previous_value = self.get()

    def leaf_is_valid(self, value) -> bool:
        """Check if a value is valid against the current widget
        by default, always true"""
        return True

    def refresh_status_display(self):
        """General update of leaves

        By default the label is changed
        On entries, it will be the entries themselves

        Also try to update error message"""
        configure_on_status(self._label, self._status)
        self._update_error_msg()

    def _update_error_msg(self):
        """What to show if there is an error

        by default, nothing.
        Numeric entries will show on self._status_lbl
        """
        pass

    def _get_status_error_msg(self):
        """Default error msg to be shown"""
        return f'Invalid input "{self.entry.get()}"'

    def destroy(self):
        # self._update_parent_status(1)
        self._holder.destroy()
        self._reset_master()

    def str2type(self, value):
        """Strongly type the value entered by the user"""
        if value is None:
            return "None"
        if self.item_type == "number":
            return float(value)
        elif self.item_type == "integer":
            return int(value)
        elif self.item_type == "boolean":
            return bool(value)
        else:
            return str(value)

    def get(self):
        try:
            return self.str2type(self.tkvar.get())
        except ValueError:
            raise GetException()

    def set(self, value, first_time: bool = False):
        try:
            value = self.str2type(value)
            if first_time:
                self.previous_value = value
            self.tkvar.set(value)
        except ValueError:
            raise SetException()


class _DeadLeafWidget(_LeafWidget):
    """seems to be used for the Name of a multiple item"""

    def __init__(self, schema, parent, name, root_frame):
        
        super().__init__(schema, parent, name, root_frame)
        self.kind = "dead_leaf"

    def leaf_create_specific_widgets(self):
        self._dynamic_title = ttk.Label(
            self._holder,
            textvariable=self.tkvar,
            wraplength=int(0.5 * WIDTH_UNIT),
            font=("TkDefaultFont", 14, "bold"),
        )  #  imag
        self._label.place_forget()
        self._dynamic_title.pack()


class OTEntry(_LeafWidget):  # , metaclass=abc.ABCMeta):
    """Factory for OpenTea Entries."""

    def leaf_create_specific_widgets(self):
        self.entry = ttk.Entry(
            self._holder,
            textvariable=self.tkvar,
            exportselection=False,
        )
        self.entry.place(relx=0.5, rely=1.0, anchor="sw")
        if self.state == "disabled" or self.dependent:  # TODO :  can it be dependent?
            self.entry.configure(style="Disabled.TLabel", state="disabled")

    def refresh_status_display(self):
        """REDEFINITION : Update of Entries, instead of label"""
        configure_on_status(self.entry, self._status)
        self._update_error_msg()


class OTNumericEntry(OTEntry):  # , metaclass=abc.ABCMeta):
    def leaf_create_specific_widgets(self):
        super().leaf_create_specific_widgets()

        self._holder.config(height=2 * LINE_HEIGHT)
        self.entry.place(relx=0.5, rely=0.5, anchor="sw")
        self._label.place(relx=0.5, rely=0.5, anchor="se")
        self._bounds = [
            self.schema.get("minimum", -float("inf")),
            self.schema.get("maximum", float("inf")),
        ]
        self._exclusive_bounds = [
            self.schema.get("exclusiveMinimum", False),
            self.schema.get("exclusiveMaximum", False),
        ]

        self._status_lbl = ttk.Label(
            self._holder, text="no status yet", style="Status.TLabel", compound="left"
        )
        self._status_lbl.place(relx=1.0, rely=0.5, anchor="ne")

    def _update_error_msg(self):
        """What to show if there is an error

        by default, nothing.
        Numeric entries will show on self._status_lbl
        """
        if self._status == -1:
            self._status_lbl.config(
                text=self._get_status_error_msg(), image=IMAGE_DICT["invalid"]
            )
        else:
            self._status_lbl.config(text="", image="")

    def _get_default(self):
        value = self.schema.get("default", None)

        if value is not None:
            return value

        # set a valid default value
        value = 0.0
        if "minimum" in self.schema:
            value = self.schema["minimum"]
            if self._exclusive_bounds[0]:
                if self._type == "integer":
                    value += 1
                else:
                    value *= 1.1
        elif "maximum" in self.schema:
            value = self.schema["maximum"]
            if self._exclusive_bounds[1]:
                if self._type == "integer":
                    value -= 1
                else:
                    value *= 0.9

        return value

    def leaf_is_valid(self, value):
        """Check if a value is valid against the current widget
        by default, always true"""
        error_msg = self._check_bounds(value)
        if error_msg:
            return False
        return True

    def _check_bounds(self, value):
        """Validate rules on entries."""
        str_operators = ["<", ">"]
        operators = [operator.le, operator.ge]
        for lim, exclusive, operator_, str_operator in zip(
            self._bounds, self._exclusive_bounds, operators, str_operators
        ):
            if operator_(value, lim):
                if not exclusive and value == lim:
                    continue

                return f"Invalid: {'=' if exclusive else ''}{str_operator}{lim}"

        return ""

    def _get_status_error_msg(self):
        """Redefine error msg"""
        try:
            err_msg = self._check_bounds(self.get())
        except GetException:
            err_msg = super()._get_status_error_msg()

        return err_msg


class OTBoolean(_LeafWidget):
    def leaf_create_specific_widgets(self):
        self._label.place(relx=0.5, rely=0.5, anchor="e")
        self._cbutt = ttk.Checkbutton(
            self._holder, variable=self.tkvar, command=play_switch
        )
        self._cbutt.place(relx=0.5, rely=0.5, anchor="w")

        # redefine void as False for boolean
        self.default = self.schema.get("default", False)
    
    def leaf_define_var(self):
        """Creation of the Tk variable to track changes
        
        Must redefine the Var, if staying to tk.Variable, 0 an 1 are not understood as False and True"""
        self.tkvar = BooleanVar()
        self.tkvar.trace_add("write", self.leaf_on_value_change)
    


class OTFileBrowser(_LeafWidget):
    def __init__(self, schema, parent, name, root_frame):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested object
        root_frame :  a Tk object were the widget will be grafted
        """
        super().__init__(schema, parent, name, root_frame)

        self._filter = []
        self._isdirectory = False
        if self.schema["ot_type"] == "folder":
            self._isdirectory = True
        
        if "ot_filter" in schema:
            filters = schema["ot_filter"]
            if "directory" in filters:         # Deprecated, use "ot_type" = Folder
                self._isdirectory = True
            else:
                for ext in filters:
                    filetype = (f"{ext} files", f"*.{ext}")
                    self._filter.append(filetype)

    def leaf_create_specific_widgets(self):
        self._label.place(relx=0.5, rely=0.5, anchor="e")
        self._holder.config(height=2 * LINE_HEIGHT)

        self._entry = ttk.Entry(
            self._holder,
            textvariable=self.tkvar,
            state="disabled",
            # foreground="black",
            justify="right",
        )
        self._entry.place(relx=0.5, rely=0.5, relwidth=0.4, anchor="sw")

        self._scroll = ttk.Scrollbar(
            self._holder, orient="horizontal", command=self.__scrollHandler
        )
        self._scroll.place(relx=0.5, rely=0.5, relwidth=0.4, anchor="nw")
        self._entry.configure(xscrollcommand=self._scroll.set)

        self._btn = ttk.Button(
            self._holder,
            image=IMAGE_DICT["load"],
            width=0.1 * WIDTH_UNIT,
            compound="left",
            style="clam.TLabel",
            command=self._browse,
        )

        self._btn.place(relx=0.9, rely=0.5, anchor="w")

    def __scrollHandler(self, *L):
        """Callback for entry scrollbar"""
        op, howMany = L[0], L[1]
        if op == "scroll":
            units = L[2]
            self._entry.xview_scroll(howMany, units)
        elif op == "moveto":
            self._entry.xview_moveto(howMany)

    def leaf_is_valid(self, value) -> bool:
        """Check if a value is valid against the current widget"""
        if value == "":
            return True
        if not os.path.isfile(value):
            return False
        return True

    # def _update_leaf_styl

    def _browse(self, event=None):
        """Browse directory or files."""
        cur_path = self.get()

        if self._isdirectory:
            path = filedialog.askdirectory(title=self.title)
        else:
            path = filedialog.askopenfilename(title=self.title, filetypes=self._filter)

        if path != cur_path:
            if path != "":
                path = os.path.relpath(path)
            self.tkvar.set(path)

    def refresh_status_display(self):
        """Adjust color of the entry"""
        # style = "TEntry" if self.status == 1 else "Highlighted.TEntry"
        # self._entry.configure(style=style)
        configure_on_status(self._entry, self._status)


class _OTChoiceAbstract(_LeafWidget, metaclass=abc.ABCMeta):
    # TODO: think about type -> OTVariable development

    def __init__(self, schema, parent, name, root_frame):
        super().__init__(schema, parent, name, root_frame)
        self._label.place(relx=0.5, rely=1, anchor="se")

        # redefine default
        value = self.schema.get("default", None)
        if value is None:
            value = self.schema.get("enum", [""])[0]
        self.default = value


class OTChoice:
    def __new__(self, schema, parent, name, root_frame):
        if "enum" in schema:
            if len(schema["enum"]) > 5:
                return _OTChoiceCombo(schema, parent, name, root_frame)
            else:
                return _OTChoiceRadio(schema, parent, name, root_frame)
        elif "ot_dyn_choice" in schema:
            return _OTChoiceDynamic(schema, parent, name, root_frame)


class _OTChoiceRadio(_OTChoiceAbstract):
    def __init__(self, schema, parent, name, root_frame):
        self.rad_btns = {}
        super().__init__(schema, parent, name, root_frame)

    def leaf_create_specific_widgets(self):
        self.radio_options = self.schema.get("enum", [])
        self.radio_titles = self.schema.get("enum_titles", self.schema["enum"])
        self._pack_with_radiobuttons()

    def _pack_with_radiobuttons(self):
        """Radiobutton version of the widget"""
        n_lines = max(len(self.radio_options), 1)
        self._holder.config(height=n_lines * LINE_HEIGHT)
        rel_step = 1.0 / n_lines
        current_rely = 1 * rel_step

        for value, title in zip(self.radio_options, self.radio_titles):
            rad_btn = ttk.Radiobutton(
                self._holder,
                text=title,
                value=value,
                variable=self.tkvar,
                command=play_switch,
            )
            rad_btn.place(relx=0.5, rely=current_rely, anchor="sw")
            self.rad_btns[value] = rad_btn
            current_rely += rel_step

        self._holder.configure(relief="sunken", padding=2)

        self._label.place(relx=0.5, rely=int(0.5 * current_rely), anchor="e")

    def leaf_is_valid(self, value) -> bool:
        """Check if a value is valid against the current widget"""
        if value not in self.radio_options:
            # logger.warning(f"Error {self.name} {value} not in options")
            return False
        return True


class _OTChoiceCombo(_OTChoiceAbstract):
    """OT choices widget."""

    def leaf_create_specific_widgets(self):
        self.combo_options = self.schema.get("enum", [])
        self.pack_with_combobox()

    def pack_with_combobox(self):
        """Combobox version of the widget"""

        self.combo = ttk.Combobox(
            self._holder,
            values=self.combo_options,
            textvariable=self.tkvar,
            state="readonly",
            postcommand=play_switch,
        )
        self.combo.place(relx=0.5, rely=1, anchor="sw")

    def leaf_is_valid(self, value) -> bool:
        """Check if a value is valid against the current widget"""
        if value not in self.combo_options:
            return False
        return True


class _OTChoiceDynamic(_OTChoiceCombo):
    """This particular class is for choosing among a variable set of option

    Example: select a patch name among a list found in an external file
    """

    def set(self, value, first_time: bool = False):
        """Reconfigure the options when set"""
        tree = Nob(self.my_root_tab_widget.get())
        key = self.schema["ot_dyn_choice"]
        self.combo_options = tree[key][:]
        self.pack_with_combobox()
        super().set(value, first_time=first_time)


class _OTAbstractList(_LeafWidget, metaclass=abc.ABCMeta):
    """Class to handle LISTS

    In the memory a list is a Python Lists
    But for the GUI, the list containes subleaves with specific behavior
    We loose here the usual perfect mapping btw memory and GUI"""

    def leaf_define_var(self):
        """We must redifine variable because ListBox is passive an connot link to tk. Variable easily"""
        self.list_variables = []

    def leaf_create_specific_widgets(self):
        entry_type = self.schema["items"].get("type", "string")
        self.leaf_entry_default = self.schema["items"].get(
            "default", infer_default(entry_type)
        )
        self.leaf_default = self.schema.get("default", [self.leaf_entry_default])
        self.entrylistholder = ttk.Frame(self._holder)
        self.entrylistholder.place(relwidth=0.5, relx=0.5, rely=0.0, anchor="nw")
        self._configure_popup_menu()

    def _configure_popup_menu(self):
        self.popup_menu = Menu(
            self.entrylistholder, tearoff=False
        )  # binding in tv bindings
        self._add_popup_commands()
        self.entrylistholder.bind("<Enter>", self._activate_popup)
        self.entrylistholder.bind("<Leave>", self._deactivate_popup)

    @abc.abstractmethod
    def _add_popup_commands(self):
        pass

    def _activate_popup(self, *args):
        self.entrylistholder.bind_all("<Button-2>", self.on_right_click)

    def _deactivate_popup(self, *args):
        self.entrylistholder.unbind_all("<Button-2>")

    def on_right_click(self, event):
        self.popup_menu.tk_popup(event.x_root, event.y_root)

    def on_copy(self, *args):
        copy_str = ", ".join([str(value) for value in self.get()])
        root = get_tkroot(self._holder)
        root.clipboard_clear()
        root.clipboard_append(copy_str)

    def _resize_holder(self, n_lines):
        self._holder.config(height=n_lines * LINE_HEIGHT)

    def once_validated(self):
        """What to do when a list entry is validated"""
        self.set_slaves(self.get())


class _ListEntry(_LeafWidget):
    """The default SUBwidget entry for lists

    This is packed Inside an OTList."""

    def __init__(self, schema, parent, holder, initial_value=None):
        """Additions to _LeafWidget init"""
        self._holder = holder
        super().__init__(schema, parent, None, holder)
        if initial_value is not None:
            self.set(initial_value)
            self._previous_value = initial_value
        

    ### VUE
    def leaf_create_specific_widgets(self):
        """RThe Tk apearance of an entry"""
        self.entry = ttk.Entry(
            self._holder, textvariable=self.tkvar, exportselection=False
        )
        self.entry.pack(side="top")
        self.entry.bind("<FocusIn>", self.on_entry_focus)

    def refresh_status_display(self):
        """Adjust display

        - change color of entry
        - update error message
        """
        configure_on_status(self.entry, self._status)
        self._update_error_msg()

    def _update_error_msg(self):
        """What to show if there is an error

        by default, nothing.
        Numeric entries will show on self._status_lbl
        ListEntrys defer to parents
        """
        if self._status == -1:
            self.parent._status_lbl.config(
                text=self._get_status_error_msg(), image=IMAGE_DICT["invalid"]
            )
        else:
            self.parent._status_lbl.config(text="", image="")

    def on_entry_focus(self, event):
        """If user focus on a cell,

        and cell is invalid, update List error message
        else make it void
        """
        pass

    def leaf_create_common_widgets(self, *args):
        """Remove the usual creation of widgets"""
        pass

    def destroy(self):
        """Destroy widget"""
        self.entry.destroy()


class OTDynamicList(_OTAbstractList):
    """List controlled by the user

    Used for input, not just feedback

    CaveAt :  this item is like a NODE,
    but can change its size a lot, and order matters
    Instead of self.children ,
      the subwidgets are stored as a list in  self.variables

    """

    def __init__(self, schema, parent, name, root_frame):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested object
        parent: ???
        name: name to be added
        root_frame :  a Tk object were the widget will be grafted
        """

        # TODO: any way to avoid having this special case?
        self.empty_widget = None
        super().__init__(schema, parent, name, root_frame)
        self.kind = "dynlist_leaf"

    def leaf_create_specific_widgets(self):
        """Add more to _OTAbstractList._create_widgets"""
        super().leaf_create_specific_widgets()

        self.min_items = self.schema.get("minItems", 1)
        self.max_items = self.schema.get("maxItems", 999)
        # adjust default to min max
        self.leaf_default = adjust_list_to_range(
            self.leaf_default, self.min_items, self.max_items, self.leaf_entry_default
        )

        self.resizable = True
        if self.min_items == self.max_items:
            self.resizable = False

        self._status_lbl = ttk.Label(
            self._holder, text="no status yet", style="Status.TLabel", compound="left"
        )
        self._status_lbl.place(relx=0.5, rely=0.2, anchor="ne")

        # By default it is void
        if self.resizable:
            self.additem_bt = ttk.Button(
                self._holder, text="+", command=self.on_add_item
            )
            self.delitem_bt = ttk.Button(
                self._holder, text="-", command=self.on_del_item
            )
            self.additem_bt.place(relwidth=0.07, relx=0.43, rely=0.4, anchor="ne")
            self.delitem_bt.place(relwidth=0.07, relx=0.5, rely=0.4, anchor="ne")

        for i, value in enumerate(self.leaf_default):
            self._add_item(value=value)

    def _add_popup_commands(self):
        """Re-define _OTAbstractList void method"""
        self.popup_menu.add_command(label="Copy", command=self.on_copy)
        self.popup_menu.add_command(label="Paste", command=self.on_paste)

    def _resize_holder(self):
        """Adjust holder size to intern variable"""
        n_lines = max(2, 1 + len(self.list_variables))
        if self.resizable:
            n_lines += 1
        super()._resize_holder(n_lines)

    def create_void_entry(self):
        """Add new entry to holder"""
        if self.empty_widget is not None:
            return
        label = ttk.Label(self.entrylistholder, text="void")
        label.pack(side="top")
        self.empty_widget = label

    def on_paste(self, *args):
        """Callback on paste"""
        try:
            paste_str = get_tkroot(self._holder).clipboard_get()
        except tkinter._tkinter.TclError:
            paste_str = ""

        if not paste_str:
            messagebox.showwarning(message="Nothing to paste")
            return

        paste_ls = [value.strip() for value in paste_str.split(",")]

        # validate clipboard
        # TODO: need to find alternative way to check clipboard (is_valid?)
        var = (
            self.list_variables[0]
            if self.list_variables
            else _ListEntry(self, self.entrylistholder, self.item_type)
        )
        for value in paste_ls:
            try:
                var.str2type(value)
            except ValueError:
                message = f"Invalid clipboard:\n'{paste_str}'"
                messagebox.showwarning(message=message)
                return

        # paste clipboard
        self.set(paste_ls)

    def on_add_item(self):
        """callback Add an item at the end of the array."""
        # TODO: can this be simplified based on new way of handling values?

        if len(self.list_variables) == self.max_items:
            return
        self._add_item()

    def on_del_item(self):
        """Callback Delete item at the end of the array."""
        if len(self.list_variables) == self.min_items:
            return
        self.remove_items(1)

    # TODO : unsure what this does
    def clear_empty_widget(self):
        """Remove empty widget"""
        if self.empty_widget is None:
            return
        self.empty_widget.destroy()
        self.empty_widget = None

    def _get_previous_item_value(self):
        """Return previous value of one list entry"""
        try:
            return self.previous_value[-1]
        except TypeError:
            return None

    def _add_item(self, value=None):
        """Add an item internally"""
        # TODO: change on add item to get None

        # previous_value = self._get_previous_item_value(len(self.dynlist_variables))
        self.clear_empty_widget()
        
        #self.ping()
        if value is None:
            value = self.leaf_default

        new_entry = _ListEntry(
            self.schema["items"],
            self,
            self.entrylistholder,
            initial_value=value,
        )
        #logger.warning(f"Add item {len(self.list_variables)} with {value} ")
        #logger.warning(f"got {new_entry.get()}")
        #new_entry.update_status()
        self.evaluate_status_ascending(changing=True)
        self.refresh_status_display_ascending()

        # Entry is added  to list_variables by add_child
        self._resize_holder()

    def remove_items(self, n_items=None):
        """Delete N_ITEMS last items of self.variables

        WARNING :
        - del all items if n_items is None
        - del no items if self.variables is []
        """
        if not self.list_variables:
            return

        if n_items is None:
            n_items = len(self.list_variables)

        i = 0
        while i < n_items:
            var = self.list_variables.pop()
            var.destroy()
            i += 1

        self._resize_holder()
        if len(self.list_variables) == 0:
            self.create_void_entry()

    def add_child(self, child):
        """New child to list"""
        self.list_variables.append(child)

    def list_children(self):
        """How to how to iterate over the list of children
        """
        return self.list_variables

    ### Controls
    def set(self, values, first_time: bool = False):
        """Opentea SET method"""

        values = adjust_list_to_range(
            values, self.min_items, self.max_items, self.leaf_entry_default
        )

        if first_time:
            self.previous_value = values

        n_set_elems = len(values)
        n_vars = len(self.list_variables)
        # delete excess items
        if n_vars > n_set_elems:
            n_del = n_vars - n_set_elems
            self.remove_items(n_del)

        # update existing variables
        for variable, value in zip(self.list_variables, values):
            variable.set(value)

        # add new variables if necessary
        for i in range(n_vars, n_set_elems):
            self._add_item(values[i])

    # def update_status_successors(self):
    #     """RECURSIVE to refresh all successors

    #     REDEFINES :  because here children are stored in a list, not a dict.
    #     """
    #     for child in self.list_variables:
    #         child.update_status()
    #         child.update_status_successors()

    def get(self):
        """Opentea GET method"""
        return [var.get() for var in self.list_variables]


class OTStaticList(_OTAbstractList):
    """List NOT controlled bby the user

    created if ot_require or disabled"""

    ### Vue
    def leaf_create_specific_widgets(self):
        """Add to _create_widgets"""
        self.kind = "dead_leaf"
        super().leaf_create_specific_widgets()
        self._configure_listbox()

    def _configure_listbox(self):
        """Packing elements"""
        nlines = 6
        self._resize_holder(nlines)

        scrollbar = ttk.Scrollbar(self.entrylistholder, orient="vertical")

        self.lbx = Listbox(
            self.entrylistholder, height=nlines, yscrollcommand=scrollbar.set
        )
        self.lbx.configure(
            state="disabled",
            highlightbackground=PARAMS["bg"],
            background=PARAMS["bg"],
            disabledforeground=PARAMS["bg_dark"],
        )

        scrollbar.config(command=self.lbx.yview)
        scrollbar.pack(side="right", fill="y")
        self.lbx.pack(side="top", fill="both", pady=2)
        self.lbx.bind("<Enter>", self._unbind_global_scroll)
        self.lbx.bind("<Leave>", self._bind_global_scroll)

    def _bind_global_scroll(self, *args):
        """Enable scroll if pointer in widget"""
        self.lbx.event_generate("<<bind_global_scroll>>")

    def _unbind_global_scroll(self, *args):
        """Disable scroll if pointer in widget"""
        self.lbx.event_generate("<<unbind_global_scroll>>")

    def _update_listbox(self):
        """Change appearence according to internal variable"""
        self.lbx.configure(state="normal")
        self.lbx.delete(0, "end")
        for item in self.list_variable:
            self.lbx.insert("end", item)
        self.lbx.configure(state="disabled")

    ### Model

    def _add_popup_commands(self):
        """Re-define _OTAbstractList void method"""
        self.popup_menu.add_command(label="Copy", command=self.on_copy)

    def get(self):
        """Opentea GET method"""
        return self.list_variable

    def set(self, value, first_time: bool = False):
        """Opentea SET method
        By assigning we trigger the variable __setter__
        """
        self.list_variable = list(value)
        self._update_listbox()


class OTDocu(_DeadLeafWidget):
    def __init__(self, schema, parent, name, root_frame):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested object
        root_frame :  a Tk object were the widget will be grafted
        """
        super().__init__(schema, parent, name, root_frame)
        self.root = get_tkroot(root_frame)
        self._dialog = None

    def leaf_create_specific_widgets(self):
        self._btn = ttk.Button(
            self._holder,
            width=0.01 * WIDTH_UNIT,
            compound="center",
            image=IMAGE_DICT["docu"],
            style="clam.TLabel",
            command=self._popup_dialog,
        )
        self._btn.place(relx=0.9, rely=0.5, anchor="center")
        self._holder.pack_configure(side="bottom", fill="x")

    def _popup_dialog(self):
        """Display content of documentation string."""
        # TODO: need to be reviewed (but deprecated)
        self._dialog = Toplevel(self.root)
        self._dialog.transient(self.root)
        self._dialog.title("Documentation")
        self._dialog.grab_set()

        self._dialog.bind("<Control-w>", self._destroy_dialog)
        self._dialog.bind("<Escape>", self._destroy_dialog)
        self._dialog.protocol("WM_DELETE_WINDOW", self._destroy_dialog)

        dlg_frame = ttk.Frame(self._dialog, width=3 * WIDTH_UNIT, height=3 * WIDTH_UNIT)
        dlg_frame.pack(side="top", fill="both", expand=True)
        dlg_frame.grid_propagate(False)
        dlg_frame.grid_rowconfigure(0, weight=1)
        dlg_frame.grid_columnconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(dlg_frame)
        scrollbar.pack(side="right", fill="y")

        text_wd = Text(
            dlg_frame,
            wrap="word",
            yscrollcommand=scrollbar.set,
            borderwidth=0.02 * WIDTH_UNIT,
            relief="sunken",
        )

        # Example of formatting
        text_wd.tag_configure("bold", font=("Times", 14, "normal"))
        text_wd.insert("end", self.tkvar.get(), "bold")
        text_wd.config(state="disabled")
        text_wd.pack()
        scrollbar.config(command=text_wd.yview)

    def _destroy_dialog(self, event=None):
        """Destroying dialog."""
        self.root.focus_set()
        self._dialog.destroy()
        self._dialog = None


class OTDescription(_DeadLeafWidget):
    def __init__(self, schema, parent, name, root_frame):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested object
        root_frame :  a Tk object were the widget will be grafted
        """
        super().__init__(schema, parent, name, root_frame)
        self._holder.pack_configure(side="bottom", fill="x")

    def leaf_create_specific_widgets(self):
        self._label.config(
            justify="left", textvariable=self.tkvar, wraplength=WIDTH_UNIT * 0.8
        )
        self._label.pack(side="bottom")


class OTComment(_DeadLeafWidget):
    def __init__(self, schema, parent, name, root_frame):
        """Startup class.

        Inputs :
        --------
        schema : a schema as a nested object
        root_frame :  a Tk object were the widget will be grafted
        """
        state = schema.get("state", "normal")
        self.disabled = state == "disabled"

        super().__init__(schema, parent, name, root_frame)
        self._holder.pack_configure(side="top", fill="x")
        self._holder.bind("<Enter>", self._unbind_global_scroll)
        self._holder.bind("<Leave>", self._bind_global_scroll)
        self.configure_display()

    def leaf_create_specific_widgets(self):
        height = self.schema.get("height", 6)

        self.text_console = TextConsole(
            self._holder, self.tkvar, height=height, width=10, disabled=self.disabled
        )

    def _bind_global_scroll(self, *args):
        self._holder.event_generate("<<bind_global_scroll>>")

    def _unbind_global_scroll(self, *args):
        self._holder.event_generate("<<unbind_global_scroll>>")

    def get(self):
        return self.tkvar.get().rstrip()

    def set(self, value, first_time: bool = True):
        if first_time:
            self.previous_value = value
        self.text_console.set_text(value)  # variable gets set automatically

    def configure_display(self):
        if self.disabled:
            return
        bgcolor = "white" if self._status == 1 else PARAMS["hl_bg"]
        fgcolor = "black" if self._status == 1 else PARAMS["bg_dark"]
        self.text_console.configure(background=bgcolor)
        self.text_console.configure(foreground=fgcolor)


class OTEmpty(_DeadLeafWidget):
    """OT widget for VOID types."""

    def leaf_create_specific_widgets(self):
        if self.schema.get("ot_type", None) != "void":
            return

        info = []
        for item in ["name", "title", "type", "ot_type"]:
            if item in self.schema:
                info.append(f"{item} = {self.schema[item]}")

        self._label.configure(text="\n".join(info))
        self._label.pack(side="top", padx=2, pady=2)

        self._holder.forget()

    def get(self):
        return None

    def set(self, *args, **kwargs):
        pass


def infer_default(item_type):
    if item_type == "number":
        default = 0.0
    elif item_type == "integer":
        default = 1
    elif item_type == "boolean":
        default = False
    elif item_type == "string":
        default = ""
    else:
        default = None
    return default


def adjust_list_to_range(
    list_: list, min_items: int, max_items: int, default: any
) -> list:
    """Ajsut list with a range"""
    _excess = len(list_) - max_items
    if _excess > 0:
        for _ in range(_excess):
            list_.pop(-1)
    _miss = min_items - len(list_)
    if _miss > 0:
        for _ in range(_miss):
            list_.append(default)
    return list_
