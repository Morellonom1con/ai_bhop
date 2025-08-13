# TODO: create tkextensions package for common objects (with neverd)

import tkinter as tk
import io

from tkinter import (
    ttk,
    Variable as Tk_Variable,
    Canvas as Tk_Canvas,
)
from tkinter.scrolledtext import ScrolledText

from opentea.gui_forms.constants import PARAMS
from opentea.gui_forms.utils import is_hierarchically_above
from opentea.gui_forms.soundboard import play_item


class SwitchForm(ttk.Frame):
    """Overriden Frame class to mimick notebooks without tabs."""

    def add(self, item_id, title=None):
        label_frame = ttk.LabelFrame(
            self,
            text=title,
            relief="sunken",
        )
        label_frame.id = item_id  # added attribute
        self.sf_raise(item_id)
        return label_frame

    def sf_del(self, item_id):
        """Destroy tab_id tab."""
        for child_widget in self.winfo_children():
            if child_widget.id == item_id:
                child_widget.destroy()

    def sf_raise(self, item_id):
        """Forget current view and repack tab_name tab."""
        for child_widget in self.winfo_children():
            if child_widget.id == item_id:
                child_widget.pack(fill="both")
            else:
                child_widget.pack_forget()


class MouseScrollableFrame(ttk.Frame):
    # TODO: repetion on neverd -> tkxtensions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pack(side="top", fill="both", expand=True)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self.canvas = Tk_Canvas(
            self,
            background=PARAMS["bg_lbl"],
            highlightbackground=PARAMS["bg_lbl"],
            highlightcolor=PARAMS["bg_lbl"],
        )

        self.canvas.configure(width=1000, height=300)

        self.scrollbar_y = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.scrollbar_x = ttk.Scrollbar(
            self, orient="horizontal", command=self.canvas.xview
        )

        self.canvas.configure(yscrollcommand=self.scrollbar_y.set)
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set)

        self.canvas.grid(row=0, column=0, sticky="news")
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.scrollbar_x.grid(row=1, column=0, sticky="we")

        # bind frame (allow scroll behavior in all tabs)
        self._bind_scroll_activation()

    def _bind_scroll_activation(self):
        self.canvas.bind("<Enter>", self._bind_scroll)
        self.canvas.bind("<Enter>", self._bind_global_scroll_event, add="+")
        self.canvas.bind("<Leave>", self._unbind_scroll)
        self.canvas.bind("<Leave>", self._unbind_global_scroll_event, add="+")

    def _bind_global_scroll_event(self, *args):
        self.canvas.bind_all("<<bind_global_scroll>>", self._bind_global_scroll)
        self.canvas.bind_all("<<unbind_global_scroll>>", self._unbind_global_scroll)

    def _unbind_global_scroll_event(self, *args):
        self.canvas.unbind_all("<<bind_global_scroll>>")
        self.canvas.unbind_all("<<unbind_global_scroll>>")

    def _bind_global_scroll(self, event):
        if is_hierarchically_above(event.widget, self.canvas):
            self._bind_scroll_y()

    def _unbind_global_scroll(self, event):
        if is_hierarchically_above(event.widget, self.canvas):
            self._unbind_scroll_y()

    def _bind_scroll_y(self):
        if PARAMS["sys"] == "Linux":
            self.canvas.bind_all("<4>", self._on_mouse_wheel)
            self.canvas.bind_all("<5>", self._on_mouse_wheel)
        else:
            self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

    def _bind_scroll_x(self):
        if PARAMS["sys"] == "Linux":
            self.canvas.bind_all("<Shift-Button-4>", self._on_shift_mouse_wheel)
            self.canvas.bind_all("<Shift-Button-5>", self._on_shift_mouse_wheel)
        else:
            self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mouse_wheel)

    def _unbind_scroll_y(self):
        if PARAMS["sys"] == "Linux":
            self.canvas.unbind_all("<4>")
            self.canvas.unbind_all("<5>")
        else:
            self.canvas.unbind_all("<MouseWheel>")

    def _unbind_scroll_x(self):
        if PARAMS["sys"] == "Linux":
            self.canvas.unbind_all("<Shift-Button-4>")
            self.canvas.unbind_all("<Shift-Button-5>")
        else:
            self.canvas.unbind_all("<Shift-MouseWheel>")

    def _bind_scroll(self, *args):
        self._bind_scroll_y()
        self._bind_scroll_x()

    def _unbind_scroll(self, *args):
        self._unbind_scroll_y()
        self._unbind_scroll_x()

    def _on_mouse_wheel(self, event):
        self.canvas.yview_scroll(self._get_delta(event), "units")

    def _on_shift_mouse_wheel(self, event):
        self.canvas.xview_scroll(self._get_delta(event), "units")

    def _get_delta(self, event):
        delta = -1 * event.delta if PARAMS["sys"] != "Linux" else -1
        if PARAMS["sys"] == "Windows":
            delta /= 120

        if PARAMS["sys"] == "Linux" and event.num == 5:
            delta *= -1

        return delta


class TextConsole:
    """Text widget with search and auto -refresh capabilities."""

    def __init__(
        self,
        holder,
        content_var,
        height=None,
        width=None,
        search=False,
        disabled=True,
    ):
        """Startup class.

        holder : Tkwidget where to pack the text
        content: Tkstring to display in the widget
        """
        self.content_var = content_var
        self.disabled = disabled
        self.search = search

        self._create_widgets(holder, height, width)
        self._set_text(self.content_var.get())

    def configure(self, **kwargs):
        self.text_holder.configure(**kwargs)

    def _create_widgets(self, holder, height, width):
        self.container = ttk.Frame(holder, relief="sunken")
        self.container.pack(
            fill="both",
        )

        self.body = ttk.Frame(
            self.container,
        )
        self.body.pack(fill="x", side="bottom", padx=2, pady=2)

        if self.disabled:
            self.text_holder = ScrolledText(
                self.body,
                background=PARAMS["bg"],
                highlightbackground=PARAMS["bg_lbl"],
            )
            self.text_holder.configure(state="disabled")

        else:
            self.text_holder = ScrolledText(self.body)
            self.text_holder.bind("<KeyRelease>", self.on_text_update)

        if height is not None:
            self.text_holder.configure(height=height)
        if width is not None:
            self.text_holder.configure(width=width)

        self.text_holder.pack(fill="both")

        if self.search:
            self._create_search_widgets()

    def _create_search_widgets(self):
        self.controls = ttk.Frame(self.container)
        self.controls.pack(fill="x", side="top")

        self.search_var = Tk_Variable()
        self.search_lbl = ttk.Label(self.controls, text="Search")
        self.search_ent = ttk.Entry(self.controls, textvariable=self.search_var)
        self.search_lbl.pack(side="right")
        self.search_ent.pack(side="right")

        self.search_var.trace("w", self.highlight_pattern)

    def on_text_update(self, *args):
        self._update_content(self.get())

    def _update_content(self, content):
        self.content_var.set(content)

        if self.search:
            self.highlight_pattern()

    def _set_text(self, content):
        self.configure(state="normal")
        self.text_holder.delete(1.0, "end")
        self.text_holder.insert(1.0, content)

        if self.disabled:
            self.configure(state="disabled")

    def set_text(self, content):
        self._set_text(content)
        self._update_content(content)

    def highlight_pattern(self, *args):
        """Highlight the pattern."""
        self.text_holder.tag_delete("highlight")

        if self.search_var.get() == "":
            return

        self.text_holder.mark_set("matchStart", 1.0)
        self.text_holder.mark_set("matchEnd", 1.0)
        self.text_holder.mark_set("searchLimit", "end")
        count = tk.StringVar()
        pattern = self.search_var.get()
        if pattern:
            while True:
                index = self.text_holder.search(
                    self.search_var.get(),
                    "matchEnd",
                    "searchLimit",
                    count=count,
                    regexp=True,
                )
                if index == "":
                    break

                if count.get() == 0:
                    break
                self.text_holder.mark_set("matchStart", index)
                self.text_holder.mark_set("matchEnd", "%s+%sc" % (index, count.get()))
                self.text_holder.tag_add("highlight", "matchStart", "matchEnd")
        self.text_holder.tag_config("highlight", background="yellow")

    def get(self):
        return self.text_holder.get("1.0", "end")



class TextRedirector(io.TextIOBase):
    """A class to redirect stdout and stderr to a tkinter Text widget"""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state=NORMAL)
        self.widget.insert(END, str, (self.tag,))
        self.widget.see(END)  # Auto-scroll to the bottom
        self.widget.configure(state=DISABLED)

    def flush(self):
        pass
