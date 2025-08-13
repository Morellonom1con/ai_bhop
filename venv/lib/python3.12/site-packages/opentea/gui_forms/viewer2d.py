from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk, Variable
from arnica.plots.hoveritems import HoverItems
from opentea.gui_forms._exceptions import SchemaException
from opentea.gui_forms.constants import load_and_run_process

PADDING = 3


class Viewer2D:
    def __init__(
        self, master: ttk.Frame, otroot, callback_2d: callable, controls: dict = None
    ):
        """Creation of a viewer for matplotlib figures"""
        self.otroot = otroot
        self.callback_2d = callback_2d
        self.ctlvars = {}

        # Header frame to refresh
        _header_frame = ttk.Frame(master)
        _header_frame.pack(side="left", padx=PADDING, pady=PADDING)
        refresh2d = ttk.Button(
            _header_frame, text="Refresh", command=self.refresh_2d_view
        )
        refresh2d.pack(side="top")
        self.add_controls(_header_frame, controls)

        # Create a Frame to hold the plot
        _canvas_frame = ttk.Frame(master)
        _canvas_frame.pack(
            side="top", padx=PADDING, pady=PADDING, fill="both", expand=True
        )
        # Create the Matplotlib figure and axes
        _fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = _fig.add_subplot(111)
        # self.fig, self.ax = plt.subplots()
        # Embed the figure into the Tkinter frame
        self.canvas = FigureCanvasTkAgg(_fig, master=_canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="top", fill="both", expand=True)

        # create the hover object
        self.hover = HoverItems(self.ax)

        # Add the Matplotlib Navigation Toolbar
        _footer_frame = ttk.Frame(master)
        _footer_frame.pack(side="top", fill="x", padx=PADDING, pady=PADDING)
        _toolbar = NavigationToolbar2Tk(self.canvas, _footer_frame)
        _toolbar.update()
        _toolbar.pack(side="top", fill="x", expand=True)

    def add_controls(self, frame, controls: dict):
        """create the control panels"""
        if controls is None:
            return
        for name, control in controls.items():
            ot_type = control.get("ot_type", None)
            title = control.get("title", name)
            lbl = ttk.Label(frame, text=title)
            lbl.pack(side="top")

            if ot_type == "radiobutton":
                self.ctlvars[name] = Variable()
                options = control.get("options", None)
                if options is None:
                    raise SchemaException(
                        f"Field 'option' is missing for radiobutton {name} in View2d specification"
                    )
                values = [opt.lower().replace(" ", "_") for opt in options]
                for opt, value in zip(options,values):
                    value = opt.lower().replace(" ", "_")
                    rad_btn = ttk.Radiobutton(
                        frame,
                        text=opt,
                        value=value,
                        variable=self.ctlvars[name],
                    )
                    rad_btn.pack(side="top", anchor="w")
                self.ctlvars[name].set(values[0])
            else:
                raise SchemaException(
                    f"Control {name} type {ot_type} not supported yet in viewerd 2D"
                )

    def get_controls(self) -> dict:
        """return the content of the controls vars"""
        out = {}
        for name, tkvar in self.ctlvars.items():
            out[name] = tkvar.get()
        return out

    def refresh_2d_view(self):
        """callback for the button refresh"""
        self.ax.clear()
        self.hover.clean_hover()
        load_and_run_process(
            self.callback_2d,
            self.ax,
            self.hover,
            self.otroot.get(),
            self.get_controls(),
        )
        self.canvas.draw_idle()
