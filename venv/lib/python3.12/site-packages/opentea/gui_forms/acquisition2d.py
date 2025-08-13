"""2D Acquisition widget


Developer notes: We want to keep this as simple as posible <400LOC (functions excluded)

Data is stored in PIXELS: indeed, frequent open/save would gradually modify the acquisition

"""


from __future__ import annotations
from tkinter import ttk
import platform

# from opentea.gui_forms.root_widget import OTRoot
from loguru import logger
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# from opentea.gui_forms.acquisition2d_utils import (
from arnica.utils.brokenlines_actions import (
    coords_pix_2_real,
    find_line_from_point,
    is_near_point,
    corner_coordinates,
    acq_lines_2_pix,
    acq_lines_2_real,
    closest_line,
    point_from_curvi,
)


COLOR_CYCLE = [  # paul tol's vivid caregory colors
    "#0077BB",
    "#33BBEE",
    "#009988",
    "#EE7733",
    "#CC3311",
    "#EE3377",
    "#BBBBBB",
]

INITIAL_CALIB_DIAG_REAL = [
            [0.1, 0.1],
            [0.9, 0.9],
        ]
class InteractivelineDrawer:
    def __init__(self, master: ttk.Frame, root, acq2d_options: dict  =None):
        self.root = root

        self.reserved_names= None
        self.reserved_colors= None
        
        if acq2d_options is not None:
            self.reserved_names = ["Unnamed"]+ acq2d_options.get("reserved_names", None)
            self.reserved_colors = ["#000000"]+acq2d_options.get("reserved_colors", COLOR_CYCLE)

        self.image_filename = None  # filename to the image
        self.acq_mode = tk.BooleanVar(value=False)  # filename to the image
        self.x1 = tk.StringVar()
        self.y1 = tk.StringVar()
        self.x2 = tk.StringVar()
        self.y2 = tk.StringVar()
        self.x1.trace_add("write", self.update_world_coords)
        self.x2.trace_add("write", self.update_world_coords)
        self.y1.trace_add("write", self.update_world_coords)
        self.y2.trace_add("write", self.update_world_coords)

        self.ratio_text = tk.StringVar()
        self.last_position_pix = [None, None]
        self.closest_curvi = [None, None, None]
        self.former_calib_diag_real = None
        self._init_create_control_panel(master, frame_width_px=300).pack(
            padx=2, pady=3, side=tk.LEFT, fill=tk.Y
        )
        self._init_create_viewport_panel(master).pack(
            side="top", fill="both", padx=2, pady=3, expand=True
        )

        # INTERNAL MEMORY
        self.image = None  # PIL image to be rendered, no image displayed if None
        self.lines_list = [[[1.0, 1.0], [9.0, 9.0]]]  # Main data holder
        # a List of Lines
        #      Lines are Lists of Points
        #            Points are list of two float coordinates
        #      -> First one is the calibration frame
        self.lines_colors = [0]  # Store the current color index for each line
        
        # If a line is added/removel, the list must be updated accordingly
        self.lines_names = ["Frame"]
        # INTERNAL MEMORY

        # dragging data
        self.dragging = False  # are we dragging an  point
        # self.start_point = None  # Used when creating a line
        self.selected_point = None  # coordinates of the closest point

        # Connect mouse and keyboard events
        self.cid_press = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )
        self.cid_release = self.fig.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cid_motion = self.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        if platform.system() == 'Darwin':  # macOS
            self.root.tksession.bind("<Control-a>", self.add_line)
            self.root.tksession.bind("<Control-r>", self.subdivide_line)
            self.root.tksession.bind("<Control-m>", self.delete_point)
            self.root.tksession.bind("<Control-d>", self.delete_line)
            self.root.tksession.bind("<Control-l>", self.change_color)
            self.root.tksession.bind('<Control-n>', self.name_line)
        else:  # Windows/Linux
            self.root.tksession.bind("<Alt-a>", self.add_line)
            self.root.tksession.bind("<Alt-r>", self.subdivide_line)
            self.root.tksession.bind("<Alt-m>", self.delete_point)
            self.root.tksession.bind("<Alt-d>", self.delete_line)
            self.root.tksession.bind("<Alt-l>", self.change_color)
            self.root.tksession.bind('<Alt-n>', self.name_line)
        
    def _init_create_control_panel(
        self, master: ttk.Frame, frame_width_px: int
    ) -> ttk.Frame:
        """Create the controls panel in the self.root frame

        NB: keep only active widgets as attributes"""

        control_frame = ttk.Frame(master, width=f"{frame_width_px}px")
        # control_frame.pack_propagate(False)

        # File selection button and label
        self.file_label = ttk.Label(
            control_frame, textvariable=self.image_filename, wraplength=frame_width_px
        )
        self.file_label.pack(pady=5)

        file_button = ttk.Button(
            control_frame, text="Select Image File", command=self.select_image_file
        )
        file_button.pack(pady=5)

        # Coordinates
        coord_frame = ttk.Frame(control_frame)
        coord_frame.pack(pady=10)

        vcmd = control_frame.register(validate_input)

        entry_width = int(frame_width_px / 30)
        x1_label = ttk.Label(coord_frame, text="X0:")
        x1_label.grid(row=0, column=0)
        self.x1_entry = ttk.Entry(
            coord_frame,
            textvariable=self.x1,
            width=entry_width,
            validate="key",
            validatecommand=(vcmd, "%P"),
        )
        self.x1_entry.grid(row=0, column=1)
        self.x1.set(INITIAL_CALIB_DIAG_REAL[0][0])

        y1_label = ttk.Label(coord_frame, text="Y0:")
        y1_label.grid(row=0, column=2)
        self.y1_entry = ttk.Entry(
            coord_frame,
            textvariable=self.y1,
            width=entry_width,
            validate="key",
            validatecommand=(vcmd, "%P"),
        )
        self.y1_entry.grid(row=0, column=3)
        self.y1.set(INITIAL_CALIB_DIAG_REAL[0][1])

        x2_label = ttk.Label(coord_frame, text="X1:")
        x2_label.grid(row=1, column=0)
        self.x2_entry = ttk.Entry(
            coord_frame,
            textvariable=self.x2,
            width=entry_width,
            validate="key",
            validatecommand=(vcmd, "%P"),
        )
        self.x2_entry.grid(row=1, column=1)
        self.x2.set(INITIAL_CALIB_DIAG_REAL[1][0])

        y2_label = ttk.Label(coord_frame, text="Y1:")
        y2_label.grid(row=1, column=2)
        self.y2_entry = ttk.Entry(
            coord_frame,
            textvariable=self.y2,
            width=entry_width,
            validate="key",
            validatecommand=(vcmd, "%P"),
        )
        self.y2_entry.grid(row=1, column=3)
        self.y2.set(INITIAL_CALIB_DIAG_REAL[1][1])

        self.ratio_label = ttk.Label(control_frame, textvariable=self.ratio_text)
        self.ratio_label.pack(pady=5)

        acq_frame = ttk.LabelFrame(control_frame, text="Creation mode")
        acq_frame.pack(pady=5)

        self.mode_button = ttk.Checkbutton(acq_frame, variable=self.acq_mode)
        self.mode_button.pack(pady=5)

        self.help = ttk.Label(
            acq_frame,
            text="""
Hotkeys:
Alt-a : Add line
Alt-d : Delete line
Alt-r : Refine pt.
Alt-m : Merge pt.
Alt-l : cycLe
Alt-n : Name

OSx key: Alt-> Control
""",
        )
        self.help.pack(pady=5)

        # Add a button to print line coordinates
        # self.print_button = ttk.Button(
        #     control_frame, text="Apply acquisition", command=self.apply_acquisition
        # )
        # self.print_button.pack(pady=10)

        # add a feedback info
        self.coord_label = ttk.Label(control_frame, text="Coordinates: (X, Y)")
        self.coord_label.pack(pady=5)
        return control_frame

    def _init_create_viewport_panel(
        self, master: ttk.Frame, frame_width_px: int = 80
    ) -> ttk.Frame:
        """Create the viewport panel in the self.root frame

        NB: keep only active widgets as attributes"""
        # Create a frame for the plot
        viewport_frame = ttk.Frame(master, width=frame_width_px)
        # viewport_frame.pack(side="top", fill="both", padx=2, pady=3)

        # viewport_frame.pack_propagate(False)

        # Create the Matplotlib figure and axes
        # self.fig, self.ax = plt.subplots()
        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.fig.subplots_adjust(
            left=0.04, right=0.99, top=0.99, bottom=0.04, wspace=None, hspace=None
        )

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect("equal")

        # Embed the figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=viewport_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # , side=tk.TOP, )

        # Add the Matplotlib Navigation Toolbar
        _toolbar = NavigationToolbar2Tk(self.canvas, viewport_frame)
        _toolbar.update()
        _toolbar.pack(side="bottom", fill="x")

        return viewport_frame

    def on_click(self, event):
        """Call back when the Mouse Button1 (Left) is clicked

        - Start dragging if there is a selected point
        - Else, store a start point to create a new line
        """
        if event.inaxes != self.ax:
            return

        if self.acq_mode.get() and self.selected_point is not None:
            self.dragging = True
        # else:
        #     self.start_point = (float(event.xdata), float(event.ydata))

    def on_motion(self, event):
        """Call back when the pointer is hovering on the canvas (clicked or not)

        > Update the coordinates feedback
        - If dragging a point, update its coordinates
        - Else check if a point is under focus
        > update viewport
        """
        if event.inaxes != self.ax:
            self.last_position_pix = [None, None]
            self.closest_curvi = [None, None, None]
            return

        self.last_position_pix = [float(event.xdata), float(event.ydata)]
        self.update_position_label()

        self.closest_curvi = closest_line(self.lines_list[1:], *self.last_position_pix)
        # # Update the coordinates display
        # xr, yr = self.get_real_coords(self.last_position_pix[0], self.last_position_pix[1])
        # coord_text = f"Loc: {xr:.2f}, {yr:.2f} ({event.xdata:.2f}, {event.ydata:.2f})"
        # id_line = find_line_from_point(self.lines_list, self.selected_point)
        # if id_line is not None:
        #     coord_text += f"\n (Line #{id_line} {self.lines_names[id_line]})"
        # self.coord_label.config(text=coord_text)

        corner0, corner1 = self.lines_list[0]
        if self.dragging and self.selected_point is not None:
            # prevent overmotion of frame
            if self.selected_point == corner0:
                if event.ydata >= corner1[1]:
                    return
                if event.xdata >= corner1[0]:
                    return
            if self.selected_point == corner1:
                if event.ydata <= corner0[1]:
                    return
                if event.xdata <= corner0[0]:
                    return

            if self.selected_point in [corner0, corner1]:
                cur_lines = acq_lines_2_real(
                    self.lines_list[1:], self.calib_diag_pix(), self.calib_diag_real()
                )

                self.selected_point[0], self.selected_point[1] = (
                    float(event.xdata),
                    float(event.ydata),
                )  # update selected points data
                self.lines_list[1:] = acq_lines_2_pix(
                    cur_lines, self.calib_diag_pix(), self.calib_diag_real()
                )
                self.update_ratio_label()
            else:
                self.selected_point[0], self.selected_point[1] = (
                    float(event.xdata),
                    float(event.ydata),
                )  # update selected p
        else:
            self.update_selected_point(event)  # check if a point is to be selected

        self.update_view()  # both cases update visual

    def on_release(self, event):
        """Callback when the MuseButton 1 (LeftButton) is released

        - If dragging, stop dragging
        - If creating a line, end the line creation, and update view
        """
        if event.inaxes != self.ax:
            return

        if self.dragging:
            self.dragging = False
            self.selected_point = None
            return

    def update_position_label(self):
        xr, yr = self.get_real_coords(
            self.last_position_pix[0], self.last_position_pix[1]
        )
        coord_text = f"Loc: {xr:.3f}, {yr:.3f} ({self.last_position_pix[0]:.3f}, {self.last_position_pix[1]:.3f})"
        id_line = find_line_from_point(self.lines_list, self.selected_point)
        if id_line is not None:
            coord_text += f"\n (Line #{id_line} {self.lines_names[id_line]})"
        self.coord_label.config(text=coord_text)

    def update_world_coords(self, *args):
        if self.former_calib_diag_real is None:
            return
        #logger.warning("UWC")
        acq_line_real = acq_lines_2_real(
            self.lines_list, self.calib_diag_pix(), self.former_calib_diag_real
        )
        self.lines_list[1:] = acq_lines_2_pix(
            acq_line_real[1:], self.calib_diag_pix(), self.calib_diag_real()
        )

        self.former_calib_diag_real = self.calib_diag_real()

        self.update_ratio_label()
        self.update_view() 



    def update_ratio_label(self):    
        #logger.warning("URL")
        [[xr0, yr0], [xr1, yr1]] = self.calib_diag_real()
        [[xp0, yp0], [xp1, yp1]] = self.calib_diag_pix()
        ratio_x = (xr1 - xr0) / (xp1 - xp0)
        ratio_y = (yr1 - yr0) / (yp1 - yp0)
        self.ratio_text.set(f"ratio x/y: ={ratio_x:.4f}/{ratio_y:.4f}")

    def update_selected_point(self, event):
        """Detect the selected point, if any"""
        self.selected_point = None
        min_dist = 1e6
        for line in self.lines_list:
            for point in line:
                dist = is_near_point(event.xdata, event.ydata, point[0], point[1])
                if dist is not None and dist < min_dist:
                    min_dist = dist
                    self.selected_point = point

    def calib_diag_real(self):
        """return the calibration data in real coordinates"""
        return [
            [float(self.x1.get()), float(self.y1.get())],
            [float(self.x2.get()), float(self.y2.get())],
        ]

    def calib_diag_pix(self):
        """return the calibration data in pix coordinates"""
        return self.lines_list[0]

    def get_real_coords(self, x: float, y: float):
        """Return coords in real world, from wiewport coords"""
        xr, yr = coords_pix_2_real(x, y, self.calib_diag_pix(), self.calib_diag_real())
        return xr, yr

    def update_view(self):
        """Update the viewport display

        > display image
        > display frame
        > display lines
        """
        # reset
        cur_x_lim = self.ax.get_xlim()
        cur_y_lim = self.ax.get_ylim()

        self.ax.clear()

        # Image
        if self.image is not None:
            imwidth = 10 * self.image.width / max(self.image.width, self.image.height)
            imheight = 10 * self.image.height / max(self.image.width, self.image.height)
            self.ax.imshow(
                self.image, extent=[0, imwidth, 0, imheight], interpolation=None
            )

        
        # display lines
        for i, line in enumerate(self.lines_list[1:]):
            x_coords = [pt_[0] for pt_ in line]
            y_coords = [pt_[1] for pt_ in line]
            color = self.reserved_colors[self.lines_colors[i + 1]]
            gain = 1
            if self.selected_point in line:
                gain = 4
            self.ax.plot(
                x_coords,
                y_coords,
                color=color,
                marker="o",
                linewidth=0.5 * gain,
                markersize=2 * gain,
            )

        if self.closest_curvi[0] is not None:
            [[xr0, yr0], [xr1, yr1]] = self.calib_diag_real()
            [[xp0, yp0], [xp1, yp1]] = self.calib_diag_pix()
            ratio_x = (xr1 - xr0) / (xp1 - xp0)
            ratio_y = (yr1 - yr0) / (yp1 - yp0)

            closest_line, closest_point, closest_curvi = self.closest_curvi
            x, y, sum_curvi = point_from_curvi(
                self.lines_list[closest_line + 1], closest_point, closest_curvi
            )
            sum_curvi *= (ratio_x + ratio_y) * 0.5
            xr, yr = self.get_real_coords(x, y)
            position_txt = f"{xr:.3f}:{yr:.2f}\n{self.lines_names[closest_line+1]} / {sum_curvi:3f}"
            self.ax.plot(
                [x],
                [y],
                color="black",
                marker="+",
                markersize=4,
            )
            self.ax.text(x+0.2, y+0.2, position_txt)

        self.ax.set_xlim(*cur_x_lim)
        self.ax.set_ylim(*cur_y_lim)


        #Must be at the end to get the whole bounding box
        # Frame
        (_x0, _y0), (_x1, _y1) = self.lines_list[0]
        x0, y0, x1, y1 = corner_coordinates(
            _x0, _y0, _x1, _y1
        )  # reorder coords if the drag is mixing min and max
        gain = 1
        x_min, x_max = cur_x_lim
        y_min, y_max = cur_y_lim
        if self.selected_point == [x0, y0]:
            gain = 2
        self.ax.plot([x0], [y0], color="black", marker="o", markersize=2 * gain)
        self.ax.plot(
            [x0, x0, None, x_min, x_max],
            [y_min, y_max,None, y0,y0],
            color="black",
            marker=None,
            linewidth=0.5 * gain,
            linestyle="dashed",
            markersize=4,
        )

        gain = 1
        if self.selected_point == [x1, y1]:
            gain = 2
        self.ax.plot([x1], [y1], color="darkgrey", marker="o", markersize=2 * gain)
        self.ax.plot(
            [x1, x1, None, x_min, x_max],
            [y_min, y_max,None, y1,y1],
            color="darkgrey",
            marker=None,
            linewidth=0.5 * gain,
            linestyle="dashed",
            markersize=4,
        )


        self.fig.canvas.draw()

    def name_line(self, event):
        """Callback to rename a line"""
        if self.acq_mode.get() is False:
            logger.warning(f"Creation mode is OFF.")
            return
    
        if self.reserved_names is not None:
            logger.warning(f"Cannot rename (n), Names are to be chosen between {str(self.reserved_names)}. Use 'Cycle' (l) instead.")
            return
    
        id_line = find_line_from_point(self.lines_list, self.selected_point)
        if id_line in [None, 0]:
            return

        user_input = simpledialog.askstring("Rename line", "Line name")
        if user_input is  None:
            print("User cancelled the input")
            return
        new_name = user_input.strip().replace(" ", "_")
            
        self.lines_names[id_line] = new_name

    def subdivide_line(self, event):
        """Callback to subdivide a line"""
        if self.acq_mode.get() is False:
            logger.warning(f"Creation mode is OFF.")
            return
        id_line = find_line_from_point(self.lines_list, self.selected_point)
        if id_line in [None, 0]:
            return

        line = self.lines_list[id_line]
        new_line = [line[0]]
        for j, ptp1 in enumerate(line[1:]):
            pt = line[j]
            if self.selected_point in [pt, ptp1]:
                mid_point = [(pt[0] + ptp1[0]) / 2, (pt[1] + ptp1[1]) / 2]
                new_line.append(mid_point)
            new_line.append(ptp1)

        self.lines_list[id_line] = new_line

        self.update_view()

    def delete_point(self, event):
        """Callback to remove a point from a line"""
        if self.acq_mode.get() is False:
            logger.warning(f"Creation mode is OFF.")
            return

        id_line = find_line_from_point(self.lines_list, self.selected_point)
        if id_line in [None, 0]:
            return

        line = self.lines_list[id_line]
        new_line = [pt_ for pt_ in line if pt_ != self.selected_point]
        self.lines_list[id_line] = new_line
        if len(new_line) <= 1:  # remove if line is below one point
            self.lines_list.pop(id_line)
            self.lines_colors.pop(id_line)
            self.lines_names.pop(id_line)

        self.update_view()

    def add_line(self, event):
        """add to remove a line"""
        if self.acq_mode.get() is False:
            logger.warning(f"Creation mode is OFF.")
            return
        x1, y1 = self.last_position_pix
        if x1 is None:
            return
        self.lines_list.append(
            [
                list([x1, y1]),
                list([x1 + 1, y1 + 1]),
            ]
        )
        self.lines_colors.append(0)
        self.lines_names.append("Unnamed")
        self.update_view()

    def delete_line(self, event):
        """Callback to remove a line"""
        if self.acq_mode.get() is False:
            logger.warning(f"Creation mode is OFF.")
            return

        id_line = find_line_from_point(self.lines_list, self.selected_point)
        if id_line in [None, 0]:
            return

        self.lines_list.pop(id_line)
        self.lines_names.pop(id_line)
        self.lines_colors.pop(id_line)
        self.selected_point = None
        self.update_view()

    def change_color(self, event):
        """Callback to cycle the color of a line"""
        if self.acq_mode.get() is False:
            logger.warning(f"Creation mode is OFF.")
            return

        if self.selected_point is None:
            logger.warning(f"No point selected.")
            return

        for id_line, line in enumerate(self.lines_list):
            if self.selected_point in line:
                self.lines_colors[id_line] = (self.lines_colors[id_line] + 1) % len(self.reserved_colors)

                if self.reserved_names is not None:
                    idx_name = self.reserved_names.index(self.lines_names[id_line])
                    idx_name +=1
                    if idx_name == len(self.reserved_names):
                        idx_name = 0

                    self.lines_names[id_line] = self.reserved_names[idx_name]
                self.update_view()
                return

    def select_image_file(self):
        """Callback to select the image file"""
        filetypes = [
            ("PNG files", "*.png"),
            ("GIF files", "*.gif"),
            ("All files", "*.*"),
        ]
        filename = filedialog.askopenfilename(
            title="Select Image File", filetypes=filetypes
        )

        if filename:
            self.update_image(filename)
            self.update_view()

    def update_image(self, filename):
        self.image_filename = filename
        if filename is not None:
            self.image = Image.open(self.image_filename)

    # def apply_acquisition(self):
    #     """Basic output of the memory content"""
    #     self.root.set(self.callback_2d_acq(self.root.get(), self.get()))

    def get(self) -> dict:
        """Basic output of the memory content"""

        data = {
            "acq_line_pix": self.lines_list[1:],
            "acq_line_real": acq_lines_2_real(
                self.lines_list[1:], self.calib_diag_pix(), self.calib_diag_real()
            ),
            "acq_colors": self.lines_colors[1:],
            "acq_names": self.lines_names[1:],
            "acq_calib_diag_pix": self.calib_diag_pix(),
            "acq_calib_diag_real": self.calib_diag_real(),
            "acq_image": self.image_filename,
        }
        return data

    def set(self, acq_data: dict):
        """Basic output of the memory content"""
        ((x1, y1), (x2, y2)) = acq_data["acq_calib_diag_real"]
        self.x1.set(x1)
        self.y1.set(y1)
        self.x2.set(x2)
        self.y2.set(y2)

        self.lines_list = acq_lines_2_pix(
            acq_data["acq_line_real"], acq_data["acq_calib_diag_pix"], self.calib_diag_real()
        )
        self.lines_list.insert(0, acq_data["acq_calib_diag_pix"])
        self.lines_colors = [0] + acq_data["acq_colors"]
        self.lines_names = ["Frame"] + acq_data["acq_names"]

        self.update_image(acq_data["acq_image"])
        self.update_ratio_label()
        self.update_view()
        
    def allow_recalibration(self):
        """this must be applied after a full load.
        
        Doing  on calibration variables from start lead to a dead end:
            there a 4 variables, as soon as the first is set everything is recalibrated
        """
        self.former_calib_diag_real = self.calib_diag_real()
            

def validate_input(new_value):
    """Allow only valid numeric coordinates (non-empty and not just '-')"""
    if new_value == "" or new_value == "-":
        return False  # Reject empty or single minus sign
    try:
        float(new_value)  # Attempt to convert to float
        return True
    except ValueError:
        return False  # Reject invalid inputs


# Initialize Tkinter application
if __name__ == "__main__":
    root = tk.Tk()
    app = InteractivelineDrawer(root)
    root.mainloop()
