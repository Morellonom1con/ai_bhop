"""Class to be added to a pyplot axis

allow the easy addition of line and markers featuring hover baloons


fig = plt.figure(figsize=(5, 4), dpi=100)
axes = fig.add_subplot(111)
hover = HoverItems(axes)
(...)
axes.plot(x,y,"bar",color="blue", linewidth=1) # line without baloons
hover.add_line(x,y,"foo",color="red", linewidth=3)
hover.add_marker(x,y,"foo",color="blue", size=10, shape="x")

plt.show()
"""

# import logging
# _log_plt = logging.getLogger(__name__)
# _log_plt.setLevel("ERROR")

# necessary to catch a boring warning message
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)


class HoverItems:
    def __init__(self, ax):
        self.ax = ax
        self.lines = []
        self.markers = []
        self.cid = self.ax.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_hover
        )
        self.current_focus = (None, None)
        # self.markers_tree=None
        # self.nodes_xy=[]

    def add_line(self, x, y, label, color=None, linewidth=None):
        """Add a line to the plot with a tooltip"""
        (line,) = self.ax.plot(
            x, y, label=label, picker=5, color=color, linewidth=linewidth
        )
        annotation = self.ax.annotate(
            label,
            xy=(x[0], y[0]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            visible=False,
        )

        self.lines.append((line, annotation))
        # self.current_focus = line, annotation

    def add_marker(self, x, y, label, color=None, size=5, shape="o", edgecolor="black"):
        """Add a marker to the plot with a tooltip"""
        (marker,) = self.ax.plot(
            x,
            y,
            label=label,
            markersize=size / 10,
            linestyle=None,
            marker=shape,
            picker=5,
            color=color,
            markeredgecolor=edgecolor,
        )  # **kwargs)

        annotation = self.ax.annotate(
            label,
            xy=(x, y),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            visible=False,
        )
        # self.nodes_xy.append((x, y))
        self.markers.append((marker, annotation))

    # self.current_focus = marker, annotation

    # def init_marker_tree(self):
    #     xy_arr = np.array(self.nodes_xy)
    #     self.markers_tree = cKDTree(xy_arr)

    def on_hover(self, event):
        # Check if the mouse is over the axes and valid data
        if event.inaxes != self.ax:
            return

        # skip if still on the same item
        try:
            past_item, _ = self.current_focus
            past_cont, _ = past_item.contains(event)
            if past_cont:
                return
        except AttributeError:
            # AttributeError: 'NoneType' object has no attribute 'contains'
            pass

        # solution using a Kdtree for fast finding, not really faster apperently
        # if self.markers_tree is not None:
        #     dd, ii = self.markers_tree.query((event.xdata, event.ydata), k=[1])
        #     marker_item, marker_annotation = self.markers[ii[0]]
        #     cont, _ = marker_item.contains(event)
        #     if cont:
        #         self._update_annotation(event.xdata, event.ydata, marker_item,marker_annotation, past_annotation)
        #         return
        # else:
        for marker_item, marker_annotation in self.markers:
            cont, _ = marker_item.contains(event)
            if cont:
                self._update_annotation(
                    marker_item,
                    marker_annotation,
                )
                return

        for line_item, line_annotation in self.lines:
            cont, _ = line_item.contains(event)
            if cont:
                self._update_annotation(
                    line_item, line_annotation, xy=(event.xdata, event.ydata)
                )
                return

        # event was neither in past item, markers nor line
        self._update_annotation(None, None)

    def clean_hover(self):
        """Remove everthing"""
        self.current_focus = (None, None)
        self.lines = []
        self.markers = []

    def _update_annotation(self, item, annotation, xy=None):
        """update the baloon help"""
        _, past_annotation = self.current_focus

        if past_annotation is not None:
            past_annotation.set_visible(False)

        if item is None:
            self.current_focus = (None, None)
        else:
            self.current_focus = (item, annotation)
            if xy is not None:
                annotation.xy = xy
            annotation.set_visible(True)

        self.ax.figure.canvas.draw_idle()
