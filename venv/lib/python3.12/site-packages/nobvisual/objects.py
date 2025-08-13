from __future__ import annotations
import tkinter as tk
from tkinter import Canvas,ttk
from typing import Tuple
from nobvisual.utils import shade_color
from nobvisual.utils import get_hex_from_color_scale



ATOL = 1e-4


class PackingCircle:
    """The typical circle used to create the Tkinter object
    
    x and y are the lower left coords of the circle, add size to get the center
    """
    def __init__(self, 
            x: float, 
            y: float, 
            r: float, 
            level: int=0, 
            color='default', 
            name: str='', 
            #short_name: str=None,
            text: str='', 
            #short_text: str='', 
            activeoutline: str='white', 
            activewidth: str=4,
        #    click_behavior: bool=None
            ):
        self.x = x
        self.y = y
        self.r = r
        self.level = level
        self._color = color
        self.name = name
        self.text = text
        self.activeoutline = activeoutline
        self.activewidth = activewidth
        self.canvas = None      # Tk canvas where the fun goes on
        self._id = None         # Tk circle object to be shown
        self._name_label = None # Tk name object to be shown
        self.parent = None
        self.children = []

    def __str__(self):
        
        try:
            pname  = self.parent.name
        except AttributeError:
            pname = None

        cname = [cp.name for cp in self.children]
        str_ = f"""
Packing Circle:
==============
name: {self.name}
text: {self.text}
parent: Packing Circle {pname}
children: Packing Circles {cname}
"""
        return str_

   

    #Propseter for id
    @property
    def id(self):
        """Return id of self"""
        return self._id
    @id.setter
    def id(self, value):
        if self._id is not None:
            raise Exception('A widget can be set only once')
        self._id = value
        self._on_widget_creation()

    #Propseter for color
    @property
    def color(self):
        """Return color of self"""
        if self.canvas is None:
            return self._color
        else:
            return self.canvas.adjust_color(self._color, self.level, self.is_leaf())
    @color.setter
    def color(self):
        return self._color

    @property
    def size(self)-> float:
        return self.canvas.map_r(self.r) / self.canvas.scale_factor

    def is_leaf(self)-> bool:
        """Test if object is a leaf or not"""
        return len(self.children) == 0

    @property
    def canvas_coords(self)-> Tuple[float,float]:
        """return absolute coordinates of object"""
        x, y, *_ = self.canvas.coords(self.id)
        return [x + self.size, y + self.size]

    def get_unscaled_canvas_coords(self):
        """return canvas coordinates of xy"""
        # assumes no movement after instatiation
        return self.canvas.map2canvas(self.x, self.y)

    

    def _on_widget_creation(self):
        """what happend at the start"""
        self._config_bindings()

    def create_widget(self, canvas: CircularPackingCanvas):
        """Addition of a widget"""
        self.canvas = canvas # keep the canvas on with the widget is presented

        x, y = self.get_unscaled_canvas_coords()
        r = self.canvas.map_r(self.r)

        (x0, y0), (x1, y1) = _get_rect_coords(x, y, r)

        self.id = self.canvas.create_oval(
            x0, y0, x1, y1, 
            fill=self.color, 
            outline=shade_color(self.color, -.3),
            activeoutline=self.activeoutline,
            activewidth=self.activewidth
        )
        return self.id

    def add_children(self, circle: PackingCircle):
        self.children.append(circle)
        circle.parent =self

    # def assign_parent(self, circle: PackingCircle):
    #     self.parent = circle

    def _config_bindings(self):
        """Addition of bindings"""
        self.canvas.tag_bind(self.id, '<Enter>', self.on_enter)
        self.canvas.tag_bind(self.id, '<Leave>', self.on_leave)
        self.canvas.tag_bind(self.id, '<Button-1>', self.on_click)

    def on_enter(self, event):
        """What happens when mouse enters"""
        x, y = self.canvas.adjust_event_coords(event.x, event.y)
        delta_x, delta_y = 0, 20
        self._highlighted_text = HighlightedText(
            self.text, x + delta_x, y + delta_y,
            delta_motion_x=delta_x, delta_motion_y=delta_y)
        self._highlighted_text.create_widget(self.canvas)
        self._highlighted_text.bind_motion()

    def on_leave(self, *args):
        """What happens when mouse leaves"""
        self._highlighted_text.destroy()

    def on_click(self, *args):
        """What happens when mouse click"""
        my_click(self)
        #self.click_behavior.act(self)


##### WTF
    def _get_max_name_label_size(self):
        x1, _, x2, _ = self.canvas.bbox(self.id)
        return (x2 - x1) - self.activewidth

    def _check_name_label_allowed_size(self, max_size):
        _, y1, _, y2 = self.canvas.bbox(self._name_label.id)
        if (y2 - y1) > max_size:
            self.hide_name()

    def show_name(self):
        """add a text over the widget"""
        text = self.name# self.short_name if short else self.name
        max_size = self._get_max_name_label_size()  # to avoid overlap
        #font = None if self.is_leaf() else ('Purisa', 12, 'bold')
        font = ('Purisa', 12, 'bold')


        if self._name_label is not None:
            self._name_label.name = text
            self._name_label.max_width = max_size
            self._check_name_label_allowed_size(max_size)
            return

        x, y = self.canvas_coords

        self._name_label = NameLabel(text, x, y, max_width=max_size, font=font)
        self._name_label.create_widget(self.canvas)

        # check y (if too large, then destroy)
        self._check_name_label_allowed_size(max_size)


    def hide_name(self):
        if self._name_label is not None:
            self._name_label.destroy()
        self._name_label = None


class CircularPackingCanvas(tk.Canvas):
    """Overloading of Canvas for circular packing"""
    def __init__(self, holder, size, bg="#ffffff", bd=5, base_color="#777777",
                 shade_factor=-0.1, highlightthickness=0, **kwargs):
        super().__init__(holder, width=size, height=size, bg=bg, bd=bd,
                         highlightthickness=highlightthickness, **kwargs)
        self.size = size
        self.base_color = base_color
        self.shade_factor = shade_factor

        self.configure(xscrollincrement=1)
        self.configure(yscrollincrement=1)

        self.circles = {}
        self._current_focus = None

    # propsetter of current focus
    @property
    def current_focus(self):
        """The Tk id of the object under current focus"""
        if self._current_focus is None:
            self.set_enclosing_circle(self.find_enclosing_circle())
        return self._current_focus
    @current_focus.setter
    def current_focus(self, circle):
        if circle.r == 0:
            return
        previous_focus = self._current_focus
        self._current_focus = circle
        x_previous, y_previous = previous_focus.get_unscaled_canvas_coords()
        x_new, y_new = circle.get_unscaled_canvas_coords()

        delta_x = int(x_new - x_previous)
        delta_y = int(y_new - y_previous)

        self.scale_view(x_previous, y_previous, previous_focus.r)  # unscale
        self.translate_view(delta_x, delta_y)
        self.scale_view(x_new, y_new, 1 / circle.r)

    @property
    def scale_factor(self):
        """the scale factor now"""
        return self.current_focus.r

    @property
    def real_size(self):
        border = float(self['bd']) + float(self['highlightthickness'])
        return self.size + 2 * border

    @property
    def draw_size(self):
        return self.size - 2 * float(self['bd'])

    @property
    def canvas_center(self):
        return [self.real_size * .5 for _ in range(2)]

    def adjust_event_coords(self, x, y):
        x_center, y_center = self.canvas_center  # original center
        new_x, new_y = x - x_center, y - y_center

        new_x_center, new_y_center = self.current_focus.get_unscaled_canvas_coords()
        new_x += new_x_center
        new_y += new_y_center

        return new_x, new_y

    def adjust_color(self, color, level, leaf):
        if color is None or color == 'default':
            color = self.base_color

        if color.startswith('colormap'):
            max_lvl = 3
            value = float(color.split(":")[1])
            norm_depth = max(0, (max_lvl - level) / max_lvl)
            cshade = 0.8 * (norm_depth)
            color = shade_color(get_hex_from_color_scale(value), cshade)

        else:
            # shade based on level
            if not leaf:
                for _ in range(level):
                    color = shade_color(color, self.shade_factor)

        return color

    def map2canvas(self, x_real, y_real):
        x_center, y_center = self.canvas_center

        x = x_center + x_real * self.draw_size * 0.5
        y = y_center + y_real * self.draw_size * 0.5

        return x, y

    def map_r(self, r_real):
        return r_real * self.draw_size * 0.5

    def add_circle(self, circle):
        circle_id = circle.create_widget(self)
        self.circles[circle_id] = circle

    def find_enclosing_circle(self):
        for circle in self.circles.values():
            if abs(circle.r - 1.0) < ATOL:
                return circle

    def set_enclosing_circle(self, circle):
        self._current_focus = circle

    def translate_view(self, delta_x=0, delta_y=0):
        self._translate_xview(delta_x)
        self._translate_yview(delta_y)

    def _translate_xview(self, delta_x):
        sign = 1 if delta_x > 0 else -1
        for _ in range(abs(delta_x)):
            self.xview_scroll(sign, 'units')

    def _translate_yview(self, delta_y):
        sign = 1 if delta_y > 0 else -1
        for _ in range(abs(delta_y)):
            self.yview_scroll(sign, 'units')

    def scale_view(self, x, y, ratio):
        self.scale('all', x, y, ratio, ratio)

    def hide_names(self):
        for circle in self.circles.values():
            circle.hide_name()

    def show_names(self, level=2):
        for circle in self.get_circle_by_level(level):
            circle.show_name()

    # def show_leaf_short_names(self):
    #     for circle in self.circles.values():
    #         if circle.is_leaf():
    #             circle.show_name(short=True)

    def get_circle_by_level(self, level):
        return [circle for circle in self.circles.values() if circle.level == level]


class UnderFocusLabel(ttk.Label):
    def configure_circle(self, circle):
        circle.canvas.tag_bind(circle.id, '<Enter>',
                               lambda e, circle=circle: self.on_enter_circle(circle, e),
                               add='+')
        circle.canvas.tag_bind(circle.id, '<Leave>', self.on_leave_circle,
                               add='+')

    def configure_circles(self, circles):
        for circle in circles:
            self.configure_circle(circle)

    def on_enter_circle(self, circle, *args):
        self.configure(text=circle.text)#, fg='black')
        self.pack()

    def on_leave_circle(self, *args):
        self.pack_forget()


class HighlightedText:
    """Highcontrast text that appear when hovering over circle.
    """
    def __init__(self, text, x, y, delta_motion_x=-5, delta_motion_y=10,
                 anchor='n'):
        self.text = text
        self._x = x
        self._y = y
        self.delta_motion_x = delta_motion_x
        self.delta_motion_y = delta_motion_y
        self.anchor = anchor

        self.ids = []
        self.canvas = None
        self._deltas = [(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0)]
        self._colors = ['white'] * 4 + ['black']

    def create_widget(self, canvas:CircularPackingCanvas):
        self.canvas = canvas
        self.ids = []
        for delta, color in zip(self._deltas, self._colors):
            self.ids.append(canvas.create_text(
                self._x + delta[0], self._y + delta[1], text=self.text,
                fill=color, state='disable', anchor=self.anchor))

    def update(self, x, y):
        for text_id, delta in zip(self.ids, self._deltas):
            self.canvas.coords(text_id, [x + delta[0], y + delta[1]])

    def destroy(self):
        for text_id in self.ids:
            self.canvas.delete(text_id)

    def bind_motion(self):
        self.canvas.bind('<Motion>', self.on_motion)

    def on_motion(self, event):
        x, y = self.canvas.adjust_event_coords(event.x, event.y)
        self.update(x + self.delta_motion_x, y + self.delta_motion_y)


class NameLabel:

    def __init__(self, name, x, y, max_width=100, font=None, text_color='black'):
        self._name = name
        self.x = x
        self.y = y
        self._max_width = max_width
        self.font = font
        self.text_color = text_color
        self.canvas = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.canvas.itemconfigure(self.id, text=value)

    @property
    def max_width(self):
        return self._max_width

    @max_width.setter
    def max_width(self, value):
        self._max_width = value
        self.canvas.itemconfigure(self.id, width=self.max_width)

    def create_widget(self, canvas:Canvas):
        self.canvas = canvas
        self.id = canvas.create_text(self.x, self.y, text=self.name,
                                     state='disabled', justify='center',
                                     width=self.max_width, font=self.font,
                                     fill=self.text_color)
        return self.id

    def destroy(self):
        self.canvas.delete(self.id)


def my_click(circle: PackingCircle):
    """
    Perform the click behavior on a circle.
    Checks for leaf or focus-specific actions and falls back to the default behavior.
    """
    # Check if the circle is a leaf
    if circle.is_leaf():
        if circle.parent is not None:
            circle.parent.on_click()
        return True

    # Check if the circle is the current focus
    if circle.canvas.current_focus is circle:
        if circle.parent is not None:
            circle.parent.on_click()
        return True

    # Default behavior
    circle.canvas.hide_names()
    circle.canvas.current_focus = circle
    
    for child in circle.children:
        child.show_name()

    return False

def _get_rect_coords(x:float, y:float, r:float)->Tuple[Tuple[float,float],Tuple[float, float]]:
        """return rect coords"""
        x0, y0 = x - r, y - r
        x1, y1 = x + r, y + r

        return (x0, y0), (x1, y1)