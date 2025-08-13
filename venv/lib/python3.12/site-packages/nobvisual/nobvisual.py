
from tkinter import ttk, Widget
from nobvisual.tkinter_circlify import tkcirclify
from nobvisual.helpers import from_nested_struct_to_nobvisual
from nobvisual.objects import CircularPackingCanvas

def nobvisual(
        nob, 
        fast:bool=False,
        holder:Widget=None,
        color: str="#777777",
        shade: float=-0.1,
        legend: dict=None,
        title: str=None,
        show_under_focus: bool=True,
    )->CircularPackingCanvas:

    circles=from_nested_struct_to_nobvisual(nob, fast=fast)

    draw_canvas=tkcirclify(
        circles=circles,
        holder=holder,
        color= color,
        shade=shade,
        legend=legend,
        title=title,
        show_under_focus=show_under_focus,
    )
    draw_canvas.show_names(level=2)
    return draw_canvas