import os
import tkinter as tk
from tkinter import (
    ttk,
    messagebox as msgbox,
    Toplevel,
)

from PIL import (
    ImageTk,
    Image,
)
import tkhtmlview as tkhtml
import markdown

from opentea.gui_forms.constants import (
    PARAMS,
    WIDTH_UNIT,
)
from opentea.gui_forms.soundboard import play_doorcls


def is_forgotten_frame(frame):
    for fnc in ["pack_info", "grid_info", "place_info"]:
        try:
            if getattr(frame, fnc)():
                return False
        except tk.TclError:
            pass

    return True


def is_hierarchically_above(ref_widget, widget):
    found = False
    parent = ref_widget
    while not found:
        if parent is None:
            break

        if parent == widget:
            found = True

        parent = parent.master

    return found


def get_tkroot(widget):
    root = widget
    while root.master is not None:
        root = root.master

    return root


def quit_dialog():
    """Quitting dialog"""
    if msgbox.askokcancel("Quit", "Do you really wish to quit?"):
        print("Get outta here and go back to your boring programs.") # une fois que l'application est ferme, le stdout redirigé est perdu. pas de print apres le destroy
        play_doorcls()
        PARAMS["top"].destroy()
        

def create_description(parent, description, size=0.9, side="top", **kwargs):
    """Interpret the description to take into account font modifiers."""
    text = description
    fontname, fontsize, fonthyphen = None, 11, "normal"

    if "<small>" in text:
        text = text.replace("<small>", "")
        fontsize = 10
    elif "<tiny>" in text:
        text = text.replace("<tiny>", "")
        fontsize = 8

    if "<bold>" in text:
        text = text.replace("<bold>", "")
        fonthyphen = "bold"

    if "<italic>" in text:
        text = text.replace("<italic>", "")
        fonthyphen = "italic"

    desc = ttk.Label(
        parent,
        text=text,
        wraplength=int(size * WIDTH_UNIT),
        font=(fontname, fontsize, fonthyphen),
    )
    desc.pack(side=side, **kwargs)

    return desc


def show_docu_webpage(markdown_str):
    """Show documentation in the web browser

    Use package `markdown`to translate markown into HTML.
    Dump is into a temporary file using `tempfile`package.
    Finally call the browser using `webbrowser`package.
    """
    md_ = markdown.Markdown()
    css_style = """
 <style type="text/css">
 body {
    font-family: Helvetica, Geneva, Arial, SunSans-Regular,sans-serif ;
    margin-top: 100px;
    margin-bottom: 100px;
    margin-right: 150px;
    margin-left: 80px;
    color: black;
    background-color: BG_COLOR
 }

 h1 {
    color: #003300
}
 </style>
"""
    css_style = css_style.replace("BG_COLOR", PARAMS["bg"])

    html = str()
    html += md_.convert(markdown_str)
    html = html.replace('src="', 'src="' + PARAMS["calling_dir"] + "/")

    # with tempfile.NamedTemporaryFile(
    #     "w", delete=False, suffix=".html"
    # ) as fout:
    #     url = "file://" + fout.name
    #     fout.write(html)

    #     html = css_style + html

    #     tmp_file = ".docu_tmp.html"
    #     with open(tmp_file, "w") as fout:
    #         fout.write(html)
    #         url = "file://" + os.path.join(os.getcwd(), tmp_file)
    #     webbrowser.open(url)
    # else:

    html = html.replace("<h1", '<h1 style="color: #003300"')
    html = html.replace("<h2", '<h2 style="color: #003300"')
    html = html.replace("<h3", '<h3 style="color: #003300"')
    top = Toplevel()
    html_label = tkhtml.HTMLScrolledText(top, html=html, background=PARAMS["bg"])
    html_label.pack(fill="both", expand=True)
    html_label.fit_height()


def create_documentation(schema, frame):
    def show_docu(event, docu_ct):
        show_docu_webpage(docu_ct)

    docu_ct = schema["documentation"]

    docu_lbl = ttk.Label(frame, text="learn more...", style="Linkable.TLabel")
    docu_lbl.pack(side="bottom")

    docu_lbl.bind("<Button-1>", lambda e, docu_ct=docu_ct: show_docu(e, docu_ct))

    return docu_lbl


def create_image(schema, frame):
    path = os.path.join(PARAMS["calling_dir"], schema["image"])
    img = ImageTk.PhotoImage(Image.open(path))
    img_lbl = ttk.Label(frame, image=img)

    img_lbl.image = img
    img_lbl.pack(side="top")

    return img_lbl
