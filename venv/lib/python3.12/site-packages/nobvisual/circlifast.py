""" module to implement a fasted circlification, using a siparl"""

#import circlify as circ
from typing import List
import circlify
from circlify import Circle
import math
from  loguru import logger
import json
EPSILON=1e-8

def circlify_silent(data, show_enclosure=False)->list:
    #with Capturing() as output:
    circles=circlify.circlify(data, show_enclosure=show_enclosure)
    
    return circles


def cast_as_bubbles(bubble: any)->dict:
    """[RECURSIVE]  convert elements of a nested object into 'bubble' dict
    
    If element is a float of an int, the information is inferred
    
    If element is a dict, the information is passed, then recursively call childrens
    
       bubble either:
            float, int (a leaf)
            dict {
                "id": ,             <- the id label (not shown)          |
                "color":            <- the color to be shown             |
                "name": d["name"]   <- the name of the bubble            | If missing
                "text": d["text"]}  <- the additional text of the bubble | a default value is found
                "datum" : 1.0       <- size of the circle                |
                "children " : [ __ other items ___]                      |
            }
    
    
    """
    # Default values
    datum = 1.0
    id = "None"
    children = []
    color = "#eeeeee"
    text = ""
    name="Unamed"

    if isinstance(bubble, float):
        datum = bubble
    elif isinstance(bubble, int):
        datum = bubble * 1.0
    
    if datum == 0:
        #logger.warning(f"Null datum leaf; Skipping")
        return None 
    elif isinstance(bubble, dict):
        if "datum" in bubble:
            datum = bubble["datum"]
            if datum == 0:
                #logger.warning(f"Null datum node {bubble['name']} ; Skipping")
                return None 
        if "id" in bubble:
            id = bubble["id"]
        if "color" in bubble:
            color = bubble["color"]
        if "text" in bubble:
            text = bubble["text"]
        if "name" in bubble:
            name = bubble["name"]
        if "children" in bubble:
            children =[]
            for child in bubble["children"]:
                child_bubble = cast_as_bubbles(child)
                if child_bubble is not None:
                    children.append(child_bubble)
    else:
        msg = f"Could not cast {bubble} as a nested bubble object"
        logger.critical(msg)
        raise ValueError(msg)
    
    return {
        "datum": datum,
        "id": id,
        "children": sorted(children, key=lambda d: d["datum"], reverse=True),
        "color": color,
        "text": text,
        "name": name,
    }


def new_circle(bubble: dict, x_p:float, y_p:float, r_p:float, level: int)-> Circle:
    """Convert a bubble into circlify Circle"""
    light_children = [
        {"id": d["id"], 
        "datum": d["datum"], 
        "color": d["color"],
        "name": d["name"],
        "text": d["text"]}
        for d in bubble["children"]
    ]
    out = Circle(
        x=x_p,
        y=y_p,
        r=r_p,
        level=level,
        ex={
            "datum": bubble["datum"],
            "id": bubble["id"],
            "children": light_children,
            "color": bubble["color"],
            "text": bubble["text"],
            "name": bubble["name"],
        },
    )
    return out


def rec_add_circle_spiral(circles: list, bubble:dict, x_p:float=0.0, y_p:float=0.0, r_p:float=1.0, level:int=0):
    """[RECURSIVE] Update recursively list circles,by adding a bubble and computing its coordinates """

    # append the buble
    circles.append(new_circle(bubble, x_p, y_p, r_p, level))

    childrens = bubble["children"]
    if not childrens:
        return None



    # Compute the position of childrens within this bubble
    x_list = [0.0]
    y_list = [0.0]
    r0 = math.sqrt(childrens[0]["datum"])
    r_list = [r0]
    r_max = r0
    r_ext = r0
   
    # first children
    if len(childrens) > 1:
        r1 = math.sqrt(childrens[1]["datum"])
        rcenter = r0 + r1
        x_list.append(0.0)
        y_list.append(rcenter)
        r_list.append(r1)
        r_max = rcenter + r1
        r_ext = r0
        r_in = r0  # spiral compression
        last_angle = math.asin(r1 / (r_ext + r1))

    # second and other children
    if len(childrens) > 2:
        for child in childrens[2:]:
            r = math.sqrt(child["datum"])
            dangle = math.asin(r / (r_ext + r))

            angle = last_angle + dangle
            r_ext = r_in + r1 * angle / math.pi  # spiral compression

            rcenter = r_ext + r

            x_list.append(math.sin(angle) * rcenter)
            y_list.append(math.cos(angle) * rcenter)
            r_list.append(r)
            r_max = max(r_max, r_ext + 2 * r)
            last_angle = last_angle + 2 * dangle

            # spiral compression
            if last_angle > 2 * math.pi:
                last_angle -= 2 * math.pi
                r1 = r
                r_in = r_ext

    # Rescale coordinates to fit everyrthing to the parent bubble
    factor = r_p / (r_max+EPSILON)
    x_list = [x_p + x * factor for x in x_list]
    y_list = [y_p + y * factor for y in y_list]
    r_list = [r * factor for r in r_list]

    # Recursive call...
    for i, child in enumerate(childrens):
        rec_add_circle_spiral(
            circles, child, x_p=x_list[i], y_p=y_list[i], r_p=r_list[i], level=level + 1
        )


def circlifast(data: dict, show_enclosure=False)-> List[dict]:
    """Convert a nested information (dict/list) to a circlify Circles list
    
    data  is a list of items [items, items]
            item = {
                "id": ,             <- the id label (not shown)          |
                "color":            <- the color to be shown             |
                "name": d["name"]   <- the name of the bubble            | If missing
                "text": d["text"]}  <- the additional text of the bubble | a default value is found
                "datum" : 1.0       <- size of the circle                |
                "children " : [ __ other items ___]                      |
            }
                
            }
    
    """
    root = {
        "datum": 1.0,
        "id": "root",
        "children": data,
        "color": "#eeeeee",
        "name": "",
        "text": "",
    }

    bubble = cast_as_bubbles(root)
    # with open("debug.json","w") as fout:
    #     json.dump(bubble, fout,indent=4)
    # exit()
    circles = []
    rec_add_circle_spiral(circles, bubble)

    if show_enclosure:
        return circles[:]
    # by default The root circle [0] is not returned
    return circles[1:]
