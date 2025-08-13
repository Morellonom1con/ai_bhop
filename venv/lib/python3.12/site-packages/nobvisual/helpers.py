"""Module for conversion between Nested representations of data"""
from typing import List
from circlify import Circle
from nobvisual.objects import PackingCircle
from nobvisual.circlifast import circlifast,circlify_silent


def from_nested_struct_to_nobvisual(nstruct, fast=True)->list:
    """turn a nested object (list/dict/etcs) into a list of packing circles """
    if fast:
        circlify_circles = circlifast(nstruct, show_enclosure=False)
    else:    
        circlify_circles = circlify_silent(nstruct, show_enclosure=False)
    return from_circlify_to_nobvisual(circlify_circles)


def from_circlify_to_nobvisual(circlify_circles: List[Circle])->List[PackingCircle]:
    """Turn a list of Cirlify CircLes 
    to a list OF  PackingCircles """

    # Translate
    pack_circles = [from_circlify_circle_to_packing_circle(circlify_circle)
               for circlify_circle in circlify_circles]
    circle_ids = [circle.ex['id'] for circle in circlify_circles]
    
    # add children to Packing Circles
    for pack_circle, circlify_circle in zip(pack_circles, circlify_circles):
        children_ids = [child['id'] for child in circlify_circle.ex.get('children', [])]
        for child_id in children_ids:
            pack_circle.add_children(pack_circles[circle_ids.index(child_id)])
    
    return pack_circles


def from_circlify_circle_to_packing_circle(circlify_circle: Circle)-> PackingCircle:
    """Turn a Cirlify circle to a Nobvisual Packing circle"""
    x, y, r = circlify_circle.circle
    level = circlify_circle.level
    data = circlify_circle.ex
    color = data.get('color', 'default')
    name = data.get('name', '')
    text = data.get('text', '')
    return PackingCircle(x, y, r, level, color=color, name=name, text=text)
