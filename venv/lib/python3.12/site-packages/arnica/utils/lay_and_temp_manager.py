"""
lay_and_temp_manager.py

Functions which deal with layouts and templates
"""

from pkg_resources import resource_listdir, resource_string

from yaml import safe_load

from showy.utils import decompact_template  # backward compatibility


__all__ = ["fetch_avbp_layouts", "fetch_avbp_templates", "decompact_template"]


def fetch_avbp_layouts():
    """It returns all the avbp layouts in a dictionary

    Output:
    -------
    avbp_layouts : nested object
    """

    avbp_layouts = {}
    layouts_files_list = resource_listdir('pyavbp.visu', 'avbp_layouts')

    for layout_file_name in layouts_files_list:
        layout_string = resource_string('pyavbp.visu', ("avbp_layouts/"
                                                        + layout_file_name))
        layout = safe_load(layout_string)
        avbp_layouts[layout["title"]] = layout

    return avbp_layouts


def fetch_avbp_templates():
    """It returns all the avbp templates in a dictionary

    Output:
    -------
    avbp_templates : nested object
    """

    avbp_templates = {}
    templates_files_list = resource_listdir('pyavbp.visu', 'avbp_templates')

    for template_file_name in templates_files_list:
        template_string = resource_string('pyavbp.visu', ("avbp_templates/"
                                                          + template_file_name))
        template = safe_load(template_string)
        avbp_templates[template["title"]] = template

    return avbp_templates
