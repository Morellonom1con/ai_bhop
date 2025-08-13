from tiny_3d_engine import load_file_as_scene
from nob.core import Tree
from opentea.process_utils import update_3d_callback
import os

def update_3d_scene(nob_in, scene):
    """Update the list of dimensions."""

    nob_tree = Tree(nob_in)
    geo_path = nob_tree.inputs.geofile[:]
    try:
        scene = load_file_as_scene(geo_path, prefix='in', scene=scene, color="#ff0000")
        os.remove(geo_path)
    except KeyError:
        print("Scene couldn't be loaded. Might be invalid .geo file.\n")
    except:
        print("Scene couldn't be loaded.\n")

    scene.add_axes()

    return scene


if __name__ == "__main__":
    print("3D CBck:")
    update_3d_callback(update_3d_scene)
