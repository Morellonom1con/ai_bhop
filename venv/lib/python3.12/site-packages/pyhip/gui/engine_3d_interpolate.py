from tiny_3d_engine import load_file_as_scene
from nob.core import Tree
from opentea.process_utils import update_3d_callback
from opentea.noob.noob import unique_dict_key

def update_3d_scene(nob_in, scene):
    """Update the list of dimensions."""

    nob_tree = Tree(nob_in)
    if nob_in.interpolation.find('interpolation'):
        geo_path = nob_tree.interpolate.geofile[:]
        scene = load_file_as_scene(geo_path, prefix='interp', scene=scene, color="#0000ff")
        scene.add_axes()

    return scene


if __name__ == "__main__":
    print("3D CBck:")
    update_3d_callback(update_3d_scene)
