from tiny_3d_engine import load_file_as_scene
from nob.core import Tree
from opentea.process_utils import update_3d_callback


def update_3d_scene(nob_in, scene):

    nob_tree = Tree(nob_in)
    geo_path = nob_tree.write.geofile[:]
    try:
        scene = load_file_as_scene(geo_path, prefix='out', scene=scene, color="#00ff00")
    except FileNotFoundError:
        print("Scene couldn't be loaded.")
    except KeyError:
        print("Scene couldn't be loaded.")
    scene.add_axes()

    return scene


if __name__ == "__main__":
    print("3D CBck:")
    update_3d_callback(update_3d_scene)
