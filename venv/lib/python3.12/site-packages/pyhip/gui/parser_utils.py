""" Parser module """ 

import h5py

def get_patch_list_AVBP(mesh_file):
    """
    *Get the list of patches of a mesh hdf5 file*
    
    :param mesh_file: Path to mesh file
    :type mesh_file: str
    
    :return:
        
        - **patch_list** - List of patches
    """
    with h5py.File(mesh_file, 'r') as fin:
        patch_list = fin['Boundary/PatchLabels'][()]
    patch_list = [patch.decode('UTF-8').strip() for patch in patch_list]
    return patch_list

def parse_meshinfo(msgs):
    """
    *Parse HIP output to get mesh informations*
    :param msgs: List of string message line

    :returns:

        - **meshinfo** - (str) Mesh informations
    """
    in_found = False
    for line in msgs:
        if 'read' in line or 'generate' in line:
            idx_in = msgs.index(line) + 1
            in_found = True
        elif 'copy 3D' in line:
            idx_in = msgs.index(line) + 1
            in_found = True
        elif 'copy q2t' in line or 'copy 2tets' in line:
            idx_in = msgs.index(line) + 1
            in_found = True
        elif in_found and ' <<hip[' in line:
            idx_out = msgs.index(line)
            in_found = False

    meshinfo = "\n".join(map(str.strip, msgs[idx_in:idx_out]))
    return meshinfo

def parse_versioninfo(msgs):
    """
    *Parse HIP output to get hip version informations*
    :param msgs: List of string message line

    :returns:

        - **meshinfo** - (str) Mesh informations
    """
    line = msgs[1].split(',')
    version = line[1][9:]
    version = "Version : " + "".join(version)[:-1]
    date = line[2].strip()

    version_info = "\n".join([version, date]) + '\n'
    return version_info

def parse_boundaries(msgs):
    """
    """
    patch_dict = {}
    found = False
    for idx, line in enumerate(msgs):
        if 'list surface' in line:
            found = True
            break
    if not found:
        return patch_dict

    idx += 2
    for line in msgs[idx:]:
        if line.strip() == '':
            break
        elements = [element.strip() for element in line.split(',')]
        idx += 1
        patch = elements[-1]
        patch_dict[patch] = {
            'nbr': elements[0].split(':')[0],
            'type': elements[2],
            'order': elements[3]}

    return patch_dict

def parse_periodicities(msgs):
    """
    *Parse HIP output to get peridic patches*
    :param msgs: List of string message line

    :returns:

        - **perio_list** - List of periodic patch dict['name', \
                                                       'leader_patch', \
                                                       'follower_patch']
    """
    perio_list = []
    found = False
    for idx, line in enumerate(msgs):
        if 'list periodic' in line:
            found = True
            break
    if not found:
        return perio_list

    idx += 1
    block = 9
    id_ = 'id'
    while '<<hip[' not in msgs[idx]:
        patches = msgs[idx].split()
        perio_patch = {'name': id_,
                       'leader_patch': patches[1],
                       'follower_patch': patches[0]}
        perio_list.append(perio_patch)
        idx += block
        id_ += '#'

    return perio_list
