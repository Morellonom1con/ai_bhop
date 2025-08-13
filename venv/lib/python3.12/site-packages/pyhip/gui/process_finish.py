""" Module executed when the **"process finish"** tab button is pressed."""
from pyhip.scenarii.complete import complete_setup
from opentea.process_utils import process_tab
from opentea.noob.noob import unique_dict_key
from nob import Nob
import yaml

def process_finish(nob_in):
    """
    *Update the inlet dict( ) and make the different setups with the*  \
    :py:func:`pyavbp.scenarii.tavbp_setup.tavbp`  *function.*

    :param nob_in: a dict( ) with IHM's parameters

    :returns: A **dict( )** with some update infos:

    """
    nob_out = Nob(nob_in.copy())

    complete_setup_data(nob_out)

    return nob_out[:]

def complete_setup_data(nob):
    """method to setup a dictionary to be read by the basic scenario"""

    setup_nob = dict()
    setup_nob.update(nob.hip[:])
    setup_nob.update(_inputs(nob.input_mode))
    setup_nob.update(_periodize(nob.periodicity))
    setup_nob.update(_transform(nob.transform))
    setup_nob.update(_interpolate(nob.interpolation))
    setup_nob.update(_adapt(nob.adaptation))
    setup_nob.update(_write(nob.write))

    setup_nob['runner'] = {'fallback': True, #nob.fallback[:], 
                           'checklevel': nob.checklvl[:]}

    with open('ihm_setup.yml', 'w') as fout:
        yaml.dump(setup_nob, fout, default_flow_style=False)

    hip_commands = complete_setup(
        Nob(setup_nob),
        fallback=True, #nob.fallback[:],
        checklevel=nob.checklvl[:],
    )

    nob.hip_cmd.set("\n".join(hip_commands))

def _inputs(nob):
    """method to setup input parameters"""

    if 'read' in nob:
        out = _read(nob.read)
    elif 'generate' in nob:
        out = _generate(nob.generate)

    return out

def _read(nob):
    """method to setup read parameters"""

    input_case = unique_dict_key(nob.input_case[:])
    case_fmt = input_case.split('_')[0]
    if 'solution' in input_case:
        avefile = nob.avefile[:]
    else:
        avefile = None

    out = {
        'files': [nob.meshfile[:], avefile],
        'meshtype': case_fmt,
        'patch_list': nob.patch_list[:],
    }

    return {'read': out}

def _generate(nob):
    """method to setup generate parameters"""

    out = {}
    if nob.structure[:] == 'quad':
        convert2tri = False
    else:
        convert2tri = True

    out = {
        'lower_corner': (nob.u_bounds[:][0], nob.v_bounds[:][0]),
        'upper_corner': (nob.u_bounds[:][1], nob.v_bounds[:][1]),
        'resolution': nob.resolution[:][:2],
        'convert2tri': convert2tri,
        'patch_list': nob.patch_list[:],
    }
    if unique_dict_key(nob.dimension[:]) in ('3d_xyz', '3d_xrtheta'):
        out['extru_range'] = nob.w_bounds[:]
        out['extru_resolution'] = nob.resolution[:][-1] - 1
        if unique_dict_key(nob.dimension[:]) == '3d_xyz':
            out['extru_axis'] = 'z'
        else:
            out['extru_axis'] = 'axi'

    return {'generate': out}

def _periodize(nob):
    """method to setup periodization parameters"""
    pairs = []
    for pair in nob.bnd_patch[:]:
        pairs.append({'leader': pair["leader_patch"],
                      'follower': pair["follower_patch"]})
    out = pairs

    return {'periodize': out}

def _transform(nob):
    """method to setup dict form transform part"""
    out = {
        'rotate': {'angles': {'x': nob.rotation_angle_x[:],
                              'y': nob.rotation_angle_y[:],
                              'z': nob.rotation_angle_z[:]}},
        'translate': {'direction': nob.translation_vector[:]},
        'scale': {'scaling_factor': nob.scaling_vector[:]},
        'duplicate': {'nbr_sector_tot': nob.nbr_tot[:],
                      'nbr_sector_tgt': nob.nbr_tgt[:]}
    }

    return {'transform': out}


def _interpolate(nob):
    """method to setup read parameters"""
    if unique_dict_key(nob[:]) == 'interpolation_off':
        return {}
    else:
        out = {
            'files': [nob.src_mesh[:], nob.src_solution[:]],
            "meshtype": "hdf5",
            'patch_list': nob.patch_list_src[:]
            }

    return {'interpolate': out}

def _adapt(nob):
    """method to setup adapt parameters"""
    if unique_dict_key(nob[:]) == "adaptation_off":
        return {}
    else:
        out = {
            "factor":{
                "coefficient": nob.refinement_coeff[:],
                "hGrad": nob.hgrad[:],
                "hmin": nob.hmin[:]
            }
        }
        return {'adapt': out}

def _write(nob):
    """method to setup write parameters"""
    output_case = unique_dict_key(nob[:]).lower()

    if output_case == 'avbp hdf5':
        case_fmt = 'hdf5'
    elif output_case == 'cgns':
        raise NotImplementedError('CGNS format is not implemented yet for writing.')

    nob.geofile.set(nob.casename[:] + '.geo')
    out = {
        'casetype': case_fmt,
        'casename': nob.casename[:],
        'options': {},
    }
    if nob.find('options'):
        out['options'] = nob.options[:]

    return {'write': out}

if __name__ == "__main__":
    process_tab(process_finish)
