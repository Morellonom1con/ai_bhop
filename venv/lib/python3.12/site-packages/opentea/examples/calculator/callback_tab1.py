"""Module for the first tab."""
from nob import Nob
from copy import deepcopy
from opentea.noob.noob import nob_get, nob_set
from opentea.process_utils import process_tab


def process_controle_existif(nob_in:dict)->dict:
    """Exemple of a Tab callback
    
    The nested object containing the GUI data is given AND returned.
    
    """
    nob_out = deepcopy(nob_in)
    # get patches  using the "noob.noob library (functionnal)"
    #npatch = nob_get(nob_in, "npatches")
    # get patches  using the "Nob library (Object oriented)"
    snob=Nob(nob_in)
    npatch = snob.npatches[:]

    list_patches = ["a1","a2","a3"]
    if npatch == 1:
        list_patches = ["p1","p2","p3","p4",]
    if npatch == 2:
        list_patches = ["p1","p3","p2","p4",]
    if npatch == 3:
        list_patches = ["p1","p2","p3"]
    if npatch == 4:
        list_patches = ["p2","p3","p4"]

    # set patches  using the "noob.noob library (functionnal)"
    nob_set(nob_out, list_patches, "list_patches")
    # set patches  using the "Nob library (Object oriented)"
    # snob_out=Nob(nob_out)
    # snob.list_patches = list_patches
    
    return nob_out


if __name__ == "__main__":
    process_tab(process_controle_existif)
