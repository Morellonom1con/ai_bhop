"""Module for the first tab."""

from opentea.noob.noob import nob_get, nob_set
from opentea.process_utils import process_tab


def process_controle_existif(nob_in):
    """Update the list of dimensions."""
    nob_out = nob_in.copy()

    # update dimensions
    ndim = nob_get(nob_in, "ndim_choice")
    if ndim == "two":
        nob_set(nob_out, ["x", "y"], "req_ndim")
    else:
        nob_set(nob_out, ["x", "y", "z"], "req_ndim")

    # update patches
    npatch = nob_get(nob_in, "npatches")
    list_patch = ["patch_#" + str(i) for i in range(npatch)]
    nob_set(nob_out, list_patch, "list_patches")

    return nob_out


if __name__ == "__main__":
    process_tab(process_controle_existif)
