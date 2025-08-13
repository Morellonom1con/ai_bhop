"""tools.py

A collection of tools for Nobs
"""
from nob import Nob
from nob.path import _keep_roots


def _allclose_compare(val_a, val_b, **kwargs):
    """Internal comparator in the np.allclose style"""
    import numpy as np

    if any(type(v) in (str, type(None)) for v in (val_a, val_b)):
        return val_a == val_b
    return np.allclose(val_a, val_b, **kwargs)


def allclose(a, b, **kwargs):
    """True if two nested objects are element-wise equal within tolerance.

    This is a direct wrapper for np.allclose
    (https://numpy.org/doc/stable/reference/generated/numpy.allclose.html),
    applied to Nob objects. `kwargs` apply to np.allclose.
    """
    na, nb = Nob(a), Nob(b)

    if set(na.paths) ^ set(nb.paths):
        return False

    for leaf in na.leaves:
        val_a, val_b = na[leaf][:], nb[leaf][:]
        if not _allclose_compare(val_a, val_b, **kwargs):
            return False

    return True


def diff(a, b, compare=None, verbose=False, **kwargs):
    """
    Diff between two Nobs using `compare` function.

    Returns (
        paths in a but not b,
        paths in a and b, where compare(a, b) = False,
        paths in b but not a
        ).
    `compare` defaults to a np.allclose clone. `kwargs` apply to `compare`.
    Can also be used simply with operator.eq (but beware of floating point rounding
    errors).

    NB: if a subtree differs between a and b, only the root of the subtree is listed.
    Use `verbose = True` to get all the differing paths.
    """
    if compare is None:
        compare = _allclose_compare

    f = (lambda x: list(x)) if verbose else _keep_roots

    return (
        f(set(a.paths) - set(b.paths)),
        [
            p
            for p in set(a.leaves) & set(b.leaves)
            if not compare(a[p][:], b[p][:], **kwargs)
        ],
        f(set(b.paths) - set(a.paths)),
    )
