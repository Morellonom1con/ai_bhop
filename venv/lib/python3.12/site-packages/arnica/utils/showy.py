
# here for backward compatibility, use `showy` instead

import warnings

import showy as showy_pkg


__all__ = ["showy", "display"]


def display(**kwargs):
    """Retro compatibility"""

    warnings.warn("Use *showy* instead of *display*", DeprecationWarning)
    showy(**kwargs)


def showy(layout, data, data_c=None, show=True):
    warnings.warn("Use *showy* package instead", DeprecationWarning)
    dataframes = [data]
    if data_c is not None:
        dataframes.append(data_c)

    return showy_pkg.showy(layout, dataframes, show=show)
