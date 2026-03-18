import numpy as np
from physipy.utils import *

__all__ = []

def WKB_seed(E, l, r, h, potential, outward = False, **kwargs):
    """
    Compute the WKB approximated step.

    Parameters
    ----------
    E : float
        Particle's energy.
    l : int
        Angular momentum quantum number.
    r : float or ndarray
        Last point of the mesh.
    h : float
        Grid's step.
    potential : callable
        Potential energy to be used in the calculation.
    outward : bool
        Boolean flag that indicates the direction of integration.
        The logic here is the following : we always integrate along 
        the direction of the decaying behaviour.
    kwargs : dict
        Additional arguments of the potential energy.

    Returns
    -------
    wkb_seed : float or ndarray
        WKB approximated starting point for the inward integration.

    """
    h = -1 * h if not outward else h
    k_1 = k_squared(r, l, E, potential, **kwargs)
    k_2 = k_squared(r + h, l, E, potential, **kwargs)
    wkb_seed = np.sqrt(k_1 / k_2) * np.exp(h / 2 * (k_1 + k_2))
    return wkb_seed