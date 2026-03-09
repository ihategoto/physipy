import numpy as np
from physipy.potentials import k

__all__ = []

def WKB_inward_seed(E, l, r, h , potential, **kwargs):
    """
    Returns the WKB decaying solution for the bound states problem.
    It can be used as the seed of the inward numerical integration

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
    kwargs : dict
        Additional arguments of the potential energy.

    Returns
    -------
    wkb_seed : float or ndarray
        WKB approximated starting point for the inward integration.

    """
    k_1 = k(r, l, E, potential, **kwargs)
    k_2 = k(r-h, l, E, potential, **kwargs)
    wkb_seed = np.sqrt(k_1 / k_2) * np.exp(h / 2 * (k_1 - k_2))
    return wkb_seed