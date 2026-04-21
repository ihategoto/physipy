import numpy as np
from physipy.utils import *

__all__ = []

def WKB_seed(E, l, r, h, potential, outward = False, scattering = False,  **kwargs):
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
    h = -h if not outward else h

    k2_1 = k_squared(r, l, E, potential, **kwargs)
    k2_2 = k_squared(r + h, l, E, potential, **kwargs)

    k_1 = np.sqrt(np.abs(k2_1))
    k_2 = np.sqrt(np.abs(k2_2))

    amplitude = np.sqrt(k_1 / k_2) if k_2 > 0 else 1.0

    if k2_1 > 0 and k2_2 > 0:
        # Oscillatory region — complex phase
        if scattering:
            wkb_seed = amplitude * np.exp(1j * h / 2 * (k_1 + k_2))
        else:
            wkb_seed = np.real(amplitude * np.exp(1j * h / 2 * (k_1 + k_2)))
    else:
        # Evanescent region — real exponential decay
        wkb_seed = amplitude * np.exp(-h / 2 * (k_1 + k_2))

    return wkb_seed