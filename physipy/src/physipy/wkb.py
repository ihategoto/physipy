import numpy as np
from physipy.utils import *
from physipy.numerics_data import Grid

__all__ = [
    'WKB_seed'
]

def WKB_seed(E, l, potential, grid = Grid(), outward = False, scattering = False, **kwargs):
    """
    Compute the WKB seed value for the first Numerov integration step.

    Uses the WKB approximation to set the initial amplitude and phase at
    the boundary of the integration domain, improving numerical stability
    compared to a naive constant seed.

    Parameters
    ----------
    E : float
        Particle's energy.
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential energy function V(r).
    grid : Grid
        Mesh parameters; r_min is used for outward, r_max for inward (default Grid()).
    outward : bool
        If True, seed is computed at r_min for outward integration.
        If False, seed is computed at r_max for inward integration (default False).
    scattering : bool
        If True, returns a complex seed for scattering states.
        If False, returns the real part for bound states (default False).
    kwargs : dict
        Additional arguments forwarded to k_squared.

    Returns
    -------
    wkb_seed : float or complex
        WKB-approximated value of ψ at the second grid point from the boundary.
    """
    h = grid.h if outward else -grid.h
    r = grid.r_min if outward else grid.r_max

    k2_1 = k_squared(r, l, E, potential, **kwargs)
    k2_2 = k_squared(r + h, l, E, potential, **kwargs)

    k_1 = np.sqrt(np.abs(k2_1))
    k_2 = np.sqrt(np.abs(k2_2))

    amplitude = np.where(k_2 > 0, np.sqrt(k_1 / k_2), 1.0)

    # If k2_1 >0 and k2_2 > 0 then we are in the oscillatory region, otherwise we have the exponential decay.
    if scattering:
        wkb_seed = np.where((k2_1 > 0) & (k2_2 > 0), amplitude * np.exp(1j * h / 2 * (k_1 + k_2)), amplitude * np.exp(-h / 2 * (k_1 + k_2)))
    else:
        wkb_seed = np.where((k2_1 > 0) & (k2_2 > 0), np.real(amplitude * np.exp(1j * h / 2 * (k_1 + k_2))), amplitude * np.exp(-h / 2 * (k_1 + k_2)))

    return wkb_seed