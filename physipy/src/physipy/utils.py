import numpy as np
import physipy.constants as constants

__all__ = [
    'k_squared',
    'isin_classical_region',
    'bessel_j',
    'bessel_n',
    'isin_asymptotic_region'
]

def k_squared(r, l, E, potential, **kwargs):
    r = np.atleast_1d(r)
    r[r < 1e-10] = 1e-10

    pot = potential(r, **kwargs)

    if 'hbar_squared_over_2_m' in kwargs:
        pre_factor = kwargs['hbar_squared_over_2_m']
        centr_barrier = pre_factor * l * (l + 1)/(r * r)
        k = (E[:, None] - pot[None, :] - centr_barrier[None, :]) / pre_factor
    else:
        m = 1 if not 'm' in kwargs else kwargs['m']
        centr_barrier = 0.5 * (constants.hbar * constants.hbar) / m * l * (l + 1)/(r * r)
        k = 2 * m / (constants.hbar * constants.hbar) * (E[:, None] - pot[None, :] - centr_barrier[None, :])

    return k 

def isin_asymptotic_region(psi, coord, k, node_tol = 1e-2, rejection_ratio = 0.1):
    """
    Estimate k²(r) = -psi''(r) / psi(r) numerically.
    Points where |psi| < node_tol * max(|psi|) are masked.
    """
    h = coord[1] - coord[0]
    psi_pp = (psi[:-2] - 2*psi[1:-1] + psi[2:]) / h**2
    
    psi_mid = psi[1:-1]
    r_mid   = coord[1:-1]
    
    # Mask near-node points
    mask = np.abs(psi_mid) > node_tol
    k2_local = np.mean(-psi_pp[mask] / psi_mid[mask])
    
    return (k2_local - k**2) / k**2 < rejection_ratio
    

def isin_classical_region(r, E, l, potential, **kwargs):
    """
    Check whether the position is within the classical region for a given potential.

    Parameters
    ----------
    r : float
        The position to be evaluated.
    E : float or ndarray
        Particle's energy(s).
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential energy to be used.
    **kwargs : dict
        - Entry 'hbar_squared_over_2_m' contains the dimensional constant to be used.
        - Entry 'm' particle's mass.
        - Additional parameters for the potential used.
    
    Returns
    -------
    f : bool or ndarray
        True if r is in the classical region for the given potential, False otherwise.
    
    """
    if r == 0:
        r = 1e-20
    
    if 'hbar_squared_over_2_m' in kwargs:
        pre_factor = kwargs['hbar_squared_over_2_m']
        v_eff = potential(r, **kwargs) + pre_factor * l * (l + 1)/(r * r)
    else:
        m = 1 if not 'm' in kwargs else kwargs['m']
        v_eff = potential(r, **kwargs) + 0.5 * (constants.hbar * constants.hbar) / m * l * (l + 1)/(r * r)

    f = (E - v_eff) > 0
    return f

def bessel_j(x, l):
    """
    Compute
    """
    x = x if x > 1e-20 else 1e-20
    n_curr = np.sin(x) / x
    n_prev = np.cos(x) / x
    
    if l == -1:
        return n_prev
    
    for i in range(0, l):
        temp = n_curr
        n_curr = (2 * i + 1) / x * n_curr - n_prev
        n_prev = temp
    
    return n_curr

def bessel_n(x, l):
    x = x if x > 1e-20 else 1e-20
    n_curr = (-1) * np.cos(x) / x
    n_prev = np.sin(x) / x
    
    if l == -1:
        return n_prev
    
    for i in range(0, l):
        temp = n_curr
        n_curr = (2 * i + 1) / x * n_curr - n_prev
        n_prev = temp
    
    return n_curr