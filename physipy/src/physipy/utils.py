import numpy as np
import physipy.constants as constants

__all__ = [
    'k_squared',
    'isin_classical_region',
    'bessel_j',
    'bessel_n',
    'isin_asymptotic_region',
    'r_asym_min',
    'n_points_needed'
]

def r_asym_min(E, l, potential, hbar2_over_2mu, safety = 2.0, pot_tol = 0.01, **kwargs):
    k = np.sqrt(E / hbar2_over_2mu)
    r_centr  = np.sqrt(hbar2_over_2mu * l * (l + 1) / E) if l > 0 else 0.0
    r_bessel = l / k if l > 0 else 0.0

    # Search from beyond the LJ repulsive core
    r_test = 5 * kwargs.get('sigma', 1)
    v = np.abs(potential(r_test, **kwargs)) 
    while v > pot_tol * E:
        r_test += 0.1 * kwargs.get('sigma', 1)
        v = np.abs(potential(r_test, **kwargs)) 

    return safety * max(r_centr, r_bessel, r_test)

def n_points_needed(E, hbar2_over_2mu, dr, n_cycles = 3):
    """Number of grid points to cover n_cycles wavelengths."""
    k = np.sqrt(E / hbar2_over_2mu)
    lambda_ = 2 * np.pi / k
    return int(np.ceil(n_cycles * lambda_ / dr))

def k_squared(r, l, E, potential, **kwargs):
    # E is a scalar here
    r = np.atleast_1d(r)
    r[r < 1e-10] = 1e-10

    pot = potential(r, **kwargs)

    if 'hbar_squared_over_2_m' in kwargs:
        pre_factor = kwargs['hbar_squared_over_2_m']
        centr_barrier = pre_factor * l * (l + 1)/(r * r)
        k = (E - pot - centr_barrier) / pre_factor
    else:
        m = 1 if not 'm' in kwargs else kwargs['m']
        centr_barrier = 0.5 * (constants.hbar * constants.hbar) / m * l * (l + 1)/(r * r)
        k = 2 * m / (constants.hbar * constants.hbar) * (E - pot - centr_barrier)

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
    r = np.atleast_1d(r)
    r[r < 1e-20] = 1e-20
    
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
    Compute spherical Bessel function of the first kind.

    Parameters
    ----------
    x : float
        Position at which the function is to be evaluated.
    l : int
        Angular momentum quantum number.
    
    Returns
    -------
    n_curr : float
        Function's value.
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
    """
    Compute spherical Bessel function of the first kind.

    Parameters
    ----------
    x : float
        Position at which the function is to be evaluated.
    l : int
        Angular momentum quantum number.
        
    Returns
    -------
    n_curr : float
        Function's value.
    """
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