import numpy as np
import physipy.constants as constants

__all__ = [
    'k_squared',
    'isin_classical_region',
    'bessel_j',
    'bessel_n'
]

def k_squared(r, l, E, potential, **kwargs):
    """
    Calculate the k function for the Schrödinger equation given the potential.
    
    Parameters
    ----------
    r : float
        Position at which k is evaluated.
    l : int
        Angular momentum quantum number.
    E : float or ndarray
        Energy(s) of the particle.
    potential : callable
        Potential energy to be used to compute k.
    **kwargs : dict
        Additional arguments of the potential energy.
    
    Returns
    -------
    k : float or None

    """
    if 'm' in kwargs:
        m = kwargs['m']
    else:
        m = 1

    if (isinstance(r, np.ndarray) and np.any(r == 0)) or np.any(r == 0):
        raise ValueError('Evaluating potential at the singular point 0.')
    
    centr_barrier = 0.5 * (constants.hbar * constants.hbar) / m * l * (l + 1)/(r * r)
    pot = potential(r, **kwargs)

    if 'k' in kwargs:
        pre_factor = kwargs['k']
        k = 2 / pre_factor * (E - pot - centr_barrier)
    else:
        k = 2 * m / (constants.hbar * constants.hbar) * (E - pot - centr_barrier)

    return k 

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
        Additional arguments of the potential energy.
    
    Returns
    -------
    f : bool or ndarray
        True if r is in the classical region for the given potential, False otherwise.
    
    """
    if 'm' in kwargs:
        m = kwargs['m']
    else:
        m = 1
    
    if r == 0:
        r = 1e-20
    
    if 'k' in kwargs:
        pre_factor = kwargs['k']
        v_eff = potential(r, **kwargs) + pre_factor * l * (l + 1)/(r * r)
    else:
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