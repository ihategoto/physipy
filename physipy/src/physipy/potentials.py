import numpy as np

__all__ = [
    "harmonic",
    "lennard_jones",
    "isin_classical_region",
    "k"
]

def harmonic(r, m = 1, omega = 1):
    """
    Calculate the harmonic potential for a given radius.
    
    Parameters
    ----------
    r : float or ndarray
        Position(s) at which the potential is evaluated.
    
    Returns
    -------
    E : float
        Harmonic potential at the given position.

    """
    E = 0.5 * m * omega * omega * r * r
    return E

def lennard_jones(r, sigma = 1, epsilon = 1):
    """
    Calculate the Lennard-Jones potential for a given radius, sigma and epsilon.
    
    Parameters
    ----------
    r : float or ndarray
        Position(s) at which the potential is evaluated.
    sigma : float
        Sigma parameter of the Lennard-Jones potential.
    epsilon : float
        Epsilon parameter of the Lennard-Jones potential.
    
    Returns
    -------
    E : float
        Lennard-Jones potential at the given position, sigma and epsilon.

    """
    E = 4 * epsilon * (np.pow(sigma/r, 12) - np.pow(sigma/r, 6))
    return E

def isin_classical_region(r, E, l, potential, **kwargs):
    """
    Check whether the position is within the classical region for a given potential.

    Parameters
    ----------
    r : float or ndarray
        The position(s) to be evaluated.
    E : float
        Particle's energy.
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
    
    if (isinstance(r, np.ndarray) and np.any(r == 0)) or np.any(r == 0):
        raise ValueError('Evaluating potential at the singular point 0.')
    
    v_eff = potential(r, **kwargs) + 0.5 * l * (l+1)/(m*r*r)
    f = (E - v_eff) > 0
    return f


def k(r, l, E, potential, **kwargs):
    """
    Calculate the k function for the Schrödinger equation given the potential.
    
    Parameters
    ----------
    r : float
        Position at which k is evaluated.
    l : int
        Angular momentum quantum number.
    E : float
        Energy of the particle.
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
    
    centr_barrier = 0.5 * l * (l+1)/(m*r*r)
    pot = potential(r, **kwargs)
    k = 2 * (E - pot - centr_barrier)
    return k 