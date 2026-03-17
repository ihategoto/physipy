import numpy as np
from physipy.numerics import Grid, SolverOpts
import physipy.constants as constants

__all__ = [
    "harmonic",
    "lennard_jones",
    "isin_classical_region",
    "k",
    "wave_vector"
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
    
    centr_barrier = 0.5 * (constants.hbar * constants.hbar) / m * l * (l + 1)/(r * r)
    pot = potential(r, **kwargs)
    k = 2 * m / (constants.hbar * constants.hbar) * (E - pot - centr_barrier)
    return k 

def helper_grid_lj(h, r_max, k = 0.4, sigma = 1):
    """
    Build a good Grid object for problems using Lennard-Jones potentials.

    Paramaters
    ----------
    h : float
        Integration step.
    r_max : float
        Last point of the mesh.
    k : float
        Fraction of sigma from which the integration must start.
    sigma : float
        Sigma parameter of the Lennard-Jones potential.

    Returns
    -------
    grid : class
        Well-initialized Grid object.
    """
    grid = Grid(k * sigma, r_max, h)
    return grid

def wave_vector(E, m = 1):
    """
    Compute the modulus of the wave vector from the energy and mass.

    Parameters
    ----------
    E : float
        Particle's energy.
    m : float
        Particle's mass.

    Returns
    -------
    k : float
        Particle's wave vector.
    """
    k = np.sqrt(2*m*E) / constants.hbar
    return k