import numpy as np
from physipy.numerics import Grid, SolverOpts
import physipy.constants as constants

__all__ = [
    "harmonic",
    "lennard_jones",
    "wave_vector"
]

def harmonic(r, **kwargs):
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
    m = 1 if 'm' not in kwargs else kwargs['m']
    omega = 1 if 'omega' not in kwargs else kwargs['omega']

    E = 0.5 * m * omega * omega * r * r
    return E

def lennard_jones(r, **kwargs):
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
    epsilon = 1 if 'epsilon' not in kwargs else kwargs['epsilon']
    sigma = 1 if 'sigma' not in kwargs else kwargs['sigma']

    E = 4 * epsilon * (np.pow(sigma/r, 12) - np.pow(sigma/r, 6))
    return E

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

def wave_vector(E, **kwargs):
    """
    Compute the modulus of the wave vector from the energy and mass.

    Parameters
    ----------
    E : float
        Particle's energy.
    m : float
        Particle's mass.
    kwargs : dict
        - Entry 'hbar_squared_over_2_m" contains the dimensional constant to be used.
        - Entry 'm' contains the particle's mass.

    Returns
    -------
    k : float
        Particle's wave vector.
    """
    if 'hbar_squared_over_2_m' in kwargs:
        k = np.sqrt(E / kwargs['hbar_squared_over_2_m'])
    else:
        m = 1 if not 'm' in kwargs else kwargs['m']
        k = np.sqrt(2*m*E) / constants.hbar
    
    return k