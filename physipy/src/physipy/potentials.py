import numpy as np
from physipy.numerics import Grid, SolverOpts
import physipy.constants as constants

__all__ = [
    "gross_pitaevskij",
    "harmonic",
    "lennard_jones",
    "effective_potential",
    "helper_grid_lj",
    "wave_vector"
]

def gross_pitaevskij(r, **kwargs):
    """
    Calculate the Gross-Pitaevskij potential a the given positions.
    
    Parameters
    ----------
    r : float or ndarray
        Position(s) at which the potential is evaluated.
    kwargs : dict
        Additional parameters for the potential:
        - g : coupling constant.
        - phi : solution of the previous iteration.
        - r_min : grid starting point.
        - r_max : grid ending point.
        - h : integration step.
    
    Returns
    -------
    E : float
        Harmonic potential at the given position.

    """
    g = 1 if 'g' not in kwargs else kwargs['g']
    phi = 1 if 'phi' not in kwargs else kwargs['phi']
    r_min = 0 if 'r_min' not in kwargs else kwargs['r_min']
    r_max = 5 if 'r_max' not in kwargs else kwargs['r_max']
    h = 1e-1 if 'is' not in kwargs else kwargs['is']

    coord = np.arange(r_min, r_max + h , h)
    phi_r = np.interp(r, coord, phi)
    r_squared = np.pow(r, 2) 

    E = 1/2 * r_squared + g * np.pow(phi_r, 2) / r_squared
    return E

def harmonic(r, **kwargs):
    """
    Calculate the harmonic potential for a given radius.
    
    Parameters
    ----------
    r : float or ndarray
        Position(s) at which the potential is evaluated.
    kwargs : dict
        Additional parameters for the potential:
        - m : particle's mass
        - omega : characteristic frequency
    
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
    kwargs : dict
        Additional parameters for the potential:
        - epsilon : potential's depth
        - omega : characteristic length

    
    Returns
    -------
    E : float
        Lennard-Jones potential at the given position, sigma and epsilon.

    """
    epsilon = 1 if 'epsilon' not in kwargs else kwargs['epsilon']
    sigma = 1 if 'sigma' not in kwargs else kwargs['sigma']

    E = 4 * epsilon * (np.pow(sigma/r, 12) - np.pow(sigma/r, 6))
    return E

def effective_potential(r, l, potential, **kwargs):
    """
    Calculate the effective potential of the particle.

    Parameters
    ----------
    r : float or ndarray
        Position(s) at which the potential is evaluated.
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential function to be used.
    kwargs : dict
        Additional paramters for both the potential and the dimensional consistency:
        - hbar_squared_over_2_m : dimensional fixing of the Schrodinger equation
    
    """
    if 'hbar_squared_over_2_m' in kwargs:
        pre_factor = kwargs['hbar_squared_over_2_m']
        v_eff = potential(r, **kwargs) + pre_factor * l * (l + 1) / (r * r)
    else:
        m = 1 if not 'm' in kwargs else kwargs['m']
        v_eff = potential(r, **kwargs) + 0.5 * (constants.hbar * constants.hbar) / m * l * (l + 1)/(r * r)

    return v_eff

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
        k = np.sqrt(2 * m * E) / constants.hbar
    
    return k