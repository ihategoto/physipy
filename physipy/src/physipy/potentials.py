import numpy as np
import math
from scipy.special import eval_hermite

from physipy.numerics_data import Grid
import physipy.constants as constants

__all__ = [
    "gross_pitaevskij",
    "harmonic",
    "lennard_jones",
    "effective_potential",
    "helper_grid_lj",
    "wave_vector",
    "ho_n_radial_wavefunction"
]

def gross_pitaevskij(r, **kwargs):
    """
    Evaluate the mean-field Gross-Pitaevskii potential at the given positions.

    The potential is V(r) = ½r² + g*(φ(r)/r)², where φ is the radial
    wavefunction from the previous self-consistent iteration.

    Parameters
    ----------
    r : float or ndarray
        Position(s) at which the potential is evaluated.
    kwargs : dict
        Required and optional parameters:
        - guess   : ndarray, radial wavefunction from the previous iteration (required).
        - g     : float, coupling constant Na (default 1).
        - r_min : float, left endpoint of the phi grid (default 0).
        - r_max : float, right endpoint of the phi grid (default 5).
        - h     : float, step size used to reconstruct the phi grid (default 1e-1).

    Returns
    -------
    V : ndarray
        GP potential evaluated at each position in r.
    """
    g = 1 if 'g' not in kwargs else kwargs['g']
    r_min = 0 if 'r_min' not in kwargs else kwargs['r_min']
    r_max = 5 if 'r_max' not in kwargs else kwargs['r_max']
    m = 1 if 'm' not in kwargs else kwargs['m']
    omega = 1 if 'omega' not in kwargs else kwargs['omega']
    h = 1e-1 if 'h' not in kwargs else kwargs['h']

    if 'guess' not in kwargs:
        raise ValueError('No initial guess provided.')
    else:
        phi = kwargs['guess']
    
    coord = np.arange(r_min, r_max + h , h)
    phi_r = np.interp(r, coord, phi)
    r_squared = np.pow(r, 2)

    E = 1/2 * m * omega * omega * r_squared + g * np.pow(phi_r, 2) / r_squared
    return E

def harmonic(r, **kwargs):
    """
    Evaluate the harmonic potential ½mω²r² at the given positions.

    Parameters
    ----------
    r : float or ndarray
        Position(s) at which the potential is evaluated.
    kwargs : dict
        Optional parameters:
        - m     : float, particle mass (default 1).
        - omega : float, angular frequency (default 1).

    Returns
    -------
    V : float or ndarray
        Harmonic potential evaluated at each position in r.
    """
    m = 1 if 'm' not in kwargs else kwargs['m']
    omega = 1 if 'omega' not in kwargs else kwargs['omega']

    E = 0.5 * m * omega * omega * r * r
    return E

def lennard_jones(r, **kwargs):
    """
    Evaluate the Lennard-Jones potential 4ε[(σ/r)¹² − (σ/r)⁶] at the given positions.

    Parameters
    ----------
    r : float or ndarray
        Position(s) at which the potential is evaluated.
    kwargs : dict
        Optional parameters:
        - epsilon : float, well depth (default 1).
        - sigma   : float, characteristic length (default 1).

    Returns
    -------
    V : float or ndarray
        Lennard-Jones potential evaluated at each position in r.
    """
    epsilon = 1 if 'epsilon' not in kwargs else kwargs['epsilon']
    sigma = 1 if 'sigma' not in kwargs else kwargs['sigma']

    E = 4 * epsilon * (np.pow(sigma/r, 12) - np.pow(sigma/r, 6))
    return E

def effective_potential(r, l, potential, **kwargs):
    """
    Evaluate the effective potential V_eff(r) = V(r) + ℏ²l(l+1)/(2mr²).

    Parameters
    ----------
    r : float or ndarray
        Position(s) at which the potential is evaluated.
    l : int
        Angular momentum quantum number.
    potential : callable
        Bare potential V(r) to which the centrifugal barrier is added.
    kwargs : dict
        Optional parameters:
        - hbar_squared_over_2_m : float, prefactor ℏ²/2m (uses natural units if absent).
        - m                     : float, particle mass (default 1, natural units).

    Returns
    -------
    V_eff : float or ndarray
        Effective potential evaluated at each position in r.
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
    Build a Grid suited for Lennard-Jones problems.

    Sets r_min = k*sigma so the integration starts safely outside the
    repulsive LJ core.

    Parameters
    ----------
    h : float
        Integration step.
    r_max : float
        Right endpoint of the grid.
    k : float
        Fraction of sigma used to set r_min (default 0.4).
    sigma : float
        LJ characteristic length (default 1).

    Returns
    -------
    grid : Grid
        Grid with r_min = k*sigma, r_max = r_max, h = h.
    """
    grid = Grid(k * sigma, r_max, h)
    return grid

def wave_vector(E, **kwargs):
    """
    Compute the magnitude of the wave vector k = sqrt(2mE)/ℏ.

    Parameters
    ----------
    E : float
        Particle's kinetic energy.
    kwargs : dict
        Optional parameters:
        - hbar_squared_over_2_m : float, prefactor ℏ²/2m (uses natural units if absent).
        - m                     : float, particle mass (default 1, natural units).

    Returns
    -------
    k : float
        Magnitude of the wave vector.
    """
    if 'hbar_squared_over_2_m' in kwargs:
        k = np.sqrt(E / kwargs['hbar_squared_over_2_m'])
    else:
        m = 1 if not 'm' in kwargs else kwargs['m']
        k = np.sqrt(2 * m * E) / constants.hbar
    
    return k

def _log_normalization_const_squared(nu, n):
    """
    Compute the log of the normalization constant squared of the n-th radial wavefunction of the 3D harmonic oscillator.

    Parameters
    ----------
    nu : float 
        nu parameter of the harmonic oscillator.
    n : int
        Radial number of the radial wavefunction.

    Returns
    -------
    log_normalization_const_squared : float
        Log of the normalization constant squared
    """
    log_normalization_const_squared = np.log(np.sqrt(2 * nu))
    log_normalization_const_squared -= np.log(np.pow(np.pi, 1/2)) 
    log_normalization_const_squared -= np.log(2) * (2 * n)
    log_normalization_const_squared -= math.lgamma(2 * n + 2)

    return log_normalization_const_squared

def ho_n_radial_wavefunction(r, nu, n, reduced = False):
    """
    Returns the n-th radial wavefunction of the 3D harmonic oscillator.

    Parameters
    ----------
    r : ndarray or float
        Locations at which the radial wavefunction must be evaluated at.
    nu : float
        nu parameter of the harmonic oscillator.
    n : int
        Radial number of the radial wavefunction.

    Returns
    -------
    n_wf : ndarray
        Wavefunction evaluated at the required locations.
    """
    r = np.atleast_1d(r)
    if n != 0:
        n_wf = np.exp(_log_normalization_const_squared(nu, n) / 2) * np.exp(-nu * r * r) / r * eval_hermite(2 * n + 1, r)
    else:
        n_wf = np.exp(_log_normalization_const_squared(nu, n) / 2) * np.exp(-nu * r * r) * 2 # eval_hermite(1, r) = 2*r, the r cancels out
    
    if reduced:
        n_wf = n_wf * r
    
    return n_wf