import numpy as np
import scipy as sp
from physipy.potentials import *
from physipy.wkb import WKB_seed
from physipy.numerics import _integrate_numerov, Grid, SolverOpts, Eigenstate

__all__ = [
    "compute_phase_shift"
]

def _integrate_scattering_state(E, l, potential, wkb = False, grid = Grid(), solver = SolverOpts(), **kwargs):
    """
    Perform Numerov integration of the radial Schrodinger equation for a scattering state.

    Parameters
    ----------
    E : float
        Particle's energy
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential energy to be used.
    wkb : bool
        Boolean flag that indicates whether we want to use the WKB seed
        over the power law seed.
    grid : class
        Mesh on which the integration is to be performed.
    solver : class
        Solver parameters to be used in the integration.
    kwargs : dict
        Additional arguments of the potential energy.

    Returns
    -------
    coord : ndarray
        Radial grid points of the eigenstate found.
    psi : ndarray
        Reconstructed wavefunction of the eigenstate found.
    """
    # check whether the problem is actually well-posed
    # meaning that on the right we're in a classical allowed region
    if not isin_classical_region(grid.r_max, E, l, potential, **kwargs):
        raise ValueError("The final point of the mesh is within a non-classical region : ill-defined scattering problem.")

    if wkb:
        psi_0 = 1
        psi_1 = WKB_seed(E, l, grid.r_min, grid.h, potential, outward = True, **kwargs)
    else:
        psi_0 = np.pow(grid.r_min, l + 1)
        psi_1 = np.pow(grid.r_min + grid.h, l + 1)

    coord, psi = _integrate_numerov(E, l, potential, psi_0, psi_1, grid, solver, outward = True, **kwargs)

    return (coord, psi)

def _tan_phase_shift(k, l, r1, r2, u1, u2):   
    """
    Compute the tangent of the phase shift associated to the l-wave.

    Parameters
    ----------
    k : float
        Particle's wave vector.
    l : int
        Angular momentum for which the phase shift is to be computed.
    r1 : float
        First point in which the wave function is evaluated.
    r2 : float
        Second point in which the wave function is evaluated.
    u1 : float
        Numerical value of the wave function at r1.
    u2 : float
        Numerical value of the wave function at r2.

    Returns
    -------
    tan_phase_shift : float
        Tangent of the phase shift associated to the l-wave.
    """
    kappa = u1 * r2 / (u2 * r1)
    tan_phase_shift = (sp.special.jv(l, k * r1) - kappa * sp.special.jv(l, k * r2))/(sp.special.yv(l, k * r1) - kappa * sp.special.yv(l, k * r2))
    return tan_phase_shift

def _phase_shift(k, l, r1, r2, u1, u2):
    """
    Compute the phase shift associated to the l-wave.

    Parameters
    ----------
    k : float
        Particle's wave vector.
    l : int
        Angular momentum for which the phase shift is to be computed.
    r1 : float
        First point in which the wave function is evaluated.
    r2 : float
        Second point in which the wave function is evaluated.
    u1 : float
        Numerical value of the wave function at r1.
    u2 : float
        Numerical value of the wave function at r2.

    Returns
    -------
    tan_phase_shift : float
        Phase shift associated to the l-wave.
    """
    phase_shift = np.arctan(_tan_phase_shift(k, l, r1, r2, u1, u2))
    return phase_shift

def compute_phase_shift(psi, coord, E, l, start_index, delta_index, N = 1, tan = True): #function, coordinates, indeces of starting, ending
    """
    Compute phase shift using two points of the asymptotic region of the scattering state.

    Parameters
    ----------
    psi : ndarray
        Wave function of the scattering state.
    coord : ndarray
        Coordinates of the wave function.
    E : float
        Particle's energy.
    l : int
        Angular momentum quantum number.
    start_index : int
        Start of the asymptotic region.
    delta_index : int
        Index step between the two points that are used to compute the phase shift.
    N : int
        Number of phase shifts to be evaluated to assess the asymptotic region.
    tan : bool
        Boolean flag indicating whether the tangent or the angle is to be returned.
    """
    k = wave_vector(E)

    # check the consistency of the input
    assert len(psi) == len(coord)
    assert start_index + N * delta_index <= len(psi)
    
    eval_function = _tan_phase_shift if tan else _phase_shift
    results = []

    for i in range (0, N):
        results.append(eval_function(k, l, coord[start_index + i * delta_index], coord[start_index + (i + 1) * delta_index], psi[start_index + i * delta_index], psi[start_index + (i + 1) * delta_index]))
    
    return (np.mean(results), np.std(results))
