import numpy as np
import math as mt

from physipy.utils import *
from physipy.cython import numerics
from physipy.numerics_data import *

def _numerov_step(psi_prev, psi_curr, k_prev, k_curr, k_next, h):
    """
    Perform a single step of the Numerov method.

    Parameters
    ----------
    psi_prev : float or ndarray
        The solution(s) at current_position - h.
    psi_curr : float or ndarray
        The solution at current_position.
    k_prev : float or ndarray
        The k(s) evaluated at current_position - h.
    k_curr : float or ndarray
        The k(s) evaluated at current_position.
    k_next : float or ndarray
        The k(s) evaluated at current_position + h.
    h : float
        The integration step.

    Returns
    -------
    step : float or ndarray
        Solution(s) at the next point in the mesh.

    """
    step = (2 * (1 - (5 * h**2 * k_curr / 12)) * psi_curr - (1 + (h**2 * k_prev / 12)) * psi_prev) / (1 + (h**2 * k_next / 12))

    return step

'''
def _integrate_numerov_vectorized(E, l, potential, psi_0, psi_1, grid = Grid(), solver = SolverOpts(), outward = True, store_wavefunction = False, n_points = 2, **kwargs):
    """
    Perform a Numerov integration of the radial Schrödinger equation.
    Not supported.

    Parameters
    ----------
    E : float or ndarray
        Particle's energy.
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential energy to be used.
    psi_0 : float
        First seed value for Numerov integration.
    psi_1 : float
        Second seed value for Numerov integration.
    grid : class
        Mesh on which the integration is to be performed.
    solver : class
        Solver parameters to be used in the integration.
    outward : bool
        Flag to select the direction of the integration.
    store_wavefunction : bool
        Boolean flag that tells if the whole wavefunction(s) is to be stored.
    n_points : int
        Number of points (>2) to be stored if store_wavefunction is False.
    kwargs : dict
        Additional arguments of the potential energy.

    Returns
    -------
    coord : ndarray or None
        Radial grid points if store_wavefunction is True.
    psi : ndarray
        Solution wavefunction.
    
    """
    r_min = grid.r_min
    r_max = grid.r_max
    h = grid.h if outward else -1 * grid.h
    r = grid.r_min if outward else grid.r_max
    n_steps = mt.ceil(abs(r_max - r_min) / abs(grid.h))
    grid  = r_min + np.arange(n_steps) * abs(grid.h)
    E = np.atleast_1d(E)

    # adjust psi_0 and psi_1 to match the shape of E
    if isinstance(E, np.ndarray):
        psi_0 = np.ones(E.shape[0]) * psi_0
        psi_1 = np.ones(E.shape[0]) * psi_1

    # initialize psi and coord with known points
    if store_wavefunction:
        coord = np.zeros(n_steps)
        psi = np.zeros((E.shape[0], n_steps))
        psi[:, 0] = psi_0
        psi[:, 1] = psi_1
        coord[0] = r
        coord[1] = r + h
    else:
        psi = np.zeros((E.shape[0], n_points))
        coord = np.zeros(n_points)
        psi[:, -2] = psi_0
        psi[:, -1] = psi_1
        coord[-2] = r
        coord[-1] = r + h

    k = k_squared(grid, l, E, potential, **kwargs)
    k_prev = k[:, 0]
    k_curr = k[:, 1]
    i = 2

    while i < n_steps:
        k_next = k[:, i]
        if store_wavefunction:
            temp = _numerov_step(psi[:, i - 2], psi[:, i - 1], k_prev, k_curr, k_next, h)
            psi[:, i] = temp
        else:
            temp = _numerov_step(psi[:, -2], psi[:, -1], k_prev, k_curr, k_next, h)
            psi[:, :-1] = psi[:, 1:]
            psi[:, -1] = temp

        # normalize to prevent blow up
        psi[np.ravel(np.abs(temp) > solver.renorm_threshold)] *= solver.renorm_factor
        k_prev = k_curr
        k_curr = k_next
        r += h
        if store_wavefunction:
            coord[i] = r + h
        else:
            coord[:-1] = coord[1:]
            coord[-1] = r + h
        i += 1
    
    if not outward:
        # reverse the lists containing the coordinates and values of the solution
        coord = coord[::-1]
        psi = psi[:, ::-1]
    
    return (coord, psi)
'''

def _integrate_numerov(E, l, potential, psi_0, psi_1, grid = Grid(), solver = SolverOpts(), outward = True, store_wavefunction = False, n_points = None, **kwargs):
    """
    Perform a Numerov integration of the radial Schrödinger equation.
    It uses Cython-based Numerov's step.

    Parameters
    ----------
    E : float or ndarray
        Particle's energy.
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential energy to be used.
    psi_0 : float
        First seed value for Numerov integration.
    psi_1 : float
        Second seed value for Numerov integration.
    grid : class
        Mesh on which the integration is to be performed.
    solver : class
        Solver parameters to be used in the integration.
    outward : bool
        Flag to select the direction of the integration.
    store_wavefunction : bool
        Boolean flag that tells if the whole wavefunction(s) is to be stored.
    n_points : int
        Number of points (>2) to be stored if store_wavefunction is False.
    kwargs : dict
        Additional arguments of the potential energy.

    Returns
    -------
    coord : ndarray or None
        Solution coordinates.
    psi : ndarray
        Solution wavefunction.
    
    """
    if not store_wavefunction and n_points is None:
        raise ValueError('Both store_wavefunction and n_points are None.')

    h = grid.h if outward else -1 * grid.h
    coord  = grid.coord
    psi = np.zeros(len(coord))
    k2 = k_squared(coord, l, E, potential, **kwargs)

    if not outward:
        k2 = k2[::-1]
    
    psi = numerics._integrate_numerov_cython(np.asarray(psi_0).item(), np.asarray(psi_1).item(), h, solver.renorm_threshold, solver.renorm_factor, psi, k2)
    
    if not store_wavefunction:
        coord = coord[-n_points:]
        psi = psi[-n_points:]

    if not outward:
        psi = psi[::-1]

    return (coord, psi)

