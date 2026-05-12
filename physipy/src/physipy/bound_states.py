import numpy as np

from physipy.utils import *
from physipy.numerics import _integrate_numerov, Grid, SolverOpts, Eigenstate
from physipy.potentials import *
from physipy.wkb import WKB_seed

__all__ = [
    "bisection",
    "energy_levels"
]


def _integrate_bound_state(E, l, potential, psi_0_outward, psi_1_outward, psi_0_inward, psi_1_inward, grid = Grid(), solver = SolverOpts(), **kwargs):
    """
    Perform outward and inward Numerov integration of the radial Schrödinger equation for a bound state and compute the 
    Wronskian mismatch at the classical turning point.

    Parameters
    ----------
    E : float
        Trial energy.
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential energy to be used.
    psi_0_outward, psi_1_outward : float
        Initial values for outward Numerov integration.
    psi_0_inward, psi_1_inward : float
        Initial values for inward Numerov integration (e.g. from WKB).
    grid : class
        Mesh on which the integration is to be performed.
    solver : class
        Solver parameters to be used in the integration.
    kwargs : dict
        Additional arguments of the potential energy.

    Returns
    -------
    coord : ndarray
        Radial grid points.
    psi : ndarray
        Reconstructed wavefunction.
    W : float
        Wronskian mismatch used for eigenvalue search.

    """
    # check whether it is possibile to perform inward intergration
    if isin_classical_region(grid.r_max, E, l, potential, **kwargs):
        # not good, r_max must be after the rightmost classical turning point
        raise ValueError("The final point of the mesh is within a classical region, cannot perform inward integration.")
    
    # identify the matching point in the grid
    # note: that wwe take the matching point as the right-most classical turning point
    f = isin_classical_region(np.arange(grid.r_min, grid.r_max + grid.h, grid.h), E, l, potential, **kwargs)
    i = len(f) - 1

    try:
        while not f[i]:
            i -= 1
    except IndexError:
        print("No classical region detected.")
        return (None, None, None)
    
    # check whether the matching point is within the classical region, if not raise an exception
    if not f[i - solver.match_buffer_steps]:
        raise IndexError('Matching region in non-classical region, please adjust the buffer size.')
    
    r_match = grid.r_min + i * grid.h - solver.match_buffer_steps * grid.h

    # outward integration (classical region)
    outward_grid = Grid(grid.r_min, r_match, grid.h)
    coord_outward, psi_outward = _integrate_numerov(E, l, potential, psi_0_outward, psi_1_outward, outward_grid, solver, outward = True, store_wavefunction = True, **kwargs)

    # inward intergration (non-classical region)
    inward_grid = Grid(r_match, grid.r_max, grid.h)
    coord_inward, psi_inward = _integrate_numerov(E, l, potential, psi_0_inward, psi_1_inward, inward_grid, solver, outward = False, store_wavefunction = True, **kwargs)
    
    # calculate matching condition
    temp = psi_outward[::-1]
    uo0, uo1, uo2 = temp[0], temp[1], temp[2]
    ui0, ui1, ui2 = psi_inward[0], psi_inward[1], psi_inward[2]

    # check for numerical stability
    if abs(uo0) < solver.tol or abs(ui0) < solver.tol:
        # match point too close to a node
        print("Numerical stability error: matching region too close to a node of the solution.")
        print("Try again adjusting the buffer.")
        return (None, None, 1e30)

    # 2nd-order one-sided derivatives at r_match
    duo = (3*uo0 - 4*uo1 + uo2) / (2*grid.h)
    dui = (-3*ui0 + 4*ui1 - ui2) / (2*grid.h)

    W = uo0 * dui - ui0 * duo

    # merge outward and inward solutions
    scale = uo0 / ui0
    psi_inward = psi_inward * scale
    psi = np.concatenate((psi_outward, psi_inward))
    coord = np.concatenate((coord_outward, coord_inward))

    return (coord, psi, W)

def bisection(err_prev, err_curr, E_prev, E_curr, l, potential, psi_0_outward, psi_1_outward, psi_0_inward, psi_1_inward, grid = Grid(), solver = SolverOpts(), **kwargs):
    """
    Perform the bisection method in order to find the zero of the Wronskian mismatch between two energies that make its sign change.

    Parameters
    ----------
    err_prev : float
        Previous Wronskian mismatch.
    err_curr : float
        Current Wronskian mismatch.
    E_prev : float
        Previous energy.
    E_curr : float
        Current energy.
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential to be used.
    psi_0_outward, psi_1_outward : float
        Initial values for outward Numerov integration.
    psi_0_inward, psi_1_inward : float
        Initial values for inward Numerov integration (e.g. from WKB).
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
    eigenvalue : float
        Estimated eigenvalue.

    """
    if np.sign(err_prev) == np.sign(err_curr):
        print('Error: invalid starting points.')
        return (None, None, None)

    while abs(E_prev - E_curr) > solver.bisection_tol:
        E_med = (E_prev + E_curr)/2
        coord, psi, new_error = _integrate_bound_state(E_med, l, potential, psi_0_outward, psi_1_outward, psi_0_inward, psi_1_inward, grid, solver, **kwargs)
        
        if np.sign(err_prev) == np.sign(new_error):
            E_prev = E_med
        else:
            E_curr = E_med
            
    eigenvalue = (E_prev + E_curr) / 2

    return (coord, psi, eigenvalue)

def energy_levels(E, l, potential, psi_0_outward , grid = Grid(), solver = SolverOpts(), **kwargs):
    """
    Perform the integration using Numerov algorithm for the specified angular momentum list and energy range.
    
    Parameters
    ----------
    E : iterable
        Energy list for which the integration is to be performed.
    l : iterable
        Angular momentum list for which the integration is to be performed.
    potential : callable
        Potential energy to be used in the integration.
    psi_0_outward : float
        Initial values for outward Numerov integration.
    grid : class
        Mesh on which the integration is to be performed.
    solver : class
        Solver parameters to be used in the integration.
    kwargs : dict
        Additional arguments of the potential energy.

    Returns
    -------
    eigenstates : list
        List containing Eigenstate objects for each eigenstate found in the working range of energy and angular momentum.

    """
    # grid parameters
    r_min = grid.r_min
    r_max = grid.r_max
    h = grid.h

    err_prev = None
    err_curr = 0
    E_prev = None

    # standard inward seed (scale does not matter)
    psi_0_inward = 1

    eigenstates = []

    for _l in l:
        for _E in E:
            psi_1_outward = WKB_seed(_E, _l, r_min, h, potential, outward = True, **kwargs)
            psi_1_inward = WKB_seed(_E, _l, r_max, h, potential, outward = False, **kwargs)
            __1, __2, error = _integrate_bound_state(_E, _l, potential, psi_0_outward, psi_1_outward, psi_0_inward, psi_1_inward, grid, solver, **kwargs)
            err_curr = error
            if err_prev is not None and E_prev is not None and np.sign(err_curr) != np.sign(err_prev):
                coord, psi, good_energy = bisection(err_prev, err_curr, E_prev, _E, _l, potential, psi_0_outward, psi_1_outward, psi_0_inward, psi_1_inward, grid, solver, **kwargs)
                print("Found energy at : E = {:.3f} l = {} h = {}".format(good_energy, _l, h))
                eigenstates.append(Eigenstate(good_energy, _l, coord, psi))
            E_prev = _E
            err_prev = err_curr
        err_prev = None
        E_prev = None

    return eigenstates