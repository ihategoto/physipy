import numpy as np
from itertools import cycle

from physipy.utils import *
from physipy.numerics import _integrate_numerov
from physipy.numerics_data import *
from physipy.potentials import *
from physipy.wkb import WKB_seed

__all__ = [
    "bisection",
    "energy_levels"
]


def _integrate_bound_state(E, l, potential, grid = Grid(), solver = SolverOpts(), **kwargs):
    """
    Perform outward and inward Numerov integration of the radial Schrödinger equation
    for a bound state and compute the Wronskian mismatch at the classical turning point.

    Outward and inward seeds are initialised internally:

    Outward seed at r_min
    ---------------------
    - Classical at r_min (regular potential, e.g. HO, Coulomb, GP): both seed points
      are set to the power-law u ~ r^{l+1}, which is the exact leading behaviour of
      the regular solution (Frobenius indicial equation). Using the hard-wall BC
      u(r_min) = 0 instead would place an artificial node at r_min rather than at
      r = 0, shifting the eigenvalue upward.
    - Non-classical at r_min (hard-core potential, e.g. Lennard-Jones): the
      wavefunction is physically zero at the core wall. The hard-wall BC
      (psi_0 = 0) is used together with a WKB exponentially decaying seed for
      the second point.

    Inward seed at r_max
    --------------------
    r_max is always in the classically forbidden region for a bound state; the WKB
    exponentially decaying form is used. The overall scale of psi_0_inward is
    irrelevant because it cancels in the Wronskian.

    Parameters
    ----------
    E : float
        Trial energy.
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential energy to be used.
    grid : Grid
        Mesh on which the integration is to be performed.
    solver : SolverOpts
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
    # check whether it is possible to perform inward integration
    if isin_classical_region(grid.r_max, E, l, potential, **kwargs):
        raise ValueError("The final point of the mesh is within a classical region, cannot perform inward integration.")

    # identify the matching point: rightmost grid point still inside the classical region
    f = isin_classical_region(grid.coord, E, l, potential, **kwargs)
    i = len(f) - 1

    try:
        while not f[i]:
            i -= 1
    except IndexError:
        #print("No classical region detected.")
        return (None, None, None)

    # if the classical region is too narrow for the buffer, treat as unusable
    if i < solver.match_buffer_steps or not f[i - solver.match_buffer_steps]:
        return (None, None, None)

    r_match = grid.r_min + i * grid.h - solver.match_buffer_steps * grid.h

    # --- outward seeds ---
    if bool(np.any(isin_classical_region(grid.r_min, E, l, potential, **kwargs))):
        # Regular potential: power-law leading behaviour
        psi_0_outward = grid.r_min ** (l + 1)
        psi_1_outward = (grid.r_min + grid.h) ** (l + 1)
    else:
        # Hard-core potential: hard-wall BC + WKB exponential decay
        psi_0_outward = 0.0
        psi_1_outward = WKB_seed(E, l, potential, grid, outward=True, **kwargs)

    outward_grid = Grid(grid.r_min, r_match, grid.h)
    coord_outward, psi_outward = _integrate_numerov(
        E, l, potential, psi_0_outward, psi_1_outward,
        outward_grid, solver, outward=True, store_wavefunction=True, **kwargs
    )

    # --- inward seeds ---
    psi_0_inward = 1.0
    psi_1_inward = WKB_seed(E, l, potential, grid, outward=False, **kwargs)

    inward_grid = Grid(r_match, grid.r_max, grid.h)
    coord_inward, psi_inward = _integrate_numerov(
        E, l, potential, psi_0_inward, psi_1_inward,
        inward_grid, solver, outward=False, store_wavefunction=True, **kwargs
    )

    # Wronskian mismatch at r_match
    temp = psi_outward[::-1]
    uo0, uo1, uo2 = temp[0], temp[1], temp[2]
    ui0, ui1, ui2 = psi_inward[0], psi_inward[1], psi_inward[2]

    if abs(uo0) < solver.tol or abs(ui0) < solver.tol:
        print("Numerical stability error: matching region too close to a node of the solution.")
        print("Try again adjusting the buffer.")
        return (None, None, 1e30)

    # 2nd-order one-sided finite differences at r_match
    duo = (3*uo0 - 4*uo1 + uo2) / (2*grid.h)
    dui = (-3*ui0 + 4*ui1 - ui2) / (2*grid.h)

    W = uo0 * dui - ui0 * duo

    # merge outward and inward solutions (rescale inward to match outward amplitude)
    scale = uo0 / ui0
    psi_inward = psi_inward * scale
    psi = np.concatenate((psi_outward, psi_inward))
    coord = np.concatenate((coord_outward, coord_inward))

    return (coord, psi, W)


def bisection(err_prev, err_curr, E_prev, E_curr, l, potential, grid = Grid(), solver = SolverOpts(), **kwargs):
    """
    Find a zero of the Wronskian mismatch via bisection between two energies at which
    the mismatch has opposite signs.

    Parameters
    ----------
    err_prev : float
        Wronskian mismatch at E_prev.
    err_curr : float
        Wronskian mismatch at E_curr.
    E_prev : float
        Lower bracket energy.
    E_curr : float
        Upper bracket energy.
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential to be used.
    grid : Grid
        Mesh on which the integration is to be performed.
    solver : SolverOpts
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
        E_med = (E_prev + E_curr) / 2
        coord, psi, new_error = _integrate_bound_state(E_med, l, potential, grid, solver, **kwargs)

        if np.sign(err_prev) == np.sign(new_error):
            E_prev = E_med
        else:
            E_curr = E_med

    eigenvalue = (E_prev + E_curr) / 2

    return (coord, psi, eigenvalue)


def energy_levels(E, l, potential, n_eigenstates = 0, grid = Grid(), solver = SolverOpts(), verbose = False, **kwargs):
    """
    Search for bound-state eigenvalues by scanning a trial-energy list for sign changes
    of the Wronskian mismatch, then refining each bracket via bisection.

    Parameters
    ----------
    E : iterable
        Energy list at which the Wronskian is evaluated.
    l : iterable
        Angular momentum quantum numbers to scan.
    potential : callable
        Potential energy function.
    n_eigenstates : int
        Maximum number of eigenstates to return.  0 means return all found
        within the scanned range (default 0).
    grid : Grid
        Radial mesh (default Grid()).
    solver : SolverOpts
        Solver parameters (default SolverOpts()).
    verbose : boolean
        Whether to print at each iteration.
    kwargs : dict
        Additional arguments forwarded to the potential.

    Returns
    -------
    eigenstates : list[Eigenstate]
        Eigenstate objects (energy, angular momentum, radial grid, wavefunction)
        for each bound state found, ordered by angular momentum then energy.
    """
    h = grid.h

    found_eigenstates = 0
    l_length = len(l)
    E_length = len(E)
    err_prev = None
    err_curr = 0
    E_prev   = None
    flag     = 1

    eigenstates = []

    l_iter = iter(l)
    E_iter = cycle(iter(E))
    _l = next(l_iter)
    _E = next(E_iter)
    i = 0
    j = 0

    while i < l_length:
        while j < E_length and flag:
            __1, __2, error = _integrate_bound_state(_E, _l, potential, grid, solver, **kwargs)
            err_curr = error

            if err_prev is not None and E_prev is not None and err_curr is not None and np.sign(err_curr) != np.sign(err_prev):
                coord, psi, good_energy = bisection(err_prev, err_curr, E_prev, _E, _l, potential, grid, solver, **kwargs)
                if verbose:
                    print("Found energy at : E = {:.3f} l = {} h = {}".format(good_energy, _l, h))
                eigenstates.append(Eigenstate(good_energy, _l, coord, psi))
                found_eigenstates += 1
                if n_eigenstates != 0 and found_eigenstates == n_eigenstates:
                    flag = 0

            E_prev   = _E
            err_prev = err_curr
            _E = next(E_iter, None)
            j += 1

        err_prev = None
        E_prev   = None
        i += 1
        j = 0
        _l = next(l_iter, None)

    return eigenstates
