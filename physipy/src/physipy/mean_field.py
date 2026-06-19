import numpy as np
from scipy.integrate import simpson

from physipy.numerics_data import Grid, SolverOpts, ConvergenceError
from physipy.bound_states import energy_levels
import physipy.potentials as potentials

__all__ = [
    "self_consistent_gp",
    "gp_energy_functional"
]

def self_consistent_gp(E, g, guess = None, n_eigenstates = 1, max_iterations = 100, tol = 1e-3, alpha = 1e-2, grid = Grid(), solver = SolverOpts(), verbose = False, convergence_crit = 'L2', keep_error = False, **kwargs):
    """
    Perform the self-consistent cycle to solve the Gross-Pitaevskii equation.

    At each iteration the ground-state wavefunction of the current mean-field
    potential is found via Numerov, normalized, and mixed with the previous
    guess using the linear mixing parameter alpha. Convergence is declared when
    the quantity selected by convergence_crit falls below tol. Both the L2
    wavefunction distance and the energy-consistency residual are computed and
    printed at every iteration regardless of which criterion drives convergence.

    Parameters
    ----------
    E : iterable
        Energy grid used to search for bound states.
    g : float
        Coupling constant Na.
    guess : ndarray or None
        Initial radial wavefunction guess. If None, the reduced HO ground
        state wavefunction is used (controlled by the 'nu' kwarg).
    n_eigenstates : int
        Number of eigenstates to track (default 1).
    max_iterations : int
        Maximum number of self-consistent iterations (default 100).
    tol : float
        Convergence threshold for the selected criterion (default 1e-3).
    alpha : float
        Linear mixing parameter: guess = (1-alpha)*guess + alpha*psi (default 1e-2).
    grid : Grid
        Radial grid (default Grid()).
    solver : SolverOpts
        Solver parameters (default SolverOpts()).
    verbose : boolean
        Whether to print at each iteration.
    convergence_crit : str
        Convergence criterion to be used, for now the following are allowed:
            - 'L2' : L2 distance.
            - 'functional' : energy-consistency residual |mu - E[psi] - <g/2 (psi/r)^2>|.
    keep_error : bool
        Whether to keep track of the chosen convergence criterion.
    kwargs : dict
        Additional parameters forwarded to the initial guess (e.g. nu for the HO).

    Returns
    -------
    E_bs : float
        Energy of the ground state from the GP functional.
    mu : float
        Chemical potential (Numerov eigenvalue).
    psi : ndarray
        Normalized radial wavefunction at convergence.
    error : ndarray
        None if keep_error is False, the chosen convergence criterion for each iteration.
    """
    if keep_error:
        error = []

    coord = grid.coord

    if guess is None:
        nu = 0.5 if 'nu' not in kwargs else kwargs['nu']
        guess = potentials.ho_n_radial_wavefunction(coord, nu, 0, reduced = True)

    # normalize the initial guess
    guess = guess / np.sqrt(simpson(guess**2, x=coord))

    # initialize the mean-field args dict
    args = {'g' : g, 'guess' : guess, 'grid_gp' : grid}

    converged = False
    diff = np.inf
    i = 0

    while not converged and i < max_iterations:
        eigenstates = energy_levels(E, [0], potentials.gross_pitaevskij, n_eigenstates = n_eigenstates, grid = grid, solver = solver, **args)
        mu = eigenstates[0].E
        psi_raw = eigenstates[0].psi
        psi_coord = eigenstates[0].coord

        # interpolate onto the full grid and normalize
        psi = np.interp(coord, psi_coord, psi_raw)
        psi = psi / np.sqrt(simpson(psi**2, x=coord))

        # wavefunction convergence: L2 distance between input guess and output psi
        l2_dist = np.sqrt(simpson((psi - guess)**2, x=coord))

        # energy diagnostics
        functional = gp_energy_functional(psi, g, grid, **kwargs)
        expected_diff = _gp_expected_difference(psi, g, grid)
        energy_residual = np.abs(mu - functional - expected_diff)
        
        if verbose:
            print(f"Iteration {i + 1}: g = {g:.2e} alpha = {alpha:.2e} mu = {mu:.4f}  E = {functional:.4f}  L2_dist = {l2_dist:.2e}  E_residual = {energy_residual:.2e}  (tol = {tol:.1e})")

        if convergence_crit == 'L2':
            diff = l2_dist
        elif convergence_crit == 'functional':
            diff = energy_residual
        else:
            raise ValueError('No good convergence criterion was provided')
        
        if keep_error:
            error.append(diff)

        if diff < tol:
            converged = True
        else:
            # mix old and new guess, then renormalize
            guess = (1 - alpha) * guess + alpha * psi
            guess = guess / np.sqrt(simpson(guess**2, x=coord))
            args['guess'] = guess

        i += 1

    if not converged:
        raise ConvergenceError(max_iterations, diff)
    
    print(f"Converged at iteration {i + 1}: g = {g:.2e} alpha = {alpha:.2e} mu = {mu:.4f}  E = {functional:.4f}  wf_diff = {diff:.2e}  E_residual = {energy_residual:.2e}  (tol = {tol:.1e})")

    if keep_error:
        return functional, mu, psi, error
    else:
        return functional, mu, psi, None

def gp_energy_functional(phi, g, grid = Grid(), **kwargs):
    """
    Evaluate the GP energy functional E[phi] for a normalized radial wavefunction.

    E = integral [ (hbar^2/2m)|dphi/dr|^2 + 1/2 m omega^2 r^2 phi^2 + g/2 phi^4/r^2 ] dr

    Parameters
    ----------
    phi : ndarray
        Normalized radial wavefunction on the grid.
    g : float
        Coupling constant Na.
    grid : Grid
        Radial grid (default Grid()).
    kwargs : dict
        Optional parameters: hbar_squared_over_2_m (default 0.5), m (default 1), omega (default 1).

    Returns
    -------
    E : float
        GP energy functional value.
    """
    hbar_squared_over_2_m = 1/2 if 'hbar_squared_over_2_m' not in kwargs else kwargs['hbar_squared_over_2_m']
    m     = 1 if 'm'     not in kwargs else kwargs['m']
    omega = 1 if 'omega' not in kwargs else kwargs['omega']

    h = grid.h
    dphi = np.zeros_like(phi)
    # 4th-order central differences for interior points, 1st-order at boundaries
    dphi[2:-2] = (-phi[4:] + 8*phi[3:-1] - 8*phi[1:-3] + phi[:-4]) / (12*h)
    dphi[0]  = (phi[1]  - phi[0])  / h
    dphi[1]  = (phi[2]  - phi[0])  / (2*h)
    dphi[-1] = (phi[-1] - phi[-2]) / h
    dphi[-2] = (phi[-1] - phi[-3]) / (2*h)

    r = grid.coord

    integrand = (hbar_squared_over_2_m * dphi**2
                 + 0.5 * m * omega**2 * r**2 * phi**2
                 + 0.5 * g * phi**4 / r**2)

    E = simpson(integrand, x=r)

    return E


def _gp_expected_difference(phi, g, grid = Grid()):
    """
    Compute the correction term <g/2 * (phi/r)^2> that relates mu to E:
    E = mu - integral[ g/2 * phi^4/r^2 dr ].

    Parameters
    ----------
    phi : ndarray
        Normalized radial wavefunction on the grid.
    g : float
        Coupling constant Na.
    grid : Grid
        Radial grid (default Grid()).

    Returns
    -------
    correction : float
        Value of the integral g/2 * integral[ phi^4/r^2 dr ].
    """
    r = grid.coord
    correction = 0.5 * g * simpson(phi**4 / r**2, x=r)

    return correction
