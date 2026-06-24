import numpy as np
from scipy.integrate import simpson

from physipy.numerics_data import Grid, SolverOpts, ConvergenceError
from physipy.bound_states import energy_levels
import physipy.potentials as potentials

__all__ = [
    "self_consistent_gp",
    "gp_energy_functional"
]

def self_consistent_gp(E, g, guess = None, n_eigenstates = 1, max_iterations = 100, tol = 1e-3, alpha = 1e-2, grid = Grid(), solver = SolverOpts(), verbose = False, convergence_crit = 'L2', keep_error = False, mixing = 'linear_wf', throw = True, **kwargs):
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
        Linear wavefunction mixing parameter: guess = (1-alpha)*guess + alpha*psi (default 1e-2).
        Linear potential mixing parameter: v = (1-alpha)*v(guess) + alpha*v(psi)
        Anderson mixing: guess = cs @ history + alpha * cs @ residuals
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
    mixing : str
        Mixing procedure to use, the following values are possible:
            - 'linear_wf' : linear mixing of the wavefunction
            - 'linear_pot' : linear mixing of the potential
            - 'anderson' : Anderson mixing procedure, the depth is expected to be in kwargs
    throw : bool
        Whether to throw an exception if convergence is not attained within max_iterations.
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

    # initialize the mean-field args dict according to the chosen mixing procedure
    if mixing == 'linear_wf':
        args = {'g' : g, 'guess' : guess, 'grid_gp' : grid}
        potential = potentials.gross_pitaevskij
    elif mixing == 'linear_pot':
        args = {'g' : g, 'guess_1' : guess, 'guess_2' : None, 'grid_gp' : grid, 'alpha' : alpha}
        potential = _mixed_gross_pitaevskij
    elif mixing == 'anderson':
        args = {'g' : g, 'guess' : guess, 'grid_gp' : grid}
        potential = potentials.gross_pitaevskij
        depth = 3 if 'depth' not in kwargs else kwargs['depth']
        residuals = np.zeros((depth, len(coord)))
        history = np.zeros((depth, len(coord)))
        history[0] = guess
    else:
        raise ValueError('No good mixing procedure was provided.')

    converged = False
    diff = np.inf
    i = 0

    while not converged and i < max_iterations:
        eigenstates = energy_levels(E, [0], potential, n_eigenstates = n_eigenstates, grid = grid, solver = solver, **args)
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
            # mixing stage
            if mixing == 'linear_wf':
                guess = (1 - alpha) * guess + alpha * psi
                guess = guess / np.sqrt(simpson(guess**2, x=coord))
                args['guess'] = guess
            elif mixing == 'linear_pot':
                if args['guess_2'] is None:
                    # first iteration
                    args['guess_2'] = psi
                    guess = psi
                else:
                    args['guess_1'] = args['guess_2']
                    args['guess_2'] = psi
                    guess = psi
            elif mixing == 'anderson':
                # update residuals
                residuals[1:, :] = residuals[0:-1, :]
                residuals[0] = psi - guess
                # find minimizing coefficients and update guess
                cs = _anderson_mixing(residuals, i)
                guess = cs @ history + alpha * cs @ residuals
                guess /= np.sqrt(simpson(guess**2, x=coord))
                # shift history
                history[1:, :] = history[0:-1, :]
                history[0] = guess
                args['guess'] = guess

        i += 1

    if not converged:
        if throw:
            raise ConvergenceError(max_iterations, diff)
        else:
            print(f'The self-consistent did not converged, last iteration: g = {g:.2e} alpha = {alpha:.2e} mu = {mu:.4f}  E = {functional:.4f}  wf_diff = {diff:.2e}  E_residual = {energy_residual:.2e}  (tol = {tol:.1e})')
    else: 
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
    r = grid.coord
    dphi = np.zeros_like(phi)
    # 4th-order central differences for interior points
    dphi[2:-2] = (-phi[4:] + 8*phi[3:-1] - 8*phi[1:-3] + phi[:-4]) / (12*h)
    # 2nd-order central differences one step from the boundaries
    dphi[1]  = (phi[2]  - phi[0])  / (2*h)
    dphi[-2] = (phi[-1] - phi[-3]) / (2*h)
    # Left boundary: exploit the power-law BC phi ~ r (l=0 regular solution),
    # giving phi'(r_min) = phi(r_min)/r_min exactly to leading order.
    dphi[0]  = phi[0] / r[0]
    # Right boundary: 2nd-order backward difference (phi is exponentially small here)
    dphi[-1] = (3*phi[-1] - 4*phi[-2] + phi[-3]) / (2*h)

    integrand = (
                hbar_squared_over_2_m * dphi**2
                 + 0.5 * m * omega**2 * r**2 * phi**2
                 + 0.5 * g * phi**4 / r**2
            )

    E = simpson(integrand, x=r)

    # Kinetic energy missing from [0, r_min]: the l=0 power-law BC gives phi ~ r,
    # so phi' ~ phi/r is finite at the origin and the kinetic density (phi')^2 is
    # constant there. The missing integral is ½(phi'(r_min))^2 * r_min = ½ phi[0]^2/r[0].
    # The HO and interaction terms vanish as r^4 and r^2 respectively near the origin,
    # so their missing contributions are O(r_min^5) and O(r_min^3) — negligible.
    E += hbar_squared_over_2_m * phi[0]**2 / r[0]

    return E

def _anderson_mixing(residuals, iteration_number):
    """
    Compute Anderson mixing coefficients from a history of SCF residuals.

    Given the last `depth` residuals F_k = ψ_k − x_k (output minus input),
    finds coefficients c_k summing to 1 that minimise ‖Σ c_k F_k‖₂.  The
    minimiser is used by the caller to form the next input guess

        x_new = Σ c_k x_k  +  β · Σ c_k F_k

    where β is the caller's step size.  This is the type-I Anderson mixing
    scheme (Walker & Ni 2011).

    Parameters
    ----------
    residuals : ndarray, shape (depth, N)
        Circular buffer of residuals in reverse chronological order:
        residuals[0] is the most recent residual, residuals[k] is k
        iterations old.  Only the first min(iteration_number, depth) rows
        contain valid data.
    iteration_number : int
        Zero-based index of the current SCF iteration.  Controls how many
        rows of `residuals` are actually used.

    Returns
    -------
    cs : ndarray, shape (depth,)
        Mixing coefficients.  cs[0] corresponds to the current iterate,
        cs[k] to the iterate k steps back.  Entries beyond the active
        window are zero.  Sum of active entries equals 1.

    Notes
    -----
    On the first active iteration (iteration_number == 1) no history is
    available, so cs = [1, 0, …, 0] is returned (pure linear mixing with
    the current residual, controlled entirely by β in the caller).

    The least-squares solve uses `numpy.linalg.lstsq` with rcond=None,
    which regularises the problem when consecutive residuals are nearly
    parallel (near-linear-dependent history).
    """
    # if we're at the first iteration no mixing is needed
    if iteration_number == 1:
        cs = np.zeros(len(residuals))
        cs[0] = 1
        return cs
    depth = len(residuals)

    if iteration_number < depth:
        last = iteration_number
    else:
        last = depth

    # compute reduced residuals
    dF = residuals[0] - residuals[1:last]

    # use lstsq to obtain minimizing coefficients
    gamma, *_ = np.linalg.lstsq(dF.T, residuals[0], rcond = None)

    # reconstruct the coefficients
    cs = np.zeros(depth)
    cs[0] = 1 - np.sum(gamma)
    cs[1:last] = gamma

    return cs

def _mixed_gross_pitaevskij(r, **kwargs):
    """
    Evaluate the mean-field Gross-Pitaevskii potential at the given positions for two different guesses.

    The potential is V(r) = (1-αV_1(r) + αV_2(r), V_1 is the potential given by
    the previous wavefunction and V_2 is the potential given by the current wavefunction.

    Parameters
    ----------
    r : float or ndarray
        Position(s) at which the potential is evaluated.
    kwargs : dict
        Required and optional parameters:
        - guess_1   : ndarray, radial wavefunction from the previous iteration (required).
        - guess_2 : ndarray, radial wavefunction from the current iteration.
        - g     : float, coupling constant Na (default 1).
        - grid_gp : ndarray, containing the grid points.
        - alpha : mixing coefficient

    Returns
    -------
    V : ndarray
        Mixed GP potential evaluated at each position in r.
    """
    g = 1 if 'g' not in kwargs else kwargs['g']
    grid = Grid() if 'grid_gp' not in kwargs else kwargs['grid_gp']
    m = 1 if 'm' not in kwargs else kwargs['m']
    omega = 1 if 'omega' not in kwargs else kwargs['omega']
    alpha = 1 if 'alpha' not in kwargs else kwargs['alpha']

    if 'guess_1' not in kwargs or 'guess_2' not in kwargs:
        raise ValueError('The function _mixed_gross_pitaevskij requires two guesses in the input dictionary.')
    
    if kwargs['guess_2'] is None:
        args = {'g' : g, 'grid_gp' : grid, 'm' : m, 'omega' : omega, 'guess' : kwargs['guess_1']}
        return potentials.gross_pitaevskij(r, **args)

    args_1 = {'g' : g, 'grid_gp' : grid, 'm' : m, 'omega' : omega, 'guess' : kwargs['guess_1']}
    args_2 = {'g' : g, 'grid_gp' : grid, 'm' : m, 'omega' : omega, 'guess' : kwargs['guess_2']}

    return (1 - alpha) * potentials.gross_pitaevskij(r, **args_1) + alpha * potentials.gross_pitaevskij(r, **args_2)

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
