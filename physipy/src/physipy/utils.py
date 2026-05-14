import numpy as np
import physipy.constants as constants
from  physipy.potentials import effective_potential
from physipy.numerics_data import NumericalOpts

__all__ = [
    'k_squared',
    'isin_classical_region',
    'bessel_j',
    'bessel_n',
    'isin_asymptotic_region',
    'r_asym_min',
    'n_points_needed'
]

def r_asym_min(E, l, potential, hbar2_over_2mu, safety = 2.0, pot_tol = 0.01, length_scale = None, **kwargs):
    """
    Estimate the start of the asymptotic region for a scattering state.

    Walks outward from 5*length_scale until |V(r)| < pot_tol*E, then
    applies a safety multiplier and takes the maximum against the
    centrifugal and de Broglie length scales.

    Parameters
    ----------
    E : float
        Particle's kinetic energy.
    l : int
        Angular momentum quantum number.
    potential : callable
        Bare potential V(r).
    hbar2_over_2mu : float
        Prefactor ℏ²/2μ for dimensional consistency.
    safety : float
        Multiplicative safety factor applied to the estimated range (default 2.0).
    pot_tol : float
        Fraction of E below which the potential is considered negligible (default 0.01).
    length_scale : float, optional
        Typical length scale of the potential. Defaults to kwargs['sigma'] or 1.
    kwargs : dict
        Additional arguments forwarded to potential.

    Returns
    -------
    asym_reg : float
        Estimated start of the asymptotic region.
    """
    k = np.sqrt(E / hbar2_over_2mu)
    r_centr  = np.sqrt(hbar2_over_2mu * l * (l + 1) / E) if l > 0 else 0.0
    r_bessel = l / k if l > 0 else 0.0

    if length_scale is None:
        length_scale = kwargs.get('sigma', 1)

    r_test = 5 * length_scale

    v = np.abs(potential(r_test, **kwargs))
    while v > pot_tol * E:
        r_test += 0.1 * length_scale
        v = np.abs(potential(r_test, **kwargs))

    asym_reg = safety * max(r_centr, r_bessel, r_test)

    return asym_reg

def n_points_needed(E, hbar2_over_2mu, dr, n_cycles = 3):
    """
    Compute the number of grid points needed to cover n_cycles wavelengths.

    Parameters
    ----------
    E : float
        Particle's kinetic energy.
    hbar2_over_2mu : float
        Prefactor ℏ²/2μ for dimensional consistency.
    dr : float
        Grid step size.
    n_cycles : int
        Number of full wavelengths to cover (default 3).

    Returns
    -------
    p : int
        Minimum number of grid points to span n_cycles wavelengths.
    """
    k = np.sqrt(E / hbar2_over_2mu)
    lambda_ = 2 * np.pi / k
    p = int(np.ceil(n_cycles * lambda_ / dr))

    return p

def k_squared(r, l, E, potential, numerical = NumericalOpts(), **kwargs):
    """
    Compute k²(r) = (E - V_eff(r)) / (ℏ²/2m) for use in the Numerov algorithm.

    Parameters
    ----------
    r : float or ndarray
        Position(s) at which k² is evaluated.
    l : int
        Angular momentum quantum number.
    E : float
        Particle's energy.
    potential : callable
        Bare potential V(r).
    numerical : NumericalOpts
        Numerical stability parameters (default NumericalOpts()).
    kwargs : dict
        Additional arguments forwarded to potential and for dimensional fixing
        (see effective_potential).

    Returns
    -------
    k_2 : float or ndarray
        k² evaluated at each position in r. Positive inside the classical region.
    """
    r = np.atleast_1d(r)
    r = np.where(r < numerical.tol, numerical.tol, r)

    pot = potential(r, **kwargs)

    if 'hbar_squared_over_2_m' in kwargs:
        pre_factor = kwargs['hbar_squared_over_2_m']
        centr_barrier = pre_factor * l * (l + 1)/(r * r)
        k_2 = (E - pot - centr_barrier) / pre_factor
    else:
        m = 1 if not 'm' in kwargs else kwargs['m']
        centr_barrier = 0.5 * (constants.hbar * constants.hbar) / m * l * (l + 1)/(r * r)
        k_2 = 2 * m / (constants.hbar * constants.hbar) * (E - pot - centr_barrier)

    return k_2

def isin_asymptotic_region(psi, h, k_2, numerical = NumericalOpts(), rejection_ratio = 0.1):
    """
    Check whether a wavefunction has reached the asymptotic (free-particle) region.

    Estimates the local k² numerically as -ψ''/ψ via finite differences and
    compares it to the expected asymptotic value. Points near nodes are masked out.

    Parameters
    ----------
    psi : ndarray
        Radial wavefunction values.
    h : float
        Grid step size.
    k_2 : float
        Expected asymptotic k² (wave vector squared).
    numerical : NumericalOpts
        Numerical stability parameters (default NumericalOpts()).
    rejection_ratio : float
        Maximum relative deviation of the local k² from k_2 to accept
        the asymptotic condition (default 0.1).

    Returns
    -------
    c : bool
        True if the wavefunction is in the asymptotic region, False otherwise.
    """
    psi_pp = (psi[:-2] - 2*psi[1:-1] + psi[2:]) / h**2

    psi_mid = psi[1:-1]

    # Mask near-node points
    mask = np.abs(psi_mid) > numerical.tol

    k2_local = np.mean(-psi_pp[mask] / psi_mid[mask])
    c = (k2_local - k_2) / k_2 < rejection_ratio

    return c


def isin_classical_region(r, E, l, potential, numerical = NumericalOpts(), **kwargs):
    """
    Check whether positions are within the classical region for a given potential.

    Parameters
    ----------
    r : float or ndarray
        Position(s) to test.
    E : float
        Particle's energy.
    l : int
        Angular momentum quantum number.
    potential : callable
        Bare potential V(r).
    numerical : NumericalOpts
        Numerical stability parameters (default NumericalOpts()).
    kwargs : dict
        Additional arguments forwarded to effective_potential.

    Returns
    -------
    f : bool or ndarray
        True at each position where E > V_eff(r) (classical region).
    """
    r = np.atleast_1d(r)
    r = np.where(r < numerical.tol, numerical.tol, r)

    v_eff = effective_potential(r, l, potential, **kwargs)
    f = (E - v_eff) > 0

    return f

def bessel_j(x, l, numerical = NumericalOpts()):
    """
    Evaluate the spherical Bessel function of the first kind j_l(x).

    Computed via the upward recurrence relation j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x),
    starting from j_0(x) = sin(x)/x and j_{-1}(x) = cos(x)/x.

    Parameters
    ----------
    x : float
        Argument at which the function is evaluated.
    l : int
        Order (angular momentum quantum number). Use l=-1 to get j_{-1}.
    numerical : NumericalOpts
        Numerical stability parameters (default NumericalOpts()).

    Returns
    -------
    j_l : float
        Value of j_l(x).
    """
    x = x if x > numerical.tol else numerical.tol
    n_curr = np.sin(x) / x
    n_prev = np.cos(x) / x

    if l == -1:
        return n_prev

    for i in range(0, l):
        temp = n_curr
        n_curr = (2 * i + 1) / x * n_curr - n_prev
        n_prev = temp

    return n_curr

def bessel_n(x, l, numerical = NumericalOpts()):
    """
    Evaluate the spherical Bessel function of the second kind y_l(x) (irregular).

    Computed via the upward recurrence relation y_{l+1}(x) = (2l+1)/x * y_l(x) - y_{l-1}(x),
    starting from y_0(x) = -cos(x)/x and y_{-1}(x) = sin(x)/x.

    Parameters
    ----------
    x : float
        Argument at which the function is evaluated.
    l : int
        Order (angular momentum quantum number). Use l=-1 to get y_{-1}.
    numerical : NumericalOpts
        Numerical stability parameters (default NumericalOpts()).

    Returns
    -------
    y_l : float
        Value of y_l(x).
    """
    x = x if x > numerical.tol else numerical.tol
    n_curr = (-1) * np.cos(x) / x
    n_prev = np.sin(x) / x

    if l == -1:
        return n_prev

    for i in range(0, l):
        temp = n_curr
        n_curr = (2 * i + 1) / x * n_curr - n_prev
        n_prev = temp

    return n_curr
