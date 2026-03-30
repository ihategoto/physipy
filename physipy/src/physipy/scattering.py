import numpy as np
import scipy as sp

from physipy.utils import *
from physipy.potentials import *
from physipy.wkb import WKB_seed
from physipy.numerics import _integrate_numerov, Grid, SolverOpts, Eigenstate

__all__ = [
    "integrate_scattering_state",
    "compute_phase_shift",
    "normalize_scattering_state",
    "normalize_lj_scattering_state"
]

def integrate_scattering_state(E, l, potential, wkb = False, grid = Grid(), solver = SolverOpts(), store_wavefunction = True, n_points = 1000, **kwargs):
    """
    Perform Numerov integration of the radial Schrodinger equation for a scattering state.

    Parameters
    ----------
    E : float or ndarray
        Particle's energy(s).
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
    # check whether the problem is well-posed
    # meaning that on the right we're in a classical allowed region
    if not np.any(isin_classical_region(grid.r_max, E, l, potential, **kwargs)):
        raise ValueError("The final point of the mesh is within a non-classical region : ill-defined scattering problem.")
    
    # check whether the window is large enough for each energy
    if not store_wavefunction:
        wavelength = 2 * np.pi / wave_vector(E, **kwargs)
        interval = grid.h * n_points
        n_cycles = interval / wavelength
        
        if n_cycles < 2:
            print('Too few cycles in the chosen region.')

    if wkb:
        psi_0 = 1
        psi_1 = WKB_seed(E, l, grid.r_min, grid.h, potential, outward = True, **kwargs)
    else:
        psi_0 = np.pow(grid.r_min, l + 1)
        psi_1 = np.pow(grid.r_min + grid.h, l + 1)

    coord, psi = _integrate_numerov(E, l, potential, psi_0, psi_1, grid, solver, outward = True, store_wavefunction = store_wavefunction, n_points = n_points, **kwargs)

    """
    for _psi, _E in zip(psi, E):
        if not isin_asymptotic_region(_psi, coord, wave_vector(_E, **kwargs)):
            print(f'For E = {_E:.2f} the wavefunction might not be in the asymptotic region.')
    """
            
    return (coord, psi)


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
    phase_shift : float or None
        Phase shift associated to the l-wave, in (-pi/2, pi/2].
        Returns None if the pair of points is numerically unstable.
    """
    if not np.isfinite([k, r1, r2, u1, u2]).all():
        return None

    # Reject points too close to a node of the numerical solution.
    amp_tol = 1e-3
    if abs(u1) < amp_tol or abs(u2) < amp_tol:
        return None

    kr1, kr2 = k * r1, k * r2

    j1 = sp.special.spherical_jn(l, kr1)
    j2 = sp.special.spherical_jn(l, kr2)
    y1 = sp.special.spherical_yn(l, kr1)
    y2 = sp.special.spherical_yn(l, kr2)

    w1 = u1 * r1
    w2 = u2 * r2

    numerator   = j1 * w2 - j2 * w1
    denominator = y1 * w2 - y2 * w1

    if not np.isfinite([numerator, denominator]).all():
        return None

    # Reject ill-conditioned windows (both projections vanish simultaneously).
    if np.hypot(numerator, denominator) < 1e-6:
        return None

    return float(np.arctan2(numerator, denominator))


def compute_phase_shift(psi, coord, E, l, **kwargs):
    """
    Compute the phase shift by averaging over a sliding window of point pairs.

    Parameters
    ----------
    psi : array-like
        Numerical radial wave function (reduced: u = r * R).
    coord : array-like
        Radial coordinate array, same length as psi.
    E : float
        Energy of the scattering state.
    l : int
        Angular momentum quantum number.
    delta_index : int
        Separation in grid steps between the two matching points.
    tan : bool, optional
        If True, return tan(delta) instead of delta.
    **kwargs
        Passed to wave_vector(E, ...).

    Returns
    -------
    results : ndarray
        All accepted raw estimates (phase shifts or tan values).
    mean : float or None
        Circular mean of the filtered estimates, None if too few points.
    std : float or None
        Standard deviation of the filtered residuals, None if too few points.
    """
    k = wave_vector(E, **kwargs)

    # find the period in terms of indexes
    peaks_idx, _ = sp.signal.find_peaks(psi)
    period_idx = int(round(np.mean(np.diff(peaks_idx))))

    estimates = []
    for peak in peaks_idx:
        i2 = peak + period_idx // 4
        if i2 >= len(coord):
            continue
        ps = _phase_shift(k, l, coord[peak], coord[i2], psi[peak], psi[i2])
        if ps is not None:
            estimates.append(ps)
            print(f"For E={E:.3f} and l={l} r1={coord[peak]:.2f}, r2={coord[i2]:.2f} → δ = {ps:.6f} rad")


    if len(estimates) == 0:
        raise ValueError("Could not be possible to estimate the phase!")
    
    estimates = np.array(estimates)
    wrapped_est = 0.5 * np.angle(np.exp(2j * estimates))
    z = np.mean(np.exp(2j * wrapped_est))
    mean_phase = 0.5 * np.angle(z)
    residuals  = 0.5 * np.angle(np.exp(2j * (wrapped_est - mean_phase)))
    std_phase  = np.std(residuals)

    return (mean_phase, std_phase)


def normalize_scattering_state(psi):
    """
    Compute the normalized wave function. 
    Also support multiple wave functions as input.

    Parameters
    ----------
    psi : ndarray
        Can either be the plain wavefunction or an ndarray containing several wavefunctions.

    Returns
    -------
    normalized_state : ndarray
        Normalized wavefunction(s).
    """
    assert isinstance(psi, np.ndarray)

    if len(psi.shape) == 1:
        normalized_state = psi/np.max(psi)
    else:
        normalized_state = psi/np.max(psi, axis = 1)
    
    return normalized_state

def normalize_lj_scattering_state(coord, psi, sigma = 1, n = 1):
    assert isinstance(psi, np.ndarray)
    
    asymptotic_regime = np.where(coord > n * sigma, 1, 0)
    temp = psi * asymptotic_regime

    normalized_state = psi / np.max(temp)

    return normalized_state

def cross_section(ls, ps_l):
    tot_cr = 4 * np.pi * np.sum((2 + ls + 1) * np.pow(np.sin(ps_l), 2))
    return tot_cr
