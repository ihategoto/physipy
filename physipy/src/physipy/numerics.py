from dataclasses import dataclass
import numpy as np

from physipy.utils import *

__all__ = [
    "Grid",
    "SolverOpts",
    "Eigenstate"
]

@dataclass(frozen = True)
class Grid:
    """
    Class containing the grid parameters.
    """
    r_min: float = 1e-4
    r_max: float = 10
    h: float = 1e-3

    def __post_init__(self):
        if self.r_min <= 0:
            raise ValueError("Grid r_min must be positive")
        if self.r_max <= self.r_min:
            raise ValueError("Grid r_max must be larger than r_min")
        if self.h <= 0:
            raise ValueError("Step size h must be positive")

@dataclass(frozen = True)
class SolverOpts:
    """
    Class containing the numerical stability parameters.
    """
    tol: float = 1e-6
    bisection_tol: float = 1e-6
    match_buffer_steps: int = 10
    renorm_threshold: float = 1e20
    renorm_factor: float = 1e-20

@dataclass(frozen = True)
class Eigenstate:
    """
    Class containing all the relevant features of an Eigenstate
    """
    E: float
    l: int
    coord: np.ndarray
    psi: np.ndarray

def _numerov_step(psi_prev, psi_curr, k_prev, k_curr, k_next, h):
    """
    Perform a single step of the Numerov method.

    Parameters
    ----------
    psi_prev : float
        The solution at current_position - h.
    psi_curr : float
        The solution at current_position.
    k_prev : float
        The k evaluated at current_position - h.
    k_curr : float
        The k evaluated at current_position.
    k_next : float
        The k evaluated at current_position + h.
    h : float
        The integration step.

    Returns
    -------
    step : float
        Solution at the next point in the mesh.

    """
    step = (2 * (1 - (5 * h**2 * k_curr / 12)) * psi_curr - (1 + (h**2 * k_prev / 12)) * psi_prev) / (1 + (h**2 * k_next / 12))
    return step

def _integrate_numerov(E, l, potential, psi_0, psi_1, grid = Grid(), solver = SolverOpts(), outward = True, **kwargs):
    """
    Perform a Numerov integration of the radial Schrödinger equation.

    Parameters
    ----------
    E : float
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
    kwargs : dict
        Additional arguments of the potential energy.

    Returns
    -------
    coord : ndarray
        Radial grid points.
    psi : ndarray
        Solution wavefunction.
    
    """
    r_min = grid.r_min
    r_max = grid.r_max
    h = grid.h if outward else -1*grid.h
    r = grid.r_min if outward else grid.r_max

    psi = []
    coord = []

    # append known points
    psi.append(psi_0)
    psi.append(psi_1)
    coord.append(r)
    coord.append(r + h)

    while (r < r_max and outward) or (r > r_min and not outward):
        k_prev = k_squared(r, l, E, potential, **kwargs)
        k_curr = k_squared(r + h, l, E, potential, **kwargs)
        k_next = k_squared(r + 2 * h, l, E, potential, **kwargs)
        temp = _numerov_step(psi_0, psi_1, k_prev, k_curr, k_next, h)
        psi.append(temp)
        if abs(temp) > solver.renorm_threshold:
            # normalize to prevent blow up (scale does not matter in the inward integration)
            psi_0 *= solver.renorm_factor
            psi_1 *= solver.renorm_factor
            psi = [x*solver.renorm_factor for x in psi]
            temp *= solver.renorm_factor
        psi_0 = psi_1
        psi_1 = temp
        r += h
        coord.append(r+h)
    
    if not outward:
        # reverse the lists containing the coordinates and values of the solution
        coord = coord[::-1]
        psi = psi[::-1]
    
    # numpize the output lists
    coord = np.array(coord)
    psi = np.array(psi)

    return (coord, psi)