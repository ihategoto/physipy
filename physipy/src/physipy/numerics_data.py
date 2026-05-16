from dataclasses import dataclass
import numpy as np

__all__ = [
    "Grid",
    "NumericalOpts",
    "SolverOpts",
    "Eigenstate"
]

@dataclass(frozen = True)
class Grid:
    """
    Immutable container for radial grid parameters.

    Attributes
    ----------
    r_min : float
        Starting point of the grid. Must be > 0 (r=0 excluded for numerical stability).
    r_max : float
        Ending point of the grid. Must be > r_min.
    h : float
        Grid step size. Must be > 0.
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
class NumericalOpts:
    """
    Immutable container for basic numerical stability parameters.

    Attributes
    ----------
    tol : float
        Values smaller than this are clamped up to it (guards against division by zero).
    threshold : float
        Values larger than this are clamped down to it (guards against overflow).
    """
    tol: float = 1e-6
    threshold: float = 1e20

@dataclass(frozen = True)
class SolverOpts(NumericalOpts):
    """
    Immutable container for Schrödinger solver parameters. Inherits from NumericalOpts.

    Attributes
    ----------
    bisection_tol : float
        Convergence tolerance on the energy interval for the bisection algorithm.
    match_buffer_steps : int
        Number of grid steps between the classical turning point and the
        outward/inward matching point.
    renorm_threshold : float
        Wavefunction amplitude above which in-flight renormalization is applied.
    renorm_factor : float
        Rescaling factor applied when the renorm_threshold is exceeded.
    """
    bisection_tol: float = 1e-6
    match_buffer_steps: int = 10
    renorm_threshold: float = 1e6
    renorm_factor: float = 1e-6

@dataclass(frozen = True)
class Eigenstate:
    """
    Immutable container for a bound-state eigenstate.

    Attributes
    ----------
    E : float
        Eigenvalue (energy) of the state.
    l : int
        Angular momentum quantum number.
    coord : ndarray
        Radial grid points on which psi is defined.
    psi : ndarray
        Radial wavefunction values (not necessarily normalised).
    """
    E: float
    l: int
    coord: np.ndarray
    psi: np.ndarray

class ConvergenceError(RuntimeError):
    def __init__(self, max_iter, delta_E):
        super().__init__(
            f"Self-consistent loop did not converge after {max_iter} iterations. "
            f"Last energy difference: {delta_E:.2e}"
        )
        self.max_iter = max_iter
        self.delta_E = delta_E