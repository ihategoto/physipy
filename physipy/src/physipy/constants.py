import numpy as np

__all__ = [
    "hbar",
    "c",
    "e",
    "alpha"
]

# Default convention : natural units with ħ = c = 1 and Heaviside–Lorentz electromagnetism

# Reduced Planck constant
hbar = 1

# Speed of light
c = 1

# Fine structure constant
alpha = 7.2973525693e-3

# Electron charge (Heaviside–Lorentz units)
e = np.sqrt(4 * np.pi * alpha)