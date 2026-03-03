# Quantum harmonic oscillator

Project goal
--------------

Solve the stationary 3D Schrödinger equation for the isotropic quantum harmonic oscillator using numerical methods. The focus is on implementing and comparing the Numerov integration method with a root-finding (bisection) approach to obtain eigenfunctions and eigenenergies.

Methods
-------

- Numerov method: integrate the radial Schrödinger equation with a suitable transformation to handle the centrifugal term and obtain wavefunction shapes.
- Bisection method: search for energy eigenvalues by matching boundary conditions (node counting or sign changes) and bracketing roots.

Repository layout
-----------------

Recommended files for this project:

- `notebooks/` — interactive exploration and plotting of eigenfunctions/eigenvalues.
- `src/` — implementation modules (Numerov integrator, bisection solver, utilities).
- `data/` — any input data or parameter sets (gitignored for large files).
- `results/` — generated figures and numeric tables.
- `env/requirements.txt` — environment dependencies.
- `README.md` — this file; documents goals and usage.

References
----------

- Numerical Recipes (Numerov method, eigenvalue shooting)
- Standard quantum mechanics texts for the analytic QHO solutions (for verification)

Contributing
------------

Open issues for feature requests or improvements, and submit PRs for new solver options, tests, or visualizations.
