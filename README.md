# Physipy — General-Purpose Computational Physics Library

Physipy is a **general-purpose Python library for computational physics**.

Its purpose is to provide a reusable and extensible collection of numerical tools, algorithms, and utilities for solving a wide range of physics problems. Rather than being tied to a single simulation or project, the repository is intended to serve as a **shared computational toolkit** for common tasks in theoretical and computational physics.

Typical use cases include:

- numerical integration of ordinary differential equations
- bound-state problems in quantum mechanics
- scattering calculations
- WKB-based approximations
- eigenvalue problems
- exploratory numerical experiments in Jupyter notebooks

The project is designed to be **modular, lightweight, and easy to extend** as new methods and problems are added.

## Goals

The main goals of this repository are:

- provide a **general-purpose library** for computational physics problems
- keep reusable numerical code separate from exploratory notebooks
- make it easy to test, extend, and reuse algorithms across different projects
- build a compact but flexible toolkit for simulations and numerical analysis

## Repository Structure

```text
physipy/
├── src/
│   └── physipy/
│       ├── __init__.py
│       ├── bound_states.py
        ├── constants.py
│       ├── numerics.py
│       ├── potentials.py
        ├── scattering.py
│       └── wkb.py
├── notebooks/
│   └── *.ipynb
├── pyproject.toml
└── README.md