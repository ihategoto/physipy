
# Computational Physics Projects

This repository hosts templates and shared resources for computational physics projects, numerical experiments, and reproducible research.

## Purpose

- Provide a consistent project layout and contribution workflow for physics simulations, numerical analysis, and data-driven experiments.

## Repository Structure

Recommended layout for each project inside this repository:

```
projects/
	└─ <project-name>/
		 ├─ notebooks/        # Jupyter notebooks for exploration and demos
		 ├─ src/              # Production-ready code and modules
		 ├─ data/             # Raw and processed datasets (gitignored large files)
		 ├─ results/          # Output figures, tables, and serialized results
		 ├─ env/              # environment files: requirements.txt, environment.yml
		 └─ README.md         # project-specific README with usage + citations
scripts/                 # helper scripts for setup, analysis, CI
docs/                    # higher-level documentation and tutorials
LICENSE
README.md
```

## Adding a New Project

1. Create `projects/<project-name>` with the structure above.
2. Add an environment spec: `requirements.txt` or `environment.yml`.
3. Include a short `README.md` describing goals, run instructions, and references.

Example Python virtualenv workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate <env-name>
```

## Coding Conventions

- Prefer clear, well-tested modules under `src/` and use notebooks only for exploration.
- Write tests in a `tests/` folder at the project root and use pytest.
- Use consistent formatting: `black` for Python, `isort` for imports.

## Data & Large Files

- Do not commit large binary datasets. Add them to `.gitignore` and provide download scripts under `scripts/`.
- Record dataset provenance and citations in the project's `README.md`.

## Reproducibility

- Pin package versions in `requirements.txt` or `environment.yml`.
- Save random seeds and document hardware/OS for numerical experiments.
- Store key results and figure-generation code in `results/` with scripts that reproduce figures.

## Collaboration & Contributions

- Open an issue to propose a new project or major change.
- Use feature branches and pull requests with descriptive titles.
- Include a brief changelog in each project's `README.md`.

## License

This repository uses the MIT License by default. Add or change a license per-project as needed.
