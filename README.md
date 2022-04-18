# Cell agent-based model pipeline

[![Build Status](https://github.com/allen-cell-animated/cell-abm-pipeline/actions/workflows/build.yml/badge.svg)](https://github.com/allen-cell-animated/cell-abm-pipeline/actions/workflows/build.yml)
[![Lint Status](https://github.com/allen-cell-animated/cell-abm-pipeline/actions/workflows/lint.yml/badge.svg)](https://github.com/allen-cell-animated/cell-abm-pipeline/actions/workflows/lint.yml)
[![Codecov](https://codecov.io/gh/allen-cell-animated/cell-abm-pipeline/branch/main/graph/badge.svg?token=1S5ZKVET7T)](https://codecov.io/gh/allen-cell-animated/cell-abm-pipeline)
[![Documentation](https://github.com/allen-cell-animated/cell-abm-pipeline/actions/workflows/documentation.yml/badge.svg)](https://allen-cell-animated.github.io/cell-abm-pipeline/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains useful modules for working with cell agent-based models.
Modules can be called via CLI or imported into a Python project.
Top-level subpackages include:

- `basic_metrics` includes modules for calculating and plotting basic simulation metrics
- `resource_usage` includes modules for quantifying resource usage (wall clock and object storage) for simulation sets
- `cell_shape` includes modules for calculating spherical harmonic coefficients and extracting cell shapes via dimensionality reduction
- `colony_dynamics` includes modules for calculating cell neighbors and quantifying colony dynamics from cell neighbor graphs
- `initial_conditions` includes modules for sampling from images and converting into input formats for various model frameworks

See relevant subpackage README for details.

The `utilities` subpackage contains various utility functions for saving and loading files (locally and on the cloud), plotting, and key management.

## Features

This repository uses the following tools for managing Python projects:

- [Poetry](https://python-poetry.org/) for packaging and dependency management
- [Tox](https://tox.readthedocs.io/en/latest/) for automated testing
- [Black](https://black.readthedocs.io/en/stable/) for code formatting
- [Pylint](https://www.pylint.org/) for linting

as well as GitHub Actions to automatically build, test, lint, and generate documentation.

## Installation

1. Clone the repo.
2. Initialize the repository with Poetry by running:

```bash
$ poetry init
```

3. Install dependencies.

```bash
$ poetry install
```

4. Activate the environment.

```bash
$ poetry shell
```

5. Run the CLI

## Commands

The `Makefile` include three commands for working with the project.

- `make clean` will clean all the build and testing files
- `make build` will run tests, lint, and type check your code (you can also just run `tox`)
- `make docs` will generate documentation
