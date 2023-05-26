[![Build Status](https://github.com/allen-cell-animated/cell-abm-pipeline/workflows/build/badge.svg)](https://github.com/allen-cell-animated/cell-abm-pipeline/actions?query=workflow%3Abuild)
[![Codecov](https://img.shields.io/codecov/c/gh/allen-cell-animated/cell-abm-pipeline?token=1S5ZKVET7T)](https://codecov.io/gh/allen-cell-animated/cell-abm-pipeline)
[![Lint Status](https://github.com/allen-cell-animated/cell-abm-pipeline/workflows/lint/badge.svg)](https://github.com/allen-cell-animated/cell-abm-pipeline/actions?query=workflow%3Alint)
[![Documentation](https://github.com/allen-cell-animated/cell-abm-pipeline/workflows/documentation/badge.svg)](https://allen-cell-animated.github.io/cell-abm-pipeline/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Installation

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and virtual environments.

1. Create the virtual environment:

```bash
$ poetry install
```

2. Activate the environment:

```bash
$ poetry shell
```

## Alternative installation using `pip`

This project also includes a `requirements.txt` generated from the `poetry.lock` file.
Install dependencies directly from this file using:

```bash
pip install -r requirements.txt
```

Install the package (note that you need pip ≥ 21.3):

```bash
pip install -e .
```

# Usage

The pipeline uses [Prefect](https://docs.prefect.io/) for workflows and [Hydra](https://hydra.cc/docs/intro/) for composable configuration.

Run the demo workflow using:

```bash
$ abmpipe demo :: parameters.name=demo context.name=demo series.name=demo
```

To use config files, create a `configs` directory with the following structure:

```
configs
├── context
│   └── demo.yaml
├── parameters
│   └── demo.yaml
└── series
    └── demo.yaml
```

Each `demo.yaml` should contain the field `name: <name>`.
Then use:

```bash
$ abmpipe demo parameters=demo context=demo series=demo
```

Use the flag `--dryrun` to display the composed configuration without running the workflow.

## Adding secrets

Configs can use [Secret](https://docs.prefect.io/concepts/blocks/?h=secret#secret-fields) fields.
In configs, any field in the form `${secret:name-of-secret}` will be resolved using the Prefect Secret loader.
These values must be configured as a Secret Block in Prefect via a script:

```python
from prefect.blocks.system import Secret

Secret(value="secret-value").save(name="name-of-secret")
```

or in the Prefect UI under Blocks.
