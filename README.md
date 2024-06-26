# Cell ABM Pipeline

[![Build status](https://allen-cell-animated.github.io/cell-abm-pipeline/_badges/build.svg)](https://github.com/allen-cell-animated/cell-abm-pipeline/actions?query=workflow%3Abuild)
[![Lint status](https://allen-cell-animated.github.io/cell-abm-pipeline/_badges/lint.svg)](https://github.com/allen-cell-animated/cell-abm-pipeline/actions?query=workflow%3Alint)
[![Documentation](https://allen-cell-animated.github.io/cell-abm-pipeline/_badges/documentation.svg)](https://allen-cell-animated.github.io/cell-abm-pipeline/)
[![Coverage](https://allen-cell-animated.github.io/cell-abm-pipeline/_badges/coverage.svg)](https://allen-cell-animated.github.io/cell-abm-pipeline/_coverage/)
[![Code style](https://allen-cell-animated.github.io/cell-abm-pipeline/_badges/style.svg)](https://github.com/psf/black)
[![License](https://allen-cell-animated.github.io/cell-abm-pipeline/_badges/license.svg)](https://github.com/allen-cell-animated/cell-abm-pipeline/blob/main/LICENSE)

## Installation

### Installation using Poetry

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and virtual environments.

1. Create the virtual environment:

```bash
poetry install
```

2. Activate the environment:

```bash
poetry shell
```

### Alternative installation using `pip`

This project also includes a `requirements.txt` generated from the `poetry.lock` file.
Install dependencies directly from this file using:

```bash
pip install -r requirements.txt
```

Install the package (note that you need pip ≥ 21.3):

```bash
pip install -e .
```

## Usage

The pipeline uses [Prefect](https://docs.prefect.io/) for workflows and [Hydra](https://hydra.cc/docs/intro/) for composable configuration.
Workflows can be run via CLI or through the Prefect UI as deployments.

When running via CLI, configurations can be passed in three ways: inline, using a single config file, or using composable config files.
Note that Hydra supports additional overriding configurations via CLI for all three options.

### Run with inline configs

Configurations can be passed directly using:

```bash
abmpipe demo :: parameters.name=demo_parameters context.name=demo_context series.name=demo_series
```

### Run using single config file

Create a config file `demo.yaml` with the following contents:

```yaml
context:
  name: demo_context
series:
  name: demo_series
parameters:
  name: demo_parameters
```

Then use:

```bash
abmpipe demo /path/to/demo.yaml
```

### Run using composable config files

Create a `configs` directory with the following structure:

```bash
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
abmpipe demo parameters=demo context=demo series=demo
```

### Additional flags

Use the flag `--dryrun` to display the composed configuration without running the workflow.

Use the flag `--deploy` to create a Prefect deployment.

### Adding secrets

Configs can use [Secret](https://docs.prefect.io/concepts/blocks/?h=secret#secret-fields) fields.
In configs, any field in the form `${secret:name-of-secret}` will be resolved using the Prefect Secret loader.
These values must be configured as a Secret Block in Prefect via a script:

```python
from prefect.blocks.system import Secret

Secret(value="secret-value").save(name="name-of-secret")
```

or in the Prefect UI under Blocks.

## Development

New flows can be added to the `flows` module with following structure:

```python
from dataclasses import dataclass

from prefect import flow


@dataclass
class ParametersConfig:
    # TODO: add parameter config


@dataclass
class ContextConfig:
    # TODO: add context config


@dataclass
class SeriesConfig:
    # TODO: add series config


@flow(name="name-of-flow")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    # TODO: add flow
```

The command:

```bash
abmpipe name-of-flow
```

will create a new flow template under the `flows` module with the name `name_of_flow`.

### Using notebooks

Notebooks can be helpful for prototyping flows.

#### Use dataclasses to specify configuration

Create dataclasses for all relevant configuration for the flow.
Specify types and default values, if relevant.
For flows in this repo, three types of configs are used:

- `ParametersConfig` specifies all parameters for the flow
- `ContextConfig` specifies the infrastructure context (e.g. local working path or S3 bucket names)
- `SeriesConfig` specifies the simulation series the flow is applied to (e.g. simulation name, conditions, seeds)

#### Load configuration into dataclasses

Configurations can be loaded in multiple ways.

1. Load entire configuration directly from an existing configuration file using the `make_config_from_file` function.
   _Works best for simple configurations without interpolation_.

```python
config = make_config_from_file(ConfigDataclass, f"/path/to/config.yaml")
```

2. Load partial configuration directly from an existing configuration file using the `make_config_from_file` function.
   Missing fields in the config can be loaded from other configuration files using `OmegaConf.load` or set directly.
   _Works best for configurations that use interpolation_.

```python
config = make_config_from_file(ConfigDataclass, f"/path/to/config.yaml")
config.field = OmegaConf.load(f"/path/to/another/config.yaml").field
config.field = "value"
```

3. Directly instantiate the config object.
   Fields in object initialization can also be loaded using `OmegaConf.load`.
   _Works best for custom configurations or testing configurations_.

```python
config = ConfigDataclass(
    field="value",
    field=OmegaConf.load(f"/path/to/config.yaml").field,
    ...
)
```

#### Call tasks from collections

Import tasks from collections in the undecorated form:

```python
from collection.module.task import task
```

Tasks can also be imported in decorated form:

```python
from collection.module import task
```

but will need to called using `task.fn()` because we are not in a Prefect flow environment.

#### Converting notebooks to flows

Make sure the main flow method has the `@flow` decorator and imports should be switched to their `@task` decorated form to take advantage of Prefect task and flow monitoring.
