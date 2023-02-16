import os
from dataclasses import dataclass, field, make_dataclass
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

defaults = [
    "_self_",
    {"context": MISSING},
    {"series": MISSING},
    {"parameters": MISSING},
]


@dataclass
class ContextConfig:
    name: str


@dataclass
class SeriesConfig:
    name: str


@dataclass
class ParametersConfig:
    name: str


def make_config_from_dotlist(module, args):
    context_config = generate_config(module.ContextConfig, "context", args)
    series_config = generate_config(module.SeriesConfig, "series", args)
    parameters_config = generate_config(module.ParametersConfig, "parameters", args)

    config = OmegaConf.create(
        {
            "series": series_config,
            "context": context_config,
            "parameters": parameters_config,
            "deploy": bool("deploy=True" in args),
        }
    )

    return config


def make_config_from_yaml(module, args):
    config_store = ConfigStore.instance()

    config = make_dataclass(
        "Config",
        [
            ("defaults", list[Any], field(default_factory=lambda: defaults)),
            ("deploy", bool, field(default=False)),
            ("context", module.ContextConfig, MISSING),
            ("series", module.SeriesConfig, MISSING),
            ("parameters", module.ParametersConfig, MISSING),
        ],
    )

    config_store.store(name="config", node=config)

    config_dir = os.path.join(os.path.abspath(os.getcwd()), "configs")
    initialize_config_dir(config_dir, version_base=None)

    config = compose(config_name="config", overrides=args)

    return config


def generate_config(config_class, group, arguments):
    dotlist = [arg.replace(group, "", 1).strip(".") for arg in arguments if arg.startswith(group)]
    config = OmegaConf.structured(config_class)
    config.merge_with_dotlist(dotlist)
    return config


def display_config(config):
    print(OmegaConf.to_yaml(config, resolve=True))
