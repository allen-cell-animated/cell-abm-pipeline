from typing import List, Any
from dataclasses import dataclass, field, make_dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

defaults = [
    "_self_",
    {"context": MISSING},
    {"series": MISSING},
    {"parameters": MISSING},
]


@dataclass
class ContextConfig:
    name: str

    location: str


@dataclass
class SeriesConfig:
    name: str

    keys: list


@dataclass
class ParametersConfig:
    name: str


def initialize_configs(module):
    config_store = ConfigStore.instance()

    config = make_dataclass(
        "Config",
        [
            ("defaults", List[Any], field(default_factory=lambda: defaults)),
            ("deploy", bool, field(default=False)),
            ("module", str, MISSING),
            ("context", module.ContextConfig, MISSING),
            ("series", module.SeriesConfig, MISSING),
            ("parameters", module.ParametersConfig, MISSING),
        ],
    )

    config_store.store(name="config", node=config)
