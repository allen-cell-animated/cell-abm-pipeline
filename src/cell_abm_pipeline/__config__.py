import os
import re
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


def make_dotlist_from_config(config):
    container = OmegaConf.to_container(OmegaConf.structured(config))
    queue = list(container.items())
    dotlist = []

    while queue:
        key, value = queue.pop()

        if isinstance(value, dict):
            queue = queue + [(f"{key}.{subkey}", subvalue) for subkey, subvalue in value.items()]
        elif isinstance(value, list):
            dotlist.append(f"{key}=[{','.join([str(v) for v in value])}]")
        elif value is None:
            dotlist.append(f"{key}=null")
        else:
            dotlist.append(f"{key}={value}")

    return dotlist


def make_config_from_dotlist(module, args):
    context_config = generate_config(module.ContextConfig, "context", args)
    series_config = generate_config(module.SeriesConfig, "series", args)
    parameters_config = generate_config(module.ParametersConfig, "parameters", args)

    config = OmegaConf.create(
        {
            "context": context_config,
            "series": series_config,
            "parameters": parameters_config,
        }
    )

    return config


def make_config_from_yaml(module, args):
    config_store = ConfigStore.instance()

    config = make_dataclass(
        "Config",
        [
            ("defaults", list[Any], field(default_factory=lambda: defaults)),
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
    active = False
    config_lines = []
    list_entries = []

    for line in OmegaConf.to_yaml(config, resolve=True).split("\n"):
        match = re.findall(r"^[\s]{2,4}\- ([\dA-z\.\*\']+)$", line)

        if match and not active:
            active = True
            list_entries.append(match[0])
        elif not match and active:
            active = False
            config_lines.append("[" + ", ".join(list_entries) + "]")
            list_entries = []
            config_lines.append(line)
        elif match and active:
            list_entries.append(match[0])
        else:
            config_lines.append(line)

    config_display = "\n".join(config_lines)
    config_display = config_display.replace(":\n[", ": [")

    print(config_display)
