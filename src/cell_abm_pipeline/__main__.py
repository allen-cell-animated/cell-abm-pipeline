import hashlib
import importlib
import os
import sys
from datetime import datetime
from types import ModuleType
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from prefect.blocks.system import Secret
from prefect.deployments import Deployment

from cell_abm_pipeline.__config__ import (
    display_config,
    make_config_from_dotlist,
    make_config_from_yaml,
)


def main() -> None:
    if len(sys.argv) < 2:
        return

    if "--dryrun" in sys.argv:
        sys.argv.remove("--dryrun")
        dryrun = True
    else:
        dryrun = False

    if "--deploy" in sys.argv:
        sys.argv.remove("--deploy")
        deploy = True
    else:
        deploy = False

    OmegaConf.register_new_resolver("secret", lambda secret: Secret.load(secret).get())
    OmegaConf.register_new_resolver("concat", lambda items: ":".join(sorted(items)))
    OmegaConf.register_new_resolver(
        "home", lambda path: os.path.join(os.path.expanduser("~"), path)
    )

    module_name = sys.argv[1].replace("-", "_")
    module = get_module(module_name)

    if module is None:
        return

    if len(sys.argv) > 2 and sys.argv[2] == "::":
        config = make_config_from_dotlist(module, sys.argv[3:])
    else:
        config = make_config_from_yaml(module, sys.argv[2:])

    display_config(config)

    if dryrun:
        return

    if deploy:
        create_deployment(module, config)
    else:
        run_flow(module, config)


def get_module(module_name: str) -> Optional[ModuleType]:
    module_spec = importlib.util.find_spec(f"..flows.{module_name}", package=__name__)

    if module_spec is not None:
        module = importlib.import_module(f"..flows.{module_name}", package=__name__)
    else:
        response = input(f"Module {module_name} does not exist. Create template for module [y/n]? ")
        if response[0] == "y":
            create_flow_template(module_name)

        module = None

    return module


def create_flow_template(module_name: str) -> None:
    path = os.path.dirname(os.path.abspath(__file__))

    with open(f"{path}/__template__.py", "r", encoding="utf-8") as file:
        template = file.read()

    template = template.replace("name-of-flow", module_name.replace("_", "-"))

    with open(f"{path}/flows/{module_name}.py", "w", encoding="utf-8") as file:
        file.write(template)


def run_flow(module: ModuleType, config: DictConfig) -> None:
    context = OmegaConf.to_object(config.context)
    series = OmegaConf.to_object(config.series)
    parameters = OmegaConf.to_object(config.parameters)

    module.run_flow(context, series, parameters)


def create_deployment(module: ModuleType, config: DictConfig) -> None:
    context = OmegaConf.to_object(config.context)
    series = OmegaConf.to_object(config.series)
    parameters = OmegaConf.to_object(config.parameters)

    assert isinstance(context, dict)
    assert isinstance(series, dict)
    assert isinstance(parameters, dict)

    flow_name = module.__name__.split(".")[-1].replace("_", "-")

    name = input("Deployment name: ")
    name = name.replace("{{timestamp}}", datetime.now().strftime("%Y-%m-%d"))
    name = name.replace("{{name}}", series["name"])

    work_queue_name = input("Deployment queue (default if None): ")
    work_queue_name = "default" if work_queue_name == "" else work_queue_name

    infra_overrides = {}

    if hasattr(context, "region"):
        infra_overrides = {"env": {"AWS_DEFAULT_REGION": context.region}}

    deployment = Deployment(name=name, flow_name=flow_name)
    checksum = hashlib.md5(OmegaConf.to_yaml(config, resolve=True).encode("utf-8")).hexdigest()

    full_name = f"\033[1m{flow_name}/{name}\033[0m"

    if deployment.load() and deployment.version != checksum:
        response = input(f"Deployment {full_name} already exists. Overwrite [y/n]? ")
        if response[0] != "y":
            return

        deployment.update(
            version=checksum,
            parameters={
                "context": context,
                "series": series,
                "parameters": parameters,
            },
            work_queue_name=work_queue_name,
            infra_overrides=infra_overrides,
        )
        deployment.apply()

        print(f"Deployment {full_name} updated.")
    elif deployment.version != checksum:
        deployment = Deployment.build_from_flow(
            flow=module.run_flow,
            name=name,
            version=checksum,
            parameters={
                "context": context,
                "series": series,
                "parameters": parameters,
            },
            work_queue_name=work_queue_name,
            infra_overrides=infra_overrides,
            apply=True,
        )

        print(f"Deployment {full_name} created.")
    elif deployment.work_queue_name != work_queue_name:
        response = input(f"Update {full_name} queue to \033[92m{ work_queue_name }\033[0m [y/n]? ")
        if response[0] != "y":
            return

        deployment.update(work_queue_name=work_queue_name)
        deployment.apply()

        print(f"Deployment {full_name} queue updated.")
    else:
        print(f"Deployment {full_name} with same configuration already exists.")


if __name__ == "__main__":
    main()
