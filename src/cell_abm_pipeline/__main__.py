import importlib
import os
import sys

from omegaconf import OmegaConf
from prefect.blocks.system import Secret
from prefect.deployments import Deployment

from cell_abm_pipeline.__config__ import (
    display_config,
    make_config_from_dotlist,
    make_config_from_yaml,
)


def main():
    if len(sys.argv) < 2:
        return

    if "--dryrun" in sys.argv:
        sys.argv.remove("--dryrun")
        dryrun = True
    else:
        dryrun = False

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

    run_flow(module, config)


def get_module(module_name):
    module_spec = importlib.util.find_spec(f"..flows.{module_name}", package=__name__)

    if module_spec is not None:
        module = importlib.import_module(f"..flows.{module_name}", package=__name__)
    else:
        response = input(f"Module {module_name} does not exist. Create template for module [y/n]? ")
        if response[0] == "y":
            create_flow_template(module_name)

        module = None

    return module


def create_flow_template(module_name):
    path = os.path.dirname(os.path.abspath(__file__))

    with open(f"{path}/__template__.py", "r", encoding="utf-8") as file:
        template = file.read()

    template = template.replace("name-of-flow", module_name.replace("_", "-"))

    with open(f"{path}/flows/{module_name}.py", "w", encoding="utf-8") as file:
        file.write(template)


def run_flow(module, config):
    context = OmegaConf.to_object(config.context)
    series = OmegaConf.to_object(config.series)
    parameters = OmegaConf.to_object(config.parameters)

    if config["deploy"]:
        deployment = Deployment.build_from_flow(
            flow=module.run_flow,
            name=series.name,
            parameters={
                "context": context,
                "series": series,
                "parameters": parameters,
            },
        )
        deployment.apply()
    else:
        module.run_flow(context, series, parameters)


if __name__ == "__main__":
    main()
