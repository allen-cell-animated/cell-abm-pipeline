import sys
import os
import importlib

import hydra
from omegaconf import OmegaConf
from prefect.deployments import Deployment

from cell_abm_pipeline.__config__ import initialize_configs


@hydra.main(version_base=None, config_name="config", config_path=None)
def run_flow(config):
    module = importlib.import_module(f"..flows.{config['module']}", package=__name__)

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


def main():
    if len(sys.argv) < 2:
        return

    module_name = sys.argv.pop(1).replace("-", "_")

    try:
        module = importlib.import_module(f"..flows.{module_name}", package=__name__)
    except ModuleNotFoundError as exception:
        print(f"Exception: {type(exception).__name__}")
        print(f"Exception message: {exception}")

        if module_name not in str(exception):
            return

        response = input("Create template for module [y/n]? ")
        if response[0] == "y":
            path = os.path.dirname(os.path.abspath(__file__))

            with open(f"{path}/__template__.py", "r", encoding="utf-8") as file:
                template = file.read()

            template = template.replace("name-of-flow", module_name.replace("_", "-"))

            with open(f"{path}/flows/{module_name}.py", "w", encoding="utf-8") as file:
                file.write(template)

        return

    sys.argv[1:1] = [
        "--config-dir",
        "configs/",
        "hydra.output_subdir=null",
        "hydra.run.dir=.",
        "hydra/job_logging=none",
        "hydra/hydra_logging=none",
        f"module={module_name}",
    ]

    initialize_configs(module)
    run_flow()


if __name__ == "__main__":
    main()
