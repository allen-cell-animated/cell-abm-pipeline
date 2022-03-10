import click

from .calculate_coefficients import CalculateCoefficients
from .compress_coefficients import CompressCoefficients
from .merge_coefficients import MergeCoefficients


class Context:
    def __init__(self, name, keys, seeds, working):
        self.name = name
        self.keys = keys
        self.seeds = seeds
        self.working = working


@click.group(invoke_without_command=True)
@click.option("-n", "--name", type=str, default="")
@click.option("-k", "--keys", type=str, multiple=True, default=[])
@click.option("-s", "--seeds", type=int, multiple=True, default=[])
@click.option("-w", "--working", type=str, default=".")
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj = Context(**kwargs)


@cli.command()
@click.option("--scale", type=int, default=1)
@click.option("--region", type=str, default=None)
@click.pass_obj
def calculate_coefficients(obj, **kwargs):
    CalculateCoefficients(obj).run(**kwargs)


@cli.command()
@click.option("--region", type=str, default=None)
@click.pass_obj
def compress_coefficients(obj, **kwargs):
    CompressCoefficients(obj).run(**kwargs)


@cli.command()
@click.option("--region", type=str, default=None)
@click.pass_obj
def merge_coefficients(obj, **kwargs):
    MergeCoefficients(obj).run(**kwargs)


if __name__ == "__main__":
    cli()
