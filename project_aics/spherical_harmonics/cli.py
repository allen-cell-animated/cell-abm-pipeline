import click
import json

from .spherical_harmonics import SphericalHarmonics


@click.group()
def cli():
    pass


@cli.command()
@click.option("--name", type=str, default="")
@click.option("--keys", type=str, multiple=True, default=[])
@click.option("--seeds", type=int, multiple=True, default=[])
@click.option("--path", type=str, default=".")
@click.option("--scale", type=int, default=1)
def calculate(**kwargs):
    SphericalHarmonics.run_calculate(**kwargs)


if __name__ == "__main__":
    cli()
