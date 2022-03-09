import click

from .spherical_harmonics import SphericalHarmonics


@click.group()
def cli():
    pass


@cli.command()
@click.argument("name", type=str)
@click.option("-k", "--keys", type=str, multiple=True, default=[])
@click.option("-s", "--seeds", type=int, multiple=True, default=[])
@click.option("-w", "--working", type=str, default=".")
@click.option("--scale", type=int, default=1)
@click.option("--region", type=str, default=None)
def calculate(**kwargs):
    SphericalHarmonics.run_calculate(**kwargs)


@cli.command()
@click.argument("name", type=str)
@click.option("-k", "--keys", type=str, multiple=True, default=[])
@click.option("-s", "--seeds", type=int, multiple=True, default=[])
@click.option("-w", "--working", type=str, default=".")
@click.option("--region", type=str, default=None)
def compress(**kwargs):
    SphericalHarmonics.run_compress(**kwargs)


@cli.command()
@click.argument("name", type=str)
@click.option("-k", "--keys", type=str, multiple=True, default=[])
@click.option("-s", "--seeds", type=int, multiple=True, default=[])
@click.option("-w", "--working", type=str, default=".")
@click.option("--region", type=str, default=None)
def merge(**kwargs):
    SphericalHarmonics.run_merge(**kwargs)


if __name__ == "__main__":
    cli()
