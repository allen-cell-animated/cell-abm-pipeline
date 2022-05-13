import click


class Context:
    def __init__(self, name, keys, seeds, working):
        self.name = name
        self.keys = keys
        self.seeds = seeds
        self.working = working


@click.group(invoke_without_command=True)
@click.option("-n", "--name", type=str, default="")
@click.option("-k", "--keys", type=str, multiple=True, default=[""])
@click.option("-s", "--seeds", type=int, multiple=True, default=[])
@click.option("-w", "--working", type=str, default=".")
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj = Context(**kwargs)


@cli.command()
@click.option("--frames", type=int, multiple=True, default=[0])
@click.option("--scale", type=int, default=1)
@click.option("--region", type=str, default=None)
@click.pass_obj
def calculate_coefficients(obj, **kwargs):
    from .calculate_coefficients import CalculateCoefficients

    CalculateCoefficients(obj).run(**kwargs)


@cli.command()
@click.option("--region", type=str, default=None)
@click.pass_obj
def compress_coefficients(obj, **kwargs):
    from .compress_coefficients import CompressCoefficients

    CompressCoefficients(obj).run(**kwargs)


@cli.command()
@click.option("--region", type=str, default=None)
@click.pass_obj
def merge_coefficients(obj, **kwargs):
    from .merge_coefficients import MergeCoefficients

    MergeCoefficients(obj).run(**kwargs)


@cli.command()
@click.option("--region", type=str, default=None)
@click.pass_obj
def perform_pca(obj, **kwargs):
    from .perform_pca import PerformPCA

    PerformPCA(obj).run(**kwargs)


@cli.command()
@click.option("--delta", type=float, default=1.0)
@click.option("--box", nargs=2, type=int, default=(100, 100))
@click.option("--scale", type=float, default=1.0)
@click.option("--region", type=str, default=None)
@click.pass_obj
def extract_shapes(obj, **kwargs):
    from .extract_shapes import ExtractShapes

    ExtractShapes(obj).run(**kwargs)


@cli.command()
@click.option("--features", type=str, multiple=True, default=[])
@click.option("--components", type=int, multiple=True, default=[])
@click.option("--region", type=str, default=None)
@click.option("--reference", type=str, default=None)
@click.pass_obj
def plot_pca(obj, **kwargs):
    from .plot_pca import PlotPCA

    PlotPCA(obj).run(**kwargs)


if __name__ == "__main__":
    cli()
