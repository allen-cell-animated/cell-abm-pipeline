import click


class Context:
    def __init__(self, name, keys, working):
        self.name = name
        self.keys = keys
        self.working = working


@click.group(invoke_without_command=True)
@click.option("-n", "--name", type=str, default="")
@click.option("-k", "--keys", type=str, multiple=True, default=[])
@click.option("-w", "--working", type=str, default=".")
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj = Context(**kwargs)


@cli.command()
@click.option(
    "-g",
    "--grid",
    type=click.Choice(["rect", "hex"], case_sensitive=False),
    default="rect",
    help="sampling grid type (default = rect)",
)
@click.option(
    "-r",
    "--resolution",
    type=float,
    default=1.0,
    help="microns between samples (default = 1.0)",
)
@click.option(
    "--contact/--no-contact",
    default=True,
    help="True if contact sheet of images is saved, False otherwise (default = True)",
)
@click.pass_obj
def sample_images(obj, **kwargs):
    from .sample_images import SampleImages

    SampleImages(obj).run(**kwargs)


@cli.command()
@click.option(
    "-g",
    "--grid",
    type=click.Choice(["rect", "hex"], case_sensitive=False),
    default="rect",
    help="sampling grid type (default = rect)",
)
@click.option(
    "--scale",
    type=float,
    default=None,
    help="Scaling factor for coordinates (default = None).",
)
@click.option(
    "--edges/--no-edges",
    default=False,
    help="True if cells touching edges are removed, False otherwise (default = False)",
)
@click.option(
    "--connected/--no-connected",
    default=False,
    help="True if unconnected voxels are removed, False otherwise (default = False)",
)
@click.option(
    "--contact/--no-contact",
    default=True,
    help="True if contact sheet of images is saved, False otherwise (default = True)",
)
@click.pass_obj
def process_samples(obj, **kwargs):
    from .process_samples import ProcessSamples

    ProcessSamples(obj).run(**kwargs)
