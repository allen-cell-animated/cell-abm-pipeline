import click


class Context:
    def __init__(self, name, keys, working):
        self.name = name
        self.keys = keys
        self.working = working


@click.group(invoke_without_command=True)
@click.option(
    "-n",
    "--name",
    type=str,
    default="",
    help="Name of data set",
    show_default=True,
)
@click.option(
    "-k",
    "--keys",
    type=str,
    multiple=True,
    default=[],
    help="Key(s) for data set",
    show_default=True,
)
@click.option(
    "-w",
    "--working",
    type=str,
    default=".",
    help="Working location (local path or S3 bucket)",
    show_default=True,
)
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj = Context(**kwargs)


@cli.command()
@click.option(
    "-n",
    "--num-images",
    type=int,
    default=0,
    help="Number of images to download.",
    show_default=True,
)
@click.pass_obj
def download_images(obj, **kwargs):
    """Download images from Quilt package."""
    from .download_images import DownloadImages

    DownloadImages(obj).run(**kwargs)


@cli.command()
@click.option("-c", "--channels", type=int, multiple=True, default=[0])
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
    """Sample image into list of id and xyz coordinates."""
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
    "--select",
    type=int,
    multiple=True,
    default=None,
    help="Specific cell ids to select (default = None).",
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
    """Process samples with selected processing steps."""
    from .process_samples import ProcessSamples

    ProcessSamples(obj).run(**kwargs)


@cli.command()
@click.option("-i", "--iterations", type=int, default=10)
@click.option("-c", "--channels", type=int, multiple=True, default=[0])
@click.pass_obj
def create_voronoi(obj, **kwargs):
    """Create Voronoi tesselation from given starting image."""
    from .create_voronoi import CreateVoronoi

    CreateVoronoi(obj).run(**kwargs)


@cli.command()
@click.option(
    "-m",
    "--margins",
    nargs=3,
    type=int,
    default=[0, 0, 0],
    help="Margin size in x, y, and z directions (default = [0, 0, 0])",
)
@click.option("--region", type=str, default=None)
@click.option("--reference", type=str, default=None)
@click.pass_obj
def convert_arcade(obj, **kwargs):
    """Convert samples into ARCADE input formats."""
    from .convert_arcade import ConvertARCADE

    ConvertARCADE(obj).run(**kwargs)
