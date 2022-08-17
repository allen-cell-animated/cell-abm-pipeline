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
@click.option(
    "-g",
    "--grid",
    type=click.Choice(["rect", "hex"], case_sensitive=False),
    default="rect",
    help="Type of sampling grid.",
    show_default=True,
)
@click.option(
    "--ds",
    type=float,
    default=1.0,
    help="Distance between elements in um.",
    show_default=True,
)
@click.option(
    "--box",
    nargs=3,
    type=int,
    default=(100, 100, 10),
    help="Bounding box size in um.",
    show_default=True,
)
@click.option(
    "--contact/--no-contact",
    default=True,
    help="True if contact sheet of images is saved, False otherwise. [default: True]",
)
@click.pass_obj
def generate_coordinates(obj, **kwargs):
    """Generate cell ids and coordinates."""
    from .generate_coordinates import GenerateCoordinates

    GenerateCoordinates(obj).run(**kwargs)


@cli.command()
@click.option(
    "-g",
    "--grid",
    type=click.Choice(["rect", "hex"], case_sensitive=False),
    default="rect",
    help="Type of sampling grid.",
    show_default=True,
)
@click.option(
    "-r",
    "--resolution",
    type=float,
    default=1.0,
    help="Distance between samples (um).",
    show_default=True,
)
@click.option(
    "-c",
    "--channels",
    type=int,
    multiple=True,
    default=[0],
    help="Image channel indices.",
    show_default=True,
)
@click.option(
    "--contact/--no-contact",
    default=True,
    help="True if contact sheet of images is saved, False otherwise.  [default: True]",
)
@click.pass_obj
def sample_images(obj, **kwargs):
    """Sample cell ids and coordinates from images."""
    from .sample_images import SampleImages

    SampleImages(obj).run(**kwargs)


@cli.command()
@click.option(
    "-g",
    "--grid",
    type=click.Choice(["rect", "hex"], case_sensitive=False),
    default="rect",
    help="Type of sampling grid.",
    show_default=True,
)
@click.option(
    "--scale",
    type=float,
    default=None,
    help="Coordinate scaling factor.",
    show_default=True,
)
@click.option(
    "--include",
    type=int,
    multiple=True,
    default=None,
    help="Specific cell ids to include.",
    show_default=True,
)
@click.option(
    "--exclude",
    type=int,
    multiple=True,
    default=None,
    help="Specific cell ids to exclude.",
    show_default=True,
)
@click.option(
    "--edges/--no-edges",
    default=False,
    help="True if cells touching edges are removed, False otherwise.  [default: False]",
)
@click.option(
    "--connected/--no-connected",
    default=False,
    help="True if unconnected voxels are removed, False otherwise.  [default: False]",
)
@click.option(
    "--contact/--no-contact",
    default=True,
    help="True if contact sheet of images is saved, False otherwise.  [default: True]",
)
@click.pass_obj
def process_samples(obj, **kwargs):
    """Process samples with selected processing steps."""
    from .process_samples import ProcessSamples

    ProcessSamples(obj).run(**kwargs)


@cli.command()
@click.option(
    "-i",
    "--iterations",
    type=int,
    default=2,
    help="Number of boundary estimation steps.",
    show_default=True,
)
@click.option(
    "-c",
    "--channels",
    type=int,
    multiple=True,
    default=[0],
    help="Image channel indices.",
    show_default=True,
)
@click.option(
    "-h",
    "--height",
    type=int,
    default=10,
    help="Target height for tesselation.",
    show_default=True,
)
@click.pass_obj
def create_voronoi(obj, **kwargs):
    """Create Voronoi tessellation from starting image."""
    from .create_voronoi import CreateVoronoi

    CreateVoronoi(obj).run(**kwargs)


@cli.command()
@click.option(
    "-m",
    "--margins",
    nargs=3,
    type=int,
    default=(0, 0, 0),
    help="Margin size in x, y, and z directions.",
    show_default=True,
)
@click.option(
    "--region",
    type=str,
    default=None,
    help="Region key to include in conversion.",
    show_default=True,
)
@click.option(
    "--reference",
    type=str,
    default=None,
    help="Path to reference data for conversion.",
    show_default=True,
)
@click.pass_obj
def convert_arcade(obj, **kwargs):
    """Convert samples into ARCADE input formats."""
    from .convert_arcade import ConvertARCADE

    ConvertARCADE(obj).run(**kwargs)


if __name__ == "__main__":
    cli()
