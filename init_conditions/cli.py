import click
from download_images import DownloadImages
from sample_images import SampleImages
from process_samples import ProcessSamples


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-n",
    "--num-images",
    type=int,
    default=0,
    help="number of FOV images to download",
)
@click.option(
    "-p",
    "--output-path",
    type=click.Path(exists=True),
    default=".",
    help="path to directory to save outputs (default = current directory)",
)
def download_images(**kwargs):
    """
    Download segmentation data and save XY projections.
    """
    DownloadImages().download(**kwargs)


@cli.command()
@click.option(
    "-p",
    "--output-path",
    type=click.Path(exists=True),
    default=".",
    help="path to directory to save outputs (default = current directory)",
)
@click.option(
    "-g",
    "--grid-type",
    type=click.Choice(["hex", "cartesian"], case_sensitive=False),
    default="hex",
    help="sampling grid type (default = hex)",
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
def sample_images(**kwargs):
    """
    Sample downloaded image files for given grid type.
    """
    SampleImages().sample(**kwargs)


@cli.command()
@click.argument("samples", type=click.Path(exists=True))
@click.option(
    "--edges/--no-edges",
    default=True,
    help="True if cells touching edges are removed, False otherwise (default = True)",
)
@click.option(
    "--connected/--no-connected",
    default=True,
    help="True if unconnected voxels are removed, False otherwise (default = True)",
)
@click.option(
    "--scale/--no-scale",
    default=True,
    help="True if coordinates are scaled, False otherwise (default = True)",
)
@click.option(
    "-s",
    "--scale-factor",
    default=1.0,
    help="Scaling factor for coordinates (default = 1.0).",
)
def process_samples(samples, **kwargs):
    """
    Process SAMPLES file with selected post-processing.
    """
    ProcessSamples(samples).process(**kwargs)


def convert_samples():
    # TODO: convert samples
    pass


if __name__ == "__main__":
    cli()
