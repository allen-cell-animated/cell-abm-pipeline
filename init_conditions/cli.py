import click
from download_images import DownloadImages
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
    Download 3D segmentation data for FOVs and save XY projections as PNGs.
    """
    DownloadImages().download(**kwargs)


def sample_images():
    # TODO: add sample images
    pass


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
    Process SAMPLES file with selected post-processing steps.
    """
    ProcessSamples(samples).process(**kwargs)


def convert_samples():
    # TODO: convert samples
    pass


if __name__ == "__main__":
    cli()
