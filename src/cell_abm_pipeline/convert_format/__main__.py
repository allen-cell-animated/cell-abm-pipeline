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
@click.option("--box", nargs=3, type=int, default=(100, 100, 10))
@click.pass_obj
def arcade_to_image(obj, **kwargs):
    from .arcade_to_image import ArcadeToImage

    ArcadeToImage(obj).run(**kwargs)


@cli.command()
@click.option("--frames", nargs=3, type=int, default=(0, 1, 1))
@click.pass_obj
def arcade_to_mesh(obj, **kwargs):
    from .arcade_to_mesh import ArcadeToMesh

    ArcadeToMesh(obj).run(**kwargs)


if __name__ == "__main__":
    cli()
