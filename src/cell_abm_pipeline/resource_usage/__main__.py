import click


class Context:
    def __init__(self, name, working):
        self.name = name
        self.working = working


@click.group(invoke_without_command=True)
@click.option("-n", "--name", type=str, default="")
@click.option("-w", "--working", type=str, default=".")
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj = Context(**kwargs)


@cli.command()
@click.pass_obj
def extract_clock(obj, **kwargs):
    from .extract_clock import ExtractClock

    ExtractClock(obj).run(**kwargs)


@cli.command()
@click.pass_obj
def calculate_storage(obj, **kwargs):
    from .calculate_storage import CalculateStorage

    CalculateStorage(obj).run(**kwargs)


if __name__ == "__main__":
    cli()
