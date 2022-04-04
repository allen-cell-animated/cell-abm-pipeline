import click


class Context:
    def __init__(self, working):
        self.working = working


@click.group(invoke_without_command=True)
@click.option("-w", "--working", type=str, default=".")
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj = Context(**kwargs)


if __name__ == "__main__":
    cli()
