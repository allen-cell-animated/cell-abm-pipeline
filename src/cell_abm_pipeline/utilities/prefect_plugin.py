from mypy.plugin import Plugin


class CustomPlugin(Plugin):
    def get_method_hook(self, fullname: str):
        # Replace the default None type of Task with the return type of
        # underlying method call
        if fullname == "prefect.tasks.Task.__call__":
            return lambda ctx: ctx.type.args[1]

        return None


def plugin(version: str):
    return CustomPlugin
