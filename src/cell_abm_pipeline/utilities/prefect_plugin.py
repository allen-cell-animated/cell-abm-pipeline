from mypy.plugin import Plugin
from mypy.types import CallableType


class CustomPlugin(Plugin):
    def get_method_hook(self, fullname: str):
        if fullname == "prefect.tasks.Task.__call__":
            return update_prefect_task_call_return

        return None

    def get_method_signature_hook(self, fullname: str):
        if fullname == "prefect.tasks.Task.submit":
            return update_prefect_task_submit_signature

        return None


def update_prefect_task_call_return(ctx):
    # Replace default None return of Task with return type of underlying method.
    return ctx.type.args[1]


def update_prefect_task_submit_signature(ctx):
    # Drop the return_state argument for submit task calls.
    if "return_state" in ctx.default_signature.arg_names:
        index = ctx.default_signature.arg_names.index("return_state")
        return CallableType(
            ctx.default_signature.arg_types[:index] + ctx.default_signature.arg_types[index + 1 :],
            ctx.default_signature.arg_kinds[:index] + ctx.default_signature.arg_kinds[index + 1 :],
            ctx.default_signature.arg_names[:index] + ctx.default_signature.arg_names[index + 1 :],
            ctx.default_signature.ret_type,
            ctx.default_signature.fallback,
        )

    return ctx.default_signature


def plugin(version: str):
    return CustomPlugin
