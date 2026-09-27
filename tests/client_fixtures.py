"""Configure isolated client instances in measurement regression fixtures."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

from acprof.host import client


def patch_client(runner, name, *args, **kwargs):
    special = {"USE_ENERGY": "use_energy", "_FIRST_PREDICT_APP_S": "first_predict_app_s"}
    if name.isupper() and hasattr(runner.config, name.lower()):
        return patch.object(runner.config, name.lower(), *args, **kwargs)
    name = special.get(name, name)
    return patch.object(runner if hasattr(runner, name) else client, name, *args, **kwargs)


@contextmanager
def patch_client_settings(runner, **settings):
    with ExitStack() as stack:
        for name, value in settings.items():
            stack.enter_context(patch_client(runner, name, value))
        yield
