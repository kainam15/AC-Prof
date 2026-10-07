"""Keep test-runner state while isolating application environment settings."""
import os
from collections.abc import Mapping


def isolated_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build values for patch.dict(clear=True) without breaking pytest shutdown.

    Session hooks may run before fixture finalizers after KeyboardInterrupt.
    Their environment state must therefore survive the entire fixture scope.
    Application settings, credentials and proxy variables are not inherited.
    """
    environment = {
        name: value for name, value in os.environ.items()
        if name.startswith(("PYTEST_", "TEXTUAL_SNAPSHOT_"))
    }
    environment.update(overrides or {})
    return environment
