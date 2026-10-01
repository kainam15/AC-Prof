"""Platform skips apply only to explicit integration markers, never ordinary bugs."""
import pytest

from acprof.platform import detect_environment


def pytest_collection_modifyitems(config, items):
    environment = detect_environment()
    for item in items:
        if not any(item.get_closest_marker(name) for name in ("wsl", "native_linux", "hardware")):
            item.add_marker(pytest.mark.unit)
        if item.get_closest_marker("native_linux") and not environment.native:
            item.add_marker(pytest.mark.skip(reason="requires Native Linux validation"))
        elif item.get_closest_marker("wsl") and environment.environment != "wsl2":
            item.add_marker(pytest.mark.skip(reason="requires WSL2 integration environment"))
