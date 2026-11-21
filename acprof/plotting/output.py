"""Publish rendered figures without modifying an earlier output's backing file."""
from collections.abc import Callable
from pathlib import Path

from acprof.artifacts import atomic_write_binary


def save_figure(path: str | Path, savefig: Callable[..., None], *, dpi: int | None = None) -> None:
    """Render to a staged stream, preserving Matplotlib's filename format selection."""
    target = Path(path)
    image_format = target.suffix[1:]
    if not image_format:
        from matplotlib import pyplot as plt

        image_format = plt.gcf().canvas.get_default_filetype()
        target = Path(f"{str(target).rstrip('.')}.{image_format}")
    atomic_write_binary(target, lambda stream: savefig(stream, format=image_format, dpi=dpi))
