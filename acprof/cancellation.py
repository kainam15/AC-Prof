"""Cooperative cancellation for synchronous artifact consumers."""
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import CancelledError


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise CancelledError()


def cancellable[T](items: Iterable[T], cancelled: Callable[[], bool] | None) -> Iterable[T]:
    """Check before advancing and yielding; keep ordinary iterators unchanged."""
    if cancelled is None:
        return items

    def checked() -> Iterator[T]:
        iterator = iter(items)
        while True:
            check_cancelled(cancelled)
            try:
                item = next(iterator)
            except StopIteration:
                return
            check_cancelled(cancelled)
            yield item

    return checked()
