"""Observe upstream loading_info during isolated loading; no log parsing."""

from contextlib import contextmanager
from functools import wraps
from threading import RLock

from acprof.quality import loading_quality

_LOCK = RLock()


@contextmanager
def capture_loading_quality(backend):
    checks = []
    if backend not in {"transformers_model", "transformers_pipeline", "sentence_transformers", "cross_encoder", "chronos"}:
        yield checks
        return
    try:
        import transformers
    except ModuleNotFoundError:
        # The actual handler retains ownership of required dependency failures.
        yield checks
        return

    base = getattr(transformers, "PreTrainedModel", None)
    if base is None:
        yield checks
        return
    # Both reviewed Transformers versions support output_loading_info. Preserve
    # the caller's return contract, and restore the descriptor even on failure.
    with _LOCK:
        descriptor = base.__dict__["from_pretrained"]
        original = descriptor.__func__

        @wraps(original)
        def observed(cls, *args, **kwargs):
            requested = kwargs.pop("output_loading_info", False)
            result = original(cls, *args, output_loading_info=True, **kwargs)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                checks.extend(loading_quality(result[1], source=f"{cls.__module__}.{cls.__name__}.from_pretrained"))
                return result if requested else result[0]
            return result

        base.from_pretrained = classmethod(observed)
        try:
            yield checks
        finally:
            base.from_pretrained = descriptor
