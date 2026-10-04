"""Apply reviewed download settings at the CLI execution boundary."""
import os

from acprof.network_policy import parse_bytes


def apply_download_arguments(args):
    for attr, key in (("model_source", "ACPROF_MODEL_SOURCE"), ("download_mode", "HF_DOWNLOAD_MODE"), ("max_download", "ACPROF_MAX_DOWNLOAD"),
                      ("model_store", "ACPROF_MODEL_STORE"), ("model_store_max", "ACPROF_MODEL_STORE_MAX")):
        value = getattr(args, attr, None)
        if value is not None:
            if attr in {"max_download", "model_store_max"}:
                parse_bytes(value)
            os.environ[key] = value
