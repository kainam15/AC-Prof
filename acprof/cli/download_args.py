"""CLI download settings shared by run/auto/probe and the TUI."""
import os

from acprof.hf_endpoints import HF_DOWNLOAD_MODES
from acprof.network_policy import parse_bytes


def add_download_arguments(parser):
    parser.add_argument("--download-mode", choices=HF_DOWNLOAD_MODES, default=None,
                        help="HF source mode (default mirror-only; official is explicit)")
    parser.add_argument("--max-download", default=None, help="Bulk download budget, e.g. 5GB; unknown size stops before download")
    parser.add_argument("--model-store", default=None, help="Host Model Store path shared across CPU/GPU runs")
    parser.add_argument("--model-store-max", default=None, help="Model Store capacity, e.g. 100GB")


def apply_download_arguments(args):
    for attr, key in (("download_mode", "HF_DOWNLOAD_MODE"), ("max_download", "ACPROF_MAX_DOWNLOAD"),
                      ("model_store", "ACPROF_MODEL_STORE"), ("model_store_max", "ACPROF_MODEL_STORE_MAX")):
        value = getattr(args, attr, None)
        if value is not None:
            if attr in {"max_download", "model_store_max"}:
                parse_bytes(value)
            os.environ[key] = value
