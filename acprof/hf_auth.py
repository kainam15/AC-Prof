"""Origin-scoped Hub credentials, resolved only during host preparation."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

from acprof.hf_endpoints import HF_OFFICIAL_ENDPOINTS, hf_origin, hf_storage_host


@dataclass(frozen=True)
class HfCredential:
    token: str | None = field(repr=False)
    source: Literal["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "local-login", "none"]


def hf_auth_origins(environ: Mapping[str, str] | None = None) -> set[tuple[str, str, int]]:
    """Explicit mirror opt-in never broadens the network destination allowlist."""
    env = os.environ if environ is None else environ
    origins = set()
    for value in env.get("ACPROF_HF_AUTH_ORIGINS", "").replace(";", ",").split(","):
        value = value.strip()
        if not value:
            continue
        origin = hf_origin(value)
        if origin is None or any(char.isspace() for char in value):
            raise ValueError("ACPROF_HF_AUTH_ORIGINS requires HTTP(S) origins without credentials, paths, queries or fragments")
        parsed = urlsplit(value)
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("ACPROF_HF_AUTH_ORIGINS requires HTTP(S) origins without credentials, paths, queries or fragments")
        origins.add(origin)
    return origins


def hf_auth_allowed(url: str) -> bool:
    origins = hf_auth_origins()
    origin = hf_origin(url)
    if origin is None or hf_storage_host(url):
        return False
    return origin in origins or origin in {hf_origin(endpoint) for endpoint in HF_OFFICIAL_ENDPOINTS}


def resolve_hf_credential() -> HfCredential:
    """Keep configured credentials distinct from a local login; never export it."""
    keys: tuple[Literal["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"], ...] = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    for key in keys:
        token = os.environ.get(key, "").strip()
        if token:
            return HfCredential(token, key)
    from huggingface_hub import constants
    from huggingface_hub.utils import get_token

    disabled = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
    if ((disabled is None and constants.HF_HUB_DISABLE_IMPLICIT_TOKEN)
            or (disabled or "").strip().upper() in {"1", "ON", "YES", "TRUE"}):
        return HfCredential(None, "none")
    login_token = (get_token() or "").strip() or None
    return HfCredential(login_token, "local-login" if login_token else "none")


def hf_token_for_url(url: str) -> str | Literal[False]:
    """Return False, not None, to suppress the SDK's own implicit token lookup."""
    if not hf_auth_allowed(url):
        return False
    return resolve_hf_credential().token or False
