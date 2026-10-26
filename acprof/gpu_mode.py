"""Pure GPU mode contract shared by configuration, runtime and analysis."""


def normalize_gpu_mode(value: object) -> str:
    """Normalize supported GPU mode aliases; reject every unknown value."""
    normalized = str(value).strip().lower()
    if normalized in {"on", "1", "1.0", "true", "yes", "gpu"}:
        return "on"
    if normalized in {"off", "0", "0.0", "false", "no", "cpu"}:
        return "off"
    raise ValueError(f"unsupported gpu_mode value: {value!r}")
