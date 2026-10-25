"""Read-only Linux process identity shared by ownership and observation."""
from pathlib import Path


def process_identity(pid: int) -> dict:
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return {"pid": pid, "start_ticks": int(fields[19]), "pgid": int(fields[2]), "state": fields[0]}


def same_process(identity: dict) -> bool:
    try:
        current = process_identity(identity["pid"])
    except (FileNotFoundError, ProcessLookupError):
        return False
    return current["start_ticks"] == identity["start_ticks"] and current["state"] != "Z"
