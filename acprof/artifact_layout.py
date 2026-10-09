"""Versioned experiment paths. Discovery is read-only; writers initialize v2 explicitly."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from acprof.artifacts import read_json_object, replace_file_durably

MANIFEST_NAME = "result_manifest.json"
METADATA_FILES = frozenset({
    "model_resolution.json", "auto_report.json", "collection_history.json",
    "input_scale_plan.json", "matrix_plan.json", "startup_oom_pruning.json",
    "compute_profile_plan.json", "execution_profile_plan.json", "runtime_validation.json", "interface_validation.json",
})
DIRECTORIES = {
    "compute_profiles": "raw/compute_profiles", "execution_profiles": "raw/execution_profiles",
    "posthoc_profiles": "raw/posthoc_profiles", "probes": "raw/probes",
    "interrupted_cases": ".acprof/recovery/interrupted_cases",
    "posthoc_backups": ".acprof/recovery/posthoc_backups",
    "debug_idle_diag": "debug/idle",
    "analysis": "plots/analysis",
}
_CASE_ID = re.compile(r"[1-9][0-9]*c_[1-9][0-9]*g_(?:on|off)\Z")


def _manifest() -> dict:
    return {
        "schema_version": 1, "layout_version": 2,
        "primary": {"results": "result_layers.json", "static_meta": "static_meta.json",
                    "capabilities": "capability_report.json"},
        "metadata": "metadata/", "raw": "raw/", "plots": "plots/",
        "logs": "logs/", "debug": "debug/", "internal": ".acprof/",
        "artifacts": {
            **{name: f"metadata/{name}" for name in sorted(METADATA_FILES)},
            "run_state.json": ".acprof/run_state.json", "result.lock": ".acprof/result.lock",
            "terminal.log": "logs/terminal.log", **DIRECTORIES,
        },
        "case_work": ".acprof/work/cases/{case_id}/",
        "request_samples": "raw/requests/{case_id}.jsonl",
    }


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path
    layout_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).expanduser().absolute())
        if type(self.layout_version) is not int or self.layout_version not in {1, 2}:
            raise ValueError(f"Unsupported artifact layout: {self.layout_version!r}")

    @classmethod
    def discover(cls, root: str | Path) -> ArtifactLayout:
        """A missing manifest means flat layout; a broken manifest never falls back."""
        root = Path(root).expanduser().absolute()
        manifest = root / MANIFEST_NAME
        if manifest.is_symlink():
            raise ValueError(f"Result manifest must not be a symlink: {manifest}")
        if not manifest.exists():
            if (root / ".acprof/run_state.json").exists():
                raise ValueError(f"Missing {MANIFEST_NAME} for a v2 experiment: {root}")
            return cls(root)
        payload = read_json_object(manifest, label="result manifest")
        expected = _manifest()
        if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
                or type(payload.get("layout_version")) is not int
                or any(payload.get(key) != value for key, value in expected.items())):
            raise ValueError(f"Unsupported or inconsistent result manifest: {manifest}")
        return cls(root, 2)

    @classmethod
    def for_new_run(cls, root: str | Path) -> ArtifactLayout:
        existing = cls.discover(root)
        return cls(existing.root, 2)

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> ArtifactLayout:
        path = Path(csv_path).absolute()
        if path.name == "result.csv" and path.parent.parent.parts[-3:] == (".acprof", "work", "cases"):
            layout = cls.discover(path.parents[4])
            if layout.layout_version != 2:
                raise ValueError(f"Missing v2 result manifest for case: {path}")
            layout.case_from_csv(path)
            return layout
        return cls.discover(path.parent)

    def path(self, name: str) -> Path:
        """Resolve a logical artifact name; unknown names remain relative to the root."""
        relative = name
        if self.layout_version == 2:
            if name in METADATA_FILES:
                relative = f"metadata/{name}"
            else:
                relative = {
                    "run_state.json": ".acprof/run_state.json",
                    "result.lock": ".acprof/result.lock",
                    ".acprof-result.lock": ".acprof/result.lock",
                    "terminal.log": "logs/terminal.log",
                    "tmux_all.log": "logs/terminal.log",
                    **DIRECTORIES,
                }.get(name, name)
        return self.contained(relative)

    def contained(self, relative: str | Path) -> Path:
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Artifact path must stay inside the result directory: {relative}")
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError(f"Artifact path escapes the result directory: {relative}")
        # Reject aliases within the directory too: recovery must never delete a shared target.
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != self.root and parent.is_relative_to(self.root)):
            raise ValueError(f"Artifact path must not contain symlinks: {relative}")
        return path

    @property
    def result_csv(self) -> Path:
        """Logical result reference (layer manifest, not a wide CSV)."""
        return self.path("result_layers.json")

    @property
    def plots_dir(self) -> Path:
        return self.path("plots") if self.layout_version == 2 else self.root

    def check_new_run(self, *, allowed_files=()) -> None:
        """Inspect a destination without creating a directory, lock or manifest."""
        if self.layout_version != 2:
            raise ValueError("Only new v2 experiments may be initialized")
        allowed = {MANIFEST_NAME, ".acprof/result.lock", *allowed_files}
        allowed_dirs = {".acprof", "metadata", "raw", "plots", "logs", "debug"}
        for name in allowed:
            allowed_dirs.update(p.as_posix() for p in Path(name).parents if p != Path("."))
        if self.root.exists():
            for path in self.root.rglob("*"):
                relative = path.relative_to(self.root).as_posix()
                if path.is_symlink():
                    raise ValueError(f"Existing experiment artifact is a symlink: {path}")
                if relative in {"probes", "raw/probes"} or relative.startswith(("probes/", "raw/probes/")):
                    continue
                if relative in allowed or (path.is_dir() and relative in allowed_dirs):
                    continue
                raise ValueError(f"结果目录已有实验产物：{path}；续跑请加 --resume，新实验请更换 --output-dir")
        manifest = self.root / MANIFEST_NAME
        if manifest.exists():
            self.discover(self.root)

    def initialize(self, *, allowed_files=()) -> None:
        """Publish the immutable routing manifest under the caller's directory lock."""
        self.check_new_run(allowed_files=allowed_files)
        manifest = self.root / MANIFEST_NAME
        if not manifest.exists():
            from acprof.artifacts import atomic_write_json
            atomic_write_json(manifest, _manifest())
        self.path("collection_history.json").parent.mkdir(parents=True, exist_ok=True)

    def case(self, model_id: str, cpu: int, mem: int, gpu: str) -> CaseArtifacts:
        case_id = f"{cpu}c_{mem}g_{gpu}"
        if not _CASE_ID.fullmatch(case_id):
            raise ValueError(f"Invalid case ID: {case_id}")
        filename = f"result_case_{model_id.replace('/', '--')}_{case_id}.csv"
        if self.layout_version == 2:
            return CaseArtifacts(self, case_id, self.contained(f".acprof/work/cases/{case_id}/result.csv"))
        return CaseArtifacts(self, case_id, self.contained(filename))

    def case_from_csv(self, csv_path: str | Path) -> CaseArtifacts:
        path = Path(csv_path).absolute()
        relative = path.relative_to(self.root)
        self.contained(relative)
        if self.layout_version == 2:
            if relative.parts[:3] != (".acprof", "work", "cases") or len(relative.parts) != 5 or path.name != "result.csv":
                raise ValueError(f"Unexpected case CSV: {path}")
            case_id = path.parent.name
        else:
            match = re.search(r"_([1-9][0-9]*c_[1-9][0-9]*g_(?:on|off))\.csv$", path.name)
            if not match or path.parent != self.root:
                raise ValueError(f"Unexpected case CSV: {path}")
            case_id = match[1]
        if not _CASE_ID.fullmatch(case_id):
            raise ValueError(f"Invalid case ID: {case_id}")
        return CaseArtifacts(self, case_id, path)


def case_sidecar(csv_path: str | Path, kind: str) -> Path:
    """Pure path routing shared by the client and packet merge; no directory scans."""
    path = Path(csv_path)
    if path.name == "result.csv" and path.parent.parent.parts[-3:] == (".acprof", "work", "cases"):
        return path.with_name({"requests": "requests.jsonl", "sniff_groups": "sniff_groups.jsonl",
                               "client_error": "client_error.json", "quality_checks": "quality_checks.json",
                               "runtime_failures": "runtime_failures.json", "cleanup_error": "cleanup_error.json"}[kind])
    return Path(f"{path}.{kind}.{'json' if kind in {'client_error', 'quality_checks', 'runtime_failures', 'cleanup_error'} else 'jsonl'}")


@dataclass(frozen=True)
class CaseArtifacts:
    layout: ArtifactLayout
    case_id: str
    csv: Path

    def sidecar(self, kind: str) -> Path:
        return case_sidecar(self.csv, kind)

    @property
    def requests(self) -> Path:
        return self.sidecar("requests")

    @property
    def retained_requests(self) -> Path:
        return self.layout.contained(f"raw/requests/{self.case_id}.jsonl") if self.layout.layout_version == 2 else self.requests

    @property
    def pcap(self) -> Path:
        return self._intermediate("sniff", "pcap")

    @property
    def latency(self) -> Path:
        return self._intermediate("lat", "json")

    def _intermediate(self, prefix: str, suffix: str) -> Path:
        if self.layout.layout_version == 2:
            return self.csv.with_name("sniff.pcap" if prefix == "sniff" else "packet_latency.json")
        return self.layout.contained(f"{prefix}_{self.csv.stem.removeprefix('result_')}.{suffix}")

    @property
    def idle(self) -> Path:
        name = f"{self.case_id}.jsonl" if self.layout.layout_version == 2 else f"{self.csv.name}.idle_diag.jsonl"
        return self.layout.path("debug_idle_diag") / name

    def temporary_files(self) -> list[Path]:
        return [self.csv, self.sidecar("sniff_groups"), self.sidecar("client_error"),
                self.requests, self.pcap, self.latency, Path(f"{self.csv}.merged"),
                self.sidecar("quality_checks"), self.sidecar("runtime_failures"), self.sidecar("cleanup_error")]

    def retain_requests(self) -> None:
        """Durably publish completed request samples before the case checkpoint."""
        source, destination = self.requests, self.retained_requests
        if source == destination:
            return
        self.layout.contained(source.relative_to(self.layout.root))
        if not source.exists():
            return
        if not source.is_file():
            raise ValueError(f"Request samples must be a regular file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        replace_file_durably(source, destination)
