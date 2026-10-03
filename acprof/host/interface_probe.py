"""Import/signature validation using only a dependency image and source bundle."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import atomic_write_json
from acprof.container.model_probe import RESULT_PREFIX
from acprof.host.command import run_command
from acprof.host.container_lifecycle import (
    ContainerCleanupError,
    container_owner_labels,
    owned_container_id,
    recover_abandoned_containers,
    remove_owned_container,
)
from acprof.host.dependency_images import prepare_environment_image
from acprof.host.source_bundle import source_bundle
from acprof.installation import resource_root
from acprof.model_contract import write_model_resolution
from acprof.model_spec import encode_model_spec, task_model_spec
from acprof.runtime_profiles import select_runtime_profile
from acprof.source_identity import service_sources, source_fingerprint


class InterfaceProbeError(RuntimeError):
    def __init__(self, message: str, *, stage: str = "interface"):
        super().__init__(message)
        self.stage = stage


def probe_interface(task_info, output_dir: str | Path, *, cpus: int = 2,
                    memory_gb: int = 4, timeout_seconds: float = 300) -> dict:
    if (type(cpus) is not int or cpus <= 0 or type(memory_gb) is not int or memory_gb <= 0
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("interface probe requires positive resource and timeout limits")
    profile = select_runtime_profile(task_info)
    project = resource_root()
    layout = ArtifactLayout.discover(output_dir)
    report = {"schema_version": 1, "status": "running", "scope": "import_and_signature_only",
              "model_id": task_info.model_id, "revision": task_info.model_revision,
              "runtime_profile": profile.profile_id, "inference": "not_run"}
    owner = container_owner_labels()
    log = ""
    try:
        # Validate and materialize the graph before preparing even dependencies.
        with source_bundle(task_info) as bundle, tempfile.TemporaryDirectory(prefix="acprof-interface-runner-") as directory:
            root = Path(directory)
            code = root / "code"
            paths = service_sources(project)
            for source in paths:
                target = code / source.relative_to(project)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            report.update(source_sha256=bundle.identity,
                          runner_sha256=source_fingerprint(code, service_sources(code), scope="interface-runner-v1"))
            from acprof.preparation_events import emit_progress
            emit_progress("environment")
            try:
                dependency = prepare_environment_image(profile.environment, project)
            except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
                raise InterfaceProbeError(str(exc), stage="environment") from exc
            emit_progress("interface")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", dependency.image_id):
                raise ValueError("interface probe requires an immutable dependency image ID")
            report["image_id"] = dependency.image_id
            recover_abandoned_containers(owner, run_command)
            cidfile = root / "container.cid"
            command = ["docker", "run", "--name", "acprof-interface-" + uuid.uuid4().hex[:16],
                       "--user", f"{os.getuid()}:{os.getgid()}",
                       "--cidfile", str(cidfile), "--network", "none", "--read-only", "--cap-drop=ALL",
                       "--security-opt", "no-new-privileges", "--pids-limit=256",
                       "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g", f"--cpus={cpus}", f"--memory={memory_gb}g",
                       "-v", f"{bundle.root}:/source:ro", "-v", f"{code}:/probe-code:ro"]
            for key, value in owner.items():
                command += ["--label", f"{key}={value}"]
            environment = {
                "HOME": "/tmp", "USER": "acprof", "LOGNAME": "acprof",
                "PYTHONPATH": "/probe-code", "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HOME": "/tmp/hf", "HF_MODULES_CACHE": "/tmp/hf-modules", "XDG_CACHE_HOME": "/tmp/cache",
                "MODEL_LOCAL_PATH": "/source", "MODEL_ID": task_info.model_id,
                "MODEL_REVISION": task_info.model_revision, "TASK_TYPE": task_info.pipeline_tag,
                "TASK_FAMILY": task_info.task_family, "RUNTIME_BACKEND": task_info.runtime_backend,
                "ACPROF_MODEL_ADAPTER": profile.adapter, "ACPROF_RUNTIME_PROFILE": profile.profile_id,
                "ACPROF_MODEL_SPEC_B64": encode_model_spec(task_model_spec(task_info)),
            }
            for key, value in environment.items():
                command += ["-e", f"{key}={value}"]
            command += ["--entrypoint", "python", dependency.image_id, "-m", "acprof.container.model_probe"]
            try:
                result = run_command(command, capture_output=True, text=True, timeout=timeout_seconds)
                log = (result.stdout or "") + "\n" + (result.stderr or "")
                records = [line[len(RESULT_PREFIX):] for line in (result.stdout or "").splitlines()
                           if line.startswith(RESULT_PREFIX)]
                observed = json.loads(records[-1]) if records else {}
                if (result.returncode or observed.get("status") != "ok"
                        or {item.get("stage") for item in observed.get("stages", [])
                            if item.get("status") == "verified"} != {"import", "signature"}):
                    raise InterfaceProbeError(observed.get("error") or log[-4000:] or "incomplete interface response")
                report.update(status="ok", validation=observed)
                if not owned_container_id(cidfile):
                    raise ContainerCleanupError("", [{"operation": "run", "detail": "missing interface container ID"}])
            except subprocess.TimeoutExpired as exc:
                log = "\n".join(value.decode(errors="replace") if isinstance(value, bytes) else value or ""
                                for value in (exc.stdout, exc.stderr))
                raise
            finally:
                if identifier := owned_container_id(cidfile):
                    remove_owned_container(identifier, run_command)
                else:
                    raise ContainerCleanupError("", [{"operation": "run", "detail": "missing interface container ID"}],
                                                run_error=sys.exc_info()[1])
    except BaseException as exc:
        report.update(status="cancelled" if isinstance(exc, KeyboardInterrupt) else "error", error=f"{type(exc).__name__}: {exc}")
        report["failed_stage"] = getattr(exc, "stage", "interface")
        if isinstance(exc, ContainerCleanupError):
            report["cleanup_error"] = exc.to_dict()
        raise
    finally:
        path = layout.path("logs") / "interface_validation.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(log)
        atomic_write_json(layout.path("interface_validation.json"), report)
        task_info.model_resolution["interface_validation"] = report
        write_model_resolution(task_info, output_dir)
    return report
