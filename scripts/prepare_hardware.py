"""Prepare exact model images before the strictly offline hardware acceptance step.

This stage may fetch artifacts and build Docker layers. It never runs a
measurement or relaxes the acceptance script's offline requirement.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acprof.artifacts import atomic_write_json  # noqa: E402
from acprof.experiment import RunConfig  # noqa: E402
from acprof.host.collection_workflow import PreparationWorkflow  # noqa: E402
from acprof.host.images.runtime import prepare_image  # noqa: E402
from acprof.host.run_state import MeasurementLock, host_identity  # noqa: E402
from scripts.check_hardware import hardware_parser, selected_hardware_cases  # noqa: E402

PINNED_REVISION = re.compile(r"[0-9a-f]{40}")


def prepare_cases(cases: list[tuple[str, RunConfig]], output: Path) -> bool:
    """Prepare every requested image and publish a fully pinned selection only on success."""
    report = {
        "schema_version": 1, "kind": "hardware_preparation", "successful": False,
        "source": None, "ci_commit": os.environ.get("GITHUB_SHA"),
        "cases": [
            {"name": name, "model": config.model, "requested_revision": config.revision,
             "status": "not_run"} for name, config in cases
        ],
    }
    report_path = output / "preparation.json"
    prepared = []
    locked = False
    try:
        # Match normal collection's host-wide lock: don't hash the source,
        # build, download or write reports during another AC-Prof measurement.
        with MeasurementLock():
            locked = True
            report["source"] = host_identity(ROOT)
            atomic_write_json(report_path, report)
            for row, (name, config) in zip(report["cases"], cases, strict=True):
                row["status"] = "running"
                atomic_write_json(report_path, report)
                try:
                    workflow = PreparationWorkflow()
                    args = config.to_namespace()
                    if not PINNED_REVISION.fullmatch(config.revision):
                        # A moving branch or empty revision must be frozen
                        # before building. The final resolver call matches the
                        # pinned arguments used by offline acceptance and thus
                        # its resolution-provenance image fingerprint.
                        discovered = workflow.resolve(args)
                        if not PINNED_REVISION.fullmatch(discovered.model_revision or ""):
                            raise RuntimeError("unresolved model revision; cannot prepare immutable image")
                        args = replace(config, revision=discovered.model_revision).to_namespace()
                    task = workflow.resolve(args)
                    if task.model_revision != args.revision:
                        raise RuntimeError("resolved revision differs from requested pinned revision")
                    task.model_download_policy = args.model_download_policy
                    image = prepare_image(task, str(ROOT))
                    revision = task.model_revision
                    if not isinstance(revision, str) or not PINNED_REVISION.fullmatch(revision):
                        raise RuntimeError("prepared model revision must be an immutable commit SHA")
                    if PINNED_REVISION.fullmatch(config.revision) and revision != config.revision:
                        raise RuntimeError("prepared revision differs from requested pinned revision")
                    if task.model_id != config.model:
                        raise RuntimeError("prepared model identity differs from requested model")
                    prepared.append({
                        "name": name, "model": task.model_id, "revision": revision,
                        "task": task.pipeline_tag, "input_scales": config.input_scales,
                    })
                    row.update(status="passed", revision=revision, image_id=image.tag, image_name=image.name)
                    print(f"[hardware-prepare] {name}: ready {image.name}", flush=True)
                except BaseException as exc:
                    row.update(status="cancelled" if not isinstance(exc, Exception) else "failed",
                               error=f"{type(exc).__name__}: {exc}")
                    print(f"[hardware-prepare] {name}: {row['error']}", file=sys.stderr, flush=True)
                    if not isinstance(exc, Exception):
                        raise
                finally:
                    atomic_write_json(report_path, report)
                if row["status"] != "passed":
                    break
            if len(prepared) == len(cases):
                # Publish the immutable plan under the same host lock; otherwise
                # another measurement might begin while we write the artifact.
                atomic_write_json(output / "prepared-matrix.json", {"schema_version": 1, "cases": prepared})
                report["successful"] = True
                atomic_write_json(report_path, report)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if locked:
            atomic_write_json(report_path, report)
        print(f"[hardware-prepare] {report['error']}", file=sys.stderr, flush=True)
        if not isinstance(exc, Exception):
            raise
    return report["successful"]


def main(argv=None) -> int:
    parser = hardware_parser()
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("硬件准备输出目录必须为空")
    if os.environ.get("ACPROF_REQUIRE_PREPARED_IMAGE") == "1":
        parser.error("镜像准备阶段不得启用 ACPROF_REQUIRE_PREPARED_IMAGE=1")
    if any(os.environ.get(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        parser.error("镜像准备阶段不得启用离线模型下载模式")
    cases, _ = selected_hardware_cases(args, output, parser)
    output.mkdir(parents=True, exist_ok=True)
    successful = prepare_cases(cases, output)
    print(f"硬件镜像准备{'完成' if successful else '失败'}，证据：{output}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
