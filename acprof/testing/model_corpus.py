"""Offline regression tooling over the real coverage and model-inspection pipeline.

Only repository I/O is substituted. Fixtures are never evidence of checkpoint
execution. The process-wide guards make this a sequential development tool.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import socket
import sys
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO, TypedDict
from unittest.mock import patch

from acprof.artifacts import atomic_write, atomic_write_json, read_json_object
from acprof.host.preparation.model_coverage import run_sample, validate_sample
from acprof.hub.repository import MetadataFile, model_source, read_metadata_file
from acprof.models.evidence import INTERFACE_RESOLVER_VERSION, RESOLVER_VERSION, content_digest
from acprof.models.files import safe_path
from acprof.testing.provenance import capture_provenance

_ACTUAL_FIELDS = {
    "inspection_status", "compatibility_status", "reason_code", "selected_task",
    "runtime_profile", "runtime_status", "supported",
}
_REQUIRED_EXPECTATIONS = {"inspection_status", "compatibility_status", "reason_code"}
_IDENTITY_FIELDS = ("case_id", "source", "model_id", "revision")
_FRAMEWORKS = {"torch", "transformers", "diffusers", "onnxruntime", "kokoro", "chronos",
               "sentence_transformers", "sklearn"}


class AttentionGroup(TypedDict):
    stage: str
    reason_code: str
    count: int
    case_ids: list[str]


def _validate_decision(decision: object, *, complete: bool) -> None:
    if not isinstance(decision, dict) or not _REQUIRED_EXPECTATIONS <= decision.keys():
        raise ValueError("expected decision needs inspection_status, compatibility_status and reason_code")
    if set(decision) - _ACTUAL_FIELDS or complete and set(decision) != _ACTUAL_FIELDS:
        raise ValueError("unknown or missing decision fields")
    if not isinstance(decision["inspection_status"], str) or decision["inspection_status"] not in {
        "candidate", "resolved", "blocked", "ambiguous", "needs_configuration", "selection_required", "unknown",
    } or not isinstance(decision["compatibility_status"], str) or decision["compatibility_status"] not in {
        "supported", "unsupported", "unknown",
    }:
        raise ValueError("invalid static decision status")
    for key in ("reason_code", "selected_task", "runtime_profile"):
        if key in decision and decision[key] is not None and not isinstance(decision[key], str):
            raise ValueError(f"{key} must be a string or null")
    if "supported" in decision and type(decision["supported"]) is not bool:
        raise ValueError("supported must be a boolean")
    if "runtime_status" in decision and decision["runtime_status"] != "not_requested":
        raise ValueError("corpus replay cannot contain runtime evidence")
    if decision.get("supported") is True and (decision["compatibility_status"] != "supported"
            or decision["inspection_status"] not in {"candidate", "resolved"} or decision["reason_code"] is not None):
        raise ValueError("supported decision conflicts with static evidence")


def load_corpus(path: Path) -> tuple[dict, dict, str]:
    """Validate all identities and exact local bytes before creating reports."""
    path = Path(path).resolve()
    manifest = read_json_object(path, label="model corpus")
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or type(manifest.get("corpus_version")) is not int or manifest["corpus_version"] != 1):
        raise ValueError("unsupported model corpus version")
    validate_sample(manifest)
    if len(manifest["models"]) > 1000:
        raise ValueError("model corpus exceeds 1000 scenarios")
    content: dict[tuple[str, str, str], dict[str, MetadataFile]] = {}
    identities, seen = [], set()
    for case in manifest["models"]:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", case_id):
            raise ValueError("case_id must be a stable lowercase identifier")
        if case_id in seen:
            raise ValueError("duplicate case_id")
        seen.add(case_id)
        if case.get("fixture_kind") not in {"synthetic", "captured"} or not case.get("description"):
            raise ValueError("case requires fixture_kind and description")
        _validate_decision(case.get("expected"), complete=False)
        hub, files, metadata = case.get("hub"), case.get("repository_files"), case.get("metadata")
        if not isinstance(hub, dict) or set(hub) - {
            "pipeline_tag", "library_name", "config", "transformers_info", "tags", "card_data", "safetensors",
        }:
            raise ValueError("invalid frozen Hub observation")
        if not isinstance(files, list) or any(not isinstance(name, str) for name in files):
            raise ValueError("repository_files must be a list of paths")
        if len(files) != len(set(files)) or len(files) > 10000:
            raise ValueError("duplicate or excessive repository files")
        for name in files:
            safe_path(name)
        if not isinstance(metadata, dict) or not set(metadata) <= set(files) or len(metadata) > 128:
            raise ValueError("metadata must name at most 128 declared repository files")
        identity: dict[str, Any] = {key: case.get(key, "huggingface" if key == "source" else None)
                    for key in (*_IDENTITY_FIELDS, "weight", "fixture_kind")}
        identity.update(hub=hub, repository_files=sorted(files), metadata={}, capture_source=case.get("capture_source"))
        key = (identity["source"], case["model_id"], case["revision"])
        content[key] = {}
        for name, reference in metadata.items():
            if not isinstance(reference, dict) or reference.get("origin") not in {"synthetic", "captured"}:
                raise ValueError("metadata must declare synthetic or captured origin")
            if reference["origin"] == "captured" and not reference.get("url"):
                raise ValueError("captured metadata requires its pinned source URL")
            filename = reference.get("path")
            if not isinstance(filename, str):
                raise ValueError("metadata requires a relative path")
            filename = safe_path(filename)
            local = (path.parent / filename).resolve()
            if not local.is_relative_to(path.parent):
                raise ValueError("metadata path escapes corpus directory")
            digest = reference.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("metadata requires SHA256")
            raw = read_metadata_file(local, max_bytes=1024 * 1024, expected_sha256=digest)
            content[key][name] = MetadataFile(Path(name), raw.content)
            identity["metadata"][name] = {key: value for key, value in reference.items() if key != "path"}
        if case["fixture_kind"] == "captured" and not case.get("capture_source"):
            raise ValueError("captured repository observations require capture_source")
        identities.append(identity)
    sample_sha = content_digest({"sampling": manifest["sampling"], "weight_basis": manifest["weight_basis"],
                                 "models": sorted(identities, key=lambda item: item["case_id"])})
    return manifest, content, sample_sha


@contextmanager
def _offline_repositories(manifest: dict, content: dict):
    violations = []
    cases = {(case.get("source", "huggingface"), case["model_id"], case["revision"]): case
             for case in manifest["models"]}

    def reject(detail):
        violations.append(detail)
        raise ValueError(f"invalid offline replay: {detail}")

    def info(model_id, revision=None, **_kwargs):
        case = cases.get((model_source(), model_id, revision))
        if case is None:
            return reject(f"unfrozen repository {model_source()}/{model_id}@{revision}")
        return SimpleNamespace(**case["hub"], sha=revision,
                               siblings=[SimpleNamespace(rfilename=name) for name in case["repository_files"]])

    def metadata(model_id, name, revision=None, *, max_bytes=1024 * 1024, source=None):
        raw = content.get((model_source(source), model_id, revision), {}).get(name)
        if raw is None:
            return reject(f"missing frozen metadata: {model_id}@{revision}/{name}")
        if len(raw.content) > max_bytes:
            return reject(f"metadata exceeds read budget: {name}")
        return raw

    def network(*_args, **_kwargs):
        return reject("network access attempted")

    class FrameworkGuard:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in _FRAMEWORKS:
                reject(f"inference framework import attempted: {fullname}")
            return None

    guard = FrameworkGuard()
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                                    "MODELSCOPE_OFFLINE": "1"}))
        stack.enter_context(patch("acprof.host.model_store.store.cached_model_info", return_value=None))
        stack.enter_context(patch("huggingface_hub.HfApi.model_info", side_effect=info))
        stack.enter_context(patch("acprof.hub.repository.modelscope_info", side_effect=info))
        stack.enter_context(patch("acprof.host.detect._download_metadata", side_effect=metadata))
        stack.enter_context(patch("acprof.host.preparation.model_inspection.validate_model_runtime",
                                  side_effect=lambda *_args, **_kwargs: reject("runtime validation attempted")))
        stack.enter_context(patch.object(socket.socket, "connect", network))
        stack.enter_context(patch.object(socket.socket, "connect_ex", network))
        stack.enter_context(patch.object(socket, "getaddrinfo", network))
        sys.meta_path.insert(0, guard)
        stack.callback(sys.meta_path.remove, guard)
        yield violations
    if violations:
        raise ValueError("invalid offline replay: " + "; ".join(dict.fromkeys(violations)))


def _actual(row: dict) -> dict:
    if row["compatibility"].get("checkpoint_verified") is not False:
        raise ValueError("static replay cannot verify a checkpoint")
    decision = {"inspection_status": row["inspection_status"],
                "compatibility_status": row["compatibility"]["status"],
                "reason_code": row.get("reason_code"), "selected_task": row.get("selected_task"),
                "runtime_profile": row.get("runtime_profile"), "runtime_status": row["runtime_status"],
                "supported": row["supported"]}
    _validate_decision(decision, complete=True)
    return decision


def _baseline(path: Path, manifest: dict, sample_sha: str) -> tuple[dict, dict]:
    report = read_json_object(path, label="corpus baseline")
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("kind") != "acprof_model_corpus" or report.get("scope") != "static_fixture_replay"
            or report.get("sample_sha256") != sample_sha or not isinstance(report.get("rows"), list)):
        raise ValueError("baseline requires the same frozen corpus and report version")
    identity = report.get("source_identity")
    if not isinstance(identity, dict) or any(not isinstance(identity.get(key), str)
            or not re.fullmatch(r"[0-9a-f]{64}", identity[key])
            for key in ("source_sha256", "tests_sha256", "locks_sha256")):
        raise ValueError("baseline is missing source identity")
    cases = {case["case_id"]: case for case in manifest["models"]}
    rows = {}
    for row in report["rows"]:
        if not isinstance(row, dict) or row.get("case_id") not in cases or row["case_id"] in rows:
            raise ValueError("baseline has unknown or duplicate case rows")
        case = cases[row["case_id"]]
        if any(row.get(key) != case.get(key, "huggingface" if key == "source" else None)
               for key in _IDENTITY_FIELDS):
            raise ValueError("baseline case identity does not match frozen corpus")
        _validate_decision(row.get("actual"), complete=True)
        rows[row["case_id"]] = row
    if rows.keys() != cases.keys():
        raise ValueError("baseline is missing frozen case rows")
    return rows, {"path": str(path), "source_identity": identity,
                  "expectations_sha256": report.get("expectations_sha256"),
                  "resolver_versions": report.get("resolver_versions")}


def _summary(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        actual = row["actual"]
        if actual["supported"]:
            continue
        reason = actual["reason_code"] or "compatibility_unknown"
        stage = (row.get("failure") or {}).get("stage", "static_compatibility")
        groups.setdefault((stage, reason), []).append(row["case_id"])
    attention: list[AttentionGroup] = [
        {"stage": stage, "reason_code": reason, "count": len(cases), "case_ids": sorted(cases)}
        for (stage, reason), cases in groups.items()]
    attention.sort(key=lambda group: (-group["count"], group["stage"], group["reason_code"]))
    matched = sum(not row["differences"] for row in rows)
    return {"cases": len(rows), "matched": matched, "mismatched": len(rows) - matched,
            "fixture_kinds": dict(sorted(Counter(row["fixture_kind"] for row in rows).items())),
            "attention": attention}


def _write_text(path: Path, content: str) -> None:
    def write(stream: TextIO) -> None:
        stream.write(content)
    atomic_write(path, write)


def _write_report(root: Path, report: dict) -> None:
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# Static model corpus regression", "",
             "Fixture replay only; not checkpoint execution, task accuracy or Hub-wide coverage.", "",
             f"Cases: {report['summary']['cases']}; matched: {report['summary']['matched']}; "
             f"mismatched: {report['summary']['mismatched']}.", "",
             "## Attention by recorded reason", "",
             "| Stage | Reason | Count | Cases |", "| --- | --- | ---: | --- |"]
    for group in report["summary"]["attention"]:
        lines.append("| " + " | ".join(escape(group[key]) for key in ("stage", "reason_code", "count", "case_ids")) + " |")
    lines += ["", "## Expected decisions", "", "| Case | Match | Differences |",
              "| --- | --- | --- |"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=[*_IDENTITY_FIELDS, "fixture_kind", *sorted(_ACTUAL_FIELDS),
                                               "matched", "differences", "baseline_changes"])
    writer.writeheader()
    changes = {item["case_id"]: item["fields"] for item in report["changes"]}
    for row in report["rows"]:
        differences = json.dumps(row["differences"], ensure_ascii=False, sort_keys=True)
        lines.append(f"| {row['case_id']} | {not row['differences']} | {escape(differences)} |")
        writer.writerow({**{key: row[key] for key in (*_IDENTITY_FIELDS, "fixture_kind")}, **row["actual"],
                         "matched": not row["differences"], "differences": differences,
                         "baseline_changes": json.dumps(changes.get(row["case_id"], {}), sort_keys=True)})
    lines += ["", "## Baseline changes", ""]
    lines += [f"- {item['case_id']}: {json.dumps(item['fields'], ensure_ascii=False, sort_keys=True)}"
              for item in report["changes"]]
    if not report["changes"]:
        lines.append("No changed decisions." if report["baseline"] else "No baseline supplied.")
    _write_text(root / "corpus.csv", stream.getvalue())
    _write_text(root / "CORPUS.md", "\n".join(lines) + "\n")
    atomic_write_json(root / "corpus.json", report)


def replay_corpus(manifest_path: Path, output: Path, *, baseline: Path | None = None) -> dict:
    """Run real static inspection, preserving expected blockers and raw evidence."""
    manifest_path, output = Path(manifest_path), Path(output)
    manifest, content, sample_sha = load_corpus(manifest_path)
    previous, baseline_identity = _baseline(baseline, manifest, sample_sha) if baseline is not None else ({}, None)
    provenance = capture_provenance([manifest_path])
    with _offline_repositories(manifest, content):
        coverage = run_sample(manifest, output, probe="none")
    if capture_provenance([manifest_path]) != provenance:
        raise ValueError("corpus replay source changed during execution")
    current_manifest, _, current_sha = load_corpus(manifest_path)
    if current_sha != sample_sha or current_manifest != manifest:
        raise ValueError("corpus input changed during execution")
    rows, changes = [], []
    for case, result in zip(manifest["models"], coverage["rows"], strict=True):
        actual = _actual(result)
        differences = {key: {"expected": value, "actual": actual[key]}
                       for key, value in case["expected"].items()
                       if content_digest(value) != content_digest(actual[key])}
        row = {**{key: case.get(key, "huggingface" if key == "source" else None) for key in _IDENTITY_FIELDS},
               "fixture_kind": case["fixture_kind"], "expected": case["expected"], "actual": actual,
               "differences": differences, "failure": result.get("failure"),
               "next_action": result.get("next_action"), "attempt_path": result["attempt_path"]}
        rows.append(row)
        if previous:
            before = previous[case["case_id"]]["actual"]
            fields = {key: {"before": before[key], "after": actual[key]} for key in sorted(_ACTUAL_FIELDS)
                      if before[key] != actual[key]}
            if fields:
                changes.append({"case_id": case["case_id"], "fields": fields})
    report = {"schema_version": 1, "kind": "acprof_model_corpus", "scope": "static_fixture_replay",
              "sample_sha256": sample_sha, "expectations_sha256": content_digest({
                  case["case_id"]: case["expected"] for case in manifest["models"]}),
              "source_identity": provenance,
              "resolver_versions": {"contract": RESOLVER_VERSION, "interface": INTERFACE_RESOLVER_VERSION},
              "summary": _summary(rows), "rows": rows, "baseline": baseline_identity, "changes": changes}
    _write_report(output, report)
    return report
