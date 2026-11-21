"""Sequential task collections with immutable per-task workload and result identity."""
from __future__ import annotations

import copy
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from acprof.artifacts import atomic_write_json, read_json_object
from acprof.experiments.configuration import RunConfig
from acprof.host.detect import TaskInfo
from acprof.host.preparation.task_selection import activate_task
from acprof.host.runtimes.contracts import RuntimeCleanupError
from acprof.models.evidence import content_digest
from acprof.results.task_collections import read_task_collection
from acprof.system.cleanup import cleanup_on_exit, format_exception_notes
from acprof.tasks.workloads import task_experiment_overrides, task_fingerprint, workload_manifest


def collection_root(config: RunConfig) -> Path:
    return Path(config.output_dir).expanduser().resolve() / config.model.replace("/", "--")


def configuration_identity(config: RunConfig, task: TaskInfo) -> dict:
    values = asdict(config)
    for name in ("resume", "output_dir", "notify", "task_run_id", "tasks", "revision",
                 "inference_config", "confirm_inference"):
        values.pop(name)
    for name in ("workload_spec", "model_spec"):
        if values.get(name):
            values[name] = content_digest(read_json_object(Path(values[name]).expanduser(), label=name))
    if task.inference_configuration:
        values["inference_configuration_sha256"] = task.inference_configuration["identity_sha256"]
    return values


class TaskCollection:
    def __init__(self, config: RunConfig, task: TaskInfo):
        self.config, self.task = config, copy.deepcopy(task)
        self.root = collection_root(config)
        self.run_id = uuid4().hex
        if config.task_run_id and not config.resume:
            raise ValueError("--task-run-id identifies an existing collection; use --resume")
        self.manifest: dict[str, Any] = {
            "schema_version": 1, "run_id": self.run_id, "status": "pending",
            "model_id": task.model_id, "model_revision": task.model_revision,
            "configuration": configuration_identity(config, task), "task_snapshot": asdict(self.task),
            "tasks": [{"task_id": item["task_id"], "fingerprint": item["fingerprint"], "status": "pending"}
                      for item in task.selected_tasks],
        }

    @property
    def path(self) -> Path:
        return self.root / "metadata" / "task_runs" / f"{self.run_id}.json"

    @classmethod
    def read_manifest(cls, config: RunConfig) -> tuple[Path, dict]:
        return read_task_collection(collection_root(config), run_id=config.task_run_id)

    @classmethod
    def restore(cls, config: RunConfig):
        path, data = cls.read_manifest(config)
        task = TaskInfo(**data["task_snapshot"])
        if task.inference_configuration:
            from acprof.host.preparation.inference_configuration import (
                validate_resume_inference_requests,
            )
            from acprof.models.inference_configuration import validate_configuration_binding

            validate_resume_inference_requests(task.inference_configuration, config.to_namespace())
            validate_configuration_binding(task.inference_configuration, task)
        elif config.inference_config or config.confirm_inference:
            raise ValueError("historical task collections cannot accept new inference requests during resume")
        if data["model_id"] != task.model_id or data["model_revision"] != task.model_revision:
            raise ValueError("task collection model identity changed")
        if config.revision and config.revision not in {task.model_revision, task.requested_revision}:
            raise ValueError("task collection revision changed")
        if config.tasks and config.tasks != "all":
            requested = [name.strip() for name in config.tasks.split(",")]
            profiles = {item["task_id"]: item for item in task.selected_tasks}
            semantics = {member["task_id"] for item in task.selected_tasks
                         for member in item.get("semantic_tasks", [item])}
            if (not requested or len(set(requested)) != len(requested)
                    or set(requested) - (profiles.keys() | semantics)):
                raise ValueError("task collection selection changed")
            expanded = {task_id for name in requested for task_id in (
                [member["task_id"] for member in profiles[name].get("semantic_tasks", [profiles[name]])]
                if name in profiles else [name])}
            if expanded != semantics:
                raise ValueError("task collection selection changed")
        if configuration_identity(config, task) != data["configuration"]:
            raise ValueError("task collection configuration changed; use the original options")
        for selected, recorded in zip(task.selected_tasks, data["tasks"], strict=True):
            if (task_fingerprint(selected) != recorded["fingerprint"]
                    or selected["fingerprint"] != recorded["fingerprint"]
                    or selected["task_id"] != recorded["task_id"]):
                raise ValueError("task workload fingerprint changed")
        instance = cls(config, task)
        instance.run_id, instance.manifest = data["run_id"], data
        if instance.path.resolve() != path.resolve():
            raise ValueError("task collection path identity changed")
        return instance

    def child_config(self, selected: dict) -> RunConfig:
        from acprof.extensions import CATALOG
        values = dict(tasks=selected["task_id"], task_run_id=self.run_id,
                      revision=self.task.model_revision, task=selected["interface"],
                      task_family=CATALOG.task_families[selected["interface"]])
        values.update(task_experiment_overrides(selected))
        if values["task_family"] == "nlp":
            supplied = (read_json_object(Path(self.config.workload_spec).expanduser(), label="workload specification")
                        if self.config.workload_spec else None)
            manifest = workload_manifest(selected, supplied=supplied, frozen=self.task.inference_configuration)
            path = self.root / "metadata" / "task_workloads" / (content_digest(manifest) + ".json")
            if path.is_file():
                if read_json_object(path, label="task workload") != manifest:
                    raise ValueError("saved task workload changed")
            elif self.config.resume and next(item["status"] for item in self.manifest["tasks"]
                                             if item["task_id"] == selected["task_id"]) != "pending":
                raise ValueError("saved task workload is missing")
            else:
                atomic_write_json(path, manifest)
            values["workload_spec"] = str(path)
        config = replace(self.config, **values)
        # Interrupted before this task started: it has no RunState to restore.
        from acprof.results.layout import ArtifactLayout
        if config.resume and not ArtifactLayout.discover(config.result_dir(Path.cwd())).path("run_state.json").exists():
            config = replace(config, resume=False)
        return config

    def run(self, execute):
        from acprof.application.run import RunOutcome
        from acprof.host.run_state import ResultDirectoryLock
        final_csvs, completed = [], 0

        def interrupted():
            self.manifest["status"] = "interrupted"
            atomic_write_json(self.path, self.manifest)

        with (ResultDirectoryLock(self.root / "metadata" / "task_run_locks" / self.run_id),
              cleanup_on_exit(interrupted, description="task collection interruption", on_error_only=True)):
            self.manifest.update(status="running", pid=os.getpid(), started_at_ns=time.time_ns())
            atomic_write_json(self.path, self.manifest)
            print(f"[tasks] collection={self.run_id}; resume with --task-run-id {self.run_id}", flush=True)
            for selected, record in zip(self.task.selected_tasks, self.manifest["tasks"], strict=True):
                print(f"[tasks] {selected['task_id']} ({selected['interface']})", flush=True)
                try:
                    config = self.child_config(selected)
                    record.update(status="running")
                    atomic_write_json(self.path, self.manifest)
                    task = activate_task(self.task, selected, backend=self.config.backend)
                    outcome = execute(config, task)
                    record.update(status=outcome.status, final_csv=outcome.final_csv)
                    record.pop("error", None)
                    completed += outcome.completed_cases
                    if outcome.final_csv:
                        final_csvs.append(outcome.final_csv)
                except (Exception, SystemExit) as exc:
                    detail = format_exception_notes(exc)
                    record.update(status="failed", error=f"{type(exc).__name__}: {detail}")
                    print(f"[tasks][ERROR] {selected['task_id']}: {detail}", flush=True)
                    if isinstance(exc, RuntimeCleanupError):
                        raise
                atomic_write_json(self.path, self.manifest)
            status = "ok" if all(item["status"] == "ok" for item in self.manifest["tasks"]) else "failed"
            self.manifest.update(status=status, final_csvs=final_csvs)
            atomic_write_json(self.path, self.manifest)
        return RunOutcome(status, final_csvs[-1] if final_csvs else None, completed, str(self.path))
