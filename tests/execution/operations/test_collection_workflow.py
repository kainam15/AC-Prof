"""Preparation pauses and retries must keep the original collection alive."""
import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from huggingface_hub.errors import RepositoryNotFoundError

import tests.models.contracts.test_model_contract as contracts
import tests.support.run_recovery_fixtures as recovery
from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES
from acprof.events.preparation import MAX_MESSAGE
from acprof.host.detect import detect_task
from acprof.host.preparation.collection_workflow import (
    PreparationWorkflow,
    retain_validation_failure,
)
from acprof.host.preparation.model_inspection import explain_resolution
from acprof.models.spec import task_model_spec
from acprof.results.csv import read_result_csv
from tests.models.contracts.test_model_lookup import hub_error


def events(output):
    return [json.loads(line.removeprefix("ACPROF_PREPARATION "))
            for line in output.getvalue().splitlines() if line.startswith("ACPROF_PREPARATION ")]


def answer_with_parameter_defaults(*decisions):
    """Simulate a user accepting the optional per-profile preparation page."""
    queued = iter(decisions)

    def respond(stage, kind, **fields):
        if fields.get("purpose") == "workload_parameters":
            review = fields["questions"] + fields.get("parameter_advanced", [])
            return {"action": "answer", "answers": {
                question["path"]: question["value"] for question in review}}
        return next(queued)

    return respond


class TestCollectionWorkflow:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.fixture_root = tmp_path
        self.run = recovery.RunRecoveryFixture()
        self.run.build(self._request, self.fixture_root)

    @pytest.mark.parametrize("status", ["resource_limit", "resource_limited", "inconclusive", "error"])
    def test_cli_never_measures_after_unsuccessful_validation(self, status):
        with pytest.raises(SystemExit):
            self.run.invoke(validation=lambda **kwargs: {"status": status})
        assert self.run.calls == []

    @pytest.mark.parametrize("failed", ["postprocess", "validate_output"])
    def test_output_failure_stops_before_matrix(self, failed):
        from contextlib import ExitStack

        from acprof.container.runtime_validate import validate
        from tests.execution.validation.test_validation_stages import TestValidationStage
        with ExitStack() as stack:
            handler = TestValidationStage().fixtures(stack, failed=failed)
            with pytest.raises(SystemExit):
                self.run.invoke(validation=lambda **kwargs: validate({"text": "hello"}))
        handler.predict.assert_called_once()
        assert self.run.calls == []
        fields, rows = read_result_csv(self.run.directory / "result_layers.json")
        assert fields
        assert rows == []

    def test_start_resolves_and_validates_before_measurement_without_questions(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch("sys.stdin", io.StringIO()):
            self.run.invoke(output=output)
        records = events(output)
        passed = [item["stage"] for item in records if item["status"] == "passed"]
        assert ("resolution") in (passed)
        assert (passed.index("resolution")) < (passed.index("runtime"))
        assert not (any(item.get("request") for item in records))
        assert (self.run.calls) == ([1, 2])

    @pytest.mark.parametrize("rebuild", [False, True])
    def test_cli_prepares_image_with_requested_policy(self, rebuild):
        policies = []
        def prepare(_task, _root, *, reuse_existing):
            policies.append(reuse_existing)
            return self.run.image
        self.run.invoke(*(["--rebuild-image"] if rebuild else []), image=prepare)
        assert policies == [not rebuild]
        assert self.run.calls == [1, 2]

    def test_image_verification_failure_rebuilds_only_after_explicit_reply(self):
        policies = []
        def prepare(_task, _root, *, reuse_existing):
            policies.append(reuse_existing)
            assert not self.run.calls
            if reuse_existing:
                raise RuntimeError("image fingerprint mismatch")
            return self.run.image
        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch(
            "sys.stdin", io.StringIO('{"id":1,"action":"rebuild"}\n'),
        ):
            self.run.invoke(image=prepare)
        assert policies == [True, False]
        assert self.run.calls == [1, 2]

    def test_runtime_retry_keeps_resolved_model_image_and_input_plan(self):
        output = io.StringIO()
        attempts = []

        def validate(**kwargs):
            attempts.append(kwargs)
            assert not (self.run.calls), "measurement started before validation passed"
            report = self.run.directory / "metadata/runtime_validation.json"
            report.write_text(json.dumps({"status": "error" if len(attempts) == 1 else "ok",
                                          "devices": {"off": {"error": "ultravox_config.py" if len(attempts) == 1 else ""}}}))
            if len(attempts) == 1:
                raise FileNotFoundError("ultravox_config.py")
            return {"status": "ok"}

        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch(
            "sys.stdin", io.StringIO('{"id":1,"action":"retry"}\n'),
        ):
            self.run.invoke(validation=validate, output=output)
        assert (len(attempts)) == (2)
        for key in ("task_info", "image_info", "planned"):
            assert (attempts[0][key]) is (attempts[1][key])
        assert (attempts[1]["cpu_list"]) == ([1, 2])
        assert (attempts[1]["gpu_list"]) == (["off"])
        records = events(output)
        for stage in ("resolution", "image", "input"):
            assert (sum(item["stage"] == stage and item["status"] == "running" for item in records)) == (1)
        request = next(item for item in records if item.get("request"))
        assert (request["request"]["kind"]) == ("error")
        assert ("questions") not in (request["request"])
        assert (self.run.calls) == ([1, 2])
        failures = list(self.run.directory.glob("**/runtime-attempts/*/runtime_validation.json"))
        assert (len(failures)) == (1)
        assert (json.loads(failures[0].read_text())["status"]) == ("error")
        assert (json.loads((self.run.directory / "metadata/runtime_validation.json").read_text())["status"]) == ("ok")

    @pytest.mark.parametrize("case", ["nonfinite", "oversized"])
    def test_validation_failure_archive_rejects_untrusted_report(self, case):
        from acprof.results.layout import ArtifactLayout

        root = self.fixture_root / f"archive-{case}"
        layout = ArtifactLayout.for_new_run(root)
        layout.initialize()
        report = layout.path("runtime_validation.json")
        if case == "nonfinite":
            raw = b'{"status":"error","ignored":NaN}'
        else:
            raw = b'{"status":"error","padding":"' + b"x" * MAX_JSON_ARTIFACT_BYTES + b'"}'
        report.write_bytes(raw)

        with pytest.raises(ValueError, match="invalid runtime validation JSON|runtime validation exceeds"):
            retain_validation_failure(str(root))

        assert not list((root / "logs/runtime-attempts").glob("*"))

    @pytest.mark.skipif(os.name != "posix", reason="directory fsync is a POSIX durability contract")
    def test_validation_failure_archive_syncs_files_and_directory_chain(self):
        import stat

        from acprof.results.layout import ArtifactLayout

        root = self.fixture_root / "archive-durability"
        layout = ArtifactLayout.for_new_run(root)
        layout.initialize()
        layout.path("runtime_validation.json").write_text(
            '{"status":"error","devices":{}}\n', encoding="utf-8"
        )
        synced_types = []
        synced_directories = []
        real_fsync = os.fsync

        def track_fsync(fd):
            file_type = stat.S_IFMT(os.fstat(fd).st_mode)
            synced_types.append(file_type)
            if file_type == stat.S_IFDIR:
                synced_directories.append(Path(os.readlink(f"/proc/self/fd/{fd}")).resolve())
            real_fsync(fd)

        with patch("os.fsync", side_effect=track_fsync):
            retain_validation_failure(str(root))

        archives = list((root / "logs/runtime-attempts").glob("*"))
        assert len(archives) == 1
        assert (archives[0] / "runtime_validation.json").is_file()
        assert stat.S_IFREG in synced_types
        assert any(path == archives[0].resolve() for path in synced_directories)
        assert root.resolve() in synced_directories

    def test_cancel_failed_validation_never_starts_measurement(self):
        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch(
            "sys.stdin", io.StringIO('{"id":1,"action":"cancel"}\n'),
        ), pytest.raises(KeyboardInterrupt):
            self.run.invoke(validation=FileNotFoundError("ultravox_config.py"))
        assert (self.run.calls) == ([])

    def test_resume_reuses_successful_validation_without_repeating_preparation(self):
        self.run.interrupt_after_first()
        output = io.StringIO()
        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch("sys.stdin", io.StringIO()):
            self.run.invoke("--resume", output=output, validation=AssertionError("validation repeated"))
        runtime = [item for item in events(output) if item["stage"] == "runtime"]
        assert ([item["status"] for item in runtime]) == (["passed"])
        assert (self.run.calls) == ([2])

    @pytest.mark.parametrize("case", ["nonfinite", "oversized"])
    def test_resume_rejects_untrusted_static_metadata_before_measurement(self, case):
        self.run.interrupt_after_first()
        meta_path = self.run.directory / "static_meta.json"
        meta = json.loads(meta_path.read_text())
        if case == "nonfinite":
            meta["corrupt_metric"] = float("nan")
        else:
            meta["padding"] = "x" * MAX_JSON_ARTIFACT_BYTES
        meta_path.write_text(json.dumps(meta))
        from acprof.host.run_state import file_sha256
        state_path = self.run.directory / ".acprof/run_state.json"
        state = json.loads(state_path.read_text())
        state["artifacts"]["static_meta.json"] = file_sha256(meta_path)
        state_path.write_text(json.dumps(state))
        calls_before = list(self.run.calls)
        with pytest.raises(SystemExit) as raised:
            self.run.invoke("--resume")

        assert raised.value.code == 2
        assert self.run.calls == calls_before

    def test_resume_does_not_treat_resource_limit_as_successful_validation(self):
        def interrupted(request, **_kwargs):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            self.run.invoke(case=interrupted)
        meta_path = self.run.directory / "static_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["runtime_validation"] = {"status": "resource_limit"}
        meta_path.write_text(json.dumps(meta))
        from acprof.host.run_state import file_sha256
        state_path = self.run.directory / ".acprof/run_state.json"
        state = json.loads(state_path.read_text())
        state["artifacts"]["static_meta.json"] = file_sha256(meta_path)
        state_path.write_text(json.dumps(state))
        output = io.StringIO()
        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch("sys.stdin", io.StringIO()):
            with pytest.raises(RuntimeError, match="saved runtime validation is not successful"):
                self.run.invoke("--resume", output=output)
        runtime = [item for item in events(output) if item["stage"] == "runtime"]
        assert ([item["status"] for item in runtime]) == (["failed"])
        assert (self.run.calls) == ([])

    def test_inspect_separates_resolution_runtime_and_measurement(self):
        task = contracts.TestModelContract().discover()
        task.model_resolution["runtime_validation"] = {
            "status": "error", "devices": {"off": {"error": "FileNotFoundError: ultravox_config.py"}},
        }
        summary = explain_resolution(task)
        assert ("Interface Resolution: resolved") in (summary)
        assert ("Runtime Validation: failed") in (summary)
        assert ("Measurement: not_started") in (summary)
        assert ("ultravox_config.py") in (summary)

    @staticmethod
    def args():
        return SimpleNamespace(model="arbitrary/audio-model", task=None, task_family=None, backend=None,
                               model_spec=None, revision=None, batch_size=1, gpus="off")

    def test_answer_continues_resolution_and_same_evidence_reuses_decision(self):
        task = contracts.TestModelContract().discover(contracts.SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["turns"]'))
        answer = {"multimodal.inputs.turns": {"template": [{"role": "user", "content": {"from": "text"}}]}}
        workflow = PreparationWorkflow(interactive=True, cache_dir=self.run.root / "decisions")
        with patch("acprof.host.detect.detect_task", return_value=task), patch.object(
            workflow, "ask", side_effect=answer_with_parameter_defaults(
                {"action": "answer", "answers": answer}, {"action": "confirm"}),
        ) as ask, patch("sys.stdout", io.StringIO()):
            reviewed = workflow.resolve(self.args())
            assert ask.call_count == 3
        assert (task_model_spec(reviewed)["multimodal"]["inputs"]["turns"]) == (answer["multimodal.inputs.turns"])
        with patch("acprof.host.detect.detect_task", return_value=task), patch.object(
            workflow, "ask", side_effect=answer_with_parameter_defaults(),
        ) as ask, patch("sys.stdout", io.StringIO()):
            cached = workflow.resolve(self.args())
            assert ask.call_count == 1
            assert ask.call_args.kwargs["purpose"] == "workload_parameters"
        assert (task_model_spec(cached)) == (task_model_spec(reviewed))
        changed = copy.deepcopy(task)
        changed.model_revision = "b" * 40
        with patch("acprof.host.detect.detect_task", return_value=changed), patch.object(
            workflow, "ask", side_effect=KeyboardInterrupt,
        ) as ask, patch("sys.stdout", io.StringIO()), pytest.raises(KeyboardInterrupt):
            workflow.resolve(self.args())
        ask.assert_called_once()

    @pytest.mark.parametrize("corruption", ["nonfinite", "top_level", "oversized"])
    def test_invalid_decision_cache_is_ignored_before_review(self, corruption):
        task = contracts.TestModelContract().discover(
            contracts.SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["turns"]')
        )
        answer = {"multimodal.inputs.turns": {"template": [
            {"role": "user", "content": {"from": "text"}}
        ]}}
        cache_dir = self.run.root / "decisions"
        workflow = PreparationWorkflow(interactive=True, cache_dir=cache_dir)
        with patch("acprof.host.detect.detect_task", return_value=task), patch.object(
            workflow, "ask", side_effect=answer_with_parameter_defaults(
                {"action": "answer", "answers": answer}, {"action": "confirm"}),
        ), patch("sys.stdout", io.StringIO()):
            workflow.resolve(self.args())

        cache = next(cache_dir.glob("*.json"))
        if corruption == "nonfinite":
            content = cache.read_text().rstrip()
            cache.write_text(content[:-1] + ', "ignored": NaN}\n')
        elif corruption == "top_level":
            cache.write_text("[]\n")
        else:
            saved = json.loads(cache.read_text())
            saved["padding"] = "x" * MAX_MESSAGE
            cache.write_text(json.dumps(saved))
            assert cache.stat().st_size > MAX_MESSAGE

        with patch("acprof.host.detect.detect_task", return_value=task), patch.object(
            workflow, "ask", side_effect=KeyboardInterrupt,
        ) as ask, patch("sys.stdout", io.StringIO()), pytest.raises(KeyboardInterrupt):
            workflow.resolve(self.args())
        ask.assert_called_once()

    def test_metadata_and_known_dependency_access_errors_are_not_choices(self):
        task = contracts.TestModelContract().discover()
        task.metadata_errors = ["repository permission denied"]
        assert (PreparationWorkflow.questions(task, self.args())) == ([])
        source = contracts.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer")'
        def denied(*_args, **_kwargs):
            raise PermissionError("permission denied")
        task = contracts.TestModelContract().discover(source, dependency_lookup=denied)
        assert (PreparationWorkflow.questions(task, self.args())) == ([])
        mixed = contracts.TestModelContract().discover(source + '\nAutoTokenizer.from_pretrained(dynamic_repo())', dependency_lookup=denied)
        assert (PreparationWorkflow.questions(mixed, self.args())) == ([])

    def test_explicit_task_is_not_replaced_by_cache_or_inference(self):
        task = contracts.TestModelContract().discover()
        args = self.args()
        args.task = "text-generation"
        workflow = PreparationWorkflow(cache_dir=self.run.root / "decisions")
        with patch("acprof.host.detect.detect_task", return_value=task), pytest.raises(ValueError, match="explicit --task"):
            workflow.resolve(args)
        assert not (list(Path(self.run.root).rglob("decisions/*.json")))

    @pytest.mark.parametrize('reply_case', range(2), ids=["('', KeyboardInterrupt)", '(\'{"id":4,"action":"retry"}\\n\', ValueError)'])
    def test_controller_disconnect_and_stale_answer_cannot_start_collection(self, reply_case):
        (reply, error) = tuple((('', KeyboardInterrupt), ('{"id":4,"action":"retry"}\n', ValueError)))[reply_case]
        with patch("sys.stdin", io.StringIO(reply)), patch("sys.stdout", io.StringIO()):
            workflow = PreparationWorkflow(interactive=True)
            with pytest.raises(error):
                workflow.ask("runtime", "error", detail="missing file")

    def test_controller_reply_rejects_nonfinite_json(self):
        reply = '{"id":1,"action":"retry","ignored":NaN}\n'
        with patch("sys.stdin", io.StringIO(reply)), patch("sys.stdout", io.StringIO()):
            workflow = PreparationWorkflow(interactive=True)
            with pytest.raises(ValueError, match="non-finite number"):
                workflow.ask("runtime", "error", detail="missing file")

    @pytest.mark.parametrize("stage", ["image", "runtime"])
    def test_rebuild_request_is_explicit_and_invalidates_preparation(self, stage):
        from acprof.host.preparation.collection_workflow import RebuildEnvironment

        workflow = PreparationWorkflow(interactive=True)
        with patch("sys.stdin", io.StringIO('{"id":1,"action":"rebuild"}\n')), pytest.raises(RebuildEnvironment):
            workflow.ask(stage, "error", detail="image verification failed")
        assert workflow.rebuild_environment

    def test_explicit_rebuild_invalidates_image_and_plan_but_keeps_resolution(self):
        output = io.StringIO()
        attempts = []
        def validate(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise FileNotFoundError("image code needs rebuilding")
            return {"status": "ok"}
        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch(
            "sys.stdin", io.StringIO('{"id":1,"action":"rebuild"}\n'),
        ):
            self.run.invoke(validation=validate, output=output)
        records = events(output)
        for stage, count in (("resolution", 1), ("image", 2), ("input", 2), ("runtime", 2)):
            assert (sum(item["stage"] == stage and item["status"] == "running" for item in records)) == (count)
        assert (self.run.calls) == ([1, 2])

    def test_conflicting_metadata_requests_configuration_without_task_menu(self):
        from acprof.models.candidates import apply_model_candidate, discover_model_candidates
        task = copy.deepcopy(self.run.task)
        task.hub_metadata = {"pipeline_tag": "fill-mask", "transformers_info": {"pipeline_tag": "text-classification"}}
        task.model_resolution = apply_model_candidate(task, discover_model_candidates(task))
        assert (task.model_resolution["status"]) == ("ambiguous")
        workflow = PreparationWorkflow(interactive=True)
        with patch("acprof.host.detect.detect_task", return_value=task), patch.object(
            workflow, "ask", side_effect=KeyboardInterrupt,
        ) as ask, patch("sys.stdout", io.StringIO()), pytest.raises(KeyboardInterrupt):
            workflow.resolve(self.args())
        questions = ask.call_args.kwargs["questions"]
        assert [item["path"] for item in questions] == ["model_spec"]
        assert not any("options" in question for question in questions)
        assert not ask.call_args.kwargs.get("advanced")
        assert "需要补充配置" in ask.call_args.kwargs["summary"]
        assert "ambiguous" not in ask.call_args.kwargs["summary"]
        assert "Task" not in ask.call_args.kwargs["fields"]
    def test_resolution_failure_preserves_typed_error_in_preparation_request(self):
        output = io.StringIO()
        workflow = PreparationWorkflow(interactive=True)
        with patch("huggingface_hub.HfApi.model_info", side_effect=hub_error(RepositoryNotFoundError, 404)), patch(
            "sys.stdin", io.StringIO('{"id":1,"action":"cancel"}\n'),
        ), patch("sys.stdout", output), pytest.raises(KeyboardInterrupt):
            workflow.run("resolution", detect_task, "asdf")
        request = next(item["request"] for item in events(output) if item.get("request"))
        assert (request["model_error"]["reason_code"]) == ("repository_unavailable")
        assert (request["model_error"]["model_id"]) == ("asdf")
        assert ("SystemExit") not in (request["detail"])
