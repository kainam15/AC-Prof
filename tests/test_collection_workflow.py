"""Preparation pauses and retries must keep the original collection alive."""
import io
import copy
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_run_recovery as recovery
import test_model_contract as contracts
from acprof.host.model_inspection import explain_resolution
from acprof.host.collection_workflow import PreparationWorkflow
from acprof.model_spec import task_model_spec


def events(output):
    return [json.loads(line.removeprefix("ACPROF_PREPARATION "))
            for line in output.getvalue().splitlines() if line.startswith("ACPROF_PREPARATION ")]


class CollectionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.run = recovery.RunRecoveryTests()
        self.run.setUp()
        self.addCleanup(self.run.doCleanups)

    def test_start_resolves_and_validates_before_measurement_without_questions(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch("sys.stdin", io.StringIO()):
            self.run.invoke(output=output)
        records = events(output)
        passed = [item["stage"] for item in records if item["status"] == "passed"]
        self.assertIn("resolution", passed)
        self.assertLess(passed.index("resolution"), passed.index("runtime"))
        self.assertFalse(any(item.get("request") for item in records))
        self.assertEqual(self.run.calls, [1, 2])

    def test_runtime_retry_keeps_resolved_model_image_and_input_plan(self):
        output = io.StringIO()
        attempts = []

        def validate(**kwargs):
            attempts.append(kwargs)
            self.assertFalse(self.run.calls, "measurement started before validation passed")
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
        self.assertEqual(len(attempts), 2)
        for key in ("task_info", "image_info", "planned"):
            self.assertIs(attempts[0][key], attempts[1][key])
        self.assertEqual(attempts[1]["cpu_list"], [1, 2])
        self.assertEqual(attempts[1]["gpu_list"], ["off"])
        records = events(output)
        for stage in ("resolution", "image", "input"):
            self.assertEqual(sum(item["stage"] == stage and item["status"] == "running" for item in records), 1)
        request = next(item for item in records if item.get("request"))
        self.assertEqual(request["request"]["kind"], "error")
        self.assertNotIn("questions", request["request"])
        self.assertEqual(self.run.calls, [1, 2])
        failures = list(self.run.directory.glob("**/runtime-attempts/*/runtime_validation.json"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(json.loads(failures[0].read_text())["status"], "error")
        self.assertEqual(json.loads((self.run.directory / "metadata/runtime_validation.json").read_text())["status"], "ok")

    def test_cancel_failed_validation_never_starts_measurement(self):
        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch(
            "sys.stdin", io.StringIO('{"id":1,"action":"cancel"}\n'),
        ), self.assertRaises(KeyboardInterrupt):
            self.run.invoke(validation=FileNotFoundError("ultravox_config.py"))
        self.assertEqual(self.run.calls, [])

    def test_resume_reuses_successful_validation_without_repeating_preparation(self):
        self.run.interrupt_after_first()
        output = io.StringIO()
        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch("sys.stdin", io.StringIO()):
            self.run.invoke("--resume", output=output, validation=AssertionError("validation repeated"))
        runtime = [item for item in events(output) if item["stage"] == "runtime"]
        self.assertEqual([item["status"] for item in runtime], ["passed"])
        self.assertEqual(self.run.calls, [2])

    def test_resume_does_not_treat_resource_limit_as_successful_validation(self):
        def interrupted(**_kwargs):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.run.invoke(case=interrupted, validation=lambda **_kwargs: {"status": "resource_limit"})
        output = io.StringIO()
        with patch.dict(os.environ, {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch("sys.stdin", io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "saved runtime validation is not successful"):
                self.run.invoke("--resume", output=output)
        runtime = [item for item in events(output) if item["stage"] == "runtime"]
        self.assertEqual([item["status"] for item in runtime], ["failed"])
        self.assertEqual(self.run.calls, [])

    def test_inspect_separates_resolution_runtime_and_measurement(self):
        task = contracts.ModelContractTests().discover()
        task.model_resolution["runtime_validation"] = {
            "status": "error", "devices": {"off": {"error": "FileNotFoundError: ultravox_config.py"}},
        }
        summary = explain_resolution(task)
        self.assertIn("Interface Resolution: resolved", summary)
        self.assertIn("Runtime Validation: failed", summary)
        self.assertIn("Measurement: not_started", summary)
        self.assertIn("ultravox_config.py", summary)

    @staticmethod
    def args():
        return SimpleNamespace(model="arbitrary/audio-model", task=None, task_family=None, backend=None,
                               model_spec=None, revision=None, batch_size=1)

    def test_answer_continues_resolution_and_same_evidence_reuses_decision(self):
        task = contracts.ModelContractTests().discover(contracts.SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["turns"]'))
        answer = {"multimodal.inputs.turns": {"template": [{"role": "user", "content": {"from": "text"}}]}}
        workflow = PreparationWorkflow(interactive=True, cache_dir=self.run.root / "decisions")
        with patch("acprof.host.detect.detect_task", return_value=task), patch.object(
            workflow, "ask", return_value={"action": "answer", "answers": answer},
        ) as ask, patch("sys.stdout", io.StringIO()):
            reviewed = workflow.resolve(self.args())
            ask.assert_called_once()
        self.assertEqual(task_model_spec(reviewed)["multimodal"]["inputs"]["turns"], answer["multimodal.inputs.turns"])
        with patch("acprof.host.detect.detect_task", return_value=task), patch.object(workflow, "ask") as ask, patch("sys.stdout", io.StringIO()):
            cached = workflow.resolve(self.args())
            ask.assert_not_called()
        self.assertEqual(task_model_spec(cached), task_model_spec(reviewed))
        changed = copy.deepcopy(task)
        changed.model_revision = "b" * 40
        with patch("acprof.host.detect.detect_task", return_value=changed), patch.object(
            workflow, "ask", side_effect=KeyboardInterrupt,
        ) as ask, patch("sys.stdout", io.StringIO()), self.assertRaises(KeyboardInterrupt):
            workflow.resolve(self.args())
        ask.assert_called_once()

    def test_metadata_and_known_dependency_access_errors_are_not_choices(self):
        task = contracts.ModelContractTests().discover()
        task.metadata_errors = ["repository permission denied"]
        self.assertEqual(PreparationWorkflow.questions(task, self.args()), [])
        source = contracts.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer")'
        def denied(*_args, **_kwargs):
            raise PermissionError("permission denied")
        task = contracts.ModelContractTests().discover(source, dependency_lookup=denied)
        self.assertEqual(PreparationWorkflow.questions(task, self.args()), [])
        mixed = contracts.ModelContractTests().discover(source + '\nAutoTokenizer.from_pretrained(dynamic_repo())', dependency_lookup=denied)
        self.assertEqual(PreparationWorkflow.questions(mixed, self.args()), [])

    def test_explicit_task_is_not_replaced_by_cache_or_inference(self):
        task = contracts.ModelContractTests().discover()
        args = self.args()
        args.task = "text-generation"
        workflow = PreparationWorkflow(cache_dir=self.run.root / "decisions")
        with patch("acprof.host.detect.detect_task", return_value=task), self.assertRaisesRegex(ValueError, "explicit --task"):
            workflow.resolve(args)
        self.assertFalse(list(Path(self.run.root).rglob("decisions/*.json")))

    def test_controller_disconnect_and_stale_answer_cannot_start_collection(self):
        for reply, error in (("", KeyboardInterrupt), ('{"id":4,"action":"retry"}\n', ValueError)):
            with self.subTest(reply=reply), patch("sys.stdin", io.StringIO(reply)), patch("sys.stdout", io.StringIO()):
                workflow = PreparationWorkflow(interactive=True)
                with self.assertRaises(error):
                    workflow.ask("runtime", "error", detail="missing file")

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
            self.assertEqual(sum(item["stage"] == stage and item["status"] == "running" for item in records), count)
        self.assertEqual(self.run.calls, [1, 2])

    def test_native_task_conflict_asks_only_task_and_preserves_choice(self):
        from acprof.model_resolution import discover_model_candidates
        task = copy.deepcopy(self.run.task)
        task.hub_metadata = {"pipeline_tag": "fill-mask", "transformers_info": {"pipeline_tag": "text-classification"}}
        task.model_resolution = discover_model_candidates(task)
        self.assertEqual(task.model_resolution["status"], "ambiguous")
        workflow = PreparationWorkflow(interactive=True)
        with patch("acprof.host.detect.detect_task", return_value=task), patch.object(
            workflow, "ask", return_value={"action": "answer", "answers": {"task": "text-classification"}},
        ) as ask, patch("sys.stdout", io.StringIO()):
            result = workflow.resolve(self.args())
        self.assertEqual([item["path"] for item in ask.call_args.kwargs["questions"]], ["task"])
        self.assertEqual(result.pipeline_tag, "text-classification")
        self.assertEqual(result.model_resolution["user_decisions"]["answers"], {"task": "text-classification"})
