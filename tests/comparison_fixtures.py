"""跨后端比较条件与续跑身份独立；审计只读取已有产物。"""
import copy
import csv
import importlib
import json

from acprof.config import CSV_FIELDS
from acprof.platform import Environment
from acprof.run_args import build_parser


class ComparisonFixture:
    def build(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.left, self.right = self.root / "torch", self.root / "onnx"
        self.contract = {
            "schema_version": 1, "task": "tabular-regression", "batch_size": 1,
            "scenario": {"type": "serial"},
            "input": {"planned_scale": 2, "actual_scale": 2, "rows": 2, "feature_dim": 2},
            "output": {"type": "regression", "shape": [2, 1], "count": 2},
        }
        for directory, backend in ((self.left, "torch"), (self.right, "onnxruntime")):
            self.write_experiment(directory, backend)

    def write_json(self, directory, filename, payload):
        (directory / filename).write_text(json.dumps(payload))

    def change_json(self, directory, filename, update):
        path = directory / filename
        payload = json.loads(path.read_text())
        update(payload)
        path.write_text(json.dumps(payload))

    def write_experiment(self, directory, backend):
        directory.mkdir()
        options = vars(build_parser().parse_args([
            "--model", "example/source", "--profiling-mode", "basic", "--cpus", "1", "--mems", "4",
            "--gpus", "off", "--warmup", "0", "--repeat", "1", "--backend", backend,
        ]))
        options["measurement_environment"] = {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "ACPROF_RUNTIME_THREADS": "1",
        }
        self.write_json(directory, "run_state.json", {
            "schema_version": 1, "run_id": backend, "status": "complete", "options": options,
            "host": {"source_sha256": backend, "packages_sha256": backend},
            "runtime": {"planned": {"scales": [2]}},
        })
        self.write_json(directory, "input_scale_plan.json", {
            "schema_version": 2, "model_id": f"example/{backend}", "pipeline_tag": "tabular-regression",
            "task_family": "structured", "scenario": {"type": "serial"},
            "quality_constraints": {"reference_id": "known-affine-v1", "atol": 1e-6, "rtol": 1e-5},
            "entries": [{"input_scale": 2, "payload": {"features": [[1, 2], [3, 4]], "batch_size": 1}}],
        })
        self.write_json(directory, "static_meta.json", {
            **Environment("native_linux").metadata(),
            "schema_version": 7, "model_name": f"example/{backend}", "model_revision": backend,
            "runtime_backend": backend, "image_id": f"sha256:{backend}",
            "runtime_environment": {"environment_id": backend},
            "runtime_validation": {"devices": {"off": {"status": "ok", "runtime_parameters": {
                "effective": {"threads": 1},
            }}}},
            "cgroup_version": "2", "cgroup_collection_mode": "v2",
        })
        self.write_json(directory, "hardware_conditions.json", {"schema_version": 1, "cases": {
            "1c_4g_off": {"host_id": "fixture-host", "cpu_model": ["fixture CPU"],
                         "cpu_affinity": ["0-3"], "cpu_policy": {"governor": "performance", "boost": "0"},
                         "gpu": "not_applicable", "runtime_threads": 1}
        }})
        self.write_rows(directory, self.contract)

    def write_rows(self, directory, contract, *, count=3):
        row = {**dict.fromkeys(CSV_FIELDS, "nan"), "cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off",
               "environment_class": "native_linux",
               "latency_app_s": 0.25,
               "input_scale": 2, "warmup": 0, "repeat_idx": 0, "status": "ok", "error": "",
               "workload_contract": json.dumps({"schema_version": 1, "request_count": count,
                    "variants": [{"count": count, "contract": contract}]})}
        with (directory / "result_all.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, CSV_FIELDS)
            writer.writeheader()
            writer.writerow(row)

    def compare(self):
        module = importlib.import_module("acprof.analysis.comparison")
        return module.compare_results(self.left, self.right)

    def write_distribution(self, directory, counts, *, output_only=False, task=None):
        variants = []
        for index, count in enumerate(counts):
            contract = copy.deepcopy(self.contract)
            if task:
                contract["task"] = task
            section, name = ("output", "count") if output_only else ("input", "actual_scale")
            contract[section][name] = index + 1
            variants.append({"count": count, "contract": contract})
        with (directory / "result_all.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["workload_contract"] = json.dumps({"schema_version": 1,
            "request_count": sum(counts), "variants": variants})
        with (directory / "result_all.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
