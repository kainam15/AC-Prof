"""Only distinct runs count as independent replicates; failed samples remain visible."""
import csv
import shutil

import comparison_fixtures as comparison_fixture


class IndependentComparisonFixture:
    def build(self, request, tmp_path):
        self._request = request
        self.fixture_root = tmp_path
        fixture = comparison_fixture.ComparisonFixture()
        fixture.build(self.fixture_root)
        self.fixture = fixture

    def replicate(self, side, index, values, *, failed=False):
        fixture = self.fixture
        path = fixture.root / f"{side}-{index}"
        shutil.copytree(fixture.left if side == "left" else fixture.right, path)
        fixture.change_json(path, "run_state.json", lambda state: state.update(run_id=f"{side}-{index}"))
        with (path / "result_all.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, template = reader.fieldnames, next(reader)
        rows = [{**template, "repeat_idx": i, "latency_app_s": value, "repeat_in_window": 3}
                for i, value in enumerate(values)]
        if failed:
            rows[-1].update(status="error", error="request failure", latency_app_s="nan")
        fixture.change_json(path, "run_state.json", lambda state: state["options"].update(repeat=len(rows)))
        with (path / "result_all.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def compare(self, left, right):
        from acprof.analysis.independent_comparison import compare_experiments
        return compare_experiments(left, right, metrics=["latency_app_s"], resamples=200, seed=3)
