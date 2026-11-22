import csv

from acprof.analysis.compatibility import write_compatibility_report


def test_exports_distinguish_requested_task_and_activated_profile(tmp_path):
    rows = [
        {"model_id": "fixture/translated", "requested_task_id": "translate-fr",
         "selected_task_id": "profile-fr", "task_selection_source": "explicit"},
        {"model_id": "fixture/rejected", "requested_task_id": "missing",
         "selected_task_id": None, "selected_task": "text2text-generation"},
        {"model_id": "fixture/legacy", "selected_task": "text-generation"},
    ]
    write_compatibility_report(tmp_path, rows)
    with (tmp_path / "models.csv").open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert [(row["requested_task_id"], row["selected_task_id"], row["task_selection_source"])
            for row in exported] == [("translate-fr", "profile-fr", "explicit"), ("missing", "", ""), ("", "", "")]

    table = (tmp_path / "REPORT.md").read_text().splitlines()
    assert table[4].endswith("| Requested task | Selected profile |")
    translated, rejected, legacy = table[-3:]
    assert translated.endswith("| translate-fr | profile-fr |")
    assert rejected.endswith("| missing |  |")
    assert legacy.endswith("|  |  |")
