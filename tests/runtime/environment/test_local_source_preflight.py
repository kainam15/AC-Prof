"""Local import paths must not hide dependencies in a pinned custom model."""
from collections import Counter
from unittest.mock import Mock

import pytest

from acprof.failures import RuntimeFailure
from acprof.models.evidence import ModelEvidence
from acprof.models.inspection.metadata import collect_source_evidence
from acprof.runtime.dependencies import dependency_preflight
from acprof.runtime.profiles import PROFILES
from tests.runtime.environment.test_runtime_preflight import model


def custom_task(files, entry="entry.Model"):
    task = model("image-classification", "cv", "vit", repository_files=tuple(files))
    task.model_config["auto_map"] = {"AutoModel": entry}
    return task


@pytest.mark.parametrize("statement", [
    "from .helpers import codec as decoder", "from helpers import codec",
    "import helpers.codec as decoder", "from helpers.codec import decode",
])
def test_nested_source_reports_its_missing_distribution(statement):
    files = {"entry.py": statement, "helpers/__init__.py": "", "helpers/codec.py": "import skimage"}
    task = custom_task(files)
    reads = Mock(side_effect=files.__getitem__)
    with pytest.raises(RuntimeFailure) as caught:
        dependency_preflight(task, PROFILES["cv-cpu"], read_source=reads)
    failure = caught.value.failure
    assert failure.reason_code == "runtime_dependency_missing"
    [problem] = failure.evidence["dependencies"]
    assert (problem["distribution"], problem["source"]) == ("scikit-image", "helpers/codec.py")
    assert set(failure.evidence["source_sha256"]) == set(files)
    assert Counter(call.args[0] for call in reads.call_args_list) == Counter(files.keys())


def test_entry_package_initializers_are_evidence_even_without_an_explicit_import():
    files = {"outer/pkg/model.py": "class Model: pass", "outer/pkg/__init__.py": "",
             "outer/__init__.py": "import skimage", "unrelated.py": "import detectron2"}
    task = custom_task(files, "outer.pkg.model.Model")
    evidence = ModelEvidence(task.model_id, task.model_revision)
    graph = collect_source_evidence(task, evidence, task.model_config, files.__getitem__)
    expected = {"outer/__init__.py", "outer/pkg/__init__.py", "outer/pkg/model.py"}
    assert set(graph) == expected
    with pytest.raises(RuntimeFailure) as caught:
        dependency_preflight(task, PROFILES["cv-cpu"], read_source=files.__getitem__)
    failure = caught.value.failure
    assert failure.reason_code == "runtime_dependency_missing"
    assert set(failure.evidence["source_sha256"]) == expected
    assert [item["source"] for item in failure.evidence["dependencies"]] == ["outer/__init__.py"]


@pytest.mark.parametrize("required", [False, True])
def test_guarded_local_imports_do_not_hide_or_overstate_missing_dependencies(required):
    files = {"entry.py": "if flag:\n    from helpers import codec\nif False:\n    import unused\n",
             "helpers/__init__.py": "", "helpers/codec.py": "from entry import Model\nimport skimage",
             "also_needed.py": "import helpers.codec", "unused.py": "import detectron2"}
    if required:
        files["entry.py"] += "import also_needed\n"
    with pytest.raises(RuntimeFailure) as caught:
        dependency_preflight(custom_task(files), PROFILES["cv-cpu"], read_source=files.__getitem__)
    failure = caught.value.failure
    assert failure.reason_code == ("runtime_dependency_missing" if required else "runtime_dependency_unknown")
    [problem] = failure.evidence["dependencies"]
    assert problem["module"] == "skimage"
    assert problem["conditional"] is not required
    assert "unused.py" not in failure.evidence["source_sha256"]


def test_proven_package_attribute_does_not_require_an_unimported_sibling():
    files = {"entry.py": "from helpers import codec", "helpers/__init__.py": "codec = 1",
             "helpers/codec.py": "import skimage", "helpers.py": "import detectron2"}
    report = dependency_preflight(custom_task(files), PROFILES["cv-cpu"], read_source=files.__getitem__)
    assert set(report["source_sha256"]) == {"entry.py", "helpers/__init__.py"}
    assert report["dependencies"] == []


def test_partially_initialized_package_does_not_hide_a_possible_submodule():
    files = {"entry.py": "from helpers import codec", "helpers/__init__.py": "from . import setup\ncodec = 1",
             "helpers/setup.py": "from . import codec", "helpers/codec.py": "import skimage"}
    task = custom_task(files)
    evidence = ModelEvidence(task.model_id, task.model_revision)
    graph = collect_source_evidence(task, evidence, task.model_config, files.__getitem__)
    assert set(graph) == set(files)
    with pytest.raises(RuntimeFailure) as caught:
        dependency_preflight(task, PROFILES["cv-cpu"], read_source=files.__getitem__)
    assert caught.value.failure.reason_code == "runtime_dependency_unknown"
    [problem] = caught.value.failure.evidence["dependencies"]
    assert (problem["module"], problem["source"], problem["conditional"]) == ("skimage", "helpers/codec.py", True)


def test_absolute_dotted_import_cannot_load_a_child_of_a_plain_module():
    files = {"entry.py": "import helpers.codec", "helpers.py": "", "helpers/codec.py": "import skimage"}
    with pytest.raises(RuntimeFailure) as caught:
        dependency_preflight(custom_task(files), PROFILES["cv-cpu"], read_source=files.__getitem__)
    assert caught.value.failure.reason_code == "runtime_dependency_unknown"
    assert "unresolved local import helpers.codec" in caught.value.failure.detail
    assert "helpers/codec.py" not in caught.value.failure.evidence["source_sha256"]


@pytest.mark.parametrize(("broken", "diagnostic"), [
    ("def broken(", "a.py"), (" " * (512 * 1024 + 1), "source exceeds dependency analysis budget"),
    (None, "read failed"),
], ids=["syntax", "oversize", "read-error"])
def test_failed_sources_count_towards_the_read_budget(monkeypatch, broken, diagnostic):
    files = {"entry.py": "from . import a, b, c", "a.py": broken, "b.py": "", "c.py": ""}

    def read(name):
        if files[name] is None:
            raise OSError("read failed")
        return files[name]

    reads = Mock(side_effect=read)
    monkeypatch.setattr("acprof.runtime.dependencies.MAX_FILES", 2)
    with pytest.raises(RuntimeFailure) as caught:
        dependency_preflight(custom_task(files), PROFILES["cv-cpu"], read_source=reads)
    assert [call.args[0] for call in reads.call_args_list] == ["entry.py", "a.py"]
    failure = caught.value.failure
    assert failure.reason_code == "runtime_dependency_unknown"
    assert diagnostic in failure.detail
    assert "source dependency file budget exceeded" in failure.detail
    assert set(failure.evidence["source_sha256"]) == {"entry.py"}


def test_unreadable_requirements_share_the_budget_with_python_sources(monkeypatch):
    files = {"requirements.txt": "", "entry.py": ""}
    reads = Mock(side_effect=OSError("requirements read failed"))
    monkeypatch.setattr("acprof.runtime.dependencies.MAX_FILES", 1)
    with pytest.raises(RuntimeFailure) as caught:
        dependency_preflight(custom_task(files), PROFILES["cv-cpu"], read_source=reads)
    reads.assert_called_once_with("requirements.txt")
    assert "requirements read failed" in caught.value.failure.detail
    assert "source dependency file budget exceeded" in caught.value.failure.detail


def test_repeated_package_lookup_keeps_the_first_read_error_until_the_next_preflight():
    files = {"entry.py": "import a, b", "a.py": "from helpers import codec", "b.py": "from helpers import codec",
             "helpers/codec.py": "", "helpers/__init__.py": ""}
    task = custom_task(files)
    counts = Counter()

    def read(name):
        counts[name] += 1
        if name == "helpers/__init__.py" and counts[name] == 1:
            raise OSError("first initializer read failed")
        return files[name]

    with pytest.raises(RuntimeFailure) as caught:
        dependency_preflight(task, PROFILES["cv-cpu"], read_source=read)
    assert caught.value.failure.reason_code == "runtime_dependency_unknown"
    assert "first initializer read failed" in caught.value.failure.detail
    assert counts["helpers/__init__.py"] == 1
    report = dependency_preflight(task, PROFILES["cv-cpu"], read_source=read)
    assert set(report["source_sha256"]) == set(files)
    assert counts["helpers/__init__.py"] == 2


def test_relative_import_cannot_be_satisfied_from_above_the_snapshot():
    files = {"entry.py": "from ..utils import Model", "utils.py": "class Model: pass"}
    reads = Mock(side_effect=files.__getitem__)
    with pytest.raises(RuntimeFailure) as caught:
        dependency_preflight(custom_task(files), PROFILES["cv-cpu"], read_source=reads)
    assert caught.value.failure.reason_code == "runtime_dependency_unknown"
    assert "relative import escapes pinned snapshot" in caught.value.failure.detail
    reads.assert_called_once_with("entry.py")
