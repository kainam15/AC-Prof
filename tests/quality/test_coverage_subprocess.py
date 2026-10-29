import os
import subprocess
import sys
import textwrap


def test_coverage_subprocess_patch_collects_child_source(tmp_path):
    (tmp_path / "child.py").write_text(
        "value = 2\n"
        "if value > 1:\n"
        "    result = value + 1\n"
        "else:\n"
        "    result = 0\n"
        "print(result)\n",
        encoding="utf-8",
    )
    (tmp_path / "parent.py").write_text(
        "import subprocess\n"
        "import sys\n"
        "subprocess.run([sys.executable, 'child.py'], check=True)\n",
        encoding="utf-8",
    )
    (tmp_path / ".coveragerc").write_text(
        textwrap.dedent(
            """\
            [run]
            branch = True
            parallel = True
            patch = subprocess
            relative_files = True
            source = child.py
            """
        ),
        encoding="utf-8",
    )
    data_file = tmp_path / ".coverage"
    env = dict(os.environ, COVERAGE_FILE=str(data_file))
    subprocess.run(
        [sys.executable, "-m", "coverage", "run", "parent.py"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [sys.executable, "-m", "coverage", "combine", "--keep"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = subprocess.run(
        [sys.executable, "-m", "coverage", "report", "-m", "child.py"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "child.py" in report.stdout
    assert "No source for child.py" not in report.stderr
