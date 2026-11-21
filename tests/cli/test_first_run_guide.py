import json
import re
import shlex
from pathlib import Path

from acprof.cli import audit
from acprof.experiments.arguments import build_parser
from acprof.experiments.configuration import RunConfig
from tests.support.comparison_fixtures import ComparisonFixture


def test_first_cpu_run_and_audit_commands_in_the_guide(tmp_path, monkeypatch, capsys):
    guide = Path(__file__).resolve().parents[2] / "docs/usage/experiments.md"
    first_section = guide.read_text(encoding="utf-8").split("\n## ", 2)[1]
    commands = [shlex.split(block.replace("\\\n", " "))
                for block in re.findall(r"```bash\n(.*?)\n```", first_section, re.S)]
    parser = build_parser()
    runs = [RunConfig.from_namespace(parser.parse_args(command[2:])) for command in commands
            if command[:2] == ["acprof", "run"]]
    [config] = [run for run in runs if run.gpus == "off"]
    assert (config.profiling_mode, config.cpus, config.mems, config.gpus) == ("best_effort", "1", "4", "off")
    assert (config.warmup, config.repeat, config.repeat_in_window) == (0, 1, 1)
    assert (config.compute_profile_tool, config.execution_profile_tool) == ("none", "none")
    [audit_command] = [command for command in commands if command[:2] == ["acprof", "audit"]]
    fixture = ComparisonFixture()
    fixture.build(tmp_path / "fixture")
    result_dir = config.result_dir(tmp_path)
    result_dir.parent.mkdir(parents=True)
    fixture.left.rename(result_dir)
    monkeypatch.chdir(tmp_path)
    assert audit.main(audit_command[2:]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["accepted"]
    assert report["counts"]["formal_ok"] == 1
    assert "cpu_energy_total_j" in report["missing_metrics"]
