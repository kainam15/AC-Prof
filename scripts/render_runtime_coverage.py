"""Render the complete Docker/Process audit from the sole metric registry."""
import argparse
import sys
from dataclasses import astuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.results.coverage import collector_contract, metric_coverage  # noqa: E402
from acprof.results.metrics import METRICS  # noqa: E402


def render() -> str:
    lines = ["# Docker / Process 指标覆盖矩阵", "",
             "由 `scripts/render_runtime_coverage.py` 从 [指标登记表](../../acprof/results/metrics.py) 和",
             "[采集来源契约](../../acprof/results/coverage.py) 生成。语义与实施边界见 [统一画像](runtime-profile.md)。", "",
             f"共 {len(METRICS)} 个登记字段，包含身份、诊断和数值，不能把字段数当作可采集指标数。",
             "`conditional` 仅表示存在实现路径；实际可采集性由权限、设备和当前运行证据决定。",
             "`unsupported` 表示该 Runtime 未接入。所有条目的硬件验收均为 `not_verified`；",
             "本表不凭代码、Mock 或历史不同版本实验宣称已通过硬件验证。", "",
             "| 来源 | 范围 | 成本级别 | 条件 / 权限 | 生产者 |",
             "| --- | --- | --- | --- | --- |"]
    contracts = {(metric.source, collector_contract(metric)) for metric in METRICS.values()}
    for source, contract in sorted(contracts, key=lambda item: (item[0], *astuple(item[1]))):
        lines.append(f"| `{source}` | `{contract.scope}` | {contract.cost} | {contract.prerequisites} | [{Path(contract.producer).name}](../../{contract.producer}) |")
    lines.extend(["", "| 指标 | Docker | Process | 单位 | 来源 | 窗口 | 推导 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"])
    for name, metric in METRICS.items():
        docker, process = (metric_coverage(name, kind) for kind in ("docker", "process"))
        values = (name, docker["collectable"], process["collectable"], metric.unit, metric.source,
                  metric.window, docker["derivation"])
        lines.append("| " + " | ".join(f"`{value.replace('|', ' / ')}`" for value in values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "docs/results/runtime-coverage.md"
    content = render()
    if args.check:
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            print("Run python scripts/render_runtime_coverage.py", file=sys.stderr)
            return 1
    else:
        path.write_text(content, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
