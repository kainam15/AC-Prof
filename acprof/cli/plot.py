"""Plot CLI: exclude collection before loading the rendering pipeline."""
from __future__ import annotations

import sys
from contextlib import ExitStack

DEFAULT_RESULT_DIR = "results"


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help"}:
        print("usage: acprof plot [result]")
        print()
        print("Plot AC-Prof result CSV files.")
        print()
        print("positional arguments:")
        print("  result  Experiment directory, result_layers.json, or result CSV (default: results/)")
        return 0

    from acprof.host.run_state import RunStateError, offline_analysis_lock

    with ExitStack() as stack:
        try:
            stack.enter_context(offline_analysis_lock())
        except RunStateError as error:
            print(f"绘图失败：{error}", file=sys.stderr)
            return 1
        from acprof.plotting.data import PlotInputNotFound
        from acprof.plotting.pipeline import plot_results

        try:
            plot_results(args[0] if args else DEFAULT_RESULT_DIR)
        except PlotInputNotFound as error:
            print(f"绘图失败：找不到输入 {error.filename!r}；请检查结果路径或实验是否已生成结果。", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
