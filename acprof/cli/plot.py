"""Plot CLI: exclude collection before loading the rendering pipeline."""
from __future__ import annotations

import sys
from contextlib import ExitStack

CSV_PATH = "results/result_layers.json"


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help"}:
        print("usage: acprof plot [result_csv]")
        print()
        print("Plot AC-Prof result CSV files.")
        print()
        print("positional arguments:")
        print("  result  Experiment directory or result_layers.json")
        return 0

    from acprof.host.run_state import RunStateError, offline_analysis_lock

    with ExitStack() as stack:
        try:
            stack.enter_context(offline_analysis_lock())
        except RunStateError as error:
            print(f"绘图失败：{error}", file=sys.stderr)
            return 1
        from acprof.plotting.pipeline import plot_results

        plot_results(args[0] if args else CSV_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
