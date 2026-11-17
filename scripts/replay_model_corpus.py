"""Replay frozen local model metadata and fail on unreviewed decision changes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Frozen corpus with reviewed expected decisions")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for reports and coverage evidence")
    parser.add_argument("--baseline", type=Path, help="Prior corpus.json for the same frozen inputs")
    args = parser.parse_args(argv)
    from acprof.testing.model_corpus import replay_corpus
    try:
        report = replay_corpus(args.manifest, args.output_dir, baseline=args.baseline)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"[corpus][ERROR] {exc}", file=sys.stderr)
        return 2
    summary = report["summary"]
    print(f"Static corpus: {summary['cases']} cases, {summary['matched']} matched, "
          f"{summary['mismatched']} mismatches; {len(report['changes'])} baseline changes")
    print(f"Report: {args.output_dir / 'CORPUS.md'}")
    return 1 if summary["mismatched"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
