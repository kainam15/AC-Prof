"""Safe layer generation, validation and on-demand wide CSV export."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from acprof.result_layers import export_result_layers, read_result_layers


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="acprof results",
        description="Verify experiment layers and explicitly export a wide CSV",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    verify = actions.add_parser("verify", help="Validate integrity and join keys")
    verify.add_argument("directory", type=Path)
    export = actions.add_parser("export", help="Reconstruct a wide CSV on demand")
    export.add_argument("directory", type=Path)
    export.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "verify":
            fields, rows = read_result_layers(args.directory)
            print(f"[results] Verified {len(rows)} measurements and {len(fields)} fields")
            return 0
        count = export_result_layers(args.directory, args.destination)
        print(f"[results] Exported {count} measurements to {args.destination}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"[results][ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
