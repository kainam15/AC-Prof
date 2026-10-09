"""Safe layer generation, validation and on-demand wide CSV export."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from acprof.artifact_layout import ArtifactLayout
from acprof.result_layers import (
    export_result_layers,
    publish_result_layers,
    read_result_layers,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="acprof results",
        description="Split, verify and export joinable result CSV layers",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    split = actions.add_parser("split", help="Split an existing result_all.csv")
    split.add_argument("source", type=Path, help="Result directory or wide result CSV")
    split.add_argument("--output-dir", type=Path, default=None,
                       help="New, empty directory for non-destructive historical migration")
    verify = actions.add_parser("verify", help="Validate integrity and join keys")
    verify.add_argument("directory", type=Path)
    export = actions.add_parser("export", help="Reconstruct a wide CSV on demand")
    export.add_argument("directory", type=Path)
    export.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "split":
            source = args.source / "result_all.csv" if args.source.is_dir() else args.source
            if not source.is_file():
                parser.error(f"missing result CSV: {source}")
            root = source.parent
            layout = ArtifactLayout.discover(root)
            if args.output_dir is None and layout.layout_version != 2:
                parser.error("historical flat results require --output-dir to preserve originals")
            if args.output_dir is not None:
                destination = args.output_dir
                if destination.resolve() == root.resolve() or (
                    destination.exists() and any(destination.iterdir())
                ):
                    parser.error("--output-dir must be a different, empty directory")
            else:
                destination = root
            manifest = publish_result_layers(source, output_dir=destination)
            print(f"[results] Published {len(manifest['layers'])} layers in {destination}")
            return 0
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
