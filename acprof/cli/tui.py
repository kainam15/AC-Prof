"""TUI 命令参数与启动。"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import replace


def _build_parser() -> argparse.ArgumentParser:
    from acprof.hub.repository import MODEL_SOURCES

    parser = argparse.ArgumentParser(
        description="AC-Prof interactive terminal interface",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Pre-fill the repository ID for the selected model source",
    )
    parser.add_argument("--model-source", choices=MODEL_SOURCES, help="Pre-fill the model source")
    parser.add_argument(
        "--preset",
        choices=("smoke", "main"),
        default=None,
        help="Initial form preset (overrides saved experiment defaults)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Initial result directory (overrides the preset or saved directory)",
    )
    parser.add_argument(
        "--color-system",
        choices=("truecolor", "256", "auto"),
        default="truecolor",
        help="Terminal colors: RGB by default; use 256 for older terminals or auto for environment detection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(cli_args)
    # Rich caches its first width table. Match VS Code's Unicode 11 default,
    # but let Windows Terminal use modern widths (Unicode 16+ makes ☰ wide).
    # WT_SESSION identifies a terminal family, not its Unicode version.
    if os.environ.get("TERM_PROGRAM") == "vscode":
        os.environ.setdefault("UNICODE_VERSION", "11.0.0")
    # Help and argument errors must not initialize the TUI runtime.
    from acprof.experiments.configuration import RunConfigError
    from acprof.tui.app import AcprofTui

    try:
        app = AcprofTui(color_system=args.color_system)
    except ValueError as exc:
        parser.error(str(exc))
    config = app.configuration.startup_config(model=args.model, model_source=args.model_source)
    if args.preset:
        try:
            config = config.with_preset(args.preset)
        except RunConfigError as exc:
            parser.error(str(exc))
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    app.configuration.initial_config = config
    app.run()
    if app.updates.restart_command:
        # Textual has restored the terminal before replacing this process.
        command = [*app.updates.restart_command, *cli_args]
        try:
            os.execv(command[0], command)
        except OSError as exc:
            import shlex
            print(f"Restart failed: {exc}\nRun: {shlex.join(command)}")


if __name__ == "__main__":
    main()
