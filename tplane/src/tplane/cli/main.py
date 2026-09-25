"""Parse the closed tp command line and dispatch to the subcommands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from tplane.cli.show import render_show
from tplane.export.otlp import render_otlp
from tplane.export.parquet import ParquetExportError, write_parquet
from tplane.files import FileExistsRefusal, publish_new_file, write_text
from tplane.integrity.metrics import IntegrityError, source_windows
from tplane.integrity.monitor import DEFAULT_INTEGRITY_OPTIONS, IntegrityOptions, detect_alarms
from tplane.integrity.report import render_integrity
from tplane.integrity.steps import Batching
from tplane.schema import Record
from tplane.store import RECORDS_DIRECTORY, FileRecordStore, StartedMarker, StoreError

EXIT_OK: Final[int] = 0
EXIT_ALARM: Final[int] = 1
EXIT_USAGE: Final[int] = 2
EXPORT_FORMATS: Final[tuple[str, ...]] = ("otlp", "parquet")


class CommandError(RuntimeError):
    """A tp command cannot run on the given run directory or settings."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run tp and return the exit code; prints messages, never stack traces."""
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "show":
            return _show(arguments.run_dir, failures_only=arguments.failures_only)
        if arguments.command == "integrity":
            options = IntegrityOptions(
                window_units=arguments.window_units,
                healthy_kl=arguments.healthy_kl,
                broken_kl=arguments.broken_kl,
                batching=Batching(arguments.batching),
            )
            return _integrity(arguments.run_dir, options=options)
        if arguments.command == "export":
            return _export(arguments.format, arguments.run_dir, output=arguments.output)
        raise CommandError(f"unknown command {arguments.command!r}")
    except (
        CommandError,
        FileExistsRefusal,
        IntegrityError,
        ParquetExportError,
        StoreError,
    ) as error:
        print(str(error), file=sys.stderr)
        return EXIT_USAGE
    except OSError as error:
        print(f"{error.filename}: {error.strerror}", file=sys.stderr)
        return EXIT_USAGE


def run() -> None:
    """Console-script entry point."""
    raise SystemExit(main())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tp", description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    show = subparsers.add_parser(
        "show", help="print the units of a run directory", allow_abbrev=False
    )
    show.add_argument("run_dir", type=Path)
    show.add_argument("--failures-only", action="store_true")
    integrity = subparsers.add_parser(
        "integrity",
        help="print per-source reward integrity and alarms; exit 1 when an alarm fired",
        allow_abbrev=False,
    )
    integrity.add_argument("run_dir", type=Path)
    integrity.add_argument(
        "--window-units", type=int, default=DEFAULT_INTEGRITY_OPTIONS.window_units
    )
    integrity.add_argument("--healthy-kl", type=float, default=DEFAULT_INTEGRITY_OPTIONS.healthy_kl)
    integrity.add_argument("--broken-kl", type=float, default=DEFAULT_INTEGRITY_OPTIONS.broken_kl)
    integrity.add_argument(
        "--batching",
        choices=[member.value for member in Batching],
        default=DEFAULT_INTEGRITY_OPTIONS.batching.value,
        help="step: one observation per 'step' tag (default); units: windows of records, for runs without steps",
    )
    export = subparsers.add_parser(
        "export",
        help="write a run's records to a new OTLP/JSON or Parquet file",
        allow_abbrev=False,
    )
    export.add_argument("format", choices=EXPORT_FORMATS)
    export.add_argument("run_dir", type=Path)
    export.add_argument("--output", type=Path, required=True)
    return parser


def _show(run_dir: Path, *, failures_only: bool) -> int:
    records, markers = _read_run(run_dir)
    print(
        "\n".join(render_show(records, started_without_record=markers, failures_only=failures_only))
    )
    return EXIT_OK


def _integrity(run_dir: Path, *, options: IntegrityOptions) -> int:
    records, _ = _read_run(run_dir)
    windows = source_windows(records, window_units=options.window_units)
    alarms = detect_alarms(records, options=options)
    print("\n".join(render_integrity(windows, alarms, options=options)))
    return EXIT_ALARM if len(alarms) != 0 else EXIT_OK


def _export(export_format: str, run_dir: Path, *, output: Path) -> int:
    records, _ = _read_run(run_dir)
    exists = f"{output.name} already exists; tp never replaces an export, choose a new path"
    if export_format == "otlp":
        text = json.dumps(render_otlp(records), sort_keys=True) + "\n"
        publish_new_file(output, write_text(text), exists_message=exists)
        return EXIT_OK
    if export_format == "parquet":
        write_parquet(records, output)
        return EXIT_OK
    raise CommandError(
        f"export format must be one of {list(EXPORT_FORMATS)}, got {export_format!r}"
    )


def _read_run(run_dir: Path) -> tuple[tuple[Record, ...], tuple[StartedMarker, ...]]:
    if not (run_dir / RECORDS_DIRECTORY).is_dir():
        raise CommandError(
            f"{run_dir} is not a run directory; it must contain a {RECORDS_DIRECTORY}/ folder written by tplane"
        )
    store = FileRecordStore(run_dir)
    return store.iterate(), store.started_without_record()


if __name__ == "__main__":
    run()
