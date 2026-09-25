"""Write records as one Parquet row each, for the training path, with dictionary-encoded categorical columns.

Categorical columns are dictionary-encoded by the Parquet writer, one dictionary per column chunk, so row-group
statistics describe only the values present and a trainer can push `failure_class IS NULL` down to them. An Arrow
dictionary array shared across row groups would give an all-null group the statistics of the whole dictionary and
make that filter skip valid rows. Read the columns back as Arrow dictionaries with read_dictionary=DICTIONARY_COLUMNS.
pyarrow is the optional `parquet` extra and is imported only here.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final

from tplane.files import FileExistsRefusal, publish_new_file
from tplane.schema import Record

DEFAULT_ROW_GROUP_ROWS: Final[int] = 64 * 1024

Getter = Callable[[Record], object]


class ParquetExportError(RuntimeError):
    """A Parquet export needs pyarrow, a positive row-group size and a path that does not exist yet."""


@dataclass(frozen=True, kw_only=True)
class Column:
    """One output column: its name, its Arrow type name, whether it is dictionary-encoded, and its value."""

    name: str
    arrow_type: str
    dictionary: bool
    value: Getter


def _failure(attribute: str) -> Getter:
    def read(record: Record) -> object:
        if record.failure is None:
            return None
        if attribute == "failure_class":
            return record.failure.failure_class.value
        return getattr(record.failure, attribute).value

    return read


COLUMNS: Final[tuple[Column, ...]] = (
    Column(name="run_id", arrow_type="string", dictionary=True, value=lambda record: record.run_id),
    Column(
        name="unit_id", arrow_type="string", dictionary=False, value=lambda record: record.unit_id
    ),
    Column(
        name="attempt", arrow_type="int32", dictionary=False, value=lambda record: record.attempt
    ),
    Column(
        name="kind", arrow_type="string", dictionary=True, value=lambda record: record.kind.value
    ),
    Column(name="source", arrow_type="string", dictionary=True, value=lambda record: record.source),
    Column(
        name="outcome",
        arrow_type="string",
        dictionary=True,
        value=lambda record: record.outcome.value,
    ),
    Column(
        name="failure_class", arrow_type="string", dictionary=True, value=_failure("failure_class")
    ),
    Column(name="failure_kind", arrow_type="string", dictionary=True, value=_failure("kind")),
    Column(name="failure_stage", arrow_type="string", dictionary=True, value=_failure("stage")),
    Column(
        name="decision",
        arrow_type="string",
        dictionary=True,
        value=lambda record: record.decision.action.value,
    ),
    Column(
        name="reward", arrow_type="float64", dictionary=False, value=lambda record: record.reward
    ),
    Column(
        name="started_at_ms",
        arrow_type="int64",
        dictionary=False,
        value=lambda record: record.started_at_ms,
    ),
    Column(
        name="ended_at_ms",
        arrow_type="int64",
        dictionary=False,
        value=lambda record: record.ended_at_ms,
    ),
    Column(
        name="cpu_seconds",
        arrow_type="float64",
        dictionary=False,
        value=lambda record: record.cost.cpu_seconds,
    ),
    Column(
        name="gpu_seconds",
        arrow_type="float64",
        dictionary=False,
        value=lambda record: record.cost.gpu_seconds,
    ),
    Column(
        name="tokens_in",
        arrow_type="int64",
        dictionary=False,
        value=lambda record: record.cost.tokens_in,
    ),
    Column(
        name="tokens_out",
        arrow_type="int64",
        dictionary=False,
        value=lambda record: record.cost.tokens_out,
    ),
    Column(
        name="usd", arrow_type="float64", dictionary=False, value=lambda record: record.cost.usd
    ),
    Column(
        name="peak_host_rss_bytes",
        arrow_type="int64",
        dictionary=False,
        value=lambda record: record.resources.peak_host_rss_bytes,
    ),
    Column(
        name="mismatch_kl",
        arrow_type="float64",
        dictionary=False,
        value=lambda record: None if record.mismatch is None else record.mismatch.kl,
    ),
    Column(
        name="mismatch_ratio_mean",
        arrow_type="float64",
        dictionary=False,
        value=lambda record: None if record.mismatch is None else record.mismatch.ratio_mean,
    ),
)
DICTIONARY_COLUMNS: Final[tuple[str, ...]] = tuple(
    sorted(column.name for column in COLUMNS if column.dictionary)
)


def write_parquet(
    records: Sequence[Record], path: Path, *, row_group_rows: int = DEFAULT_ROW_GROUP_ROWS
) -> None:
    """Write the records in the order given to a new Parquet file at path; an existing path is an error."""
    if row_group_rows < 1:
        raise ParquetExportError(f"row_group_rows must be at least 1, got {row_group_rows}")
    arrow = _import("pyarrow")
    parquet = _import("pyarrow.parquet")
    table = arrow.table({column.name: _array(arrow, column, records) for column in COLUMNS})
    try:
        publish_new_file(
            path,
            lambda temporary: parquet.write_table(
                table,
                temporary,
                row_group_size=row_group_rows,
                use_dictionary=list(DICTIONARY_COLUMNS),
            ),
            exists_message=f"{path.name} already exists; tp never replaces an export, choose a new path",
        )
    except FileExistsRefusal as error:
        raise ParquetExportError(str(error)) from error


def _array(arrow: ModuleType, column: Column, records: Sequence[Record]) -> object:
    return arrow.array(
        [column.value(record) for record in records], type=getattr(arrow, column.arrow_type)()
    )


def _import(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ParquetExportError(
            "parquet export requires pyarrow; install tplane[parquet]"
        ) from error
