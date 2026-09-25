"""Parquet export writes one row per record with dictionary-encoded categorical columns and row-group statistics a reader can use to skip failures, and never replaces a file."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from tests.support import make_unit_record
from tplane.export.parquet import COLUMNS, DICTIONARY_COLUMNS, ParquetExportError, write_parquet
from tplane.schema import Action, FailureKind, Record


def records() -> list[Record]:
    failed = [
        make_unit_record(index, failure_kind=FailureKind.SANDBOX, action=Action.MASK)
        for index in range(10)
    ]
    scored = [make_unit_record(index, reward=float(index % 2)) for index in range(10, 30)]
    return failed + scored


def test_one_row_per_record_with_dictionary_encoded_categories(tmp_path: Path) -> None:
    path = tmp_path / "records.parquet"

    write_parquet(records(), path)

    table = pq.read_table(path, read_dictionary=list(DICTIONARY_COLUMNS))
    chunks = pq.ParquetFile(path).metadata.row_group(0)
    names = [column.name for column in COLUMNS]
    assert table.num_rows == 30
    assert table.column_names == names
    assert DICTIONARY_COLUMNS == (
        "decision",
        "failure_class",
        "failure_kind",
        "failure_stage",
        "kind",
        "outcome",
        "run_id",
        "source",
    )
    for name in DICTIONARY_COLUMNS:
        assert "RLE_DICTIONARY" in chunks.column(names.index(name)).encodings, name
        assert pa.types.is_dictionary(table.schema.field(name).type), name
    assert table.column("reward").to_pylist()[:11] == [None] * 10 + [0.0]
    assert table.column("failure_kind").to_pylist()[:11] == ["sandbox"] * 10 + [None]


def test_row_group_statistics_let_a_reader_skip_failed_rows(tmp_path: Path) -> None:
    path = tmp_path / "records.parquet"
    write_parquet(records(), path, row_group_rows=10)

    metadata = pq.ParquetFile(path).metadata
    index = [column.name for column in COLUMNS].index("failure_class")
    groups = [
        metadata.row_group(group).column(index).statistics
        for group in range(metadata.num_row_groups)
    ]
    kept = ds.dataset(path).to_table(filter=ds.field("failure_class").is_null())
    infra = pq.read_table(path, filters=[("failure_class", "=", "infra")])

    assert metadata.num_row_groups == 3
    # Invariant: an all-null row group has no min/max; a shared Arrow dictionary once gave it min = max = "infra",
    # and the null filter then skipped all 20 kept rows.
    assert (groups[0].min, groups[0].max, groups[0].null_count) == ("infra", "infra", 0)
    assert [(group.has_min_max, group.null_count) for group in groups[1:]] == [
        (False, 10),
        (False, 10),
    ]
    assert kept.num_rows == 20
    assert infra.column("failure_class").to_pylist() == ["infra"] * 10


def test_an_existing_file_is_never_replaced(tmp_path: Path) -> None:
    path = tmp_path / "records.parquet"
    path.write_text("keep me")

    with pytest.raises(
        ParquetExportError, match="records.parquet already exists; tp never replaces an export"
    ):
        write_parquet(records(), path)

    assert path.read_text() == "keep me"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["records.parquet"]


def test_row_group_rows_below_one_is_rejected_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ParquetExportError, match="row_group_rows must be at least 1, got 0"):
        write_parquet(records(), tmp_path / "records.parquet", row_group_rows=0)

    assert list(tmp_path.iterdir()) == []
