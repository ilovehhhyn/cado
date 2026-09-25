"""Attach a tplane record to a Harbor ATIF trajectory as extra.tplane, and read it back.

ATIF v1.8 (Harbor RFC 0001, tag v0.23.0) has no outcome or error field, and its models forbid unknown keys, so the
record goes in the root `extra` object that ATIF allows. The block is the record's own wire form (schema version 2),
which parse_record reads back exactly.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Final

from tplane.schema import Record
from tplane.wire import RecordParseError, parse_record, render_record

ATIF_VERSION_PREFIX: Final[str] = "ATIF-v1."
EXTRA_KEY: Final[str] = "tplane"


class AtifError(ValueError):
    """An ATIF trajectory is a v1 document with at least one step and at most one tplane outcome."""


def annotate_atif(trajectory: Mapping[str, object], record: Record) -> dict[str, object]:
    """Return a copy of the trajectory whose root extra holds the record under 'tplane'."""
    document = copy.deepcopy(dict(trajectory))
    _require_atif(document)
    extra = _extra(document)
    if EXTRA_KEY in extra:
        raise AtifError(
            f"trajectory already has extra.{EXTRA_KEY}; refusing to overwrite an earlier outcome"
        )
    document["extra"] = {**extra, EXTRA_KEY: render_record(record)}
    return document


def record_from_atif(trajectory: Mapping[str, object]) -> Record:
    """Read the record back from an annotated trajectory."""
    document = dict(trajectory)
    _require_atif(document)
    extra = _extra(document)
    if EXTRA_KEY not in extra:
        raise AtifError(
            f"trajectory has no extra.{EXTRA_KEY} outcome; annotate it with annotate_atif first"
        )
    try:
        return parse_record(extra[EXTRA_KEY])
    except RecordParseError as error:
        raise AtifError(f"extra.{EXTRA_KEY}: {error}") from error


def _require_atif(document: Mapping[str, object]) -> None:
    version = document.get("schema_version")
    if type(version) is not str or not version.startswith(ATIF_VERSION_PREFIX):
        raise AtifError(f"schema_version must start with {ATIF_VERSION_PREFIX!r}, got {version!r}")
    steps = document.get("steps")
    if type(steps) is not list or len(steps) == 0:
        raise AtifError("an ATIF trajectory must have at least one step")


def _extra(document: Mapping[str, object]) -> dict[str, object]:
    extra = document.get("extra", {})
    if type(extra) is not dict:
        raise AtifError(f"extra must be a JSON object, got {type(extra).__name__}")
    return extra
