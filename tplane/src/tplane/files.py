"""Publish a new file atomically under a name that must not exist yet."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Final

TEMPORARY_PREFIX: Final[str] = ".tmp-"


class FileExistsRefusal(FileExistsError):
    """A published file is never replaced; the caller's message says what already exists."""


def publish_new_file(final: Path, write: Callable[[Path], None], *, exists_message: str) -> None:
    """Call write(temporary) in final's directory, fsync, then hard-link to final, which fails if final exists.

    A crash before the link leaves only a dot-prefixed temporary file, which readers skip.
    """
    descriptor, temporary_name = tempfile.mkstemp(dir=final.parent, prefix=TEMPORARY_PREFIX)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, final)
    except FileExistsError as error:
        raise FileExistsRefusal(exists_message) from error
    finally:
        temporary.unlink()


def write_text(text: str) -> Callable[[Path], None]:
    """Return a writer of text in UTF-8, for publish_new_file."""

    def write(path: Path) -> None:
        path.write_text(text, encoding="utf-8")

    return write
