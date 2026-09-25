"""The guard warns exactly when free GPU memory is below the threshold and names the knob to lower."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.support import MemoryRecordStore
from tplane.guard import DEFAULT_WARN_PERCENT, GuardError, headroom_warning
from tplane.schema import UnitKind
from tplane.unit import Run, RunOptions

GIB = 1024 * 1024 * 1024


@pytest.mark.parametrize(
    ("used", "expected_warning"),
    (
        (89 * GIB // 100, False),
        (90 * GIB // 100, False),
        (90 * GIB // 100 + 1, True),
        (91 * GIB // 100, True),
    ),
)
def test_warning_boundary_at_the_threshold(used: int, expected_warning: bool) -> None:
    warning = headroom_warning(gpu_used_bytes=used, gpu_total_bytes=GIB, longest_tokens=4096)

    assert (warning is not None) is expected_warning


def test_warning_text_names_headroom_threshold_longest_sample_and_fix() -> None:
    warning = headroom_warning(
        gpu_used_bytes=76 * GIB, gpu_total_bytes=80 * GIB, longest_tokens=131_072
    )

    assert warning == (
        f"gpu memory headroom is 5% of 80.0 GiB, below the {DEFAULT_WARN_PERCENT}% warning threshold while "
        "the longest sample is 131072 tokens; lower max_tokens_per_gpu or gpu_memory_utilization before "
        "the next step"
    )


def test_guard_rejects_used_above_total_and_bad_percent() -> None:
    with pytest.raises(GuardError, match="gpu_used_bytes must be between 0 and gpu_total_bytes"):
        headroom_warning(gpu_used_bytes=2, gpu_total_bytes=1, longest_tokens=1)
    with pytest.raises(GuardError, match="warn_percent must be between 1 and 99, got 0"):
        headroom_warning(gpu_used_bytes=0, gpu_total_bytes=1, longest_tokens=1, warn_percent=0)


def make_run(
    read_gpu: Callable[[], tuple[int, ...]], lines: list[str], store: MemoryRecordStore
) -> Run:
    return Run(
        run_id="run-1",
        store=store,
        options=RunOptions(sample_interval_ms=10),
        read_host=lambda: 1,
        read_gpu=read_gpu,
        gpu_total_bytes=(100,),
        log=lines.append,
    )


def test_unit_check_headroom_logs_and_tags_the_warning() -> None:
    lines: list[str] = []
    store = MemoryRecordStore()
    run = make_run(lambda: (95,), lines, store)

    with run.unit("s-1", kind=UnitKind.TRAIN_STEP) as unit:
        warning = unit.check_headroom(longest_tokens=8192)

    assert warning is not None and warning.startswith("gpu memory headroom is 5%")
    assert lines[0] == f"tplane s-1 headroom: {warning}"
    assert ("headroom_warning", warning) in store.iterate()[0].tags


def test_unit_check_headroom_without_a_gpu_sample_reports_unknown_and_keeps_the_unit_alive() -> (
    None
):
    lines: list[str] = []
    store = MemoryRecordStore()

    calls = {"count": 0}

    def lost_after_the_probe() -> tuple[int, ...]:
        calls["count"] += 1
        if calls["count"] > 1:
            raise RuntimeError("NVML_ERROR_GPU_IS_LOST")
        return (1,)

    run = make_run(lost_after_the_probe, lines, store)

    with run.unit("s-1", kind=UnitKind.TRAIN_STEP) as unit:
        warning = unit.check_headroom(longest_tokens=1)

    assert warning is None
    assert lines[0] == "tplane s-1 headroom: unknown; no gpu sample has been taken"
    (record,) = store.iterate()
    assert record.failure is None
    assert ("headroom_warning", "unknown; no gpu sample has been taken") in record.tags


def test_unit_check_headroom_rejects_a_device_index_that_was_not_sampled() -> None:
    run = make_run(lambda: (1,), [], MemoryRecordStore())

    with run.unit("s-1", kind=UnitKind.TRAIN_STEP) as unit:
        with pytest.raises(
            GuardError, match="device_index 3 is not sampled; 1 device is configured"
        ):
            unit.check_headroom(longest_tokens=1, device_index=3)
