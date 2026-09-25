"""Per-source windows count units, not time, and each metric uses exactly the records its definition names."""

from __future__ import annotations

import pytest

from tests.support import make_unit_record
from tplane.integrity.metrics import IntegrityError, source_windows
from tplane.schema import Action, FailureKind, Record, UnitKind


def test_one_window_over_every_outcome_and_failure_class_combination() -> None:
    records = [
        make_unit_record(0, reward=1.0),
        make_unit_record(1, failure_kind=FailureKind.AGENT_EXCEPTION, action=Action.ZERO),
        make_unit_record(2, failure_kind=FailureKind.SANDBOX, action=Action.MASK),
        make_unit_record(3, failure_kind=FailureKind.NETWORK, action=Action.ZERO),
        make_unit_record(4, failure_kind=FailureKind.MISSING_REWARD, action=Action.MASK),
        make_unit_record(5, failure_kind=FailureKind.WALL_CLOCK, action=Action.ZERO),
    ]

    (window,) = source_windows(records, window_units=6)

    assert window.source == "swe" and window.units == 6
    assert window.avg_reward == pytest.approx(
        1 / 4, rel=1e-12, abs=0.0
    )  # rewards 1, 0, 0, 0 reached the trainer
    assert window.avg_reward_no_infra == pytest.approx(
        1 / 3, rel=1e-12, abs=0.0
    )  # excludes the zeroed network failure
    assert window.zero_reward_rate == pytest.approx(3 / 4, rel=1e-12, abs=0.0)
    assert window.infra_error_rate == pytest.approx(2 / 6, rel=1e-12, abs=0.0)
    assert window.grader_error_rate == pytest.approx(1 / 6, rel=1e-12, abs=0.0)
    assert window.poisoned_units == 1


def test_a_window_where_nothing_reached_the_trainer_has_no_averages() -> None:
    records = [
        make_unit_record(index, failure_kind=FailureKind.SANDBOX, action=Action.MASK)
        for index in range(3)
    ]

    (window,) = source_windows(records, window_units=3)

    assert (window.avg_reward, window.avg_reward_no_infra, window.zero_reward_rate) == (
        None,
        None,
        None,
    )
    assert window.infra_error_rate == 1.0
    assert window.poisoned_units == 0


@pytest.mark.parametrize(("count", "sizes"), ((0, []), (1, [1]), (2, [2]), (3, [3]), (4, [3, 1])))
def test_windows_split_by_unit_count_at_the_boundaries(count: int, sizes: list[int]) -> None:
    records = [make_unit_record(index) for index in range(count)]

    windows = source_windows(records, window_units=3)

    assert [window.units for window in windows] == sizes
    assert [window.index for window in windows] == list(range(len(sizes)))


def test_windows_are_per_source_in_time_order_and_skip_train_steps() -> None:
    records: list[Record] = [
        make_unit_record(2, source="b"),
        make_unit_record(1, source="a", reward=0.0),
        make_unit_record(0, source="a", reward=1.0),
        make_unit_record(3, source=None),
        make_unit_record(4, source="a", kind=UnitKind.TRAIN_STEP),
    ]

    windows = source_windows(records, window_units=1)

    assert [(window.source, window.first_unit_id) for window in windows] == [
        ("a", "u-00000"),
        ("a", "u-00001"),
        ("b", "u-00002"),
        (None, "u-00003"),
    ]


def test_window_units_below_one_is_rejected() -> None:
    with pytest.raises(IntegrityError, match="window_units must be at least 1, got 0"):
        source_windows([], window_units=0)
