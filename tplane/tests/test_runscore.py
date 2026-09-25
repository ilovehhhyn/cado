"""A whole-run score compares each feature with a normal profile of per-step means and one pooled spread, and the alarm threshold comes from normal runs scored out of sample.

Oracle: the Bayes factor and profile statistics computed by hand for small inputs.
"""

from __future__ import annotations

import math
import statistics

import pytest

from tplane.integrity.runscore import (
    RunScoreError,
    build_profile,
    calibrate_threshold,
    out_of_sample_scores,
    score_run,
)

NORMALS = (
    {"reward": [0.1, 0.2, 0.3], "length": [10.0, 10.0, 10.0]},
    {"reward": [0.3, 0.4, 0.5], "length": [12.0, 12.0, 12.0]},
)


def log_bayes_factor(z_values: list[float], tau: float) -> float:
    n, total = len(z_values), sum(z_values)
    return -0.5 * math.log(1 + n * tau * tau) + total * total * tau * tau / (
        2 * (1 + n * tau * tau)
    )


def test_profile_holds_the_per_step_mean_and_one_pooled_spread_per_feature() -> None:
    profile = build_profile(NORMALS)

    assert [profile.means[("reward", step)] for step in range(3)] == pytest.approx(
        [0.2, 0.3, 0.4], rel=1e-12, abs=1e-15
    )
    per_step_variance = statistics.variance([0.1, 0.3])
    assert profile.spreads["reward"] == pytest.approx(
        math.sqrt(per_step_variance), rel=1e-12, abs=0.0
    )
    assert profile.normal_runs == 2


def test_a_feature_that_never_varies_gets_a_relative_floor_not_a_zero_spread() -> None:
    profile = build_profile(
        ({"errors": [0.0, 0.0]}, {"errors": [0.0, 0.0]}, {"errors": [4.0, 4.0]})
    )
    constant = build_profile(({"errors": [0.0, 0.0]}, {"errors": [0.0, 0.0]}))

    assert profile.spreads["errors"] > 0
    assert constant.spreads["errors"] > 0 and math.isfinite(constant.spreads["errors"])


def test_a_run_at_the_profile_mean_scores_the_pure_occam_penalty() -> None:
    profile = build_profile(NORMALS)
    run = {"reward": [0.2, 0.3, 0.4], "length": [11.0, 11.0, 11.0]}

    score = score_run(run, profile, prior_scale=0.3)

    assert score == pytest.approx(log_bayes_factor([0.0, 0.0, 0.0], 0.3), rel=1e-12, abs=1e-12)


def test_the_run_score_is_the_log_mean_of_the_per_feature_bayes_factors() -> None:
    profile = build_profile(NORMALS)
    run = {"reward": [0.5, 0.6, 0.7], "length": [11.0, 11.0, 11.0]}
    spread = profile.spreads["reward"]
    reward = log_bayes_factor([0.3 / spread] * 3, 0.3)
    length = log_bayes_factor([0.0] * 3, 0.3)

    score = score_run(run, profile, prior_scale=0.3)

    assert score == pytest.approx(
        math.log((math.exp(reward) + math.exp(length)) / 2), rel=1e-12, abs=1e-12
    )


def test_missing_values_and_steps_beyond_the_profile_are_skipped() -> None:
    profile = build_profile(NORMALS)
    run = {"reward": [0.2, None, 0.4, 9.9], "length": [11.0, 11.0, 11.0]}

    assert score_run(run, profile, prior_scale=0.3) == pytest.approx(
        math.log(
            (
                math.exp(log_bayes_factor([0.0, 0.0], 0.3))
                + math.exp(log_bayes_factor([0.0] * 3, 0.3))
            )
            / 2
        ),
        rel=1e-12,
        abs=1e-12,
    )


def test_a_run_sharing_nothing_with_the_profile_is_rejected() -> None:
    with pytest.raises(RunScoreError, match="run shares no feature and step with the profile"):
        score_run({"entropy": [1.0]}, build_profile(NORMALS), prior_scale=0.3)


def test_a_profile_needs_at_least_two_normal_runs() -> None:
    with pytest.raises(RunScoreError, match="a profile needs at least 2 normal runs, got 1"):
        build_profile(NORMALS[:1])


def test_out_of_sample_scores_use_a_profile_built_without_each_fold() -> None:
    normals = tuple({"reward": [0.1 * index, 0.1 * index + 0.05]} for index in range(6))

    scores = out_of_sample_scores(normals, folds=3, prior_scale=0.3)

    for position, run in enumerate(normals):
        rest = [other for index, other in enumerate(normals) if index % 3 != position % 3]
        assert scores[position] == pytest.approx(
            score_run(run, build_profile(rest), prior_scale=0.3), rel=1e-12, abs=1e-12
        )


def test_leave_one_out_scoring_works_when_normals_are_few() -> None:
    normals = tuple({"reward": [0.1 * index]} for index in range(3))

    scores = out_of_sample_scores(normals, folds=3, prior_scale=0.3)

    assert scores[0] == pytest.approx(
        score_run(normals[0], build_profile(normals[1:]), prior_scale=0.3), rel=1e-12, abs=1e-12
    )


def test_out_of_sample_scoring_rejects_folds_that_leave_fewer_than_two_normals() -> None:
    with pytest.raises(
        RunScoreError,
        match="2 folds over 2 normal runs leave 1 run per profile; at least 2 are needed",
    ):
        out_of_sample_scores(tuple({"reward": [0.1]} for _ in range(2)), folds=2, prior_scale=0.3)


def test_the_threshold_is_the_mean_plus_multiplier_times_sd_of_normal_scores() -> None:
    assert calibrate_threshold((1.0, 2.0, 3.0), multiplier=1.5) == pytest.approx(
        2.0 + 1.5 * 1.0, rel=1e-12, abs=0.0
    )


def test_threshold_calibration_rejects_invalid_input() -> None:
    with pytest.raises(RunScoreError, match="calibration needs at least 2 normal scores, got 1"):
        calibrate_threshold((1.0,), multiplier=1.0)
    with pytest.raises(RunScoreError, match="multiplier must be non-negative, got -1"):
        calibrate_threshold((1.0, 2.0), multiplier=-1)
    with pytest.raises(RunScoreError, match="prior_scale must be positive, got 0"):
        score_run({"reward": [0.2]}, build_profile(NORMALS), prior_scale=0)
