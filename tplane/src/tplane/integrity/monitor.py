"""Raise an alarm when the Bayesian posterior of a change in failure rate, reward or train-inference mismatch reaches the alarm probability, and attribute reward drops.

Every stream runs the Shiryaev recursion (changepoint.py) with hazard rho per observation, and alarms at
P(change) >= alarm_probability. That bounds the probability of alarming before a real change by
alpha = 1 - alarm_probability when the likelihoods are right. An observation is one training step (steps.py):
rollouts of one prompt are correlated, so per-rollout likelihoods would overcount evidence and overflag.
    failure_rate  per source and step: k infra or grader failures of n units, Binomial(p1) against Binomial(p0),
                  with the log-likelihood ratio divided by the Rao-Scott design effect
                  deff = cluster-robust variance of k/n divided by its binomial variance, clipped to [1, n / G].
                  Failures confined to one prompt group give deff near n / G, independent failures near 1
    reward        per source and step: the clean-reward mean y ~ t_nu(m, s) healthy and t_nu(m - d * s_u, s)
                  changed. m is the mean of the r steps that end l steps before this one, so it follows learning;
                  s comes from all lagged steps (up to reward_scale_steps) by the mean squared successive
                  difference, because the scale is structural and a short window reused for many steps would
                  share one estimation error; see reward_llr(). s_u is the mean per-rollout sd. The t ratio is
                  bounded, so one extreme step cannot decide a change. Only a drop alarms: learning raises the
                  reward, poisoning or collapse lowers it. A step with fewer than two prompt groups gives no
                  evidence, because its variance is not identifiable
    mismatch      per train step, an equal mixture of two failure modes against health:
                  kl mode     log10(kl) ~ N(log10 kl1, s) against N(log10 kl0, s)
                  ratio mode  mean ratio ~ N(1 - shift, se) against N(1, se); E_q[p/q] = 1 exactly when healthy
                              and p(nucleus) < 1 under top-p truncation, so only a drop is a failure
    poisoned      per source: an infra or grader failure whose zero reached the trainer, observed directly
A reward alarm is attributed with P(config) = 1 - (1 - P(failure-rate change)) * (1 - P(mismatch change)),
read attribution_steps steps after the alarm so that a cause detected just after its effect still counts. This
treats the two configuration causes as independent and the reward drop as equally likely under both. Defaults
come from docs/references.md and docs/results; they are starting points to be tuned on real runs.
"""

from __future__ import annotations

import bisect
import dataclasses
import itertools
import math
import statistics
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tplane.integrity.changepoint import (
    NO_CHANGE_LOG_ODDS,
    gaussian_llr,
    log_mean_exp,
    log_odds_of,
    probability_of,
    shiryaev_update,
    student_t_llr,
)
from tplane.integrity.metrics import IntegrityError, is_poisoned, start_order, units_by_source
from tplane.integrity.steps import Batching, StepObservation, step_observations
from tplane.schema import MismatchSummary, Record

# A bitwise-identical rollout and trainer give kl = 0, and log10 needs a floor.
KL_FLOOR: Final[float] = 1e-12
EVEN_ODDS: Final[float] = 0.5
# A mean ratio known to better than 1e-9 carries no more information, and the floor keeps its likelihood ratio finite.
RATIO_STANDARD_ERROR_FLOOR: Final[float] = 1e-9
POISONED_DETAIL: Final[str] = (
    "an infra or grader failure was scored as zero and entered the reward; "
    "set Policy(on_infra=mask, on_grader=mask) or fix the trainer that zeroes it"
)


class Signal(StrEnum):
    """The stream an alarm came from."""

    FAILURE_RATE = "failure_rate"
    MISMATCH = "mismatch"
    POISONED = "poisoned"
    REWARD = "reward"


class Verdict(StrEnum):
    """The likely cause of a reward drop: configuration at or above the alarm probability, algorithm below even odds."""

    ALGORITHM = "algorithm"
    CONFIGURATION = "configuration"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True, kw_only=True)
class IntegrityOptions:
    """Detector settings; each default is a published figure or a stated starting point."""

    alarm_probability: float = 0.999  # alpha = 0.001
    hazard: float = 1e-3  # prior probability that a change starts at any one step
    batching: Batching = Batching.STEP
    batch_units: int = 32  # observation size under unit batching only
    failure_rate_healthy: float = 0.02  # KAT-Coder sandbox error rate after its fix
    failure_rate_broken: float = 0.16  # KAT-Coder sandbox error rate before its fix
    reward_reference_steps: int = 20
    reward_lag_steps: int = 10
    reward_scale_steps: int = 200  # steps behind the lag that estimate the step-mean scale
    reward_shift_sd: float = (
        0.25  # the drop to detect, in per-rollout reward sds (0.125 for a 0.5 pass rate)
    )
    reward_sd_floor: float = 0.05
    healthy_kl: float = 1e-3  # upper end of healthy dense bf16 k3
    broken_kl: float = 1e-2  # lower end of broken stacks and uncorrected fp8 rollouts
    kl_sd_decades: float = 0.5
    ratio_shift: float = 0.01  # a 1% nucleus mass left out by top-p 0.99
    attribution_steps: int = 5
    window_units: int = 50  # rows of the report table only

    def __post_init__(self) -> None:
        if not EVEN_ODDS < self.alarm_probability < 1:
            raise IntegrityError(
                f"alarm_probability must be in (0.5, 1), got {self.alarm_probability}"
            )
        if not 0 < self.hazard < 1:
            raise IntegrityError(f"hazard must be in (0, 1), got {self.hazard}")
        if self.batch_units < 2:
            raise IntegrityError(f"batch_units must be at least 2, got {self.batch_units}")
        if not 0 < self.failure_rate_healthy < self.failure_rate_broken < 1:
            raise IntegrityError(
                "failure rates must satisfy 0 < failure_rate_healthy < failure_rate_broken < 1, "
                f"got {self.failure_rate_healthy} and {self.failure_rate_broken}"
            )
        if not 0 < self.healthy_kl < self.broken_kl:
            raise IntegrityError(
                f"kl levels must satisfy 0 < healthy_kl < broken_kl, got {self.healthy_kl} and {self.broken_kl}"
            )
        if self.reward_scale_steps < self.reward_reference_steps:
            raise IntegrityError(
                f"reward_scale_steps must be at least reward_reference_steps, got {self.reward_scale_steps} "
                f"and {self.reward_reference_steps}"
            )
        if self.reward_reference_steps < 2 or self.reward_lag_steps < 0:
            raise IntegrityError(
                "reward_reference_steps must be at least 2 and reward_lag_steps at least 0, "
                f"got {self.reward_reference_steps} and {self.reward_lag_steps}"
            )
        scales = (self.reward_shift_sd, self.reward_sd_floor, self.kl_sd_decades, self.ratio_shift)
        if any(not value > 0 for value in scales):
            raise IntegrityError(
                f"reward_shift_sd, reward_sd_floor, kl_sd_decades and ratio_shift must be positive, got {scales}"
            )
        if self.attribution_steps < 0:
            raise IntegrityError(
                f"attribution_steps must be at least 0, got {self.attribution_steps}"
            )
        if self.window_units < 1:
            raise IntegrityError(f"window_units must be at least 1, got {self.window_units}")


DEFAULT_INTEGRITY_OPTIONS: Final[IntegrityOptions] = IntegrityOptions()


@dataclass(frozen=True, kw_only=True)
class Attribution:
    """The posterior probability that a reward drop comes from a configuration failure, and its parts."""

    configuration_probability: float
    failure_rate_probability: float
    mismatch_probability: float | None  # None when no train step recorded a mismatch summary
    verdict: Verdict


@dataclass(frozen=True, kw_only=True)
class Alarm:
    """The first time one stream's change posterior reached the alarm probability."""

    signal: Signal
    source: str | None
    unit_id: str
    at_ms: int
    probability: float
    detail: str
    attribution: Attribution | None = None


@dataclass(frozen=True, kw_only=True)
class _MismatchHistory:
    times_ms: tuple[int, ...]
    log_odds: tuple[float, ...]

    def probability_at(self, at_ms: int) -> float | None:
        """Return P(mismatch change) after the last train step at or before at_ms; None when none was recorded."""
        if len(self.times_ms) == 0:
            return None
        position = bisect.bisect_right(self.times_ms, at_ms)
        return 0.0 if position == 0 else probability_of(self.log_odds[position - 1])


def detect_alarms(
    records: Iterable[Record], *, options: IntegrityOptions = DEFAULT_INTEGRITY_OPTIONS
) -> tuple[Alarm, ...]:
    """Return the first alarm of every stream, ordered by time, then signal and source."""
    records = tuple(records)
    history, mismatch_alarm = _mismatch_stream(records, options)
    alarms = [] if mismatch_alarm is None else [mismatch_alarm]
    for _, units in units_by_source(records):
        alarms.extend(_source_alarms(units, history, options))
    return tuple(
        sorted(alarms, key=lambda alarm: (alarm.at_ms, alarm.signal.value, alarm.source or ""))
    )


def mismatch_llr(summary: MismatchSummary, *, options: IntegrityOptions) -> float:
    """Return log(mean(L_kl, L_ratio)) for one train step; only the kl mode when the ratio has no spread."""
    kl_term = gaussian_llr(
        math.log10(max(summary.kl, KL_FLOOR)),
        mean_healthy=math.log10(options.healthy_kl),
        mean_changed=math.log10(options.broken_kl),
        standard_deviation=options.kl_sd_decades,
    )
    if summary.ratio_standard_error == 0:
        return kl_term
    ratio_term = gaussian_llr(
        summary.ratio_mean,
        mean_healthy=1.0,
        mean_changed=1.0 - options.ratio_shift,
        standard_deviation=max(summary.ratio_standard_error, RATIO_STANDARD_ERROR_FLOOR),
    )
    return log_mean_exp([kl_term, ratio_term])


def failure_rate_llr(observation: StepObservation, *, options: IntegrityOptions) -> float:
    """Return the step's binomial log-likelihood ratio divided by its estimated Rao-Scott design effect."""
    k, n = observation.failures, observation.units
    p0, p1 = options.failure_rate_healthy, options.failure_rate_broken
    binomial = k * (math.log(p1) - math.log(p0)) + (n - k) * (math.log1p(-p1) - math.log1p(-p0))
    return binomial / design_effect(observation)


def design_effect(observation: StepObservation) -> float:
    """Return var_cluster(k/n) / (p(1 - p)/n) clipped to [1, n / G]; 1 when k is 0 or n, where it is undefined."""
    k, n = observation.failures, observation.units
    largest = n / observation.clusters
    if k == 0 or k == n or observation.failure_variance_of_mean is None:
        return 1.0
    proportion = k / n
    ratio = observation.failure_variance_of_mean / (proportion * (1 - proportion) / n)
    return min(max(ratio, 1.0), largest)


def _mismatch_stream(
    records: Sequence[Record], options: IntegrityOptions
) -> tuple[_MismatchHistory, Alarm | None]:
    steps = sorted(
        ((record, record.mismatch) for record in records if record.mismatch is not None),
        key=lambda step: start_order(step[0]),
    )
    threshold = log_odds_of(options.alarm_probability)
    log_odds = NO_CHANGE_LOG_ODDS
    trace: list[float] = []
    alarm: Alarm | None = None
    for record, summary in steps:
        llr = mismatch_llr(summary, options=options)
        log_odds = shiryaev_update(log_odds, log_likelihood_ratio=llr, hazard=options.hazard)
        trace.append(log_odds)
        if alarm is None and log_odds >= threshold:
            detail = (
                f"kl {summary.kl:.2e} against healthy {options.healthy_kl:.0e} and broken {options.broken_kl:.0e}; "
                f"mean importance ratio {summary.ratio_mean:.4f} +- {summary.ratio_standard_error:.4f}, expected 1"
            )
            alarm = Alarm(
                signal=Signal.MISMATCH,
                source=record.source,
                unit_id=record.unit_id,
                at_ms=record.started_at_ms,
                probability=probability_of(log_odds),
                detail=detail,
            )
    times = tuple(record.started_at_ms for record, _ in steps)
    return _MismatchHistory(times_ms=times, log_odds=tuple(trace)), alarm


def _source_alarms(
    units: Sequence[Record], history: _MismatchHistory, options: IntegrityOptions
) -> list[Alarm]:
    source = units[0].source
    poisoned = next((record for record in units if is_poisoned(record)), None)
    alarms: list[Alarm] = []
    if poisoned is not None:
        alarms.append(
            Alarm(
                signal=Signal.POISONED,
                source=source,
                unit_id=poisoned.unit_id,
                at_ms=poisoned.started_at_ms,
                probability=1.0,
                detail=POISONED_DETAIL,
            )
        )
    failure_rate = _FailureRateStream(options)
    reward = _RewardStream(options)
    pending: tuple[Alarm, int] | None = (
        None  # a reward alarm waiting attribution_steps for its causes
    )
    reward_alarm: Alarm | None = None
    observations = step_observations(
        units, batching=options.batching, batch_units=options.batch_units
    )
    for position, observation in enumerate(observations):
        failure_rate.observe(observation, source)
        fresh = (
            reward.observe(observation, source)
            if pending is None and reward_alarm is None
            else None
        )
        if fresh is not None:
            pending = (fresh, position + options.attribution_steps)
        if pending is not None and position >= pending[1]:
            reward_alarm = _with_attribution(
                pending[0], failure_rate, history, observation.at_ms, options
            )
            pending = None
    if pending is not None:
        reward_alarm = _with_attribution(
            pending[0], failure_rate, history, observations[-1].at_ms, options
        )
    alarms.extend(alarm for alarm in (failure_rate.alarm, reward_alarm) if alarm is not None)
    return alarms


class _FailureRateStream:
    """The per-step infra or grader failure count of one source."""

    def __init__(self, options: IntegrityOptions) -> None:
        self._options = options
        self._threshold = log_odds_of(options.alarm_probability)
        self.log_odds = NO_CHANGE_LOG_ODDS
        self.alarm: Alarm | None = None

    def observe(self, observation: StepObservation, source: str | None) -> None:
        llr = failure_rate_llr(observation, options=self._options)
        self.log_odds = shiryaev_update(
            self.log_odds, log_likelihood_ratio=llr, hazard=self._options.hazard
        )
        if self.alarm is None and self.log_odds >= self._threshold:
            detail = (
                f"infra or grader failures {observation.failures} of {observation.units} units in step "
                f"{observation.label}; healthy rate {self._options.failure_rate_healthy}, "
                f"broken {self._options.failure_rate_broken}"
            )
            self.alarm = _alarm(
                Signal.FAILURE_RATE, observation, source, probability_of(self.log_odds), detail
            )


class _RewardStream:
    """The per-step clean-reward mean of one source, against its recent mean and its long-run scale."""

    def __init__(self, options: IntegrityOptions) -> None:
        self._options = options
        self._threshold = log_odds_of(options.alarm_probability)
        self._log_odds = NO_CHANGE_LOG_ODDS
        self._history: deque[StepObservation] = deque(
            maxlen=options.reward_scale_steps + options.reward_lag_steps
        )

    def observe(self, observation: StepObservation, source: str | None) -> Alarm | None:
        """Update with one informative step and return an alarm the first time the drop posterior crosses."""
        if observation.clean_mean is None or observation.clean_variance_of_mean is None:
            return None
        history = list(self._history)
        lagged = history[: max(0, len(history) - self._options.reward_lag_steps)]
        self._history.append(observation)
        if len(lagged) < self._options.reward_reference_steps:
            return None
        llr, mean = reward_llr(observation, lagged, options=self._options)
        self._log_odds = shiryaev_update(
            self._log_odds, log_likelihood_ratio=llr, hazard=self._options.hazard
        )
        if self._log_odds < self._threshold:
            return None
        detail = (
            f"clean mean reward {observation.clean_mean:.3f} in step {observation.label} against a reference of "
            f"{mean:.3f} over {self._options.reward_reference_steps} earlier steps"
        )
        return _alarm(Signal.REWARD, observation, source, probability_of(self._log_odds), detail)


def reward_llr(
    observation: StepObservation, lagged: Sequence[StepObservation], *, options: IntegrityOptions
) -> tuple[float, float]:
    """Return the step's t log-likelihood ratio of a drop, and the reference mean it was measured against.

    The location m is the mean of the last r lagged steps, so it follows learning. The scale is structural
    (prompt clustering and batch size) and comes from all lagged steps through the mean squared successive
    difference s^2 = mean((y_t - y_(t-1))^2) / 2, which slow drift does not inflate; its degrees of freedom are
    2 (n - 1)^2 / (3n - 4). A window reused for many steps would share one estimation error across all of them.
    """
    means = [step.clean_mean for step in lagged if step.clean_mean is not None]
    variances = [
        step.clean_variance_of_mean for step in lagged if step.clean_variance_of_mean is not None
    ]
    unit_sds = [step.clean_unit_sd for step in lagged if step.clean_unit_sd is not None]
    current_mean, current_variance = observation.clean_mean, observation.clean_variance_of_mean
    if current_mean is None or current_variance is None:
        raise IntegrityError(
            f"step {observation.label} has no reward evidence; call reward_llr only on informative steps"
        )
    recent = means[-options.reward_reference_steps :]
    location = statistics.fmean(recent)
    successive = [(later - earlier) ** 2 for earlier, later in itertools.pairwise(means)]
    scale_squared = statistics.fmean(successive) / 2
    count = len(means)
    degrees = max(1.0, 2 * (count - 1) ** 2 / (3 * count - 4))
    excess = max(0.0, current_variance - statistics.fmean(variances))
    scale = math.sqrt(scale_squared * (1 + 1 / len(recent)) + excess)
    scale = max(scale, options.reward_sd_floor / math.sqrt(observation.clean_count))
    unit_sd = max(
        statistics.fmean(unit_sds) if len(unit_sds) != 0 else 0.0, options.reward_sd_floor
    )
    llr = student_t_llr(
        current_mean,
        mean_healthy=location,
        mean_changed=location - options.reward_shift_sd * unit_sd,
        scale=scale,
        degrees=degrees,
    )
    return llr, location


def _with_attribution(
    alarm: Alarm,
    failure_rate: _FailureRateStream,
    history: _MismatchHistory,
    at_ms: int,
    options: IntegrityOptions,
) -> Alarm:
    """Attach the configuration posteriors as they stand at at_ms, the end of the attribution window."""
    attribution = _attribute(
        probability_of(failure_rate.log_odds), history.probability_at(at_ms), options
    )
    return dataclasses.replace(alarm, attribution=attribution)


def _attribute(
    failure_probability: float, mismatch_probability: float | None, options: IntegrityOptions
) -> Attribution:
    configuration = 1 - (1 - failure_probability) * (1 - (mismatch_probability or 0.0))
    if configuration >= options.alarm_probability:
        verdict = Verdict.CONFIGURATION
    elif configuration < EVEN_ODDS:
        verdict = Verdict.ALGORITHM
    else:
        verdict = Verdict.UNDETERMINED
    return Attribution(
        configuration_probability=configuration,
        failure_rate_probability=failure_probability,
        mismatch_probability=mismatch_probability,
        verdict=verdict,
    )


def _alarm(
    signal: Signal,
    observation: StepObservation,
    source: str | None,
    probability: float,
    detail: str,
) -> Alarm:
    return Alarm(
        signal=signal,
        source=source,
        unit_id=observation.last_unit_id,
        at_ms=observation.at_ms,
        probability=probability,
        detail=detail,
    )
