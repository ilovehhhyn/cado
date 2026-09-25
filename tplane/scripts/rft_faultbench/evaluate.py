"""Evaluate tplane.integrity.runscore on RFT-FaultBench with nested cross-validation, and print the results table.

Outer 5-fold CV over all runs (seed 0). Inside each outer training set, the threshold multiplier k is chosen
empirically: inner 5-fold CV, with normals scored out of sample, picks the k with the best average F1 whose
inner false-alarm rate on normals is at most MAX_INNER_FALSE_ALARMS. The outer test fold never influences its k.
Protocol A uses the paper's labels (every run in a fault directory is faulty, the 11 baselines are normal);
protocol B counts the un-injected controls as normal. Features are trajectory aggregates only; the ablations
(step metrics, trend and variance tests, per-step spreads) are recorded in docs/results/rft-faultbench.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import statistics

from tplane.integrity.runscore import build_profile, calibrate_threshold, score_run

FAMILIES = ("rf", "pg", "od", "ca", "te")
FEATURES = (
    "reward_mean", "reward_sd", "response_length_mean", "response_length_sd", "advantage_std_mean",
    "repetition_ratio_mean", "unique_token_ratio_mean", "tool_call_count_mean", "tool_error_count_mean",
    "advantage_mean_mean", "return_mean", "empty_response_rate", "short_response_rate", "long_response_rate",
)
MULTIPLIERS = tuple(step / 4 for step in range(0, 17))  # k from 0.0 to 4.0
MAX_INNER_FALSE_ALARMS = 0.10
MIN_GROUP_NORMALS = 5
FOLDS = 5


def load(cache: pathlib.Path) -> list[dict]:
    runs = []
    for line in cache.read_text().splitlines():
        row = json.loads(line)
        group = row["group"]
        family = group[:2] if group[:2] in FAMILIES else None
        steps = len(row["steps"])
        trajectory = [row["trajectory"].get(str(step + 1), {}) for step in range(steps)]
        runs.append({
            "family": family,
            "setting": "hard" if group.endswith("hard_fault") else "easy" if group.endswith("easy_fault") else None,
            "config": (row["label"]["task_group"], row["trajectory"].get("1", {}).get("count", 0)),
            "task": row["label"]["task_group"],
            "paper_faulty": family is not None,
            "injected": row["label"]["injected"],
            "series": {feature: [entry.get(feature) for entry in trajectory] for feature in FEATURES},
        })
    return runs


def group_of(run: dict, normals: list[dict]) -> tuple:
    """Return the run's (task, batch) group with 5 shared normals, else its task with 2, else all normals."""
    if sum(1 for normal in normals if normal["config"] == run["config"]) >= MIN_GROUP_NORMALS:
        return run["config"]
    if sum(1 for normal in normals if normal["task"] == run["task"]) >= 2:
        return (run["task"],)
    return ("all",)


def profiles(normals: list[dict]) -> dict:
    groups = collections.defaultdict(list)
    for run in normals:
        groups[run["config"]].append(run["series"])
        groups[(run["task"],)].append(run["series"])
        groups[("all",)].append(run["series"])
    return {key: build_profile(members) for key, members in groups.items() if len(members) >= 2}


def normal_scores(normals: list[dict]) -> list[float]:
    """Score every normal out of sample: each fold by group profiles built without it (leave-one-out when few)."""
    folds = FOLDS if len(normals) >= 2 * FOLDS else len(normals)
    scores = []
    for fold in range(folds):
        rest = [run for index, run in enumerate(normals) if index % folds != fold]
        table = profiles(rest)
        scores += [score_run(run["series"], table[group_of(run, rest)]) for run in normals[fold::folds]]
    return scores


def f1_by_group(runs: list[dict], flagged: list[bool], faulty) -> float:
    values = []
    for setting in ("easy", "hard"):
        for family in FAMILIES:
            members = [i for i, run in enumerate(runs) if (run["family"] == family and run["setting"] == setting) or run["family"] is None]
            if not any(faulty(runs[i]) for i in members):
                continue
            tp = sum(flagged[i] and faulty(runs[i]) for i in members)
            fp = sum(flagged[i] and not faulty(runs[i]) for i in members)
            fn = sum(not flagged[i] and faulty(runs[i]) for i in members)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.fmean(values) if values else 0.0


def predict(train: list[dict], test: list[dict], faulty, multiplier: float) -> list[bool]:
    normals = [run for run in train if not faulty(run)]
    threshold = calibrate_threshold(normal_scores(normals), multiplier=multiplier)
    table = profiles(normals)
    return [score_run(run["series"], table[group_of(run, normals)]) > threshold for run in test]


def choose_multiplier(train: list[dict], faulty) -> float:
    """Pick k by inner CV: best average F1 among k whose inner false-alarm rate is at most the limit."""
    inner_folds = [train[fold::FOLDS] for fold in range(FOLDS)]
    outcomes = {k: ([], [], []) for k in MULTIPLIERS}
    for fold in range(FOLDS):
        inner_train = [run for other, part in enumerate(inner_folds) if other != fold for run in part]
        normals = [run for run in inner_train if not faulty(run)]
        scores = normal_scores(normals)
        table = profiles(normals)
        test = inner_folds[fold]
        test_scores = [score_run(run["series"], table[group_of(run, normals)]) for run in test]
        for k in MULTIPLIERS:
            threshold = calibrate_threshold(scores, multiplier=k)
            outcomes[k][0].extend(test)
            outcomes[k][1].extend(score > threshold for score in test_scores)
    ranked = []
    for k, (runs, flagged, _unused) in outcomes.items():
        negatives = [i for i, run in enumerate(runs) if not faulty(run)]
        false_alarms = sum(flagged[i] for i in negatives) / len(negatives)
        ranked.append((false_alarms <= MAX_INNER_FALSE_ALARMS, f1_by_group(runs, flagged, faulty), k))
    return max(ranked)[2]


def evaluate(runs: list[dict], protocol: str) -> dict:
    faulty = (lambda run: run["paper_faulty"]) if protocol == "A" else (lambda run: run["injected"])
    order = list(range(len(runs)))
    random.Random(0).shuffle(order)
    flagged = [False] * len(runs)
    chosen = []
    for fold in range(FOLDS):
        held = set(order[fold::FOLDS])
        train = [runs[i] for i in order if i not in held]
        k = choose_multiplier(train, faulty)
        chosen.append(k)
        test_indices = sorted(held)
        for i, flag in zip(test_indices, predict(train, [runs[i] for i in test_indices], faulty, k)):
            flagged[i] = flag
    table = {}
    for setting in ("easy", "hard"):
        for family in FAMILIES:
            members = [i for i, run in enumerate(runs) if (run["family"] == family and run["setting"] == setting) or run["family"] is None]
            tp = sum(flagged[i] and faulty(runs[i]) for i in members)
            fp = sum(flagged[i] and not faulty(runs[i]) for i in members)
            fn = sum(not flagged[i] and faulty(runs[i]) for i in members)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            table[f"{family} {setting}"] = [round(100 * precision, 2), round(100 * recall, 2), round(100 * f1, 2)]
    negatives = [i for i, run in enumerate(runs) if not faulty(run)]
    return {
        "protocol": protocol,
        "multipliers": chosen,
        "average_f1": {s: round(statistics.fmean(table[f"{f} {s}"][2] for f in FAMILIES), 2) for s in ("easy", "hard")},
        "false_alarms": f"{sum(flagged[i] for i in negatives)} of {len(negatives)}",
        "families": table,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    loaded = load(arguments.cache)
    for protocol in ("A", "B"):
        print(json.dumps(evaluate(loaded, protocol)))
