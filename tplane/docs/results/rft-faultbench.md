# RFT-FaultBench anomaly detection

This page scores `tplane.integrity.runscore` on RFT-FaultBench (arXiv 2605.04431; Zenodo record 20044882, md5 afceded9bbd8f2859833df8ba4a6f924). The task is to decide whether a whole RL fine-tuning run is normal or faulty. The run was made on 2026-09-24 on Della (`gputest`, job 14388472), with `scripts/rft_faultbench/evaluate.sbatch`.

## Result

Under the paper's own labels (protocol A), the detector's average F1 over the five fault families is **87.38 on easy runs and 73.73 on hard runs**. That is above every method in the paper's Table III:

| Average F1 over 5 families | Easy | Hard | False alarms on normals |
|---|---|---|---|
| IF / LOF (paper) | 68.60 / 35.06 | 7.15 / 8.77 | not reported |
| TranAD (paper) | 86.55 | 68.90 | not reported |
| Anomaly Transformer (paper) | 80.66 | 69.53 | not reported |
| RFT-FM (paper) | 80.66 | 70.75 | not reported |
| **tplane runscore** | **87.38** | **73.73** | 2 of 11 |
| flag every run faulty | 92.00 | 94.15 | 11 of 11 |

Read the "flag every run faulty" row before trusting any F1 here: **flagging everything beats every method**. Protocol A has 11 negative runs against 60 to 84 positive runs per family, so F1 rewards false alarms. Our 2 of 11 false alarms correspond to 96 to 97% precision; the paper's methods report 86 to 100% average precision.

Per family (P / R / F1):

| Family | Protocol A easy | Protocol A hard | Protocol B hard |
|---|---|---|---|
| RF | 96.6 / 95.0 / 95.8 | 96.4 / 64.3 / 77.1 | 91.3 / 100.0 / 95.5 |
| PG | 96.8 / 75.0 / 84.5 | 97.5 / 68.8 / 80.6 | 91.3 / 75.0 / 82.4 |
| OD | 95.8 / 76.7 / 85.2 | 96.1 / 58.3 / 72.6 | 88.2 / 71.4 / 79.0 |
| CA | 94.9 / 61.7 / 74.8 | 96.0 / 57.1 / 71.6 | 87.5 / 66.7 / 75.7 |
| TE | 96.7 / 96.7 / 96.7 | 95.6 / 51.2 / 66.7 | 91.3 / 100.0 / 95.5 |

## Label finding and protocol B

Each hard fault directory holds 28 runs, but only 14 carry an injected fault. The other 14 are the registry's `control` runs, with no `fault_injection` in any trajectory. The paper counts all 28 as faulty (for example, 53/84 recall for RF hard). Protocol B counts the 224 un-injected controls as normal, giving 235 normal runs and 544 faulty runs:

| | Easy F1 | Hard F1 | False alarms on normals |
|---|---|---|---|
| tplane runscore, protocol B | 91.64 | 85.58 | 4 of 235 (1.7%) |
| flag every run faulty, protocol B | 92.00 | 61.57 | 235 of 235 |

Hard under protocol B is the most informative cell. The easy splits contain no controls, so their only negatives are the 11 baselines under either protocol.

## Method

- **Unit.** A whole run of 5 to 40 steps (median 20). Every fault is injected from step 1, so this is not change-point detection.
- **Features.** 14 per-step trajectory aggregates: mean and sd of reward, mean and sd of response length, advantage sd and mean, repetition ratio, unique-token ratio, tool calls, tool errors, return, and the empty, short and long response rates. `fault_injection` is read only into labels.
- **Profile.** `build_profile` gives a per-step mean and one pooled spread per feature, from training-fold normals. Profiles are grouped by (task, batch size) when at least 5 normals share them, else by task when at least 2 do, else over all normals.
- **Score.** `score_run` computes log mean_f BF_f, where BF_f is the Bayes factor for a run-long shift δ ~ N(0, 0.3²) in feature f.
- **Threshold.** `calibrate_threshold` returns mean + k·sd of normal scores computed **out of sample**: 5 folds, or leave-one-out when normals are few. The multiplier k is chosen **empirically inside each outer training fold**. An inner 5-fold cross-validation picks the k in {0, 0.25, ..., 4} with the best average F1 whose inner false-alarm rate on normals is at most 10%. The chosen k per outer fold was 0, 1.75, 0, 0, 0 (protocol A) and 0, 0.25, 0.25, 0, 0 (protocol B).
- **Evaluation.** Outer 5-fold cross-validation over all 779 runs, seed 0. The outer test fold never influences its own profile or k.

## Ablations

These variants lost and were removed from the detector. The numbers come from the earlier exploration scripts, which used a fixed quantile or multiplier rather than nested selection. The scripts are at commit 93e9b99 (`git show 93e9b99:scripts/rft_faultbench/sweep.py`).

| Variant | Protocol A easy / hard F1 | Protocol B hard F1 | Finding |
|---|---|---|---|
| threshold from normals scored in sample | 90.69 / 79.16 | 79.93 | Invalid: it flagged 9 of the 11 normals. In-sample scores are optimistically low. |
| per-step spread, all features, out of sample | 55.13 / 37.41 | 79.93 | Baseline for the rows below. |
| step-level trainer metrics only (reward, KL, entropy, loss, lengths) | 14.50 / 3.59 | 79.70 | Much weaker than trajectory aggregates; removed. |
| + linear-trend Bayes factor | 60.32 / 49.49 | 79.07 | No gain; removed. |
| + variance-inflation Bayes factor | 55.10 / 47.48 | 79.07 | No gain; removed. |
| summed evidence across features instead of the mixture | about ±1 F1 | about ±1 F1 | No consistent gain; the mixture is kept. |
| **one pooled spread per feature instead of per step** | 78.23 / 69.31 | 86.38 | The largest gain. Per-step spreads from about 9 normal runs were noise. |
| **empirical k by nested cross-validation** (this page) | 87.38 / 73.73 | 85.58 | Replaces the hand-set k = 1.25. |

## What this does not show

- The published baselines were not re-run here. Their numbers are copied from Table III, whose threshold protocol is only partly described.
- The hard controls differ from the 11 baselines in configuration (128 against 64 trajectories per step). Under protocol A, a detector can therefore gain recall by noticing the configuration change rather than the fault. Protocol B does not reward that.
- The earlier exploration evaluated 738 variants on the same outer folds. The two design choices kept from it (trajectory features and the pooled spread) were chosen with that data, so they carry some selection bias. The threshold multiplier does not: it is chosen by nested cross-validation.
- The chosen k sits at its floor of 0 in most folds, which is the module's rule against thresholds below the mean normal score. The inner 10% false-alarm limit is what binds it.
- This benchmark has no change points: every fault starts at step 1. It does not test tplane's streaming change detectors.

## Better benchmarks

As of 2026-09-24, a search found no free public dataset of real RL runs with labelled infrastructure faults or reward poisoning that start mid-run, and none holding both rollout and trainer log-probabilities for the same tokens. The closest options are generators, not datasets:

- `ariahw/rl-rewardhacking-ext` (verl): reward hacking appears naturally around step 80 to 100, with matched healthy runs.
- VeXact against vLLM on/off pairs (arXiv 2605.14220): train-inference mismatch switched by one factor.
- Precision-RL FP16 against BF16 (arXiv 2510.26788): graded mismatch.

Building a labelled benchmark from these, with injected sandbox failures at a random step, is the next step for evaluating mid-run detection.
