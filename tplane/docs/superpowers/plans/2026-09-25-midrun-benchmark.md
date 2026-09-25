# Mid-run integrity benchmark plan

**Goal:** a benchmark of real RL training runs, with faults that start partway through, that tests whether `tp integrity` detects configuration failures quickly, attributes them correctly, and does not overflag healthy training.

**Why:** RFT-FaultBench injects every fault from step 1, has 11 normal runs, and has no train-inference mismatch. No public dataset has real runs with labelled mid-run infrastructure faults (`docs/results/rft-faultbench.md`, benchmark search of 2026-09-24).

**Substrate:** `ariahw/rl-rewardhacking-ext` at commit `0938cb6`. That is verl 0.6.1, Qwen3-4B with LoRA and GRPO, on the `leetcode_rh` coding environment. Each step runs 16 prompts × 16 rollouts, with temperature 0.7 and a maximum of 1536 completion tokens. Code is graded by local subprocess execution. In the loophole variant, reward hacking appears on its own partway through training.

## Desired behaviours

These are fixed before any run. Each becomes a check in the analysis script.

| ID | Behaviour | Pass condition |
|---|---|---|
| D1 | No overflagging on healthy training | No alarm on any healthy run, null branch, or pre-onset segment. The false-alarm rate is reported per 1,000 steps, with its interval. |
| D2 | Poisoning is seen at once | A branch whose typed sandbox failures are zeroed raises `poisoned` at the first zeroed failure. A masked branch never raises it. |
| D3 | Infra failure rate | Typed sandbox and grader faults raise `failure_rate`. The delay is reported. |
| D4 | Attribution: configuration | A reward drop caused by partly typed sandbox failures, or by a mismatch, gets the verdict `configuration`. |
| D5 | Attribution: algorithm | A reward drop caused by a learning-rate spike gets the verdict `algorithm`, never `configuration`. |
| D6 | Reward hacking is not called infra | Loophole runs raise no `failure_rate`, `poisoned` or `mismatch` alarm, and never get a `configuration` verdict. |
| D7 | Mismatch | A top-p (ratio mode) or temperature (KL mode) rollout mismatch raises `mismatch`. Healthy runs, loophole runs and the learning-rate branch never do. |
| D8 | Calibration | Real healthy runs set the empirical false-alarm rate, which is compared with the model's alpha. Changing a detector setting must be justified by theory, never by this benchmark's fault branches. |

One case is not identifiable, and the benchmark records it as a limit. When sandbox failures are 0% typed, nothing distinguishes them from a learning collapse. The expected verdict there is `algorithm` or `undetermined`, and that is not a pass.

## Arms

All arms use the repository defaults. Faults are one factor each, forked from a healthy checkpoint so that everything else is identical.

| Arm | Runs | Steps | Purpose |
|---|---|---|---|
| H: `rl_baseline` (no loophole) | 3 seeds | 200, checkpoint every 25 | Negatives, and the fork points |
| R: `no_intervention` (loophole) | 2 seeds | 200 | Reward hacking that appears on its own (D6) |
| N: null branch | 1 per fork | 60 | Resume with no fault; controls for resume effects (D1) |
| F1a: sandbox outage, 30% of executions fail, 30% of those typed and masked, the rest silently scored 0 | 1 per fork | 60 | MiMo (D3, D4) |
| F1b: sandbox outage, typed and masked | 1 per fork | 60 | Correct handling (D2, D3) |
| F1c: sandbox outage, typed and zeroed | 1 per fork | 60 | Poisoning (D2) |
| F1d: sandbox outage, 0% typed | 1 per fork | 60 | The unidentifiable case (recorded limit) |
| F2: grader missing reward on 20% of samples | 1 per fork | 60 | Grader failure (D3) |
| F3a: rollout `top_p` 1.0 → 0.9, trainer unchanged | 1 per fork | 60 | Ratio-mode mismatch (D7) |
| F3b: rollout temperature 0.7 → 1.0, trainer unchanged | 1 per fork | 60 | KL-mode mismatch (D7) |
| F4: learning rate × 20 | 1 per fork | 60 | Algorithm drop (D5) |

- Forks are taken at step 100 of H seeds 1 and 2: 2 forks × 9 branches = 18 branches.
- Total: 3×200 + 2×200 + 18×60 ≈ 2,080 steps. The step time is measured by a validation job before any long run is submitted.
- Semi-synthetic replay: the injections are re-applied at random onsets to the logged H runs on CPU. This gives many replicates for delay curves. The real branches check that replay matches reality.

## Model changes, justified by theory before any fault result (done 2026-09-25)

The per-rollout model raised 23 false alarms in 100,000 healthy synthetic steps once prompt correlation and learning drift were added. After the changes below it raises 1 (`docs/results/integrity-validation.md`).

1. **Step-level observations** (`integrity/steps.py`), with prompt groups (verl `uid`) as clusters, and the hazard counted per step.
2. **Failure rate:** a binomial ratio per step divided by an estimated Rao-Scott design effect.
3. **Reward:** a Student-t predictive. Its location comes from the recent lagged steps. Its scale comes from up to 200 lagged steps by mean squared successive difference, because a short window reused for many steps shares one estimation error.

## Instrumentation (patch on the substrate, kept in `benchmarks/midrun/`)

- In the trainer, after `old_log_prob`, record one train-step record with `summarise_mismatch`. The per-sequence sums come from `rollout_log_probs`, `old_log_probs` and `response_mask` (`calculate_log_probs: true`).
- After `compute_reward`, record one trajectory record per rollout: `source=leetcode_rh`, reward, typed failure, decision and `uid` tag. Records go to one JSONL file per step.
- Fault switches are read from environment variables at each step: `FAULT`, `FAULT_START`, `FAULT_RATE`, `FAULT_TYPED`.

## Compute and operations

- Della, under `/scratch/gpfs/ARORA/hh9077/cado`.
- Every long run is resumable from its checkpoint. A validation job on `gputest` must first reach every phase: rollout, reward, records, checkpoint and resume, with every fault switched on once.
- Launch, compare and submit are sequenced by a detached script on the cluster, with no long-lived SSH.
- The benchmark and caches are deleted when the project is done.
