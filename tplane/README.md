# tplane

`tplane` records one typed outcome per unit of RL work: a trajectory, a train step or an eval. When a unit fails for a reason that is not the algorithm's, it tells the trainer what to do: mask the unit, zero it with a named cause, retry it, or abort. It runs on a Slurm compute node with no daemon and no network, and it has no runtime dependencies.

## Install

    pip install -e ".[dev]"        # core, tests, linters
    pip install -e ".[gpu]"        # adds NVML sampling (nvidia-ml-py)
    pip install -e ".[parquet]"    # adds Parquet export (pyarrow)

## Ten-minute tour

    import os
    from pathlib import Path
    import tplane

    gpu = tplane.create_nvml_gpu_reader()          # or tplane.no_gpu_reader()
    run = tplane.Run(
        run_id=os.environ.get("SLURM_JOB_ID", "local"),
        store=tplane.FileRecordStore(Path("runs/local")),
        options=tplane.RunOptions(policy=tplane.Policy(on_infra=tplane.Action.MASK, max_attempts=2), gpus_held=1),
        read_host=tplane.proc_status_reader(os.getpid()),
        read_gpu=gpu.read,
        gpu_total_bytes=gpu.total_bytes,
    )

    def rollout(unit: tplane.Unit) -> None:
        with unit.stage(tplane.Stage.ROLLOUT):
            response = policy_model.generate(prompt)
            unit.tokens(tokens_in=len(prompt), tokens_out=len(response))
        with unit.stage(tplane.Stage.GRADE):
            unit.reward(grade(response))

    result = run.attempt("traj-000001", kind=tplane.UnitKind.TRAJECTORY, source="swe-tasks", body=rollout)
    print(result.record.decision.action, result.record.failure)

Then read the run from the shell:

    tp show runs/local --failures-only

## What a unit guarantees

- Each attempt writes exactly one record, atomically, before control returns. A second write of the same attempt is an error.
- An exception inside the unit is classified by walking its explicit cause chain (`raise ... from ...`) and matching class names, including base classes. An error raised while handling another is classified as itself. Known infrastructure signatures become `infra`: CUDA out of memory, host `MemoryError`, NCCL, connection errors, sandbox errors and Xid. `TimeoutError` becomes `timeout`. Anything else is `agent` and is never masked.
- A trajectory or eval that ends without `unit.reward(...)` is a `grader/missing_reward` failure. It is never a score of zero.
- `KeyboardInterrupt` and `SystemExit` propagate without a record. The start marker stays, so `tp show` reports the unit as started with no record.
- `unit.exec(argv, ...)` runs a child from exactly the environment you pass. A timeout kills and reaps the child. Only the last 64 KiB of each stream is kept. A non-zero exit code is returned to you. A timeout, a signal or an OOM signature ends the unit.
- Memory is sampled every 250 ms on a background thread. A failed read is counted in the `sampler_errors` tag and never raised into the unit.
- Misuse of tplane inside a unit, such as `unit.reward(nan)`, is your bug, not the agent's. It propagates without a record, and is never zeroed into the training data.
- An `abort` decision raises `UnitAborted` after the record is written, even when no exception was in flight.
- Starting an attempt that already has a record is refused before its body runs. A start marker left by a killed job is refused with the fix named: run the next attempt number, or delete the marker if no process still runs it.
- `Run` reads each memory source once at construction, so an unreadable cgroup or GPU reader fails before any work.
- A timeout kills the child's whole process group, grandchildren included.

## Policy

`Policy(on_agent, on_grader, on_infra, on_timeout, max_attempts)` takes one of `abort`, `mask` or `zero` for each failure class.
- Defaults: agent `zero`, grader `mask`, infra `mask`, timeout `zero`, one attempt.
- Network, sandbox, killed-process and engine-crash failures are retried while attempts remain.
- A GPU OOM is not retried, because the same batch usually runs out of memory again.

Your trainer applies the decision. The adapters below write it into each framework's own sample. The trainer-side filter that drops masked samples from the group baseline is yours to add in every framework.

## Adapters

None of the adapters imports its framework. Each maps fields whose names were read from the version listed in `docs/references.md`, and each is tested against fakes with exactly those fields. Check the names again when you pin a different version.

| Framework | Module | What it does |
|---|---|---|
| verl v0.9.1 | `tplane.adapters.verl` | `guarded_rollout` runs a rollout in a unit, with retries. `masked_agent_loop_fields` returns `AgentLoopOutput` fields with no response tokens, and the decision in `extra_fields`. |
| slime v0.3.2, miles #2802 | `tplane.adapters.slime` | `sample_fields(record)` returns `reward` and `remove_sample`, plus `status="failed"` for a masked failure (never `aborted`, which means resume), and `metadata["exit_status"]` in the miles vocabulary. |
| prime-rl, verifiers v0.3.1 | `tplane.adapters.prime_rl` | `record_trace` and `record_dispatch_failure` write a record from a `Trace` or `DispatchFailure`, inside the orchestrator, before errored traces are dropped. `is_trainable(record)` tells the orchestrator what to keep. When the decision is `retry`, dispatch the task again with the next attempt number. |

## Export

| Format | How | Notes |
|---|---|---|
| Harbor ATIF v1.8 | `tplane.export.atif.annotate_atif(trajectory, record)` | Adds the record under root `extra.tplane`; `record_from_atif` reads it back. ATIF has no outcome field. |
| OTLP/JSON | `tp export otlp RUN_DIR --output traces.json` | One `invoke_agent` span per attempt, with GenAI and `rl.*` attributes. POST the file to a collector's `/v1/traces`. |
| Parquet | `tp export parquet RUN_DIR --output records.parquet` | One row per attempt. Categorical columns are dictionary-encoded per column chunk, so `failure_class IS NULL` can skip row groups. |

`tp export` never replaces an existing file.

The OpenEnv RFC draft that proposes this outcome as a standard is `docs/rfc/0001-typed-rollout-outcomes.md`. It has not been submitted.

## Reward integrity

`tp integrity RUN_DIR` answers two questions: are the rewards still honest, and when a reward changes, was it the configuration or the algorithm? It prints per-source windows of MiMo's dashboard metrics, then every alarm. It exits 1 when any alarm fired, so a cron job can watch a run.

    tp integrity runs/local --healthy-kl 1e-3 --broken-kl 1e-2

Each signal is a Bayesian change detector, the Shiryaev posterior P(a change has happened | data). An alarm fires when that posterior reaches 0.999. When the model's likelihoods hold, at most 0.1% of changes are preceded by a false alarm.

**Evidence is counted per training step, with prompt groups as clusters.** Tag each trajectory with its training step (`unit.tag("step", str(step))`) and its prompt group (`unit.tag("group", uid)`). Rollouts of one prompt share its difficulty, so per-rollout evidence would overcount and overflag. For runs without steps, pass `--batching units`.

| Signal | Scope | Evidence per step |
|---|---|---|
| `failure_rate` | per source | infra or grader failures: Binomial 2% (healthy) against 16% (broken), the KAT-Coder before-and-after rates. The log-likelihood ratio is divided by a Rao-Scott design effect estimated from the prompt groups, so a burst inside one prompt counts once. |
| `poisoned` | per source | an infra or grader failure whose zero reached the trainer; observed directly, no probability needed |
| `reward` | per source | the clean-reward step mean under a Student-t predictive. Its location is the mean of the 20 steps that end 10 steps earlier. Its scale comes from up to 200 lagged steps by mean squared successive difference. Only a drop of 0.25 per-rollout sd alarms. |
| `mismatch` | run | a mixture of two failure modes: log10 KL(rollout, train) near 1e-2 instead of 1e-3, or a mean importance ratio below 1 |

**Why the importance ratio.** A token sampled from the rollout engine q with trainer probability p satisfies E_q[p/q] = 1 exactly when both share support, however large the numeric mismatch. Top-p or top-k truncation without renormalising the trainer makes the mean ratio p(nucleus) < 1. The same truncation biases the k3 KL estimate low, so the ratio catches what the KL misses. The derivation is in `src/tplane/mismatch.py`.

**Recording mismatch.** Inside a train step, compute per-sequence sums in your tensor library and pass them in:

    delta = (old_log_prob - rollout_log_probs).clamp(-20, 20)          # same weights, so this is pure mismatch
    ratio, k3 = delta.exp() * mask, (delta.expm1() - delta) * mask
    sums = [tplane.SequenceSums(tokens=t, ratio_sum=r, k3_sum=k) for t, r, k in
            zip(mask.sum(-1).tolist(), ratio.sum(-1).tolist(), k3.sum(-1).tolist())]
    unit.mismatch(tplane.summarise_mismatch(sums))

Use trainer log-probabilities computed with the weights that generated the rollout: verl's decoupled `old_log_prob`, not the current policy. Otherwise the KL also contains the policy update.

**Attribution.** A reward alarm carries P(config) = 1 - (1 - P(failure-rate change)) (1 - P(mismatch change)), read 50 units after the alarm. The verdict is:
- `configuration` when P(config) >= 0.999;
- `algorithm` when P(config) < 0.5;
- `undetermined` otherwise.

The attribution assumes the two configuration causes are independent, and that a reward drop is equally likely under either hypothesis. `algorithm` means that tplane saw no configuration evidence. It does not prove the algorithm is at fault.

**Measured on synthetic batched runs** (`docs/results/integrity-validation.md`, 50 seeds per row, unfavourable results first). The runs have 8 prompts × 4 rollouts per step, correlated prompts and learning drift:
- **Small drops are often missed.** A pass-rate drop from 0.5 to 0.35 was missed in 26 of 50 runs within 150 steps. A drop from 30% sandbox failures (70% untyped) was missed in 36 of 50. This is the price of not overflagging.
- **Burst runs raise one false alarm.** On 100,000 healthy steps, the reward stream raised 1 false alarm, in the run where one prompt group fails every fourth step. The other streams raised none. The per-rollout model it replaced raised 23.
- **Configuration drops are called configuration.** For 50% sandbox failures with 70% untyped, the reward alarm fires after a median of 9 steps, and all 50 verdicts were `configuration`.
- **Algorithm drops are called algorithm.** A learning collapse (0.5 to 0.2) is attributed to `algorithm` in 50 of 50 runs. When every failure is untyped, nothing distinguishes it from a collapse, and all 50 verdicts were `algorithm`.
- **Fast signals stay fast.** A typed failure rate of 20% alarms within 1 step (median). A KL jump to 3e-2 alarms within 4 steps.

These runs come from generators that match the detector's assumptions. They show the detector works as designed, not how well it does on a real run. The defaults are starting points to tune on real runs. MoE stacks sit near 1e-3 to 3e-3 KL when healthy, so raise `--healthy-kl` for them.

**Runs broken from the start.** The change detectors above look for a change within a run. For a run that is faulty from its first step, `tplane.integrity.runscore` compares the whole run with a profile of normal runs: a per-step mean and one pooled spread per feature, and a Bayes factor for a run-long shift. Its threshold is calibrated on normal runs scored out of sample. On RFT-FaultBench, under the paper's labels, it averages F1 87.38 (easy) and 73.73 (hard), above every published method there; see `docs/results/rft-faultbench.md` for the imbalance caveat and the label finding.

## Development gate

    ruff format --check src tests scripts && ruff check src tests scripts && mypy src && pytest -q

`python scripts/integrity_validation.py` regenerates `docs/results/integrity-validation.md` in about 30 seconds.

## On Della

`scripts/della_smoke.sbatch` submits a ten-minute job. The job fills one GPU inside a unit until CUDA runs out of memory, then prints the failure timeline. Submit it from the repository root, after `pip install -e ".[gpu]"` in your module environment.

The script finds the Slurm job cgroup from `/proc/self/cgroup` and reads the host memory limit from `memory.max`. If the node is not cgroup v2, it stops and says so.

## What tplane does not do

- It does not prevent an OOM. `unit.check_headroom(longest_tokens=...)` warns when free GPU memory is under 10% and names the setting to lower. It does not predict the next step's peak.
- It does not survive `scancel` of the whole job. The start marker tells you which unit was running.
- It does not snapshot, fork or replay environments. It does not price tokens without a `PriceTable`.
- It does not measure grader latency. Records hold the failure stage, not per-stage timings.
- It does not record steps, tool calls or token ids, so its ATIF and OTLP exports carry the outcome of a unit, not its trajectory.
- A reward drop that `tp integrity` misses within 100 units starts to enter its reference and can go unreported.
