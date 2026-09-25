# Trajectory plane build roadmap

> **For agentic workers:** This is the roadmap across all seven parts. Part 1 has its own step-level plan at `docs/superpowers/plans/2026-09-24-core-sdk.md`; execute that with superpowers:subagent-driven-development or superpowers:executing-plans. Parts 2 to 7 are specified here at task level and each gets its own step-level plan when its turn comes.

**Goal:** A Python SDK, command-line tool and runtime so that a researcher running RL never has to diagnose, mask or recover from a non-algorithm failure by hand.

**Architecture:** A typed outcome record per unit of work (a trajectory, a train step, an eval) is the centre of the design. Everything else consumes it: the policy that tells the trainer what to do with a failed unit, the cost ledger, the integrity monitor, fork and replay, fault injection, and the scale layer. The first part builds that record and the classifier that fills it, on a process backend that runs on a Slurm cluster with no virtual machines.

**Tech Stack:** Python 3.11, standard library only in the core (dataclasses, enum, json, subprocess, threading), optional `nvidia-ml-py` for GPU memory, `pytest` and `ruff` pinned for development. Later parts add `pyarrow`, `river`, `ruptures`, an OpenTelemetry exporter, and a microVM control plane (AgentENV or E2B runtime).

**Spec:** `rl-rollout-layer-strategy.md` (positioning, architecture, roadmap) and `rl-rollout-layer-repo-catalogue.md` (what to build on), both one directory above this repository.

## Global Constraints

- Python `>=3.11,<3.13`. No dependency in the core package; optional extras are pinned exactly.
- House style: the `mdx` skill applies to every file. Frozen keyword-only dataclasses, closed enums matched exhaustively, no `Any`, no silent fallback, errors that name the fix, tests named as sentences with at least half negative.
- Every wire format carries `schema_version` and rejects unknown keys.
- Every child process starts from an empty environment plus an explicit list.
- Output that reaches a file is sorted and deterministic; no timestamps in generated text except the recorded start and end of a unit.
- The core must run on a Slurm compute node with no daemon, no network and no `/dev/kvm`: file-based storage under the run directory.
- Package name `tplane`, command name `tp`.

## Status (2026-09-24)

Parts 1 to 3 are built on branch `parts-1-3`. The deviations from this roadmap, each for a stated reason:

- **Part 1.**
  - The Della smoke run (`docs/results/della-smoke.md`) found a leak: a failed unit pinned its step's GPU memory until the next garbage collection. It is fixed and confirmed on hardware.
  - A GPU OOM is not retried by default, so the "retry then mask" output in section 3 does not occur. Retrying the same batch usually runs out of memory again.
  - The classifier covers about 60 exception classes from 14 libraries (`docs/references.md`).
- **Part 2.**
  - The alarms use the Bayesian Shiryaev posterior instead of `river`'s ADWIN or Page-Hinkley. It gives the probability of a change directly, and an alarm threshold of 0.999 bounds false alarms by 0.001 when the model holds. `river` would also bring numpy, scipy and a compiled extension into the core.
  - Records gained a train-inference `mismatch` block (schema version 2; version 1 still reads).
  - A reward alarm is attributed to configuration or to the algorithm.
  - `grader_latency_p50_ms` is not computed, because records hold no per-stage timings.
  - RFT-FaultBench was run on Della (`docs/results/rft-faultbench.md`). Every fault starts at step 1, so the task needs a run-level test rather than the shipped change detectors. A run-level Bayesian shift test is level with the published hard results and below the best published easy result, under the paper's labels. The paper counts un-injected controls as faults.
- **Part 3.**
  - The adapters are tested against fakes with the field names of the pinned versions, not against the frameworks themselves, so the gate "green against pinned framework versions" is met only in that sense.
  - ATIF and OTLP carry the unit's outcome, not its steps, because tplane does not record steps.
  - The RFC is drafted (`docs/rfc/0001-typed-rollout-outcomes.md`) and not submitted.

---

## 1. Which function is the highest leverage, from first principles

A researcher feels a non-algorithm failure in three ways. The run dies and they must work out what died, where, and why. Progress is lost and they restart from a checkpoint or from scratch. Or, worst, nothing dies: a sandbox or grader failure is scored as a wrong answer and the reward signal is poisoned for hours (MiMo lost three hours of a $2.6M run this way; KAT-Coder found 16% of early trajectories tainted).

All three costs are addressed by one artifact: a **typed outcome record per unit of work**. The record says what the unit was, when it ran, what resources it used, whether it succeeded, and if not, whether the cause was the agent, the grader, the infrastructure or a timeout, with the evidence. A **policy** then turns that record into an action the trainer can apply without a human: score it, mask it out of the loss and the group baseline, zero it with a named exit status, retry it, or abort.

Every other part consumes that record:

| Part | What it adds | What it consumes from Part 1 |
|---|---|---|
| 2 Integrity monitor | Per-source pass rates with and without infra failures, change-point alarms | Records with `source`, `outcome`, `failure.kind`, `reward` |
| 3 Adapters and standards | prime-rl and slime adapters, OpenEnv RFC, ATIF and OTLP export | The record schema and the policy contract |
| 4 Fork and restore | MicroVM backend with snapshot DAG and fork-with-reseed | The `Unit` lifecycle and `Stage` boundaries |
| 5 Replay | Recorded LLM and tool responses, divergence report | Unit identity, stages, exec results |
| 6 Fault injection | Injectors at process, sandbox and grader layer | The classifier as the oracle under test |
| 7 Scale | Scheduler, density, 100k+ concurrency | Records as the telemetry stream |

Part 1 also needs no virtual machine, so it works today on a Slurm cluster. That is why it is first. A second reason: it is the smallest part that a researcher can adopt in one afternoon and feel the difference on the next run.

## 2. SDK design principles

**One nouns-and-verbs table, with every verb in exactly one group.** The API is small on purpose. A researcher should learn it in ten minutes.

| Group | Noun | Verbs | Where it lives |
|---|---|---|---|
| Lifecycle | `Run`, `Unit` | `tplane.Run(...)`, `run.unit(...)` as a context manager, `run.attempt(...)` for the retry loop | `unit.py` |
| Execution | `Unit` | `unit.stage(...)` context, `unit.exec(argv, ...)`, `unit.tokens(...)` | `unit.py`, `exec.py` |
| Outcome | `Unit`, `Failure` | `unit.reward(value)`, `unit.fail(failure)`; automatic classification of exceptions and exec results | `unit.py`, `classify.py` |
| Policy | `Policy`, `Decision` | `decide(failure, policy=..., attempt=...)` | `policy.py` |
| Observation | `Record`, `ResourceSummary`, `CostLine` | `unit.result`, `store.iterate()` | `schema.py`, `resources.py`, `cost.py`, `store.py` |
| Reading | command line | `tp show RUN_DIR`, later `tp integrity`, `tp replay`, `tp inject` | `cli/` |

**Robustness rules that hold in every part.**
- The classifier defaults unknown exceptions to `agent`, never to `infra`. A bug in the researcher's code must not be masked out of training (NeMo-RL's rule).
- A unit either produces a record or re-raises. It never exits with neither.
- Records are written atomically and create-exclusive. A crash mid-write leaves nothing.
- Every wait has a timeout. Every captured output is bounded. Child processes are killed and reaped on timeout.
- Resource sampling never blocks the unit; a sampler failure is recorded in the record, not raised in the middle of a rollout.
- A setting that cannot be honoured (GPU sampling requested with no NVML) is an error at `Run` construction, before any work.

**Modularity for scale.** Pure decision functions (`classify_exception`, `classify_exec`, `decide`, `summarise_samples`, `cost_line`, `build_record`, `render_show`) live in files that import no I/O. Effects (clock, samplers, process execution, storage) are parameters with production defaults. A later backend (microVM, Slurm job step, E2B) implements the same `exec` and sampler shape. A later store (Parquet, a service) implements the same two-method `RecordStore` protocol. The `Record` gains fields only by bumping `schema_version`.

**Pain points it solves directly.**
- "Was that my bug or the cluster?" The record says `infra.oom_gpu` with the exception line, the peak memory against the device total, and the longest sample in the batch.
- "It died at step 17 and I lost the step." The policy retried or masked and the run continued; the record shows what happened.
- "My rewards drifted and I only noticed a day later." Infra failures never reach the reward; Part 2 alarms per data source within minutes.
- "What did that run cost?" The ledger sums GPU-seconds and tokens per unit, and per run.

## 3. Using it on Della to see when you OOM

Della is a Slurm cluster. Compute nodes have no daemon you can run, GPFS home and scratch, Python from modules, and NVIDIA GPUs with NVML available through the driver. There is no `/dev/kvm` for you, so Parts 4 to 7 are not for Della. Part 1 is.

**What you do.** In the RL script, wrap each rollout batch and each train step in a unit, and construct one `Run` per Slurm job whose run directory is under `/scratch/gpfs/ARORA/hh9077/runs/<jobid>`:

```python
run = tplane.Run(
    run_id=os.environ["SLURM_JOB_ID"],
    store=tplane.FileRecordStore(Path("/scratch/gpfs/ARORA/hh9077/runs") / os.environ["SLURM_JOB_ID"]),
    options=tplane.RunOptions(policy=tplane.Policy(on_infra=tplane.Action.MASK, max_attempts=2), gpus_held=4, host_limit_bytes=host_limit),
    read_host=tplane.read_cgroup_memory_current_bytes_for(Path("/sys/fs/cgroup") / slurm_cgroup),
    read_gpu=gpu_reader.read,
    gpu_total_bytes=gpu_reader.total_bytes,
)
for step in range(steps):
    result = run.attempt(f"step-{step:05d}", kind=tplane.UnitKind.TRAIN_STEP, source="swe-tasks", body=lambda unit: train_step(unit, batch))
```

Inside `train_step`, the rollout is `with unit.stage(tplane.Stage.ROLLOUT): ...`, the training update is `with unit.stage(tplane.Stage.TRAIN): ...`, and the adapter records the sample lengths with `unit.tokens(tokens_in=..., tokens_out=...)`.

**What happens when a step OOMs on the GPU.** PyTorch raises `torch.OutOfMemoryError` inside the `TRAIN` stage. The unit catches it, classifies it as `infra` / `oom_gpu` with evidence lines (the exception type, the first line of the message, the peak GPU memory sampled every 250 ms against the device total, the longest sample in the batch), stops the sampler, computes the cost of the lost step, asks the policy, and gets `RETRY` on attempt 1 (because you set `max_attempts=2` and OOM on a fresh attempt sometimes clears fragmentation) then `MASK` on attempt 2. The record is written before control returns. Your loop continues. Nothing is scored zero and nothing enters the loss.

**What happens when the host OOM killer takes the process.** The sampler's last host reading is near the cgroup limit, the child exits with signal 9, and `classify_exec` returns `infra` / `oom_host` with the memory series as evidence. If the whole Python process is killed, the unit cannot write its record; the guard writes a `unit-started` marker file at entry so `tp show` can print "step-00017 started 14:02:11, no record: process died" instead of silence.

**What you see afterwards.**

```
$ tp show /scratch/gpfs/ARORA/hh9077/runs/4812233 --failures-only
unit         attempt kind        outcome class  kind      stage    peak_gpu        decision  cost
step-00017   1       train_step  failed  infra  oom_gpu   train    79.4/80.0 GiB   retry     4 gpu-min
step-00017   2       train_step  failed  infra  oom_gpu   train    79.6/80.0 GiB   mask      4 gpu-min
step-00023   1       train_step  failed  infra  network   rollout  41.0/80.0 GiB   retry     1 gpu-min
failures: 3 of 120 units; infra 3, agent 0, grader 0, timeout 0; gpu-hours lost 0.15
evidence step-00017/2: exception_type=OutOfMemoryError; message=CUDA out of memory. Tried to allocate 2.50 GiB; longest_sample_tokens=131072; peak_gpu_used_bytes=85459698688
headroom warning at step-00016: gpu memory headroom is 4% of 80.0 GiB, below the 10% warning threshold while the longest sample is 131072 tokens; lower max_tokens_per_gpu or gpu_memory_utilization before the next step
```

The last line is the guard: it warned one step before the OOM, in the log and in the record, with the fix named.

**What the record does not do on Della.** It does not prevent the OOM. Prevention is a configuration change the guard can only recommend. It does not survive a `scancel` of the whole job; the marker file tells you where it was.

## 4. Part 1: core SDK (highest leverage, build first)

**Deliverable.** `tplane` package with the outcome record, classifier, policy, resource sampling, cost ledger, file store, `Run`/`Unit` lifecycle, process execution, headroom guard, `tp show`, a verl-shaped rollout guard, and a Della smoke script. Step-level plan: `2026-09-24-core-sdk.md`.

**Files.**
- `src/tplane/schema.py`: enums and frozen dataclasses (`Failure`, `Record`, `ResourceSummary`, `CostLine`, `Decision`, `ExecResult`). Pure.
- `src/tplane/wire.py`: `render_record`, `parse_record`, `RecordParseError`. Pure.
- `src/tplane/classify.py`: `classify_exception`, `classify_exec`, signature table. Pure.
- `src/tplane/policy.py`: `Policy`, `decide`, `DEFAULT_POLICY`. Pure.
- `src/tplane/cost.py`: `PriceTable`, `cost_line`, `NO_PRICES`. Pure.
- `src/tplane/resources.py`: `ResourceSample`, `summarise_samples` (pure); file readers for `/proc` and cgroup v2; NVML reader; `Sampler` thread.
- `src/tplane/store.py`: `RecordStore` protocol, `FileRecordStore`, `StoreError`.
- `src/tplane/exec.py`: `exec_process`, `BASE_ENV`.
- `src/tplane/unit.py`: `RunOptions`, `Run`, `Unit`, `UnitFailed`, `UnitResult`, `build_record` (pure).
- `src/tplane/guard.py`: `headroom_warning` (pure).
- `src/tplane/cli/main.py`, `src/tplane/cli/show.py`: `render_show` (pure) and the entry point.
- `src/tplane/adapters/verl.py`: `guarded_rollout`, `masked_agent_loop_fields`.
- `scripts/della_smoke.py`, `README.md`.

**Tricky parts, and what to reference.**
- Exception chains: verl and Ray wrap exceptions; classify must walk `__cause__` and `__context__` (bounded depth) and match on type name, not on imported classes, so the core never imports torch. Reference: Python data model on exception chaining; `torch.OutOfMemoryError` naming (PyTorch 2.5+ renamed from `torch.cuda.OutOfMemoryError`).
- Host OOM evidence: cgroup v2 `memory.events` `oom_kill` counter and `memory.current`; Slurm's cgroup path per job differs by site (`/sys/fs/cgroup/system.slice/slurmstepd.scope/job_<id>` on cgroup v2 sites). Probe once at `Run` construction and error if the configured source is unreadable. Reference: Slurm cgroup.conf documentation; kernel `Documentation/admin-guide/cgroup-v2.rst`.
- NVML: `nvidia-ml-py` (the `pynvml` module) `nvmlDeviceGetMemoryInfo`; device order follows `CUDA_VISIBLE_DEVICES` only if mapped; record NVML indices and the visible-devices string in the record tags.
- Sampling thread and CUDA: NVML calls are safe from a thread; never call torch from the sampler.
- Suppressing exceptions in `__exit__`: only for `Exception`, never `KeyboardInterrupt` or `SystemExit`; only when the decision is not `ABORT`; and the record must be written before suppression. Test all three.
- Atomic create-exclusive write: temp file in the same directory, fsync, `os.link` to the final name (fails if it exists), unlink temp. Reference: the `mdx` systems reference on atomic writes.
- Bounded output capture: redirect child stdout and stderr to temporary files and read only the tail. `communicate()` is unbounded.
- verl field names: `AgentLoopOutput` fields (`prompt_ids`, `response_ids`, `response_mask`, `response_logprobs`, `reward_score`, `num_turns`, `metrics`, `extra_fields`) must be verified against the pinned verl version at integration time. Reference: verl docs, "Agent Loop" page.
- The guard's warning is a fraction of device memory. It does not predict the next step's peak; say so in the message and the docs.

**Acceptance.** `pytest` green with every gating flag combination covered; a MiMo-style replay test where a rollout that raises a sandbox connection error is recorded `infra/sandbox`, masked, and its reward is `None`; a Della smoke run (opt-in) that allocates until CUDA OOM and produces the `tp show` output above.

## 5. Part 2: reward-integrity monitor (second)

**Why second.** Silent reward poisoning is the most expensive failure and the one no open-source tool detects. It needs only Part 1's records.

**Deliverable.** `tplane.integrity` with per-source metrics (`avg_reward`, `avg_reward_no_infra`, `zero_pass_rate`, `infra_error_rate`, `grader_error_rate`, `grader_latency_p50_ms`), online change-point alarms, and `tp integrity RUN_DIR`.

**Files.** `src/tplane/integrity/metrics.py` (pure aggregation over records, one window), `src/tplane/integrity/alarms.py` (pure: ADWIN or Page-Hinkley from `river`, wrapped so the core stays testable with hand-written series), `src/tplane/integrity/report.py` (pure text), `src/tplane/cli/integrity.py`, `tests/`.

**Interfaces.** Consumes `Record` (`source`, `outcome`, `failure`, `reward`, `started_at_ms`). Produces `SourceWindow`, `Alarm(source, metric, at_ms, before, after, reason)`.

**Tricky.** Windows must be by unit count, not wall time, because rollouts are bursty. The "no infra" average must exclude both `infra` and `grader` failures and never `agent` failures, or the metric hides reward hacking. Alarm thresholds are config with a small legal set; validate the RFT-FaultBench dataset detection delay and false-alarm rate and record both in `docs/`. References: MiMo dashboard metric definitions (redreamality analysis, 2026-09-21); `river` drift detectors (`ADWIN`, `PageHinkley`); `ruptures` for offline localisation; RFT-FaultBench (arXiv 2605.04431) for validation data.

**Tasks.** (1) `SourceWindow` aggregation with a table test over outcome and failure-class combinations. (2) Rolling windows by count with boundary tests at zero, one, N-1, N, N+1. (3) `river`-backed alarm with a fake series where the alarm fires at the planted change and a control series where it does not. (4) Report rendering. (5) `tp integrity` CLI. (6) Validation script against RFT-FaultBench with the numbers committed.

**Acceptance.** On a synthetic MiMo-incident series (300 units, infra failures scored as zero from unit 200), the alarm fires within 20 units; on the corrected series (infra masked) it does not fire.

## 6. Part 3: adapters and the standard (third)

**Why third.** Distribution. The record only compounds if verl, prime-rl and slime users get it by default, and if the schema is the one OpenEnv adopts.

**Deliverable.** `tplane.adapters.prime_rl`, `tplane.adapters.slime`, ATIF export (`tplane.export.atif`), OTLP export (`tplane.export.otlp`), an OpenEnv RFC draft in `docs/rfc/`, Parquet export for the training path.

**Interfaces.** Adapters consume `Run` and produce the framework's native sample with the decision applied: prime-rl's `DispatchFailure` mapped to our `Failure`; slime's `Sample.status` `FAILED` mapped with a named `exit_status` (the miles convention). ATIF export adds an `extra.tplane` block per step and an episode-level `failure` object; OTLP export emits `invoke_agent` and `execute_tool` spans with `rl.failure_class`, `rl.failure_kind`, `rl.decision`, `rl.unit_id`.

**Tricky.** slime's `ABORTED` means "partial rollout to resume", not "infra"; do not conflate. prime-rl drops failed rollouts before the trainer sees them; the adapter must run inside `verifiers`' rollout, not after. ATIF has no error field; extend under `extra` and propose the field upstream. References: verifiers error hierarchy and issue #2660; miles PRs #2801 and #2802; NeMo-RL `experience.failures`; Harbor ATIF RFC 0001; OTel GenAI issue #338; OpenEnv RFC 002, 004, 012.

**Tasks.** (1) prime-rl adapter with fakes for `Episode` and `DispatchFailure`. (2) slime adapter mapping `Status` and `exit_status`. (3) ATIF writer with a round-trip test against the published example. (4) OTLP exporter with an in-memory span collector. (5) Parquet writer with dictionary-encoded `failure_kind` and a predicate-pushdown test. (6) RFC draft with the schema, the discard rule and the reference adapter list.

## 7. Part 4: sandbox backend with fork-with-reseed (fourth)

**Why fourth.** It is the first part that needs KVM and a control plane. It unlocks tree search, restore-to-fixed-state and cheap parallel rollouts, which are real pains but felt after the classification pain.

**Deliverable.** `tplane.backends.microvm` over AgentENV or E2B runtime (decided by a two-week spike measuring restore p50 and p99, memory per idle instance, fork fan-out to 64, and reseed correctness), a content-addressed snapshot DAG, `unit.seal()` and `run.fork(snapshot, count)`, and per-branch copy-on-write accounting in the cost line.

**Interfaces.** The backend implements the same `exec` shape as `exec.py` plus `seal() -> SnapshotId` and `fork(SnapshotId, count) -> tuple[SandboxHandle, ...]`. `Record.sandbox = SandboxInfo(provider, image_digest, snapshot_id, fork_parent)` under a new `schema_version`.

**Tricky.** Firecracker's documentation warns that restoring one snapshot into many VMs duplicates RNG state and tokens; every branch must reseed the guest RNG, regenerate machine identifiers and rotate any credentials, and a test must prove two forks produce different random bytes and different session tokens. The kernel clock after restore must be pinned or advanced deliberately. NVML has no place here; GPU tier is a separate backend on gVisor plus cuda-checkpoint. References: Firecracker snapshot docs and VMGenID note; AgentENV README; e2b-dev/runtime orchestrator and `fork`; PandaStack blog numbers; DSec paper for density.

**Tasks.** (1) Spike harness script with the four measurements and a committed results file. (2) Backend `exec` over the chosen control plane with the hermetic environment rule. (3) `seal` and snapshot DAG with content-addressed ids. (4) `fork` with reseeding and the two-forks-differ test. (5) Cost accounting per branch. (6) Schema bump with a parse test that rejects the old version's missing `sandbox` block only when the version says it must exist.

## 8. Part 5: replay and divergence (fifth)

**Deliverable.** An egress recorder (mitmproxy-class, also the allowlist point), a cassette of LLM and tool responses keyed by a canonical request hash, `tp replay UNIT` that restores the sealed snapshot, replays responses, re-runs only the grader, and reports the first divergent step.

**Interfaces.** Consumes `Record`, snapshot ids from Part 4, the `Stage` boundaries. Produces `ReplayReport(unit_id, divergent_step, expected_hash, actual_hash)`.

**Tricky.** LLM sampling is not bit-exact across batch compositions; replay from the cassette by default and only compare when the user asks, reporting divergence as a metric, not a failure. Matching must ignore volatile fields (timestamps, request ids). Side effects outside HTTP need the snapshot, which is why this part follows Part 4. References: Inspect's cache key; Helicone's request fingerprint; mitmproxy server replay options; SGLang `--enable-deterministic-inference`; Belayer's prefix-consistent recovery.

**Tasks.** (1) Canonical request hash with a table test over volatile fields. (2) Cassette store, append-only, content-addressed. (3) Recorder proxy with an allowlist and a test that a non-allowlisted host is refused and recorded. (4) Replayer that serves from the cassette and reports misses. (5) `tp replay` with a divergence report on a planted change.

## 9. Part 6: fault injection (sixth)

**Deliverable.** `tp inject` with injectors at the process layer (kill, freeze, slow disk via a wrapper, clock skew), the sandbox layer (TAP netem, block-device delay, pause), and the grader layer (corrupt, delay, drop responses using the WireMock fault vocabulary), plus a pre-flight report that runs a recorded batch against a scaled-down pipeline and grades the classifier's answers.

**Interfaces.** Consumes the classifier and the policy as the system under test. Produces `InjectionReport(fault, expected_kind, observed_kind, decision)`.

**Tricky.** The pre-flight harness is only meaningful if the classifier can be wrong; every fault case needs a planted misclassification control. Grader-layer faults must be reproducible: seed the fault schedule. References: Toxiproxy toxic list; Chaos Mesh fault classes; RFT-FaultBench's 16 reward faults; Belayer's LD_PRELOAD approach.

**Tasks.** (1) Process-layer injectors with real subprocesses. (2) Grader-layer injector as a wrapper around any reward function. (3) Fault schedule with a seed and a determinism test. (4) Report with the confusion matrix of expected versus observed failure kinds. (5) `tp inject` CLI.

## 10. Part 7: scale (last)

**Deliverable.** A scheduler that keeps Kubernetes out of the hot path, warm pools per image, placement by snapshot locality, per-tenant nested quotas, preemption-tolerant sandbox-and-worker pairs, and the density recipe (virtio-pmem DAX, DAMON, free-page reporting, SCHED_IDLE with core scheduling, EROFS images). Targets: 10k concurrent by month 9, 100k by month 18.

**Tricky.** Everything here is measured, not designed: each change ships with a benchmark that names hardware, method, raw samples and the unfavourable result first. References: DSec paper for the recipe and numbers; Modal's "1M concurrent" post for the no-k8s scheduler; AgentENV for overcommit; Aquifer for the CXL future.

**Tasks.** (1) Load generator that replays Part 1 records as synthetic units. (2) Warm pool with a fill-rate benchmark. (3) Placement by snapshot locality with a hit-rate test. (4) Quotas with a table test over nested limits. (5) Preemption survival test with a killed worker and a surviving sandbox. (6) Density recipe applied one knob at a time with before-and-after numbers.

## 11. Sequencing and gates

| Gate | Evidence required |
|---|---|
| Part 1 done | `pytest` green; the sandbox-error replay test; the Della smoke run output committed to `docs/results/` with the job id and the node type |
| Part 2 done | Alarm fires on the synthetic incident within 20 units and stays silent on the control; RFT-FaultBench numbers committed |
| Part 3 done | Three adapters green against pinned framework versions; ATIF round trip; RFC submitted |
| Part 4 done | Spike results committed; two-forks-differ test; restore p50 under 100 ms on the chosen control plane, method and raw samples committed |
| Part 5 done | Replay of 1,000 recorded units with the divergence rate and its breakdown committed |
| Part 6 done | Confusion matrix over every `FailureKind` with no planted misclassification missed |
| Part 7 done | 10k concurrent on one cluster with the benchmark file |

Parts 2 and 3 can run in parallel after Part 1. Part 4 can start its spike during Part 2. Parts 5 and 6 depend on Part 4. Part 7 depends on everything.
