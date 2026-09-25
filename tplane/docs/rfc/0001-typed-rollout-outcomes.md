# RFC: Typed rollout outcomes

- Status: draft, not submitted
- Target: OpenEnv (huggingface/OpenEnv), after RFC 002 (the environment API) and RFC 012 (Harbor capture)
- Reference implementation: `tplane` 0.1.0 (this repository)

## Summary

This RFC proposes a typed outcome for every rollout, eval and train step, and one rule for what a trainer does with it. The outcome says whether the unit succeeded. If it did not, the outcome says who failed: the agent, the grader, the infrastructure or the clock. It also records the stage and the evidence. The rule stops infrastructure and grader failures from entering the reward or the group baseline.

## Motivation

Today a sandbox that runs out of memory and a wrong answer look the same to the trainer.
- OpenEnv's error model is transport-level only (`WSErrorCode`). A step result has no error field (RFC 002). A rubric exception propagates (RFC 004).
- verl's `AgentLoopOutput` (v0.9.1) has no status or error field.
- slime's `Sample.Status.FAILED` is documented but never set.
- prime-rl drops errored traces before the trainer. This keeps the loss clean, but it hides the failure rate.
- verifiers issue #2660 scores an unreadable reward file as `solved = 0.0`.

The cost is measured. MiMo-V2.6 restarted a run after an undetected infra error on one dataset. KAT-Coder found about 16% of early trajectories tainted by the sandbox.

RFC 012 already requires that "missing or failed grading must stay distinguishable from zero". This RFC gives that rule a schema.

## Specification

### The outcome

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | 2 for this draft; readers accept 1 and 2 and reject unknown keys |
| `unit_id`, `attempt` | string, int >= 1 | identity; one outcome per attempt |
| `kind` | `trajectory`, `eval`, `train_step` | trajectories and evals must carry a reward or a failure |
| `outcome` | `ok`, `failed` | `failed` exactly when `failure` is present |
| `failure.class` | `agent`, `grader`, `infra`, `timeout` | who is responsible; derived from `failure.kind` |
| `failure.kind` | closed set of 13, for example `oom_gpu`, `sandbox`, `missing_reward` | the specific cause |
| `failure.stage` | `setup`, `rollout`, `tool`, `grade`, `train` | where it happened |
| `failure.retryable` | bool | whether a new attempt can succeed |
| `failure.evidence` | sorted unique strings | for example `exception_type=OutOfMemoryError` |
| `decision.action` | `score`, `zero`, `mask`, `retry`, `abort` | what the trainer does |
| `reward` | float or null | the reward the trainer sees: its own on `score`, 0 on `zero`, null otherwise |
| `mismatch` | optional | train-inference KL and mean importance ratio per train step |

The full wire format is `src/tplane/wire.py`. The kind-to-class table is `src/tplane/schema.py`.

### The rules

1. An unknown exception is an `agent` failure, never `infra`. A bug in the researcher's code must not be masked out of training. This is NeMo-RL's rule.
2. A trajectory or eval that ends without a reward is `grader/missing_reward`. It is never a reward of 0.
3. `infra` and `grader` failures are excluded from the loss and from the group baseline (`mask`). `agent` and `timeout` failures score 0 with a named cause (`zero`). A retryable failure is retried while attempts remain.
4. Every attempt writes exactly one outcome before control returns. A process killed mid-unit leaves a start marker, so the missing outcome is visible.
5. A policy maps each class to `abort`, `mask` or `zero`. The per-class defaults are given in rule 3.

### Mapping to existing formats

| Framework | Where the outcome goes | Reference |
|---|---|---|
| verl `AgentLoopOutput` | `extra_fields["tp_decision"]`, `tp_failure_class`, `tp_failure_kind`; masked samples carry no response tokens | `tplane.adapters.verl` |
| slime `Sample` | `status="failed"` and `remove_sample=True` for a masked failure, never `aborted`; miles `metadata["exit_status"]` | `tplane.adapters.slime` |
| prime-rl and verifiers | a record per `Trace` and per `DispatchFailure`, classified from `vf.Error.type` and `status_code` | `tplane.adapters.prime_rl` |
| Harbor ATIF v1.8 | root `extra.tplane`, because the core schema has no outcome field and forbids unknown keys | `tplane.export.atif` |
| OpenTelemetry | `invoke_agent` span with `error.type`, error status, and `rl.failure_class`, `rl.failure_kind`, `rl.decision` | `tplane.export.otlp` |
| Parquet | one row per outcome; dictionary-encoded categorical columns | `tplane.export.parquet` |

## What we ask OpenEnv to adopt

1. An optional `outcome` object on the step result and the episode result, with the fields above.
2. Rules 1 to 3 as normative text for trainers that consume OpenEnv environments.
3. A `failure.kind` registry that environments can extend with namespaced kinds. The 13 kinds above are the core set.

## Open questions

- Whether a timeout belongs to the agent or to the infrastructure. NeMo-RL retries timeouts as infrastructure; miles scores a wall-clock timeout as 0. This draft gives timeouts their own class, so a policy can choose.
- HTTP status errors (5xx against 4xx) need the status as well as the class name. This draft classifies an unknown name with status 5xx, 408 or 429 as infrastructure.
- Whether `mismatch` belongs in this RFC or in a separate RFC on training-inference parity, next to RFC 012's exact log-probability requirement.

## Not in this RFC

Seeding, snapshots, replay and cost are separate RFCs. The `cost` block that tplane records is an implementation detail here.
