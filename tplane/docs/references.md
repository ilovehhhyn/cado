# References

This page lists the open-source code and papers that tplane's tables, adapters and detectors are taken from. Each entry names the version that was read. All were read on 2026-09-24.

## Exception classes in `classify.EXCEPTION_SIGNATURES`

| Library | Version read | Classes used |
|---|---|---|
| torch | 2.14.0 source; 2.10 imported | `OutOfMemoryError` (named so since 2.4; `torch.cuda.OutOfMemoryError` is the same class), `DistBackendError`, `DistNetworkError`, `DistStoreError` |
| Ray | 2.58.0 | `OutOfMemoryError` (host memory monitor, `ray.exceptions`), `ObjectStoreFullError`, `RayActorError`, `WorkerCrashedError`, `NodeDiedError`, `LocalRayletDiedError`, `OwnerDiedError`, `ObjectLostError`, `RaySystemError`, `RpcError` |
| vLLM | v0.30.0 | `EngineDeadError`, `EngineGenerateError`; `AsyncEngineDeadError` up to v0.10.2 |
| httpx | 0.28.1 | `TransportError`, `ConnectTimeout`, `TimeoutException`, `ReadTimeout` |
| aiohttp | 3.13.2 imported, 3.14.3 source | `ClientConnectionError`, `ClientPayloadError`, `ConnectionTimeoutError` |
| requests, urllib3 | 2.32.3 / 2.34.2; 2.5.0 | `ConnectionError`, `ConnectTimeout`, `Timeout`, `ReadTimeout`; `NewConnectionError`, `MaxRetryError`, `NameResolutionError`, `ConnectTimeoutError` |
| e2b | 2.51.0 | `SandboxException`, `TimeoutException`, `ServiceBusyException` |
| Modal | 1.5.5 | `SandboxTerminatedError`, `SandboxTimeoutError`, `ExecTimeoutError`, `InternalFailure` |
| Daytona | v0.190.0 | `DaytonaError`, `DaytonaTimeoutError`, `DaytonaConnectionError` |
| docker-py | 7.2.0 | `DockerException`, `ContainerError` (the command failed, so it is the agent's) |
| verifiers | v0.3.1 (`b2e4e81`) | legacy `InfraError`, `SandboxError`, `TunnelError`, `ToolError`, `ModelError`; v1 `ProviderError`, `OverlongPromptError`, `SandboxError`, `EnvError`, `TaskError`, `InterceptionError` |
| Harbor | v0.23.0 (`1e5c5c6`) | `RewardFileNotFoundError`, `RewardFileEmptyError`, `VerifierOutputParseError`, `VerifierTimeoutError`, `AgentSetupTimeoutError`, `EnvironmentStartTimeoutError` |
| NeMo-RL | main `0aa7bf5` | `RolloutInfraFailure`, `RolloutTimeout`, `GenerationUnavailable`, `NoHealthyShards`, `GymTransportError`, `RolloutDataFailure` |
| miles | PR #2801 (open, `f8fa921`) | `InfraAbort` |

Not covered, because the class alone does not decide the answer:
- HTTP status errors: httpx `HTTPStatusError`, aiohttp `ClientResponseError` and kubernetes `ApiException`. A 5xx is infrastructure; a 4xx usually is not.
- SGLang, which has no typed engine-dead exception.

## Train-inference mismatch

- Estimators: verl `verl/trainer/ppo/rollout_corr_helper.py`, `compute_offpolicy_metrics`, at commit `6093e00`. This is the global token mean of k3. The ±20 safety bound comes from slime `examples/train_infer_mismatch_helper/mis.py`, at commit `8ee9c1e`.
- Identities: verl `docs/algo/rollout_corr_math.md`, section 3.3.5: E[rho] = 1, and E[k3] = KL(rollout || old).
- Published levels. **[T]** means stated in the text; **[F]** means read off a figure and approximate.
  - Healthy dense bf16: k3 of about 1e-4 to 1e-3. Sources: the slime/Miles mismatch blog [F]; Liu et al. §3.5, 5e-4 to 1e-3 on H20 [T].
  - Broken stacks: 1e-3 to 1e-2 on L20, and 1e-2 to 1 on A100, where training was infeasible [T]. Uncorrected fp8 rollouts give 5e-3 to 2e-2 (Feng Yao, Fig. 4 [F]).
  - MoE runs sit higher: up to about 3e-3 before a collapse, with a spike to 0.039 [F].
- Top-p truncation: E_q[rho] equals p(nucleus), and k3 is biased low. This is our own derivation. slime avoids the problem by renormalising the trainer onto the rollout nucleus (`_build_topp_keep_mask`).

## Change detection

- The Shiryaev recursion and its false-alarm bound: Veeravalli and Banerjee, "Quickest Change Detection", arXiv 1210.5552, eq. 13-14 and Theorem 3.2.
- Infra failure rates: KAT-Coder-V2.5, arXiv 2607.05471, §4.2.2. The sandbox feedback error rate was about 16% before its fix and below 2% after.
- The MiMo-V2.6 dashboard incident: the notice `n-b60f90` on 2026-09-17. For about 3 hours, an infra error on one dataset was not detected.

## Formats

- verl `AgentLoopOutput`: `verl/experimental/agent_loop/agent_loop.py`, tag v0.9.1 (`1876b06`).
- slime `Sample`: `slime/utils/types.py`, tag v0.3.2 (`3778dbf`). miles `exit_status` vocabulary: PR #2802 (open, `840bca6`).
- prime-rl `DispatchFailure`: `src/prime_rl/orchestrator/types.py`, `1556db6`.
- Harbor ATIF: RFC 0001, v1.8, tag v0.23.0.
- OTLP/JSON: `opentelemetry-proto` v1.11.0. GenAI conventions: `open-telemetry/semantic-conventions-genai` main, `8ffdf56`.
