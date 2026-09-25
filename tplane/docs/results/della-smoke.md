# Della smoke run

This page records the Part 1 acceptance run. The job filled one GPU until CUDA ran out of memory, with the allocation made inside a tplane unit. It ran three train steps, each allocating 1 GiB every 0.3 s until an allocation failed.

- Command: `TPLANE_PYTHON=/scratch/gpfs/ARORA/hh9077/envs/vllm/bin/python sbatch --qos=gpu-test scripts/della_smoke.sbatch`
- Python 3.12.14, torch 2.13.0+cu130, nvidia-ml-py 13.610.43.
- Hardware: one NVIDIA A100-SXM4-80GB, partition `gputest`.

| Job | Node | Commit | Result |
|---|---|---|---|
| 14383608 | della-l07g7 | 5191a24 | All three steps were recorded as `infra/oom_gpu@train` and masked. Steps 2 and 3 ran out of memory at their first allocation, costing 0.0 gpu-min each: step 1's 79 GiB of tensors were still alive. |
| 14383803 | della-l07g4 | 98f62da | All three steps were recorded as `infra/oom_gpu@train` and masked. Each step filled the card to 79.1/80.0 GiB and cost 0.4 gpu-min. Headroom warnings came before each OOM (9%, then 8%, and so on). |
| 14385954 | della-l09g5 | a44d494 | The same result on the code after the review fixes: monotonic durations, reader probes, the unit state checks and process-group kills. |

**Cause of the first result.** A failed unit kept its exception in order to label the failing stage. The exception's traceback holds the failing frame, and that frame holds the unit, so the step's locals lived until the next garbage collection. The fix is commit 98f62da. The regression test `test_a_failed_unit_releases_the_failing_frame_without_waiting_for_the_garbage_collector` fails on the unfixed code.

**What this run does not show.**
- It does not show a host OOM. A host OOM needs a job that exceeds its cgroup memory limit, and this job did not.
- The OOM is synthetic. A training step would reach it through activations, not through empty tensors.
- The full job outputs are in `della-smoke-14383608-before-fix.txt`, `della-smoke-14383803.txt` and `della-smoke-14385954.txt`.
