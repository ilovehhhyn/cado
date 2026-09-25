"""Allocate GPU memory until CUDA runs out inside a tplane unit, so tp show demonstrates an oom_gpu record on a real node."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Final

import tplane

EXIT_USAGE: Final[int] = 2
GIB: Final[int] = 1024 * 1024 * 1024
CHUNK_BYTES: Final[int] = 1 * GIB
# torch.empty returns at once; the pause lets the 250 ms sampler see each allocation.
CHUNK_PAUSE_S: Final[float] = 0.3
CGROUP_ROOT: Final[Path] = Path("/sys/fs/cgroup")
PROC_SELF_CGROUP: Final[Path] = Path("/proc/self/cgroup")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    arguments = parser.parse_args(argv)
    opt_in = os.environ.get("TPLANE_DEVICE_TESTS")
    if opt_in is None:
        print("TPLANE_DEVICE_TESTS must be set to 1 to run the gpu smoke test", file=sys.stderr)
        return EXIT_USAGE
    if opt_in != "1":
        print(f"TPLANE_DEVICE_TESTS must be 1 or unset, got {opt_in!r}", file=sys.stderr)
        return EXIT_USAGE
    return _smoke(arguments.run_dir, steps=arguments.steps)


def _smoke(run_dir: Path, *, steps: int) -> int:
    import torch

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    read_host, host_limit_bytes = _host_memory(job_id)
    gpu = tplane.create_nvml_gpu_reader()
    run = tplane.Run(
        run_id=f"smoke-{job_id}",
        store=tplane.FileRecordStore(run_dir),
        options=tplane.RunOptions(
            policy=tplane.Policy(on_infra=tplane.Action.MASK, max_attempts=2),
            gpus_held=1,
            host_limit_bytes=host_limit_bytes,
        ),
        read_host=read_host,
        read_gpu=gpu.read,
        gpu_total_bytes=gpu.total_bytes,
        log=lambda line: print(line, file=sys.stderr),
    )
    for step in range(steps):
        run.attempt(
            f"step-{step:05d}",
            kind=tplane.UnitKind.TRAIN_STEP,
            source="smoke",
            body=lambda unit: _fill_gpu(unit, torch),
        )
    print(f"records written to {run_dir}; inspect with: tp show {run_dir} --failures-only")
    return 0


def _fill_gpu(unit: tplane.Unit, torch: ModuleType) -> None:
    held = []
    with unit.stage(tplane.Stage.TRAIN):
        while True:
            unit.check_headroom(longest_tokens=0)
            held.append(torch.empty(CHUNK_BYTES, dtype=torch.uint8, device="cuda"))
            time.sleep(CHUNK_PAUSE_S)


def _host_memory(job_id: str) -> tuple[Callable[[], int], int | None]:
    """Return the host memory reader and limit: the Slurm job cgroup on a node, this process's RSS locally."""
    if job_id == "local":
        return tplane.proc_status_reader(os.getpid()), None
    job_dir = _job_cgroup_dir(job_id)
    limit_text = (job_dir / "memory.max").read_text().strip()
    return tplane.cgroup_memory_reader(job_dir), None if limit_text == "max" else int(limit_text)


def _job_cgroup_dir(job_id: str) -> Path:
    """Find the cgroup v2 directory named job_<id> above this process's own cgroup."""
    lines = PROC_SELF_CGROUP.read_text().splitlines()
    unified = [line.split("::", 1)[1] for line in lines if line.startswith("0::")]
    if len(unified) != 1:
        raise SystemExit(
            f"{PROC_SELF_CGROUP} has no cgroup v2 line; this script requires a cgroup v2 node, got {lines!r}"
        )
    own = CGROUP_ROOT / unified[0].lstrip("/")
    for directory in (own, *own.parents):
        if directory.name == f"job_{job_id}" and (directory / "memory.current").exists():
            return directory
    raise SystemExit(
        f"no job_{job_id} cgroup with memory.current above {own}; pass the site's path to cgroup_memory_reader"
    )


if __name__ == "__main__":
    raise SystemExit(main())
