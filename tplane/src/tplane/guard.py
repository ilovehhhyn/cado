"""Warn, with the fix named, when GPU memory headroom falls below a percentage of the device total."""

from __future__ import annotations

from typing import Final

DEFAULT_WARN_PERCENT: Final[int] = 10
GIB: Final[int] = 1024 * 1024 * 1024


class GuardError(ValueError):
    """Headroom is computed from a used value within the device total and a percent between 1 and 99."""


def headroom_warning(
    *,
    gpu_used_bytes: int,
    gpu_total_bytes: int,
    longest_tokens: int,
    warn_percent: int = DEFAULT_WARN_PERCENT,
) -> str | None:
    """Return a warning when free_percent = (total - used) * 100 // total is below warn_percent, else None."""
    if gpu_total_bytes < 1:
        raise GuardError(f"gpu_total_bytes must be positive, got {gpu_total_bytes}")
    if gpu_used_bytes < 0 or gpu_used_bytes > gpu_total_bytes:
        raise GuardError(
            f"gpu_used_bytes must be between 0 and gpu_total_bytes, got {gpu_used_bytes} of {gpu_total_bytes}"
        )
    if warn_percent < 1 or warn_percent > 99:
        raise GuardError(f"warn_percent must be between 1 and 99, got {warn_percent}")
    free_percent = (gpu_total_bytes - gpu_used_bytes) * 100 // gpu_total_bytes
    if free_percent >= warn_percent:
        return None
    total_gib = gpu_total_bytes / GIB
    return (
        f"gpu memory headroom is {free_percent}% of {total_gib:.1f} GiB, below the {warn_percent}% warning "
        f"threshold while the longest sample is {longest_tokens} tokens; lower max_tokens_per_gpu or "
        "gpu_memory_utilization before the next step"
    )
