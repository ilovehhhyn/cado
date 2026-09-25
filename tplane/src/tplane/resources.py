"""Sample host and GPU memory on a background thread during a unit, and summarise the peaks."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypeAlias, cast

from tplane.schema import ResourceSummary

DEFAULT_MAX_SAMPLES: Final[int] = 100_000  # 250 ms samples for about seven hours
KILOBYTE: Final[int] = 1024
STOP_JOIN_TIMEOUT_S: Final[float] = 5.0

NvmlHandle: TypeAlias = object  # NVML's opaque device pointer; only NVML reads it


class ResourceError(RuntimeError):
    """A configured memory source must be readable in the kernel's own format."""


@dataclass(frozen=True, kw_only=True)
class ResourceSample:
    """Memory in use at one instant, with one GPU entry per device."""

    at_ms: int
    host_rss_bytes: int
    gpu_used_bytes: tuple[int, ...]


class GpuReader(Protocol):
    """Read used bytes for every configured GPU in a fixed device order."""

    @property
    def total_bytes(self) -> tuple[int, ...]: ...

    def read(self) -> tuple[int, ...]: ...


class NvmlMemoryInfo(Protocol):
    """The two fields of nvmlMemory_t that tplane reads."""

    @property
    def total(self) -> int: ...

    @property
    def used(self) -> int: ...


class NvmlModule(Protocol):
    """The four NVML calls tplane makes, as exposed by the pynvml module of nvidia-ml-py."""

    def nvmlInit(self) -> None: ...

    def nvmlDeviceGetCount(self) -> int: ...

    def nvmlDeviceGetHandleByIndex(self, index: int) -> NvmlHandle: ...

    def nvmlDeviceGetMemoryInfo(self, handle: NvmlHandle) -> NvmlMemoryInfo: ...


def summarise_samples(
    samples: Sequence[ResourceSample], *, gpu_total_bytes: tuple[int, ...]
) -> ResourceSummary:
    """Reduce samples to the peak host RSS and the peak used bytes per GPU device."""
    device_count = len(gpu_total_bytes)
    for item in samples:
        if len(item.gpu_used_bytes) != device_count:
            raise ResourceError(
                f"sample at {item.at_ms} ms has {len(item.gpu_used_bytes)} gpu entries but "
                f"{device_count} devices are configured"
            )
    peak_host = max((item.host_rss_bytes for item in samples), default=0)
    peak_gpu = tuple(
        max((item.gpu_used_bytes[index] for item in samples), default=0)
        for index in range(device_count)
    )
    return ResourceSummary(
        sample_count=len(samples),
        peak_host_rss_bytes=peak_host,
        peak_gpu_used_bytes=peak_gpu,
        gpu_total_bytes=gpu_total_bytes,
    )


def read_proc_status_rss_bytes(status_path: Path) -> int:
    """Read resident set size from a Linux /proc/<pid>/status file."""
    for line in _read_lines(status_path):
        if line.startswith("VmRSS:"):
            return _parse_int(status_path, line.split()[1]) * KILOBYTE
    raise ResourceError(
        f"{status_path}: no VmRSS line; this reader requires a Linux /proc/<pid>/status file"
    )


def read_cgroup_memory_current_bytes(current_path: Path) -> int:
    """Read the cgroup v2 memory.current file."""
    lines = _read_lines(current_path)
    if len(lines) == 0:
        raise ResourceError(f"{current_path}: empty; expected one integer")
    return _parse_int(current_path, lines[0].strip())


def read_cgroup_oom_kill_count(events_path: Path) -> int:
    """Read the oom_kill counter from a cgroup v2 memory.events file."""
    for line in _read_lines(events_path):
        if line.startswith("oom_kill "):
            return _parse_int(events_path, line.split()[1])
    raise ResourceError(
        f"{events_path}: no oom_kill line; this reader requires a cgroup v2 memory.events file"
    )


def proc_status_reader(pid: int) -> Callable[[], int]:
    """Return a reader of one process's resident set size."""
    status_path = Path("/proc") / str(pid) / "status"
    return lambda: read_proc_status_rss_bytes(status_path)


def cgroup_memory_reader(cgroup_dir: Path) -> Callable[[], int]:
    """Return a reader of the cgroup's current memory; the caller names the Slurm job cgroup directory."""
    current_path = cgroup_dir / "memory.current"
    return lambda: read_cgroup_memory_current_bytes(current_path)


def no_gpu_reader() -> GpuReader:
    """Return a reader for a unit that samples no GPU."""
    return _NoGpu()


def create_nvml_gpu_reader(
    *,
    indices: tuple[int, ...] | None = None,
    load_nvml: Callable[[], NvmlModule] | None = None,
) -> GpuReader:
    """Create an NVML-backed reader over the given NVML device indices, or every device when None."""
    nvml = _import_pynvml() if load_nvml is None else load_nvml()
    nvml.nvmlInit()
    count = nvml.nvmlDeviceGetCount()
    chosen = tuple(range(count)) if indices is None else indices
    for index in chosen:
        if index < 0 or index >= count:
            raise ResourceError(f"gpu index {index} is not present; NVML reports {count} devices")
    handles = tuple(nvml.nvmlDeviceGetHandleByIndex(index) for index in chosen)
    totals = tuple(nvml.nvmlDeviceGetMemoryInfo(handle).total for handle in handles)
    return _NvmlGpu(nvml=nvml, handles=handles, totals=totals)


class _NoGpu:
    @property
    def total_bytes(self) -> tuple[int, ...]:
        return ()

    def read(self) -> tuple[int, ...]:
        return ()


class _NvmlGpu:
    def __init__(
        self, *, nvml: NvmlModule, handles: tuple[NvmlHandle, ...], totals: tuple[int, ...]
    ) -> None:
        self._nvml = nvml
        self._handles = handles
        self._totals = totals

    @property
    def total_bytes(self) -> tuple[int, ...]:
        return self._totals

    def read(self) -> tuple[int, ...]:
        return tuple(self._nvml.nvmlDeviceGetMemoryInfo(handle).used for handle in self._handles)


class Sampler:
    """Sample memory every interval until stopped, keeping at most max_samples and never raising from the thread."""

    def __init__(
        self,
        *,
        read_host: Callable[[], int],
        read_gpu: Callable[[], tuple[int, ...]],
        interval_ms: int,
        clock_ms: Callable[[], int],
        gpu_devices: int,
        max_samples: int = DEFAULT_MAX_SAMPLES,
    ) -> None:
        if interval_ms < 1:
            raise ResourceError(f"interval_ms must be at least 1, got {interval_ms}")
        if max_samples < 1:
            raise ResourceError(f"max_samples must be at least 1, got {max_samples}")
        self._read_host = read_host
        self._read_gpu = read_gpu
        self._gpu_devices = gpu_devices
        self._interval_s = interval_ms / 1000
        self._clock_ms = clock_ms
        self._max_samples = max_samples
        self._samples: list[ResourceSample] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def errors(self) -> tuple[str, ...]:
        """Return the reader errors seen so far, in order."""
        with self._lock:
            return tuple(self._errors)

    def start(self) -> None:
        """Take one sample now, then sample on a daemon thread every interval."""
        if self._thread is not None:
            raise ResourceError("sampler must be started exactly once")
        self._take_sample()
        self._thread = threading.Thread(target=self._loop, name="tplane-sampler", daemon=True)
        self._thread.start()

    def snapshot(self) -> tuple[ResourceSample, ...]:
        """Return the samples taken so far."""
        with self._lock:
            return tuple(self._samples)

    def stop(self) -> tuple[ResourceSample, ...]:
        """Stop the thread, waiting at most STOP_JOIN_TIMEOUT_S, and return every sample."""
        if self._thread is None:
            raise ResourceError("sampler must be started before it is stopped")
        self._stop.set()
        self._thread.join(timeout=STOP_JOIN_TIMEOUT_S)
        return self.snapshot()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._take_sample()

    def _take_sample(self) -> None:
        try:
            at_ms = self._clock_ms()
            host = self._read_host()
            gpu = self._read_gpu()
            if len(gpu) != self._gpu_devices:
                raise ResourceError(
                    f"gpu reader returned {len(gpu)} entries, expected {self._gpu_devices}"
                )
            item = ResourceSample(at_ms=at_ms, host_rss_bytes=host, gpu_used_bytes=gpu)
        # The sampler must never raise into the unit, so every reader failure becomes a record.
        except Exception as error:
            with self._lock:
                self._errors.append(_describe(error))
            return
        with self._lock:
            if len(self._samples) < self._max_samples:
                self._samples.append(item)


def _describe(error: Exception) -> str:
    if isinstance(error, ResourceError):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _import_pynvml() -> NvmlModule:
    try:
        module = importlib.import_module("pynvml")
    except ImportError as error:
        raise ResourceError(
            "gpu memory sampling requires the nvidia-ml-py package; "
            "install tplane[gpu] or pass no_gpu_reader()"
        ) from error
    return cast(
        NvmlModule, module
    )  # importlib returns ModuleType; pynvml provides these four calls


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text().splitlines()
    except OSError as error:
        raise ResourceError(f"{path}: {error.strerror}") from error


def _parse_int(path: Path, text: str) -> int:
    try:
        return int(text)
    except ValueError as error:
        raise ResourceError(f"{path}: expected an integer, got {text!r}") from error
