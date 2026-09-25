"""Samples summarise to per-device peaks, readers parse the kernel's own file formats or fail with the path, and the sampler keeps sampling through reader errors without raising."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from tplane.resources import (
    ResourceError,
    ResourceSample,
    Sampler,
    create_nvml_gpu_reader,
    no_gpu_reader,
    read_cgroup_memory_current_bytes,
    read_cgroup_oom_kill_count,
    read_proc_status_rss_bytes,
    summarise_samples,
)


def sample(at_ms: int, host: int, gpu: tuple[int, ...]) -> ResourceSample:
    return ResourceSample(at_ms=at_ms, host_rss_bytes=host, gpu_used_bytes=gpu)


def test_summary_takes_the_peak_per_device_and_counts_samples() -> None:
    samples = (sample(0, 10, (1, 9)), sample(1, 30, (5, 2)), sample(2, 20, (3, 4)))

    summary = summarise_samples(samples, gpu_total_bytes=(100, 100))

    assert summary.sample_count == 3
    assert summary.peak_host_rss_bytes == 30
    assert summary.peak_gpu_used_bytes == (5, 9)


def test_empty_samples_summarise_to_zero_peaks_with_one_entry_per_device() -> None:
    summary = summarise_samples((), gpu_total_bytes=(100, 100))

    assert summary.sample_count == 0
    assert summary.peak_gpu_used_bytes == (0, 0)


def test_summary_rejects_a_sample_with_the_wrong_device_count() -> None:
    with pytest.raises(
        ResourceError, match="sample at 1 ms has 1 gpu entries but 2 devices are configured"
    ):
        summarise_samples((sample(1, 1, (1,)),), gpu_total_bytes=(100, 100))


def test_proc_status_reader_parses_vmrss_in_kilobytes(tmp_path: Path) -> None:
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmPeak:\t  200 kB\nVmRSS:\t   123 kB\nThreads:\t4\n")

    assert read_proc_status_rss_bytes(status) == 123 * 1024


def test_proc_status_reader_rejects_a_file_without_vmrss(tmp_path: Path) -> None:
    status = tmp_path / "status"
    status.write_text("Name:\tpython\n")

    with pytest.raises(ResourceError, match="no VmRSS line"):
        read_proc_status_rss_bytes(status)


def test_cgroup_readers_parse_current_and_oom_kill(tmp_path: Path) -> None:
    (tmp_path / "memory.current").write_text("4096\n")
    (tmp_path / "memory.events").write_text("low 0\nhigh 2\nmax 0\noom 1\noom_kill 1\n")

    assert read_cgroup_memory_current_bytes(tmp_path / "memory.current") == 4096
    assert read_cgroup_oom_kill_count(tmp_path / "memory.events") == 1


def test_cgroup_reader_names_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ResourceError, match="memory.current"):
        read_cgroup_memory_current_bytes(tmp_path / "memory.current")


def test_cgroup_reader_rejects_a_non_integer_value_and_names_the_file(tmp_path: Path) -> None:
    (tmp_path / "memory.current").write_text("max\n")

    with pytest.raises(ResourceError, match="memory.current: expected an integer, got 'max'"):
        read_cgroup_memory_current_bytes(tmp_path / "memory.current")


def test_no_gpu_reader_reports_zero_devices() -> None:
    reader = no_gpu_reader()

    assert reader.total_bytes == ()
    assert reader.read() == ()


@dataclass(frozen=True)
class FakeMemoryInfo:
    total: int
    used: int


class FakeNvml:
    """NVML stand-in with two devices; records every call."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.used = [10, 20]

    def nvmlInit(self) -> None:
        self.calls.append("init")

    def nvmlDeviceGetCount(self) -> int:
        return 2

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> FakeMemoryInfo:
        return FakeMemoryInfo(total=100 * (handle + 1), used=self.used[handle])


def test_nvml_reader_reads_every_device_in_index_order() -> None:
    nvml = FakeNvml()

    reader = create_nvml_gpu_reader(load_nvml=lambda: nvml)
    nvml.used = [30, 40]

    assert reader.total_bytes == (100, 200)
    assert reader.read() == (30, 40)
    assert nvml.calls == ["init"]


def test_nvml_reader_limited_to_indices_reads_only_those_devices() -> None:
    reader = create_nvml_gpu_reader(indices=(1,), load_nvml=FakeNvml)

    assert reader.total_bytes == (200,)
    assert reader.read() == (20,)


def test_nvml_reader_rejects_an_index_the_node_does_not_have() -> None:
    with pytest.raises(ResourceError, match="gpu index 2 is not present; NVML reports 2 devices"):
        create_nvml_gpu_reader(indices=(2,), load_nvml=FakeNvml)


def test_nvml_reader_without_the_package_names_the_extra_to_install() -> None:
    with pytest.raises(
        ResourceError, match="requires the nvidia-ml-py package; install tplane\\[gpu\\]"
    ):
        create_nvml_gpu_reader()


def test_sampler_takes_a_sample_at_start_and_then_every_interval() -> None:
    ticks = iter(range(0, 10_000, 100))
    hosts = iter(range(1, 1_000))
    sampler = Sampler(
        read_host=lambda: next(hosts),
        read_gpu=lambda: (0,),
        interval_ms=10,
        clock_ms=lambda: next(ticks),
        gpu_devices=1,
    )

    sampler.start()
    first = sampler.snapshot()
    threading.Event().wait(0.05)
    samples = sampler.stop()

    assert len(first) >= 1
    assert first[0].host_rss_bytes == 1
    assert len(samples) >= 2
    assert [item.host_rss_bytes for item in samples] == list(range(1, len(samples) + 1))


def test_sampler_records_reader_errors_of_any_type_and_keeps_going() -> None:
    calls = {"count": 0}

    def flaky_host() -> int:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ResourceError("/proc/1/status: gone")
        if calls["count"] == 2:
            raise RuntimeError("NVML_ERROR_GPU_IS_LOST")
        return 7

    sampler = Sampler(
        read_host=flaky_host, read_gpu=lambda: (), interval_ms=10, clock_ms=lambda: 0, gpu_devices=0
    )

    sampler.start()
    threading.Event().wait(0.08)
    samples = sampler.stop()

    assert sampler.errors == ("/proc/1/status: gone", "RuntimeError: NVML_ERROR_GPU_IS_LOST")
    assert len(samples) >= 1
    assert all(item.host_rss_bytes == 7 for item in samples)


def test_sampler_caps_the_number_of_samples() -> None:
    sampler = Sampler(
        read_host=lambda: 1,
        read_gpu=lambda: (),
        interval_ms=10,
        clock_ms=lambda: 0,
        gpu_devices=0,
        max_samples=2,
    )

    sampler.start()
    threading.Event().wait(0.08)
    samples = sampler.stop()

    assert len(samples) == 2


def test_sampler_stop_before_start_is_an_error() -> None:
    sampler = Sampler(
        read_host=lambda: 1, read_gpu=lambda: (), interval_ms=10, clock_ms=lambda: 0, gpu_devices=0
    )

    with pytest.raises(ResourceError, match="sampler must be started before it is stopped"):
        sampler.stop()


def test_sampler_rejects_a_second_start() -> None:
    sampler = Sampler(
        read_host=lambda: 1, read_gpu=lambda: (), interval_ms=10, clock_ms=lambda: 0, gpu_devices=0
    )
    sampler.start()

    with pytest.raises(ResourceError, match="sampler must be started exactly once"):
        sampler.start()
    sampler.stop()


def test_sampler_records_a_sample_with_the_wrong_device_count_as_an_error() -> None:
    sampler = Sampler(
        read_host=lambda: 1,
        read_gpu=lambda: (1, 2),
        interval_ms=10,
        clock_ms=lambda: 0,
        gpu_devices=1,
    )

    sampler.start()
    samples = sampler.stop()

    assert samples == ()
    assert sampler.errors[0] == "gpu reader returned 2 entries, expected 1"
