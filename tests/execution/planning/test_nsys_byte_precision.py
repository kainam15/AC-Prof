"""Nsys byte counts must survive its stats presentation formatter."""
from types import SimpleNamespace

from acprof.host.profiling.tools import execution_parsers, nsys


def test_nsys_stats_preserve_exact_memcpy_bytes(tmp_path, monkeypatch):
    report = tmp_path / "sample.nsys-rep"
    cache = report.with_suffix(".sqlite")
    reports = {
        "cuda_api_sum": "Total Time (ns),Num Calls,Name\n1000,1,cudaLaunchKernel\n",
        "cuda_gpu_kern_sum": "Total Time (ns),Instances,Name\n1000,1,kernel\n",
        "cuda_gpu_mem_time_sum": (
            "Total Time (ns),Count,Operation\n1000,635,[CUDA memcpy Host-to-Device]\n"
        ),
    }

    def fake_stats(command, check=False):
        cache.write_bytes(b"exported sqlite cache")
        name = command[command.index("--report") + 1]
        if name == "cuda_gpu_mem_size_sum":
            # Captured Nsys 2026.1 output for one report, before/after unit selection.
            if command[command.index("--format") + 1] == "csv:mem=B":
                output = (
                    "Total (B),Count,Operation\n"
                    "15947771,191,[CUDA memcpy Device-to-Device]\n"
                    "47368,62,[CUDA memset]\n"
                    "18112,309,[CUDA memcpy Device-to-Host]\n"
                    "1232,135,[CUDA memcpy Host-to-Device]\n"
                )
            else:
                output = (
                    "Total (MB),Count,Operation\n"
                    "15.948,191,[CUDA memcpy Device-to-Device]\n"
                    "0.047,62,[CUDA memset]\n"
                    "0.018,309,[CUDA memcpy Device-to-Host]\n"
                    "0.001,135,[CUDA memcpy Host-to-Device]\n"
                )
        else:
            output = reports[name]
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(nsys, "run_command", fake_stats)

    outputs = nsys._run_nsys_stats("/unused/nsys", str(report))
    parsed = execution_parsers.parse_nsys_stats_reports(outputs)

    assert parsed["gpu_memcpy_bytes_per_request_nsys"] == 15_967_115
    assert parsed["gpu_memcpy_count_per_request_nsys"] == 635
    assert not cache.exists()
