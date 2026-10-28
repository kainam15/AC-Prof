"""Input-scaling plots with inherited limits retained as missing numeric axes."""
from pathlib import Path


def plot_process_results(frame, directory, *, xlabel="input_scale"):
    import matplotlib.pyplot as plt

    from acprof.plotting.data import aggregate_metric_series

    directory = Path(directory) / "process"
    directory.mkdir(parents=True, exist_ok=True)
    panels = (("latency_app_s", "Client latency (s)"),
              ("throughput_samples_per_s", "Throughput (samples/s)"),
              ("process_cpu_util_avg_pct", "Observed process CPU (%, one core = 100%)"),
              ("process_rss_peak_bytes", "Observed process RSS peak (bytes)"),
              ("gpu_util_avg_pct", "Device GPU utilization (%, sharing unknown)"),
              ("gpu_mem_used_peak_bytes", "Device GPU memory (bytes, sharing unknown)"))
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), squeeze=False)
    try:
        for axis, (metric, title) in zip(axes.flat, panels, strict=True):
            axis.set(title=title, xlabel=xlabel)
            available = (
                aggregate_metric_series(frame, metric, config_columns=("config",), sort=False)
                if metric in frame else frame.iloc[:0]
            )
            if available.empty:
                axis.text(.5, .5, "Unavailable / not requested", transform=axis.transAxes, ha="center")
                continue
            for label, case in available.groupby("config", sort=False):
                series = case.sort_values("input_scale")
                axis.plot(series["input_scale"], series[metric], marker="o", label=label)
            axis.legend(fontsize="small")
        figure.suptitle("Process Runtime: input scaling; inherited resources are not imposed quotas")
        figure.tight_layout()
        target = directory / "input_scaling.png"
        figure.savefig(target)
        return target
    finally:
        plt.close(figure)
