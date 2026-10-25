"""CSV 指标的唯一登记表；标准库声明，不初始化任何采集器。"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Metric:
    name: str
    unit: str
    source: str
    window: str
    applicability: str = "all"
    kind: str = "number"
    tool: str = ""
    posthoc: bool = False
    label: str = ""
    group: str = "Other"
    direction: str = "neutral"
    scale: str = "linear"
    summary: bool = False
    aggregation: str = "mean"
    # Internal row assembly membership is not persisted metric metadata.
    row_group: str = ""
    row_order: int | None = None

    def __post_init__(self):
        if not self.label:
            object.__setattr__(self, "label", self.name)
        if self.direction not in {"higher", "lower", "neutral"}:
            raise ValueError(f"Invalid direction: {self.direction}")
        if self.scale not in {"linear", "log"}:
            raise ValueError(f"Invalid scale: {self.scale}")

    def to_dict(self):
        metadata = asdict(self)
        metadata.pop("row_group")
        metadata.pop("row_order")
        return metadata


# 顺序就是 CSV 协议顺序；单位和窗口在此显式声明，消费者不得按列名猜测。
_DECLARATIONS = (
    Metric('environment_class', 'text', 'platform_identity', 'experiment', kind='text'),
    Metric('runtime_kind', 'text', 'execution_runtime', 'experiment', kind='text'),
    Metric('runtime_id', 'text', 'immutable_runtime_artifact', 'experiment', kind='text'),
    Metric('resource_limit_source', 'text', 'verified_runtime_limits', 'case', kind='text'),
    Metric('resource_scope', 'text', 'collector_contract', 'case', kind='text'),
    Metric('environment_sharing', 'text', 'allocation_evidence', 'case', kind='text'),
    Metric('gpu_metric_scope', 'text', 'collector_contract', 'case', applicability='gpu', kind='text'),
    Metric('process_resource_error', 'text', 'psutil', 'request_window', kind='text'),
    Metric('process_resource_status', 'text', 'psutil', 'request_window', kind='text'),
    Metric('gpu_utilization_status', 'text', 'nvml_selected_device', 'request_window', kind='text'),
    Metric('gpu_memory_status', 'text', 'nvml_selected_device', 'request_window', kind='text'),
    Metric('process_startup_s', 's', 'managed_process_to_ready', 'case_startup'),
    Metric('process_cpu_time_s', 's/window', 'psutil_observed_process_tree', 'request_window', row_group='process_usage', row_order=0),
    Metric('process_cpu_util_avg_pct', '% one logical CPU', 'psutil_observed_process_tree', 'request_window', row_group='process_usage', row_order=1, label='进程 CPU 平均占用', group='CPU'),
    Metric('process_rss_avg_bytes', 'byte', 'psutil_observed_process_tree', 'request_window', row_group='process_usage', row_order=2, label='进程 RSS 平均用量', group='Memory'),
    Metric('process_rss_peak_bytes', 'byte', 'psutil_observed_process_tree', 'request_window', row_group='process_usage', row_order=3, label='进程 RSS 峰值用量', group='Memory', summary=True, aggregation='max'),
    # 资源配置、输入、输出与网络。
    Metric('cpu_cores', 'core', 'protocol', 'case'),
    Metric('mem_cap_gb', 'GiB', 'protocol', 'case'),
    Metric('gpu_mode', 'text', 'protocol', 'case', applicability='gpu', kind='text'),
    Metric('gpu_device_uuid', 'text', 'docker_device_request', 'case', applicability='gpu', kind='text'),
    Metric('input_scale', 'input_scale_type', 'protocol', 'row'),
    Metric('input_units_per_request', 'input_unit/request', 'input_plan', 'materialized_request'),
    Metric('input_num_samples', 'sample/request', 'input_plan', 'materialized_request', applicability='audio'),
    Metric('input_pixels_per_request', 'pixel/request', 'input_plan', 'materialized_request', applicability='pixel_geometry'),
    Metric('output_pixels_per_request', 'pixel/request', 'input_plan', 'materialized_request', applicability='pixel_geometry'),
    Metric('request_payload_bytes', 'byte/request', 'client_http', 'request_window'),
    Metric('packet_request_wire_bytes_per_request', 'byte/request', 'pcap', 'request_window'),
    Metric('packet_response_wire_bytes_per_request', 'byte/request', 'pcap', 'request_window'),
    Metric('packet_total_wire_bytes_per_request', 'byte/request', 'pcap', 'request_window'),
    Metric('packet_tcp_payload_bytes_per_request', 'byte/request', 'pcap', 'request_window'),
    Metric('packet_protocol_overhead_bytes_per_request', 'byte/request', 'pcap', 'request_window'),
    Metric('packet_protocol_overhead_ratio', 'ratio', 'pcap', 'request_window'),
    Metric('task_param', 'text', 'protocol', 'row', kind='text'),
    Metric('output_length_avg', 'task_output_unit/request', 'client_http', 'request_window'),
    Metric('output_token_count_avg', 'token/request', 'client_http', 'request_window', applicability='text_tokenizer'),
    Metric('repeat_idx', 'index', 'protocol', 'row'),
    Metric('warmup', '0|1', 'protocol', 'row'),
    Metric('repeat_in_window', 'request/window', 'protocol', 'row'),
    # 请求窗口的延迟分布与吞吐。
    Metric('latency_s', 's', 'pcap', 'request_window', label='Packet 延迟均值', group='Performance', direction='lower', summary=False, aggregation='mean'),
    Metric('latency_s_per_input_unit', 's/input_unit', 'pcap', 'request_window'),
    Metric('latency_s_per_input_megapixel', 's/Mpixel', 'pcap', 'request_window', applicability='pixel_geometry'),
    Metric('latency_s_per_output_megapixel', 's/Mpixel', 'pcap', 'request_window', applicability='pixel_geometry'),
    Metric('latency_request_count', 'request/window', 'pcap', 'request_window', row_group='latency_packet_distribution', row_order=0),
    Metric('latency_p50_s', 's', 'pcap', 'request_window', label='Packet P50 · 窗口均值', group='Performance', direction='lower', summary=False, aggregation='mean', row_group='latency_packet_distribution', row_order=1),
    Metric('latency_p90_s', 's', 'pcap', 'request_window', row_group='latency_packet_distribution', row_order=2),
    Metric('latency_p95_s', 's', 'pcap', 'request_window', label='Packet P95 · 窗口均值', group='Performance', direction='lower', summary=False, aggregation='mean', row_group='latency_packet_distribution', row_order=3),
    Metric('latency_tail_ratio', 'ratio', 'pcap', 'request_window', row_group='latency_packet_distribution', row_order=4),
    Metric('latency_std_s', 's', 'pcap', 'request_window', row_group='latency_packet_distribution', row_order=5),
    Metric('latency_cv', 'ratio', 'pcap', 'request_window', row_group='latency_packet_distribution', row_order=6),
    Metric('latency_iqr_s', 's', 'pcap', 'request_window', row_group='latency_packet_distribution', row_order=7),
    Metric('latency_max_s', 's', 'pcap', 'request_window', row_group='latency_packet_distribution', row_order=8),
    Metric('latency_slow_ratio', 'ratio', 'pcap', 'request_window', row_group='latency_packet_distribution', row_order=9),
    Metric('latency_app_s', 's', 'client_http', 'request_window', label='应用延迟均值', group='Performance', direction='lower', summary=False, aggregation='mean'),
    Metric('latency_app_s_per_input_unit', 's/input_unit', 'client_http', 'request_window'),
    Metric('latency_app_s_per_input_megapixel', 's/Mpixel', 'client_http', 'request_window', applicability='pixel_geometry'),
    Metric('latency_app_s_per_output_megapixel', 's/Mpixel', 'client_http', 'request_window', applicability='pixel_geometry'),
    Metric('latency_app_request_count', 'request/window', 'client_http', 'request_window', row_group='latency_app_distribution', row_order=0),
    Metric('latency_app_p50_s', 's', 'client_http', 'request_window', label='应用 P50 · 窗口均值', group='Performance', direction='lower', summary=True, aggregation='mean', row_group='latency_app_distribution', row_order=1),
    Metric('latency_app_p90_s', 's', 'client_http', 'request_window', row_group='latency_app_distribution', row_order=2),
    Metric('latency_app_p95_s', 's', 'client_http', 'request_window', label='应用 P95 · 窗口均值', group='Performance', direction='lower', summary=True, aggregation='mean', row_group='latency_app_distribution', row_order=3),
    Metric('latency_app_tail_ratio', 'ratio', 'client_http', 'request_window', row_group='latency_app_distribution', row_order=4),
    Metric('latency_app_std_s', 's', 'client_http', 'request_window', row_group='latency_app_distribution', row_order=5),
    Metric('latency_app_cv', 'ratio', 'client_http', 'request_window', row_group='latency_app_distribution', row_order=6),
    Metric('latency_app_iqr_s', 's', 'client_http', 'request_window', row_group='latency_app_distribution', row_order=7),
    Metric('latency_app_max_s', 's', 'client_http', 'request_window', row_group='latency_app_distribution', row_order=8),
    Metric('latency_app_slow_ratio', 'ratio', 'client_http', 'request_window', row_group='latency_app_distribution', row_order=9),
    Metric('throughput_samples_per_s', 'sample/s', 'derived', 'request_window', label='吞吐量', group='Performance', direction='higher', summary=True, aggregation='mean'),
    Metric('throughput_samples_per_s_per_cpu_core', 'sample/s/core', 'derived', 'request_window'),
    # 独立 Profiler；各工具的诊断紧随对应指标。
    Metric('model_logical_mflop_per_request_torch_profiler_eager', 'MFLOP/request', 'torch', 'independent_profiler', tool='torch', posthoc=True),
    Metric('model_logical_mflops_app_torch_profiler_eager', 'MFLOP/s', 'torch', 'independent_profiler', tool='torch'),
    Metric('model_logical_mflops_packet_torch_profiler_eager', 'MFLOP/s', 'torch', 'independent_profiler', tool='torch'),
    Metric('compute_profile_error_torch_profiler_eager', 'text', 'torch', 'independent_profiler', kind='text', tool='torch', posthoc=True),
    Metric('gpu_executed_mflop_per_request_ncu', 'MFLOP/request', 'ncu', 'independent_profiler', applicability='gpu', tool='ncu', posthoc=True),
    Metric('gpu_executed_tensor_mflop_per_request_ncu', 'MFLOP/request', 'ncu', 'independent_profiler', applicability='gpu', tool='ncu', posthoc=True),
    Metric('gpu_executed_scalar_mflop_per_request_ncu', 'MFLOP/request', 'ncu', 'independent_profiler', applicability='gpu', tool='ncu', posthoc=True),
    Metric('gpu_executed_tensor_share_pct_ncu', '%', 'ncu', 'independent_profiler', applicability='gpu', tool='ncu', posthoc=True),
    Metric('gpu_executed_mflops_app_ncu', 'MFLOP/s', 'ncu', 'independent_profiler', applicability='gpu', tool='ncu', posthoc=True),
    Metric('gpu_executed_mflops_packet_ncu', 'MFLOP/s', 'ncu', 'independent_profiler', applicability='gpu', tool='ncu', posthoc=True),
    Metric('gpu_kernel_launch_count_per_request_ncu', 'count/request', 'ncu', 'independent_profiler', applicability='gpu', tool='ncu', posthoc=True),
    Metric('gpu_kernel_time_sum_ms_per_request_ncu', 'ms/request', 'ncu', 'independent_profiler', applicability='gpu', tool='ncu', posthoc=True),
    Metric('compute_profile_error_ncu', 'text', 'ncu', 'independent_profiler', applicability='gpu', kind='text', tool='ncu', posthoc=True),
    Metric('cpu_heap_peak_bytes_massif', 'byte', 'massif', 'profiler_process_lifetime', applicability='cpu', tool='massif', posthoc=True),
    Metric('cpu_heap_extra_peak_bytes_massif', 'byte', 'massif', 'profiler_process_lifetime', applicability='cpu', tool='massif', posthoc=True),
    Metric('cpu_stack_peak_bytes_massif', 'byte', 'massif', 'profiler_process_lifetime', applicability='cpu', tool='massif', posthoc=True),
    Metric('cpu_heap_peak_total_bytes_massif', 'byte', 'massif', 'profiler_process_lifetime', applicability='cpu', tool='massif', posthoc=True),
    Metric('cpu_heap_peak_at_ms_massif', 'ms', 'massif', 'profiler_process_lifetime', applicability='cpu', tool='massif', posthoc=True),
    Metric('compute_profile_error_massif', 'text', 'massif', 'profiler_process_lifetime', applicability='cpu', kind='text', tool='massif', posthoc=True),
    Metric('host_inference_wall_time_ms_per_request_nsys', 'ms/request', 'nsys', 'independent_profiler', applicability='gpu', tool='nsys', posthoc=True),
    Metric('cuda_api_time_sum_ms_per_request_nsys', 'ms/request', 'nsys', 'independent_profiler', applicability='gpu', tool='nsys', posthoc=True),
    Metric('cuda_api_call_count_per_request_nsys', 'count/request', 'nsys', 'independent_profiler', applicability='gpu', tool='nsys', posthoc=True),
    Metric('gpu_kernel_time_sum_ms_per_request_nsys', 'ms/request', 'nsys', 'independent_profiler', applicability='gpu', tool='nsys', posthoc=True),
    Metric('gpu_kernel_launch_count_per_request_nsys', 'count/request', 'nsys', 'independent_profiler', applicability='gpu', tool='nsys', posthoc=True),
    Metric('gpu_memcpy_time_sum_ms_per_request_nsys', 'ms/request', 'nsys', 'independent_profiler', applicability='gpu', tool='nsys', posthoc=True),
    Metric('gpu_memcpy_count_per_request_nsys', 'count/request', 'nsys', 'independent_profiler', applicability='gpu', tool='nsys', posthoc=True),
    Metric('gpu_memcpy_bytes_per_request_nsys', 'byte/request', 'nsys', 'independent_profiler', applicability='gpu', tool='nsys', posthoc=True),
    Metric('compute_profile_error_nsys', 'text', 'nsys', 'independent_profiler', applicability='gpu', kind='text', tool='nsys', posthoc=True),
    # GPU 功率与能耗，包括采样来源和回退原因。
    Metric('gpu_idle_power_w', 'W', 'nvml_selected_device', 'matched_control', applicability='gpu', row_group='gpu_energy', row_order=0),
    Metric('gpu_idle_measured_at', 'ISO-8601', 'nvml_selected_device', 'matched_control', applicability='gpu_idle_debug', kind='text'),
    Metric('gpu_idle_rel_range_so_far', 'ratio', 'nvml_selected_device', 'case_controls_so_far', applicability='gpu_idle_debug'),
    Metric('gpu_idle_energy_source', 'text', 'nvml_selected_device', 'matched_control', applicability='gpu', kind='text'),
    Metric('gpu_energy_iters', 'sample/window', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='gpu_energy', row_order=1),
    Metric('gpu_avg_power_total_w', 'W', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='gpu_energy', row_order=2),
    Metric('gpu_peak_power_total_w', 'W', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='gpu_energy', row_order=3),
    Metric('gpu_energy_total_j', 'J/request', 'nvml_selected_device', 'request_window', applicability='gpu', label='GPU 总能耗 / request', group='Energy', direction='lower', summary=False, aggregation='mean', row_group='gpu_energy', row_order=4),
    Metric('gpu_avg_power_eff_w', 'W', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='gpu_energy', row_order=5),
    Metric('gpu_peak_power_eff_w', 'W', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='gpu_energy', row_order=6),
    Metric('gpu_energy_eff_j', 'J/request', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='gpu_energy', row_order=7),
    Metric('gpu_energy_source', 'text', 'nvml_selected_device', 'request_window', applicability='gpu', kind='text'),
    Metric('gpu_energy_fallback_reason', 'text', 'nvml_selected_device', 'request_window', applicability='gpu', kind='text'),
    # CPU package、DRAM、估算 vCPU 与容器归因能效。
    Metric('cpu_idle_power_w', 'W', 'rapl_package', 'matched_control', row_group='cpu_energy', row_order=0),
    Metric('cpu_idle_measured_at', 'ISO-8601', 'rapl_package', 'matched_control', applicability='idle_debug', kind='text'),
    Metric('cpu_idle_rel_range_so_far', 'ratio', 'rapl_package', 'case_controls_so_far', applicability='idle_debug'),
    Metric('cpu_energy_iters', 'sample/window', 'rapl_package', 'request_window', row_group='cpu_energy', row_order=1),
    Metric('cpu_avg_power_total_w', 'W', 'rapl_package', 'request_window', row_group='cpu_energy', row_order=2),
    Metric('cpu_peak_power_total_w', 'W', 'rapl_package', 'request_window', row_group='cpu_energy', row_order=3),
    Metric('cpu_energy_total_j', 'J/request', 'rapl_package', 'request_window', label='CPU package 总能耗 / request', group='Energy', direction='lower', summary=False, aggregation='mean', row_group='cpu_energy', row_order=4),
    Metric('cpu_avg_power_eff_w', 'W', 'rapl_package', 'request_window', row_group='cpu_energy', row_order=5),
    Metric('cpu_peak_power_eff_w', 'W', 'rapl_package', 'request_window', row_group='cpu_energy', row_order=6),
    Metric('cpu_energy_eff_j', 'J/request', 'rapl_package', 'request_window', row_group='cpu_energy', row_order=7),
    Metric('dram_window_energy_j', 'J/window', 'rapl_dram', 'request_window', applicability='dram', row_group='dram_energy', row_order=0),
    Metric('dram_window_duration_s', 's', 'rapl_dram', 'request_window', applicability='dram', row_group='dram_energy', row_order=1),
    Metric('dram_energy_per_request_j', 'J/request', 'rapl_dram', 'request_window', applicability='dram', label='DRAM 能耗 / request', group='Energy', direction='lower', summary=False, aggregation='mean', row_group='dram_energy', row_order=2),
    Metric('dram_avg_power_w', 'W', 'rapl_dram', 'request_window', applicability='dram', row_group='dram_energy', row_order=3),
    Metric('dram_peak_power_w', 'W', 'rapl_dram', 'request_window', applicability='dram', row_group='dram_energy', row_order=4),
    Metric('dram_idle_power_w', 'W', 'rapl_dram', 'control_window', applicability='dram', row_group='dram_energy', row_order=5),
    Metric('dram_window_effective_energy_j', 'J/window', 'rapl_dram', 'request_window', applicability='dram', row_group='dram_energy', row_order=6),
    Metric('dram_effective_energy_per_request_j', 'J/request', 'rapl_dram', 'request_window', applicability='dram', row_group='dram_energy', row_order=7),
    Metric('dram_energy_status', 'text', 'rapl_dram', 'request_window', applicability='dram', kind='text'),
    Metric('dram_energy_error', 'text', 'rapl_dram', 'request_window', applicability='dram', kind='text'),
    Metric('vcpu_cpu_share', 'ratio', 'rapl_cgroup_attribution', 'request_window', row_group='cpu_attribution', row_order=0),
    Metric('vcpu_cpu_time_s', 's/request', 'rapl_cgroup_attribution', 'request_window', row_group='cpu_attribution', row_order=1),
    Metric('vcpu_avg_power_total_w', 'W', 'rapl_cgroup_attribution', 'request_window', row_group='cpu_attribution', row_order=2),
    Metric('vcpu_peak_power_total_w', 'W', 'rapl_cgroup_attribution', 'request_window', row_group='cpu_attribution', row_order=3),
    Metric('vcpu_energy_total_j', 'J/request', 'rapl_cgroup_attribution', 'request_window', row_group='cpu_attribution', row_order=4),
    Metric('vcpu_avg_power_eff_w', 'W', 'rapl_cgroup_attribution', 'request_window', row_group='cpu_attribution', row_order=5),
    Metric('vcpu_peak_power_eff_w', 'W', 'rapl_cgroup_attribution', 'request_window', row_group='cpu_attribution', row_order=6),
    Metric('vcpu_energy_eff_j', 'J/request', 'rapl_cgroup_attribution', 'request_window', row_group='cpu_attribution', row_order=7),
    Metric('container_attributed_energy_eff_j', 'J/request', 'derived', 'request_window', label='容器归因有效能耗（估算）', group='Energy', direction='lower', summary=False, aggregation='mean', row_group='efficiency', row_order=0),
    Metric('container_attributed_samples_per_j', 'sample/J', 'derived', 'request_window', row_group='efficiency', row_order=1),
    Metric('container_attributed_edp_app_js', 'J*s/request', 'derived', 'request_window', row_group='efficiency', row_order=2),
    Metric('output_tokens_per_s_app', 'token/s', 'client_http', 'request_window', applicability='text_tokenizer', label='输出 Token 吞吐量', group='Performance', direction='higher', summary=False, aggregation='mean', row_group='efficiency', row_order=3),
    Metric('container_attributed_j_per_output_token', 'J/token', 'derived', 'request_window', applicability='text_tokenizer', row_group='efficiency', row_order=4),
    Metric('container_attributed_j_per_input_unit', 'J/input_unit', 'derived', 'request_window', row_group='efficiency', row_order=5),
    Metric('container_attributed_j_per_input_megapixel', 'J/Mpixel', 'derived', 'request_window', applicability='pixel_geometry'),
    Metric('container_attributed_j_per_output_megapixel', 'J/Mpixel', 'derived', 'request_window', applicability='pixel_geometry'),
    # CPU 资源、频率与 PMU。
    Metric('resource_usage_iters', 'sample/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=0),
    Metric('container_cpu_util_avg_pct', '%', 'cgroup', 'request_window', label='CPU 平均占用', group='Resource', direction='neutral', summary=False, aggregation='mean', row_group='resource_usage', row_order=1),
    Metric('container_cpu_util_peak_pct', '%', 'cgroup', 'request_window', label='CPU 峰值占用', group='Resource', direction='neutral', summary=True, aggregation='max', row_group='resource_usage', row_order=2),
    Metric('container_cpu_nr_periods_delta', 'count/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=3),
    Metric('container_cpu_nr_throttled_delta', 'count/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=4),
    Metric('container_cpu_throttled_period_ratio_pct', '%', 'cgroup', 'request_window', row_group='resource_usage', row_order=5),
    Metric('container_cpu_throttled_time_s_per_request', 's/request', 'cgroup', 'request_window', row_group='resource_usage', row_order=6),
    Metric('container_cpu_pressure_some_stall_pct', '%', 'cgroup', 'request_window', row_group='resource_usage', row_order=7),
    Metric('container_cpu_pressure_full_stall_pct', '%', 'cgroup', 'request_window', row_group='resource_usage', row_order=8),
    Metric('cpu_freq_avg_hz', 'Hz', 'host_cpufreq', 'request_window', row_group='resource_usage', row_order=9),
    Metric('cpu_freq_peak_hz', 'Hz', 'host_cpufreq', 'request_window', row_group='resource_usage', row_order=10),
    Metric('cpu_cycles_est_app', 'cycle/request', 'derived', 'request_window'),
    Metric('cpu_cycles_est_packet', 'cycle/request', 'derived', 'request_window'),
    Metric('cpu_cycles_per_request', 'cycle/request', 'perf_stat', 'request_window', label='CPU cycles', group='CPU PMU', direction='lower', summary=False, aggregation='mean', row_group='cpu_pmu', row_order=0),
    Metric('cpu_ref_cycles_per_request', 'ref_cycle/request', 'perf_stat', 'request_window', label='CPU ref-cycles', group='CPU PMU', direction='lower', summary=False, aggregation='mean', row_group='cpu_pmu', row_order=1),
    Metric('cpu_instructions_per_request', 'count/request', 'perf_stat', 'request_window', label='CPU instructions', group='CPU PMU', direction='neutral', summary=False, aggregation='mean', row_group='cpu_pmu', row_order=4),
    Metric('cpu_ipc', 'instruction/cycle', 'perf_stat', 'request_window', label='IPC', group='CPU PMU', direction='higher', summary=False, aggregation='mean', row_group='cpu_pmu', row_order=2),
    Metric('cpu_mips_app', 'Minstruction/s', 'perf_stat', 'request_window', row_group='cpu_pmu', row_order=5),
    Metric('cpu_mips_packet', 'Minstruction/s', 'perf_stat', 'request_window', row_group='cpu_pmu', row_order=6),
    Metric('cpu_perf_elapsed_s', 's', 'perf_stat', 'request_window', row_group='cpu_pmu', row_order=7),
    Metric('cpu_perf_running_pct', '%', 'perf_stat', 'request_window', label='PMU running', group='CPU PMU', direction='neutral', summary=False, aggregation='mean', row_group='cpu_pmu', row_order=3),
    Metric('cpu_cache_references_per_request', 'count/request', 'perf_stat', 'request_window', row_group='cpu_pmu', row_order=8),
    Metric('cpu_cache_misses_per_request', 'count/request', 'perf_stat', 'request_window', row_group='cpu_pmu', row_order=9),
    Metric('cpu_cache_miss_rate_pct', '%', 'perf_stat', 'request_window', row_group='cpu_pmu', row_order=10),
    Metric('cpu_dtlb_loads_per_request', 'count/request', 'perf_stat', 'request_window', row_group='cpu_pmu', row_order=11),
    Metric('cpu_dtlb_load_misses_per_request', 'count/request', 'perf_stat', 'request_window', row_group='cpu_pmu', row_order=12),
    Metric('cpu_dtlb_load_miss_rate_pct', '%', 'perf_stat', 'request_window', row_group='cpu_pmu', row_order=13),
    # 容器内存、swap、I/O 与 PID。
    Metric('container_mem_usage_avg_bytes', 'byte', 'cgroup', 'request_window', label='内存平均用量', group='Resource', direction='lower', summary=False, aggregation='mean', row_group='resource_usage', row_order=11),
    Metric('container_mem_usage_peak_bytes', 'byte', 'cgroup', 'request_window', label='内存峰值用量', group='Resource', direction='lower', summary=True, aggregation='max', row_group='resource_usage', row_order=12),
    Metric('container_mem_util_avg_pct', '%', 'cgroup', 'request_window', row_group='resource_usage', row_order=13),
    Metric('container_mem_util_peak_pct', '%', 'cgroup', 'request_window', row_group='resource_usage', row_order=14),
    Metric('container_mem_peak_cgroup_bytes', 'byte', 'cgroup', 'cgroup_lifetime', label='cgroup 生命周期内存峰值', group='Resource', direction='lower', summary=False, aggregation='max', row_group='resource_usage', row_order=15),
    Metric('container_mem_anon_bytes_end', 'byte', 'cgroup', 'request_window_end', row_group='resource_usage', row_order=16),
    Metric('container_mem_file_bytes_end', 'byte', 'cgroup', 'request_window_end', row_group='resource_usage', row_order=17),
    Metric('container_mem_slab_bytes_end', 'byte', 'cgroup', 'request_window_end', row_group='resource_usage', row_order=18),
    Metric('container_mem_pgfault_delta', 'count/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=19),
    Metric('container_mem_pgmajfault_delta', 'count/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=20),
    Metric('container_mem_workingset_refault_delta', 'count/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=21),
    Metric('container_mem_high_events_delta', 'count/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=22),
    Metric('container_mem_max_events_delta', 'count/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=23),
    Metric('container_mem_oom_events_delta', 'count/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=24),
    Metric('container_mem_oom_kill_events_delta', 'count/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=25),
    Metric('container_mem_pressure_some_stall_pct', '%', 'cgroup', 'request_window', row_group='resource_usage', row_order=26),
    Metric('container_mem_pressure_full_stall_pct', '%', 'cgroup', 'request_window', row_group='resource_usage', row_order=27),
    Metric('container_swap_limit_bytes', 'byte', 'cgroup', 'request_window', row_group='resource_usage', row_order=28),
    Metric('container_swap_usage_avg_bytes', 'byte', 'cgroup', 'request_window', row_group='resource_usage', row_order=29),
    Metric('container_swap_usage_peak_bytes', 'byte', 'cgroup', 'request_window', row_group='resource_usage', row_order=30),
    Metric('container_io_read_bytes_per_request', 'byte/request', 'cgroup', 'request_window', row_group='resource_usage', row_order=31),
    Metric('container_io_write_bytes_per_request', 'byte/request', 'cgroup', 'request_window', row_group='resource_usage', row_order=32),
    Metric('container_io_read_ops_per_request', 'count/request', 'cgroup', 'request_window', row_group='resource_usage', row_order=33),
    Metric('container_io_write_ops_per_request', 'count/request', 'cgroup', 'request_window', row_group='resource_usage', row_order=34),
    Metric('container_io_pressure_some_stall_pct', '%', 'cgroup', 'request_window', row_group='resource_usage', row_order=35),
    Metric('container_io_pressure_full_stall_pct', '%', 'cgroup', 'request_window', row_group='resource_usage', row_order=36),
    Metric('container_pids_current_end', 'count', 'cgroup', 'request_window_end', row_group='resource_usage', row_order=37),
    Metric('container_pids_peak_cgroup', 'count', 'cgroup', 'cgroup_lifetime', row_group='resource_usage', row_order=38),
    Metric('container_pids_max_events_delta', 'count/window', 'cgroup', 'request_window', row_group='resource_usage', row_order=39),
    # GPU 资源与运行状态。
    Metric('gpu_sm_clock_mhz', 'MHz', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='gpu_runtime_state', row_order=0),
    Metric('gpu_memory_clock_mhz', 'MHz', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='gpu_runtime_state', row_order=1),
    Metric('gpu_pstate', 'text', 'nvml_selected_device', 'request_window', applicability='gpu', kind='text', row_group='gpu_runtime_state', row_order=2),
    Metric('gpu_temp_c', 'degC', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='gpu_runtime_state', row_order=3),
    Metric('gpu_util_avg_pct', '%', 'nvml_selected_device', 'request_window', applicability='gpu', label='GPU 平均占用', group='GPU', direction='neutral', summary=False, aggregation='mean', row_group='resource_usage', row_order=40),
    Metric('gpu_util_peak_pct', '%', 'nvml_selected_device', 'request_window', applicability='gpu', label='GPU 峰值占用', group='GPU', direction='neutral', summary=True, aggregation='max', row_group='resource_usage', row_order=41),
    Metric('gpu_mem_used_avg_bytes', 'byte', 'nvml_selected_device', 'request_window', applicability='gpu', label='VRAM 平均用量', group='GPU', direction='lower', summary=False, aggregation='mean', row_group='resource_usage', row_order=42),
    Metric('gpu_mem_used_peak_bytes', 'byte', 'nvml_selected_device', 'request_window', applicability='gpu', label='VRAM 峰值用量', group='GPU', direction='lower', summary=True, aggregation='max', row_group='resource_usage', row_order=43),
    Metric('gpu_mem_util_avg_pct', '%', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='resource_usage', row_order=44),
    Metric('gpu_mem_util_peak_pct', '%', 'nvml_selected_device', 'request_window', applicability='gpu', row_group='resource_usage', row_order=45),
    # 容器冷启动。
    Metric('cold_start_started_at', 'ISO-8601', 'docker_and_server', 'container_startup', kind='text'),
    Metric('cold_start_ready_at', 'ISO-8601', 'docker_and_server', 'container_startup', kind='text'),
    Metric('cold_start_container_launch_s', 's', 'docker_and_server', 'container_startup', label='Container launch', group='Startup', direction='lower', summary=False, aggregation='lifecycle'),
    Metric('cold_start_server_setup_s', 's', 'docker_and_server', 'container_startup', label='Server setup', group='Startup', direction='lower', summary=False, aggregation='lifecycle'),
    Metric('cold_start_cuda_init_s', 's', 'docker_and_server', 'container_startup', label='CUDA init', group='Startup', direction='lower', summary=False, aggregation='lifecycle'),
    Metric('cold_start_model_load_s', 's', 'docker_and_server', 'container_startup', label='Model load', group='Startup', direction='lower', summary=False, aggregation='lifecycle'),
    Metric('cold_start_ready_wait_s', 's', 'docker_and_server', 'container_startup', label='Ready wait', group='Startup', direction='lower', summary=False, aggregation='lifecycle'),
    Metric('cold_start_first_predict_app_s', 's', 'docker_and_server', 'container_startup', label='First predict', group='Startup', direction='lower', summary=False, aggregation='lifecycle'),
    Metric('cold_start_s', 's', 'docker_and_server', 'container_startup', label='Cold start', group='Startup', direction='lower', summary=True, aggregation='lifecycle'),
    # 请求契约与结果来源；整体状态和错误固定在最后两列。
    Metric('workload_contract', 'JSON', 'handler', 'actual_request', kind='text'),
    Metric('result_origin', 'text', 'protocol', 'row', kind='text'),
    Metric('status', 'text', 'protocol', 'row', kind='text'),
    Metric('error', 'text', 'protocol', 'row', kind='text'),
)
# 展示语义与采集声明共用一个登记表；不改变 CSV 字段、单位或顺序。
# utilization、频率和总计数不自动解释成效率；未审定的指标保持 neutral。



METRICS = MappingProxyType({metric.name: metric for metric in _DECLARATIONS})
if len(METRICS) != len(_DECLARATIONS):
    raise ValueError("指标登记表包含重复字段")
CSV_FIELDS = list(METRICS)
NUMERIC_FIELDS = tuple(name for name, metric in METRICS.items() if metric.kind == "number")


def row_group_fields(group: str) -> list[str]:
    """Select explicit assembly roles; preserve their historical field order."""
    selected = (metric for metric in METRICS.values() if metric.row_group == group)
    ordered = sorted(selected, key=lambda metric: (metric.row_order is None, metric.row_order or 0))
    return [metric.name for metric in ordered]


GPU_RUNTIME_STATE_FIELDS = row_group_fields("gpu_runtime_state")

# 分析专用派生量不会写入采集 CSV。QPS 只接收显式记录，不由 samples/s 猜测。
_ANALYSIS_ONLY = (
    Metric("qps", "request/s", "recorded_qps", "request_window", label="QPS",
           group="Performance", direction="higher", summary=True),
    Metric("observed_energy_j", "J", "derived_package_gpu", "successful_windows",
           label="CPU package + GPU 窗口总能量", group="Energy", summary=True,
           aggregation="sum"),
    Metric("observed_energy_per_request_j", "J/request", "derived_package_gpu", "request_window",
           label="CPU package + GPU 能耗 / request", group="Energy", direction="lower",
           summary=True, aggregation="request_weighted"),
)
ANALYSIS_METRICS = MappingProxyType({**METRICS, **{m.name: m for m in _ANALYSIS_ONLY}})
VIEW_METRICS = tuple(name for name, metric in ANALYSIS_METRICS.items() if metric.group != "Other")


def order_csv_fields(fields: Iterable[str]) -> list[str]:
    """按登记顺序排列已有列；未知扩展列保留在 status、error 之前。"""
    fields = list(fields)
    present = set(fields)
    diagnostics = ("status", "error")
    ordered = [name for name in CSV_FIELDS if name in present and name not in diagnostics]
    ordered.extend(name for name in fields if name not in METRICS)
    ordered.extend(name for name in diagnostics if name in present)
    return ordered


def tool_fields(tool: str, *, numeric_only: bool = False) -> tuple[str, ...]:
    return tuple(name for name, metric in METRICS.items()
                 if metric.tool == tool and metric.posthoc and (not numeric_only or metric.kind == "number"))
