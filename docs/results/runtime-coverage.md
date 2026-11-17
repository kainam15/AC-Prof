# Docker / Process 指标覆盖矩阵

由 `scripts/render_runtime_coverage.py` 从 [指标登记表](../../acprof/results/metrics.py) 和
[采集来源契约](../../acprof/results/coverage.py) 生成。语义与实施边界见 [统一画像](runtime-profile.md)。

共 250 个登记字段，包含身份、诊断和数值，不能把字段数当作可采集指标数。
`conditional` 仅表示存在实现路径；实际可采集性由权限、设备和当前运行证据决定。
`unsupported` 表示该 Runtime 未接入。所有条目的硬件验收均为 `not_verified`；
本表不凭代码、Mock 或历史不同版本实验宣称已通过硬件验证。

| 来源 | 范围 | 成本级别 | 条件 / 权限 | 生产者 |
| --- | --- | --- | --- | --- |
| `allocation_evidence` | `protocol_identity` | none | valid run identity | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `cgroup` | `container_cgroup` | medium | local Docker; cgroup v2; readable metric files | [resource_usage.py](../../acprof/monitors/resource_usage.py) |
| `cgroup` | `container_cgroup` | low | local Docker; cgroup v2; readable metric files | [resource_usage.py](../../acprof/monitors/resource_usage.py) |
| `client_http` | `client_http_requests` | none | successful requests; tokenizer/geometry when applicable | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `collector_contract` | `protocol_identity` | none | valid run identity | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `derived` | `cgroup_cpu_share_plus_selected_gpu` | none | finite source fields and positive denominator | [collector_metrics.py](../../acprof/host/measurement/collector_metrics.py) |
| `derived` | `cgroup_cpu_time_times_host_frequency` | none | finite source fields and positive denominator | [collector_metrics.py](../../acprof/host/measurement/collector_metrics.py) |
| `derived` | `client_http_requests` | none | finite source fields and positive denominator | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `docker_and_server` | `container_service_lifetime` | none | Docker readiness timestamps | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `execution_runtime` | `protocol_identity` | none | valid run identity | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `handler` | `actual_request` | none | handler response contract | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `host_cpufreq` | `host_visible_cpus` | medium | readable cpufreq or cpuinfo; unavailable on WSL2 | [resource_usage.py](../../acprof/monitors/resource_usage.py) |
| `immutable_runtime_artifact` | `protocol_identity` | none | valid run identity | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `input_plan` | `materialized_request` | none | frozen input plan | [__init__.py](../../acprof/host/input_plan/__init__.py) |
| `managed_process_to_ready` | `managed_service_lifetime` | none | Process readiness | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `massif` | `independent_process_heap` | high | explicit profiler; CPU; Valgrind | [execution_plan.py](../../acprof/host/profiling/execution_plan.py) |
| `ncu` | `independent_gpu_kernels` | high | explicit profiler; GPU counter permission | [compute_plan.py](../../acprof/host/profiling/compute_plan.py) |
| `nsys` | `independent_cuda_timeline` | high | explicit profiler; NVIDIA GPU; Nsys | [execution_plan.py](../../acprof/host/profiling/execution_plan.py) |
| `nvml_selected_device` | `selected_gpu_device` | medium | Full; NVML counter or power integration; matched control | [energy_nvml.py](../../acprof/monitors/energy_nvml.py) |
| `nvml_selected_device` | `selected_gpu_device` | medium | GPU UUID; NVML query support and permission | [resource_usage.py](../../acprof/monitors/resource_usage.py) |
| `pcap` | `correlated_http_connection_frames` | high | Full; capture permission; complete request matching | [merge_packet_latency.py](../../acprof/packet/merge_packet_latency.py) |
| `perf_stat` | `container_cgroup_pmu` | medium | Full; perf; PMU permission; valid running ratio | [perf_mips.py](../../acprof/monitors/perf_mips.py) |
| `platform_identity` | `protocol_identity` | none | valid run identity | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `procfs_observed_process_tree` | `observed_process_tree` | low | Linux procfs; stable process membership | [process_usage.py](../../acprof/monitors/process_usage.py) |
| `protocol` | `protocol_identity` | none | valid run identity | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `psutil` | `observed_process_tree` | low | target procfs access | [process_usage.py](../../acprof/monitors/process_usage.py) |
| `psutil_observed_process_tree` | `observed_process_tree` | medium | procfs access; stable PID + start time | [process_usage.py](../../acprof/monitors/process_usage.py) |
| `psutil_observed_process_tree` | `observed_process_tree` | low | procfs access; stable PID + start time | [process_usage.py](../../acprof/monitors/process_usage.py) |
| `psutil_observed_process_tree` | `observed_process_tree_leaders` | low | procfs access; stable PID + start time | [process_usage.py](../../acprof/monitors/process_usage.py) |
| `rapl_cgroup_attribution` | `cgroup_cpu_time_share_of_host_packages` | medium | Full; RAPL + cgroup + host CPU counters | [energy_cpu.py](../../acprof/monitors/energy_cpu.py) |
| `rapl_dram` | `host_dram_domains` | medium | Full; DRAM requested; complete readable domains | [energy_cpu.py](../../acprof/monitors/energy_cpu.py) |
| `rapl_package` | `host_cpu_packages` | medium | Full; readable RAPL package domains | [energy_cpu.py](../../acprof/monitors/energy_cpu.py) |
| `resource_quality` | `per_metric` | none | finalized collector evidence | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `selected_device_identity` | `protocol_identity` | none | valid run identity | [metrics.py](../../acprof/host/measurement/metrics.py) |
| `service_cgroup` | `owned_service_cgroup` | low | verified owned cgroup; membership; readable metric files | [process_cgroup.py](../../acprof/monitors/process_cgroup.py) |
| `service_cgroup` | `owned_service_cgroup` | medium | verified owned cgroup; membership; readable metric files | [process_cgroup.py](../../acprof/monitors/process_cgroup.py) |
| `torch` | `independent_model_forward` | high | explicit profiler; supported backend | [compute_plan.py](../../acprof/host/profiling/compute_plan.py) |
| `verified_runtime_limits` | `protocol_identity` | none | valid run identity | [metrics.py](../../acprof/host/measurement/metrics.py) |

| 指标 | Docker | Process | 单位 | 来源 | 窗口 | 推导 |
| --- | --- | --- | --- | --- | --- | --- |
| `environment_class` | `conditional` | `conditional` | `text` | `platform_identity` | `experiment` | `recorded` |
| `runtime_kind` | `conditional` | `conditional` | `text` | `execution_runtime` | `experiment` | `recorded` |
| `runtime_id` | `conditional` | `conditional` | `text` | `immutable_runtime_artifact` | `experiment` | `recorded` |
| `resource_limit_source` | `conditional` | `conditional` | `text` | `verified_runtime_limits` | `case` | `recorded` |
| `resource_scope` | `conditional` | `conditional` | `text` | `collector_contract` | `case` | `recorded` |
| `environment_sharing` | `conditional` | `conditional` | `text` | `allocation_evidence` | `case` | `recorded` |
| `gpu_metric_scope` | `conditional` | `conditional` | `text` | `collector_contract` | `case` | `recorded` |
| `process_resource_error` | `conditional` | `conditional` | `text` | `psutil` | `request_window` | `recorded` |
| `process_resource_status` | `conditional` | `conditional` | `text` | `psutil` | `request_window` | `recorded` |
| `gpu_utilization_status` | `conditional` | `conditional` | `text` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_memory_status` | `conditional` | `conditional` | `text` | `nvml_selected_device` | `request_window` | `observed` |
| `process_startup_s` | `unsupported` | `conditional` | `s` | `managed_process_to_ready` | `case_startup` | `observed` |
| `process_cpu_time_s` | `conditional` | `conditional` | `s/window` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_cpu_util_avg_pct` | `conditional` | `conditional` | `% one logical CPU` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_rss_avg_bytes` | `conditional` | `conditional` | `byte` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_rss_peak_bytes` | `conditional` | `conditional` | `byte` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_cpu_user_time_s` | `conditional` | `conditional` | `s/window` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_cpu_system_time_s` | `conditional` | `conditional` | `s/window` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_io_read_bytes` | `conditional` | `conditional` | `byte/window` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_io_write_bytes` | `conditional` | `conditional` | `byte/window` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_io_read_ops` | `conditional` | `conditional` | `count/window` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_io_write_ops` | `conditional` | `conditional` | `count/window` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_leader_voluntary_ctx_switches` | `conditional` | `conditional` | `count/window` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_leader_involuntary_ctx_switches` | `conditional` | `conditional` | `count/window` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `process_minor_faults` | `conditional` | `conditional` | `count/window` | `procfs_observed_process_tree` | `request_window` | `observed` |
| `process_major_faults` | `conditional` | `conditional` | `count/window` | `procfs_observed_process_tree` | `request_window` | `observed` |
| `process_swap_bytes_end` | `conditional` | `conditional` | `byte` | `procfs_observed_process_tree` | `request_window_end` | `observed` |
| `process_count_end` | `conditional` | `conditional` | `count` | `psutil_observed_process_tree` | `request_window_end` | `observed` |
| `process_threads_end` | `conditional` | `conditional` | `count` | `psutil_observed_process_tree` | `request_window_end` | `observed` |
| `process_fds_end` | `conditional` | `conditional` | `count` | `psutil_observed_process_tree` | `request_window_end` | `observed` |
| `process_rss_growth_bytes` | `conditional` | `conditional` | `byte/window` | `psutil_observed_process_tree` | `request_window` | `observed` |
| `resource_metric_quality` | `conditional` | `conditional` | `JSON` | `resource_quality` | `request_window` | `recorded` |
| `service_cgroup_cpu_time_s` | `unsupported` | `conditional` | `s/window` | `service_cgroup` | `request_window` | `observed` |
| `service_cgroup_mem_usage_avg_bytes` | `unsupported` | `conditional` | `byte` | `service_cgroup` | `request_window` | `observed` |
| `service_cgroup_mem_usage_peak_bytes` | `unsupported` | `conditional` | `byte` | `service_cgroup` | `request_window` | `observed` |
| `service_cgroup_io_read_bytes` | `unsupported` | `conditional` | `byte/window` | `service_cgroup` | `request_window` | `observed` |
| `service_cgroup_io_write_bytes` | `unsupported` | `conditional` | `byte/window` | `service_cgroup` | `request_window` | `observed` |
| `run_id` | `conditional` | `conditional` | `text` | `protocol` | `run` | `recorded` |
| `case_id` | `conditional` | `conditional` | `text` | `protocol` | `case` | `recorded` |
| `case_attempt_id` | `conditional` | `conditional` | `text` | `protocol` | `case_attempt` | `recorded` |
| `window_id` | `conditional` | `conditional` | `text` | `protocol` | `request_window` | `recorded` |
| `packet_status` | `conditional` | `unsupported` | `text` | `pcap` | `request_window` | `observed` |
| `cpu_cores` | `conditional` | `conditional` | `core` | `protocol` | `case` | `recorded` |
| `mem_cap_gb` | `conditional` | `conditional` | `GiB` | `protocol` | `case` | `recorded` |
| `gpu_mode` | `conditional` | `conditional` | `text` | `protocol` | `case` | `recorded` |
| `gpu_device_uuid` | `conditional` | `conditional` | `text` | `selected_device_identity` | `case` | `recorded` |
| `input_scale` | `conditional` | `conditional` | `input_scale_type` | `protocol` | `row` | `recorded` |
| `input_units_per_request` | `conditional` | `conditional` | `input_unit/request` | `input_plan` | `materialized_request` | `recorded` |
| `input_num_samples` | `conditional` | `conditional` | `sample/request` | `input_plan` | `materialized_request` | `recorded` |
| `input_pixels_per_request` | `conditional` | `conditional` | `pixel/request` | `input_plan` | `materialized_request` | `recorded` |
| `output_pixels_per_request` | `conditional` | `conditional` | `pixel/request` | `input_plan` | `materialized_request` | `recorded` |
| `request_payload_bytes` | `conditional` | `conditional` | `byte/request` | `client_http` | `request_window` | `observed` |
| `packet_request_wire_bytes_per_request` | `conditional` | `unsupported` | `byte/request` | `pcap` | `request_window` | `observed` |
| `packet_response_wire_bytes_per_request` | `conditional` | `unsupported` | `byte/request` | `pcap` | `request_window` | `observed` |
| `packet_total_wire_bytes_per_request` | `conditional` | `unsupported` | `byte/request` | `pcap` | `request_window` | `observed` |
| `packet_tcp_payload_bytes_per_request` | `conditional` | `unsupported` | `byte/request` | `pcap` | `request_window` | `observed` |
| `packet_protocol_overhead_bytes_per_request` | `conditional` | `unsupported` | `byte/request` | `pcap` | `request_window` | `observed` |
| `packet_protocol_overhead_ratio` | `conditional` | `unsupported` | `ratio` | `pcap` | `request_window` | `observed` |
| `task_param` | `conditional` | `conditional` | `text` | `protocol` | `row` | `recorded` |
| `output_length_avg` | `conditional` | `conditional` | `task_output_unit/request` | `client_http` | `request_window` | `observed` |
| `output_token_count_avg` | `conditional` | `conditional` | `token/request` | `client_http` | `request_window` | `observed` |
| `repeat_idx` | `conditional` | `conditional` | `index` | `protocol` | `row` | `recorded` |
| `warmup` | `conditional` | `conditional` | `0 / 1` | `protocol` | `row` | `recorded` |
| `repeat_in_window` | `conditional` | `conditional` | `request/window` | `protocol` | `row` | `recorded` |
| `latency_s` | `conditional` | `unsupported` | `s` | `pcap` | `request_window` | `observed` |
| `latency_s_per_input_unit` | `conditional` | `unsupported` | `s/input_unit` | `pcap` | `request_window` | `observed` |
| `latency_s_per_input_megapixel` | `conditional` | `unsupported` | `s/Mpixel` | `pcap` | `request_window` | `observed` |
| `latency_s_per_output_megapixel` | `conditional` | `unsupported` | `s/Mpixel` | `pcap` | `request_window` | `observed` |
| `latency_request_count` | `conditional` | `unsupported` | `request/window` | `pcap` | `request_window` | `observed` |
| `latency_p50_s` | `conditional` | `unsupported` | `s` | `pcap` | `request_window` | `observed` |
| `latency_p90_s` | `conditional` | `unsupported` | `s` | `pcap` | `request_window` | `observed` |
| `latency_p95_s` | `conditional` | `unsupported` | `s` | `pcap` | `request_window` | `observed` |
| `latency_tail_ratio` | `conditional` | `unsupported` | `ratio` | `pcap` | `request_window` | `observed` |
| `latency_std_s` | `conditional` | `unsupported` | `s` | `pcap` | `request_window` | `observed` |
| `latency_cv` | `conditional` | `unsupported` | `ratio` | `pcap` | `request_window` | `observed` |
| `latency_iqr_s` | `conditional` | `unsupported` | `s` | `pcap` | `request_window` | `observed` |
| `latency_max_s` | `conditional` | `unsupported` | `s` | `pcap` | `request_window` | `observed` |
| `latency_slow_ratio` | `conditional` | `unsupported` | `ratio` | `pcap` | `request_window` | `observed` |
| `latency_app_s` | `conditional` | `conditional` | `s` | `client_http` | `request_window` | `observed` |
| `latency_app_s_per_input_unit` | `conditional` | `conditional` | `s/input_unit` | `client_http` | `request_window` | `observed` |
| `latency_app_s_per_input_megapixel` | `conditional` | `conditional` | `s/Mpixel` | `client_http` | `request_window` | `observed` |
| `latency_app_s_per_output_megapixel` | `conditional` | `conditional` | `s/Mpixel` | `client_http` | `request_window` | `observed` |
| `latency_app_request_count` | `conditional` | `conditional` | `request/window` | `client_http` | `request_window` | `observed` |
| `latency_app_p50_s` | `conditional` | `conditional` | `s` | `client_http` | `request_window` | `observed` |
| `latency_app_p90_s` | `conditional` | `conditional` | `s` | `client_http` | `request_window` | `observed` |
| `latency_app_p95_s` | `conditional` | `conditional` | `s` | `client_http` | `request_window` | `observed` |
| `latency_app_tail_ratio` | `conditional` | `conditional` | `ratio` | `client_http` | `request_window` | `observed` |
| `latency_app_std_s` | `conditional` | `conditional` | `s` | `client_http` | `request_window` | `observed` |
| `latency_app_cv` | `conditional` | `conditional` | `ratio` | `client_http` | `request_window` | `observed` |
| `latency_app_iqr_s` | `conditional` | `conditional` | `s` | `client_http` | `request_window` | `observed` |
| `latency_app_max_s` | `conditional` | `conditional` | `s` | `client_http` | `request_window` | `observed` |
| `latency_app_slow_ratio` | `conditional` | `conditional` | `ratio` | `client_http` | `request_window` | `observed` |
| `throughput_samples_per_s` | `conditional` | `conditional` | `sample/s` | `derived` | `request_window` | `derived` |
| `throughput_samples_per_s_per_cpu_core` | `conditional` | `conditional` | `sample/s/core` | `derived` | `request_window` | `derived` |
| `model_logical_mflop_per_request_torch_profiler_eager` | `conditional` | `unsupported` | `MFLOP/request` | `torch` | `independent_profiler` | `observed` |
| `model_logical_mflops_app_torch_profiler_eager` | `conditional` | `unsupported` | `MFLOP/s` | `torch` | `independent_profiler` | `observed` |
| `model_logical_mflops_packet_torch_profiler_eager` | `conditional` | `unsupported` | `MFLOP/s` | `torch` | `independent_profiler` | `observed` |
| `compute_profile_error_torch_profiler_eager` | `conditional` | `unsupported` | `text` | `torch` | `independent_profiler` | `observed` |
| `gpu_executed_mflop_per_request_ncu` | `conditional` | `unsupported` | `MFLOP/request` | `ncu` | `independent_profiler` | `observed` |
| `gpu_executed_tensor_mflop_per_request_ncu` | `conditional` | `unsupported` | `MFLOP/request` | `ncu` | `independent_profiler` | `observed` |
| `gpu_executed_scalar_mflop_per_request_ncu` | `conditional` | `unsupported` | `MFLOP/request` | `ncu` | `independent_profiler` | `observed` |
| `gpu_executed_tensor_share_pct_ncu` | `conditional` | `unsupported` | `%` | `ncu` | `independent_profiler` | `observed` |
| `gpu_executed_mflops_app_ncu` | `conditional` | `unsupported` | `MFLOP/s` | `ncu` | `independent_profiler` | `observed` |
| `gpu_executed_mflops_packet_ncu` | `conditional` | `unsupported` | `MFLOP/s` | `ncu` | `independent_profiler` | `observed` |
| `gpu_kernel_launch_count_per_request_ncu` | `conditional` | `unsupported` | `count/request` | `ncu` | `independent_profiler` | `observed` |
| `gpu_kernel_time_sum_ms_per_request_ncu` | `conditional` | `unsupported` | `ms/request` | `ncu` | `independent_profiler` | `observed` |
| `compute_profile_error_ncu` | `conditional` | `unsupported` | `text` | `ncu` | `independent_profiler` | `observed` |
| `cpu_heap_peak_bytes_massif` | `conditional` | `unsupported` | `byte` | `massif` | `profiler_process_lifetime` | `observed` |
| `cpu_heap_extra_peak_bytes_massif` | `conditional` | `unsupported` | `byte` | `massif` | `profiler_process_lifetime` | `observed` |
| `cpu_stack_peak_bytes_massif` | `conditional` | `unsupported` | `byte` | `massif` | `profiler_process_lifetime` | `observed` |
| `cpu_heap_peak_total_bytes_massif` | `conditional` | `unsupported` | `byte` | `massif` | `profiler_process_lifetime` | `observed` |
| `cpu_heap_peak_at_ms_massif` | `conditional` | `unsupported` | `ms` | `massif` | `profiler_process_lifetime` | `observed` |
| `compute_profile_error_massif` | `conditional` | `unsupported` | `text` | `massif` | `profiler_process_lifetime` | `observed` |
| `host_inference_wall_time_ms_per_request_nsys` | `conditional` | `unsupported` | `ms/request` | `nsys` | `independent_profiler` | `observed` |
| `cuda_api_time_sum_ms_per_request_nsys` | `conditional` | `unsupported` | `ms/request` | `nsys` | `independent_profiler` | `observed` |
| `cuda_api_call_count_per_request_nsys` | `conditional` | `unsupported` | `count/request` | `nsys` | `independent_profiler` | `observed` |
| `gpu_kernel_time_sum_ms_per_request_nsys` | `conditional` | `unsupported` | `ms/request` | `nsys` | `independent_profiler` | `observed` |
| `gpu_kernel_launch_count_per_request_nsys` | `conditional` | `unsupported` | `count/request` | `nsys` | `independent_profiler` | `observed` |
| `gpu_memcpy_time_sum_ms_per_request_nsys` | `conditional` | `unsupported` | `ms/request` | `nsys` | `independent_profiler` | `observed` |
| `gpu_memcpy_count_per_request_nsys` | `conditional` | `unsupported` | `count/request` | `nsys` | `independent_profiler` | `observed` |
| `gpu_memcpy_bytes_per_request_nsys` | `conditional` | `unsupported` | `byte/request` | `nsys` | `independent_profiler` | `observed` |
| `compute_profile_error_nsys` | `conditional` | `unsupported` | `text` | `nsys` | `independent_profiler` | `observed` |
| `gpu_idle_power_w` | `conditional` | `unsupported` | `W` | `nvml_selected_device` | `matched_control` | `observed` |
| `gpu_idle_measured_at` | `conditional` | `unsupported` | `ISO-8601` | `nvml_selected_device` | `matched_control` | `observed` |
| `gpu_idle_rel_range_so_far` | `conditional` | `unsupported` | `ratio` | `nvml_selected_device` | `case_controls_so_far` | `observed` |
| `gpu_idle_energy_source` | `conditional` | `unsupported` | `text` | `nvml_selected_device` | `matched_control` | `observed` |
| `gpu_energy_iters` | `conditional` | `unsupported` | `sample/window` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_avg_power_total_w` | `conditional` | `unsupported` | `W` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_peak_power_total_w` | `conditional` | `unsupported` | `W` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_energy_total_j` | `conditional` | `unsupported` | `J/request` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_avg_power_eff_w` | `conditional` | `unsupported` | `W` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_peak_power_eff_w` | `conditional` | `unsupported` | `W` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_energy_eff_j` | `conditional` | `unsupported` | `J/request` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_energy_source` | `conditional` | `unsupported` | `text` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_energy_fallback_reason` | `conditional` | `unsupported` | `text` | `nvml_selected_device` | `request_window` | `observed` |
| `cpu_idle_power_w` | `conditional` | `unsupported` | `W` | `rapl_package` | `matched_control` | `observed` |
| `cpu_idle_measured_at` | `conditional` | `unsupported` | `ISO-8601` | `rapl_package` | `matched_control` | `observed` |
| `cpu_idle_rel_range_so_far` | `conditional` | `unsupported` | `ratio` | `rapl_package` | `case_controls_so_far` | `observed` |
| `cpu_energy_iters` | `conditional` | `unsupported` | `sample/window` | `rapl_package` | `request_window` | `observed` |
| `cpu_avg_power_total_w` | `conditional` | `unsupported` | `W` | `rapl_package` | `request_window` | `observed` |
| `cpu_peak_power_total_w` | `conditional` | `unsupported` | `W` | `rapl_package` | `request_window` | `observed` |
| `cpu_energy_total_j` | `conditional` | `unsupported` | `J/request` | `rapl_package` | `request_window` | `observed` |
| `cpu_avg_power_eff_w` | `conditional` | `unsupported` | `W` | `rapl_package` | `request_window` | `observed` |
| `cpu_peak_power_eff_w` | `conditional` | `unsupported` | `W` | `rapl_package` | `request_window` | `observed` |
| `cpu_energy_eff_j` | `conditional` | `unsupported` | `J/request` | `rapl_package` | `request_window` | `observed` |
| `dram_window_energy_j` | `conditional` | `unsupported` | `J/window` | `rapl_dram` | `request_window` | `observed` |
| `dram_window_duration_s` | `conditional` | `unsupported` | `s` | `rapl_dram` | `request_window` | `observed` |
| `dram_energy_per_request_j` | `conditional` | `unsupported` | `J/request` | `rapl_dram` | `request_window` | `observed` |
| `dram_avg_power_w` | `conditional` | `unsupported` | `W` | `rapl_dram` | `request_window` | `observed` |
| `dram_peak_power_w` | `conditional` | `unsupported` | `W` | `rapl_dram` | `request_window` | `observed` |
| `dram_idle_power_w` | `conditional` | `unsupported` | `W` | `rapl_dram` | `control_window` | `observed` |
| `dram_window_effective_energy_j` | `conditional` | `unsupported` | `J/window` | `rapl_dram` | `request_window` | `observed` |
| `dram_effective_energy_per_request_j` | `conditional` | `unsupported` | `J/request` | `rapl_dram` | `request_window` | `observed` |
| `dram_energy_status` | `conditional` | `unsupported` | `text` | `rapl_dram` | `request_window` | `observed` |
| `dram_energy_error` | `conditional` | `unsupported` | `text` | `rapl_dram` | `request_window` | `observed` |
| `vcpu_cpu_share` | `conditional` | `unsupported` | `ratio` | `rapl_cgroup_attribution` | `request_window` | `estimated` |
| `vcpu_cpu_time_s` | `conditional` | `unsupported` | `s/request` | `rapl_cgroup_attribution` | `request_window` | `estimated` |
| `vcpu_avg_power_total_w` | `conditional` | `unsupported` | `W` | `rapl_cgroup_attribution` | `request_window` | `estimated` |
| `vcpu_peak_power_total_w` | `conditional` | `unsupported` | `W` | `rapl_cgroup_attribution` | `request_window` | `estimated` |
| `vcpu_energy_total_j` | `conditional` | `unsupported` | `J/request` | `rapl_cgroup_attribution` | `request_window` | `estimated` |
| `vcpu_avg_power_eff_w` | `conditional` | `unsupported` | `W` | `rapl_cgroup_attribution` | `request_window` | `estimated` |
| `vcpu_peak_power_eff_w` | `conditional` | `unsupported` | `W` | `rapl_cgroup_attribution` | `request_window` | `estimated` |
| `vcpu_energy_eff_j` | `conditional` | `unsupported` | `J/request` | `rapl_cgroup_attribution` | `request_window` | `estimated` |
| `container_attributed_energy_eff_j` | `conditional` | `unsupported` | `J/request` | `derived` | `request_window` | `estimated` |
| `container_attributed_samples_per_j` | `conditional` | `unsupported` | `sample/J` | `derived` | `request_window` | `estimated` |
| `container_attributed_edp_app_js` | `conditional` | `unsupported` | `J*s/request` | `derived` | `request_window` | `estimated` |
| `output_tokens_per_s_app` | `conditional` | `conditional` | `token/s` | `client_http` | `request_window` | `observed` |
| `container_attributed_j_per_output_token` | `conditional` | `unsupported` | `J/token` | `derived` | `request_window` | `estimated` |
| `container_attributed_j_per_input_unit` | `conditional` | `unsupported` | `J/input_unit` | `derived` | `request_window` | `estimated` |
| `container_attributed_j_per_input_megapixel` | `conditional` | `unsupported` | `J/Mpixel` | `derived` | `request_window` | `estimated` |
| `container_attributed_j_per_output_megapixel` | `conditional` | `unsupported` | `J/Mpixel` | `derived` | `request_window` | `estimated` |
| `resource_usage_iters` | `conditional` | `unsupported` | `sample/window` | `cgroup` | `request_window` | `observed` |
| `container_cpu_util_avg_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `container_cpu_util_peak_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `container_cpu_nr_periods_delta` | `conditional` | `unsupported` | `count/window` | `cgroup` | `request_window` | `observed` |
| `container_cpu_nr_throttled_delta` | `conditional` | `unsupported` | `count/window` | `cgroup` | `request_window` | `observed` |
| `container_cpu_throttled_period_ratio_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `container_cpu_throttled_time_s_per_request` | `conditional` | `unsupported` | `s/request` | `cgroup` | `request_window` | `observed` |
| `container_cpu_pressure_some_stall_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `container_cpu_pressure_full_stall_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `cpu_freq_avg_hz` | `conditional` | `unsupported` | `Hz` | `host_cpufreq` | `request_window` | `observed` |
| `cpu_freq_peak_hz` | `conditional` | `unsupported` | `Hz` | `host_cpufreq` | `request_window` | `observed` |
| `cpu_cycles_est_app` | `conditional` | `unsupported` | `cycle/request` | `derived` | `request_window` | `estimated` |
| `cpu_cycles_est_packet` | `conditional` | `unsupported` | `cycle/request` | `derived` | `request_window` | `estimated` |
| `cpu_cycles_per_request` | `conditional` | `unsupported` | `cycle/request` | `perf_stat` | `request_window` | `observed` |
| `cpu_ref_cycles_per_request` | `conditional` | `unsupported` | `ref_cycle/request` | `perf_stat` | `request_window` | `observed` |
| `cpu_instructions_per_request` | `conditional` | `unsupported` | `count/request` | `perf_stat` | `request_window` | `observed` |
| `cpu_ipc` | `conditional` | `unsupported` | `instruction/cycle` | `perf_stat` | `request_window` | `observed` |
| `cpu_mips_app` | `conditional` | `unsupported` | `Minstruction/s` | `perf_stat` | `request_window` | `observed` |
| `cpu_mips_packet` | `conditional` | `unsupported` | `Minstruction/s` | `perf_stat` | `request_window` | `observed` |
| `cpu_perf_elapsed_s` | `conditional` | `unsupported` | `s` | `perf_stat` | `request_window` | `observed` |
| `cpu_perf_running_pct` | `conditional` | `unsupported` | `%` | `perf_stat` | `request_window` | `observed` |
| `cpu_cache_references_per_request` | `conditional` | `unsupported` | `count/request` | `perf_stat` | `request_window` | `observed` |
| `cpu_cache_misses_per_request` | `conditional` | `unsupported` | `count/request` | `perf_stat` | `request_window` | `observed` |
| `cpu_cache_miss_rate_pct` | `conditional` | `unsupported` | `%` | `perf_stat` | `request_window` | `observed` |
| `cpu_dtlb_loads_per_request` | `conditional` | `unsupported` | `count/request` | `perf_stat` | `request_window` | `observed` |
| `cpu_dtlb_load_misses_per_request` | `conditional` | `unsupported` | `count/request` | `perf_stat` | `request_window` | `observed` |
| `cpu_dtlb_load_miss_rate_pct` | `conditional` | `unsupported` | `%` | `perf_stat` | `request_window` | `observed` |
| `container_mem_usage_avg_bytes` | `conditional` | `unsupported` | `byte` | `cgroup` | `request_window` | `observed` |
| `container_mem_usage_peak_bytes` | `conditional` | `unsupported` | `byte` | `cgroup` | `request_window` | `observed` |
| `container_mem_util_avg_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `container_mem_util_peak_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `container_mem_peak_cgroup_bytes` | `conditional` | `unsupported` | `byte` | `cgroup` | `cgroup_lifetime` | `observed` |
| `container_mem_anon_bytes_end` | `conditional` | `unsupported` | `byte` | `cgroup` | `request_window_end` | `observed` |
| `container_mem_file_bytes_end` | `conditional` | `unsupported` | `byte` | `cgroup` | `request_window_end` | `observed` |
| `container_mem_slab_bytes_end` | `conditional` | `unsupported` | `byte` | `cgroup` | `request_window_end` | `observed` |
| `container_mem_pgfault_delta` | `conditional` | `unsupported` | `count/window` | `cgroup` | `request_window` | `observed` |
| `container_mem_pgmajfault_delta` | `conditional` | `unsupported` | `count/window` | `cgroup` | `request_window` | `observed` |
| `container_mem_workingset_refault_delta` | `conditional` | `unsupported` | `count/window` | `cgroup` | `request_window` | `observed` |
| `container_mem_high_events_delta` | `conditional` | `unsupported` | `count/window` | `cgroup` | `request_window` | `observed` |
| `container_mem_max_events_delta` | `conditional` | `unsupported` | `count/window` | `cgroup` | `request_window` | `observed` |
| `container_mem_oom_events_delta` | `conditional` | `unsupported` | `count/window` | `cgroup` | `request_window` | `observed` |
| `container_mem_oom_kill_events_delta` | `conditional` | `unsupported` | `count/window` | `cgroup` | `request_window` | `observed` |
| `container_mem_pressure_some_stall_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `container_mem_pressure_full_stall_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `container_swap_limit_bytes` | `conditional` | `unsupported` | `byte` | `cgroup` | `request_window` | `observed` |
| `container_swap_usage_avg_bytes` | `conditional` | `unsupported` | `byte` | `cgroup` | `request_window` | `observed` |
| `container_swap_usage_peak_bytes` | `conditional` | `unsupported` | `byte` | `cgroup` | `request_window` | `observed` |
| `container_io_read_bytes_per_request` | `conditional` | `unsupported` | `byte/request` | `cgroup` | `request_window` | `observed` |
| `container_io_write_bytes_per_request` | `conditional` | `unsupported` | `byte/request` | `cgroup` | `request_window` | `observed` |
| `container_io_read_ops_per_request` | `conditional` | `unsupported` | `count/request` | `cgroup` | `request_window` | `observed` |
| `container_io_write_ops_per_request` | `conditional` | `unsupported` | `count/request` | `cgroup` | `request_window` | `observed` |
| `container_io_pressure_some_stall_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `container_io_pressure_full_stall_pct` | `conditional` | `unsupported` | `%` | `cgroup` | `request_window` | `observed` |
| `container_pids_current_end` | `conditional` | `unsupported` | `count` | `cgroup` | `request_window_end` | `observed` |
| `container_pids_peak_cgroup` | `conditional` | `unsupported` | `count` | `cgroup` | `cgroup_lifetime` | `observed` |
| `container_pids_max_events_delta` | `conditional` | `unsupported` | `count/window` | `cgroup` | `request_window` | `observed` |
| `gpu_sm_clock_mhz` | `conditional` | `conditional` | `MHz` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_memory_clock_mhz` | `conditional` | `conditional` | `MHz` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_pstate` | `conditional` | `conditional` | `text` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_temp_c` | `conditional` | `conditional` | `degC` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_util_avg_pct` | `conditional` | `conditional` | `%` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_util_peak_pct` | `conditional` | `conditional` | `%` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_mem_used_avg_bytes` | `conditional` | `conditional` | `byte` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_mem_used_peak_bytes` | `conditional` | `conditional` | `byte` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_mem_util_avg_pct` | `conditional` | `conditional` | `%` | `nvml_selected_device` | `request_window` | `observed` |
| `gpu_mem_util_peak_pct` | `conditional` | `conditional` | `%` | `nvml_selected_device` | `request_window` | `observed` |
| `cold_start_started_at` | `conditional` | `unsupported` | `ISO-8601` | `docker_and_server` | `container_startup` | `observed` |
| `cold_start_ready_at` | `conditional` | `unsupported` | `ISO-8601` | `docker_and_server` | `container_startup` | `observed` |
| `cold_start_container_launch_s` | `conditional` | `unsupported` | `s` | `docker_and_server` | `container_startup` | `observed` |
| `cold_start_server_setup_s` | `conditional` | `unsupported` | `s` | `docker_and_server` | `container_startup` | `observed` |
| `cold_start_cuda_init_s` | `conditional` | `unsupported` | `s` | `docker_and_server` | `container_startup` | `observed` |
| `cold_start_model_load_s` | `conditional` | `unsupported` | `s` | `docker_and_server` | `container_startup` | `observed` |
| `cold_start_ready_wait_s` | `conditional` | `unsupported` | `s` | `docker_and_server` | `container_startup` | `observed` |
| `cold_start_first_predict_app_s` | `conditional` | `unsupported` | `s` | `docker_and_server` | `container_startup` | `observed` |
| `cold_start_s` | `conditional` | `unsupported` | `s` | `docker_and_server` | `container_startup` | `observed` |
| `workload_contract` | `conditional` | `conditional` | `JSON` | `handler` | `actual_request` | `recorded` |
| `result_origin` | `conditional` | `conditional` | `text` | `protocol` | `row` | `recorded` |
| `status` | `conditional` | `conditional` | `text` | `protocol` | `row` | `recorded` |
| `error` | `conditional` | `conditional` | `text` | `protocol` | `row` | `recorded` |
