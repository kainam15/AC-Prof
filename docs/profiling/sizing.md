# 实验行数与时间预算

[← 返回专题目录](protocol.md)

## 结果行数和时间成本估算

### CSV 行数与请求数

完整计划的行数为：

```text
资源 case 数 = len(cpus) × len(mems) × len(gpus)
总行数 = 资源 case 数 × 实际 input scale 数 × (warmup + repeat)
正式测量行数 = 资源 case 数 × 实际 input scale 数 × repeat
```

这里使用 `input_scale_plan.json` 中实际确定的档数；自定义音频清单可能不是 6 档。
默认 6 档时，计划行数为 `4 × 4 × 2 × 6 × (2 + 5) = 1,344`，其中 warmup 384 行、
正式测量 960 行。OOM、超时或剪枝可能生成错误占位，异常中断也可能留下部分结果，
因此计划行数不等于成功实测行数。

一行 CSV 是一个请求窗口，不等于一个 batch 或一次 `/predict`：

- 固定 `--repeat-in-window N`：每个成功窗口发送 `N` 次请求，完整成功矩阵的窗口内请求数为 `总行数 × N`。
- 默认 `--repeat-in-window 0`：每行至少发送一次请求，累计请求耗时达到 `--repeat-window-seconds` 后停止；CSV 的 `repeat_in_window` 记录该行实际请求数。
- `--batch-size` 决定一次请求的样本数，不乘入 CSV 行数。样本总数还需按任务支持情况乘以 batch size。

auto 模式在每个资源 case / input scale 开始前默认额外发送 5 次 auto warmup 请求；
它们不写成 CSV 行，也不同于 `--warmup 2` 的两个完整测量窗口。默认完整矩阵另有
`32 × 6 × 5 = 960` 次 auto warmup 请求。固定 `--repeat-in-window N` 时不执行这段预热。

### 每行测量窗口

当前 client 按“冷却 → 无请求对照 → workload”执行；CPU、GPU 与资源监控共享对照窗口：

```text
row_window_s ≈ idle_cooldown_s + matched_control_s + active_workload_s
               + monitor_start_stop_and_csv_overhead_s

默认快速请求场景：5 + 20 + 10 = 35 秒/行
默认完整矩阵：1,344 × 35 / 3,600 ≈ 13.07 小时
```

GPU 开启和关闭都使用同一组 `5 + 20` 秒默认基线开销，不会再为 GPU 单独追加一个对照窗口。
`10` 秒是 auto workload 的目标，结束条件在完整请求返回后判断：若单次请求需要 120 秒，
该行至少约 `5 + 20 + 120 = 145` 秒。它不会在第 10 秒截断请求。
固定请求数时，active workload 约为 `N × 平均请求延迟`；已有成功 CSV 可用
`latency_app_s × repeat_in_window` 估算该行累计请求耗时。

单请求超时默认是 `--request-timeout-seconds 300`；超时会形成错误行，不能当作一次
成功的 300 秒测量。上述 13.07 小时仅适用于完整成功矩阵、请求足够快且默认窗口接近目标的情况，
还未包含 auto warmup、启动、抓包解析、构建、分析器和清理等开销。

### 整条命令的耗时

```text
main_collection_s ≈ sum_over_cases(
  cold_start_s
  + sum_over_scales(auto_warmup_s)
  + sum_over_scales((warmup + repeat) × row_window_s)
  + sniff_parse_and_container_cleanup_s
) + final_merge_s

total_wall_s ≈ preflight_s + model_detection_s + docker_build_and_download_s
               + input_scale_planning_s + runtime_validation_s + compute_profile_s
               + execution_profile_s + main_collection_s
```

auto warmup 约为每个 case/scale 的 `5 × 平均请求延迟`，长耗时模型应单独计入。
镜像复用、失败 case、启动 OOM 剪枝和续跑会改变实际工作量；启用通知和日志收尾也有额外耗时。
内存/PID 峰值、cgroup 增量、冷启动分解和派生能效使用已有采样路径，不增加 `/predict` 数量；
网络字节复用已有 PCAP，但仍需离线解析时间。
