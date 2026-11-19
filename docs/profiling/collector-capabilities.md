# 能力驱动采集：P2 调研与接入边界

[← 统一画像契约](../results/runtime-profile.md) · [当前覆盖矩阵](../results/runtime-coverage.md)

本文为接入设计，不代表 Process 已接入高级采集器，也不代表硬件验收。本轮实现 P1、
P3 的代码与离线验证，保持现有 Basic / Full 的配置、恢复与结果读取兼容。
后续目标是取消两种运行模式，改为按采集器能力和已验证成本独立启用；不是增加 Process Full。
单项无硬件、无权限或运行失败时，保留其他有效采集和逐项警告，不整体降级。

## 已核实的复用点

| 能力 | 当前实现与 Process 接入方式 | 范围、前提与禁止的替代 |
| --- | --- | --- |
| GPU 能耗 | `monitors/energy_nvml.py::GPUEnergyMonitor` 不要求 Docker；复用 UUID 绑定、累计能量优先/功率积分回退、matched-control 与 raw 重算 | 整卡 J，不是进程能量。累计计数 mJ 转 J；不支持、GPU 丢失、重载/回退分别记录。两窗口须使用一致来源，不能中途混合 |
| CPU / DRAM 能耗 | 复用 `energy_cpu.py` 的 RAPL topology、去重、wrap、domain 完整性与控制窗口；将容器 CPU reader 的发现从 RAPL 读取中解耦 | package/DRAM 总量可独立存在；没有服务 CPU 计数只缺归因项。DRAM 不是所有 CPU 都有；无权限不能写零或自动 chmod |
| 进程能耗归因 | 后续向归因计算注入同窗 `process CPU / host CPU`；已验证独立 cgroup 可保留另一种归因来源 | 仅为 estimated CPU 时间份额，不是实测进程能耗；进程成员缺口、调度迁移、共享设备与负残差必须保留。不得写入旧 cgroup 归因字段 |
| PMU | `perf_mips.py` 当前已经用 `perf stat -p <container-root-pid>`，并非 cgroup PMU；将 PID 发现改为 Runtime 提供身份，复用输出解析、事件支持、running ratio 和 `--control=fd` ACK 生命周期 | 首先按 PID/TID 验证权限、已有线程枚举、后续继承和退出子进程。attach 根 PID 不能直接声明整个进程树完整；instructions/cycles 的窗口及事件组须相同才计算 IPC |
| HTTP 字节 | 共享请求执行器记录实际发送的 PreparedRequest body 和实际响应正文；流式路径复用既有 chunk 字节累计 | 区分 UTF-8 正文、HTTP 头、压缩前后；不能用再次 JSON 序列化的大小冒充实际发送量。复用已完成请求，原值进入请求 journal；`response_payload_bytes` 按解码后字节计数 |
| PCAP | 复用 `measurement/capture.py`、request ID journal、packet 解析和回填；Process 用已绑定 loopback 地址/端口过滤，清理绑定本次抓包进程 | 需 tcpdump/tshark 与当前接口权限。保留 link type、丢包、重传、截断和请求覆盖；`any`/loopback 可能重复观察。PCAP frame bytes 不宣称物理链路全部开销，不用 HTTP 字节回填缺失 PCAP |

RAPL/NVML 设备计量与进程归因分开提供能力状态。同语义的 NVML 累计能量与功率积分
可以回退，但要保留方法和质量；Process RSS 与 cgroup memory、PID PMU 与 cgroup PMU
不构成直接回退。没有可靠进程归属的 GPU 利用率、频率、温度和能耗持续标记整卡范围。

## 独立 Profiler

| 工具 | 可行性与复用 | Process 边界 |
| --- | --- | --- |
| Torch Profiler | 复用 `host/profiling/tools/torch.py`、容器内现有 probe 入口、compute plan 解析 | 在冻结 Process Python/代码/模型快照中启动独立分析任务；仅支持实际 backend。schedule/step 与 forward 数写分析证据，不替换正式请求延迟；shape/stack/memory 追踪成本单独评估 |
| NCU | 复用 `tools/ncu.py` 的工具版本、命令构造和解析；以主机 CLI 包装独立 Process probe | GPU counter 权限、驱动和 replay 模式须独立探测。可能重放 kernel 或应用，属于高开销分析；不能为了补指标重跑正式推理窗口 |
| Nsys | 复用 `tools/nsys.py`、execution plan 与 timeline 解析 | 独立进程用 capture range / NVTX / CUDA profiler API 限定分析区；CUDA tracing 与 CPU sampling 分别判断权限，不能一个子能力失败抹去全部 timeline |
| Massif | 复用 `tools/massif.py`、执行计划与 ms_print 解析，Valgrind 启动冻结 Python 的 CPU probe | heap、stack、pages-as-heap 是不同来源；不能当 RSS/cgroup memory。Python/native backend 兼容、fork/exec 子进程和清理需要验证；不对正式服务进行高开销注入 |

统一接口应复用 `ExecutionRuntime.restore/start/stop` 的制品身份与所有权清理，以及既有
profiler plan/receipt/post-hoc 来源。只抽出各工具需要的 launch target 和 artifact 路径，
不复制第二套解析器或引入常驻服务。正式窗口完成并释放 measurement lock 后，分析任务
按现有锁协议隔离；在没有明确分析授权时记录 `not_requested`，不得为自动采集增加模型推理次数。

## 后续能力选择与迁移

1. 准备阶段生成逐采集器 capability：Runtime target、权限、硬件/工具版本、支持的字段、
   source/scope/window/derivation、设计成本与成本验收引用。探测不启动模型或完整 Profiler。
2. 用现有 `capabilities` 状态及 MonitorGroup 生命周期生成执行计划。只在有能力且成本已获
   验证的边界内自动启用；依赖失败只阻断依赖它的派生项。必需协议项仍 fail closed。
3. 采集期间查询失败保留已知原始证据和 partial/error，不动态切换为不同范围。相同语义
   回退要覆盖完整同窗/控制窗证据，不能拼接不兼容样本。输出每项计划状态及实际状态。
4. 新执行计划单独版本化。历史 Basic / Full 仅作为原实验的冻结选择读取，恢复时不自动
   多启用采集器；新 CLI/TUI 的模式入口在独立变更中迁移并删除，不在本轮偷换配置含义。
5. 比较继续检查设备、模型、输入、窗口、采样频率、来源和质量；Runtime 名称不同不意味着
   所有值不可并列，但相同指标名也不是受控差值或科研可比性的证明。

## 开发与硬件验收门槛

开发先覆盖无 Docker、CPU-only、权限不足、单查询 unsupported、PID 重用/成员退出、
counter reset、raw 截断、取消、超时和 cleanup 失败；这些可用 Fixture/Mock 完成。
随后由用户明确安排同硬件、同模型、同输入的配对实验：既有 Docker Full 回归、Process
各能力单独与组合、NVML 方法切换、RAPL/DRAM domain 完整性、perf 线程覆盖/running ratio、
loopback 抓包匹配、Profiler 窗口隔离。记录分布与不确定性，不预设零开销或固定可接受百分比。

本轮 `measure_overhead.py` 已支持恢复 Docker 或 Process 制品；Process 对照仅包含现有
进程/独立 cgroup/NVML 资源观察，尚不包含上述 P2 采集器。`successful` 表示诊断请求和
收尾完成；`collector_coverage_complete` 独立反映逐项覆盖，不能作为硬件精度认证。

## 参考来源与采用决定

- [NVIDIA NVML header](https://github.com/NVIDIA/go-nvml/blob/main/pkg/nvml/nvml.h)：确认累计 GPU 能量的 mJ、设备支持和错误类型；复用现有 Python monitor，不引入 Go 服务。
- [Linux powercap](https://github.com/torvalds/linux/blob/master/Documentation/power/powercap/powercap.rst)：确认 domain 层级与能量范围；保留现有 RAPL 去重和 wrap 处理。
- [Linux perf stat](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-stat.txt)：确认 PID/TID、inherit 与 enable/disable ACK；不将 attach 范围夸大为整个 cgroup。
- [libpcap 手册源码](https://github.com/the-tcpdump-group/libpcap/blob/master/pcap.3pcap.in)：参考接口、权限和链路类型边界；继续使用现有抓包与关联解析器。
- [PyTorch Profiler 源码](https://github.com/pytorch/pytorch/blob/main/torch/profiler/profiler.py)：参考 schedule/step 和高成本选项，保持独立分析窗口。
- [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)、[Nsys User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Massif manual](https://valgrind.org/docs/manual/ms-manual.html)：确认工具启动方式、分析范围与开销边界。以上结论是接入可行性判断，不是本机实测结果。
