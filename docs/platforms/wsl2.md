# WSL2 开发与部分采集

WSL2 是 AC-Prof 的一等开发平台，也是 `PARTIAL` 采集平台。Native Linux 的环境等级为
`FULL`。环境等级与 `--profiling-mode full/basic` 是两个独立维度：Native 的 basic 结果仍属于
native_linux，但不会因此成为完整 full 画像；WSL2 结果始终属于 wsl2。

第一阶段只开放已有 basic collector 和功能验证路径，不开放全部硬件或独立 profiler。
需要 WSL 发行版内的本机 Docker Engine、Linux x86_64 和统一 cgroup v2。Docker Desktop
及远程 daemon 仍会被拒绝：当前采集器必须看到同一 daemon 的容器 PID、cgroup 与挂载路径。
CUDA 本身支持 Docker Desktop 不表示 AC-Prof 的这些主机观测条件已经成立。

## 能力与缺失值

| 能力 | WSL2 支持状态 | 结果解释 |
| --- | --- | --- |
| 模型加载、功能正确性、preprocess / inference / postprocess | supported | 使用原有 workload / runtime 证据；不保证所有模型或 backend 可运行 |
| request latency、throughput / QPS / concurrency | supported | 现有请求协议中的实际值；仅用于相同环境与实验约束下分析 |
| 镜像大小、模型制品大小 | supported | 实际逻辑字节数，沿用当前定义 |
| 进程内存 | supported | 仅限采集器真正提供的进程数据；不把容器 memory.current 改名为进程 RSS |
| 容器 CPU / memory、CPU utilization、cgroup | partial | WSL guest 内的 cgroup v2，不代表 Windows host 或严格 Native cgroup 行为 |
| GPU inference、GPU memory、GPU utilization、NVML | partial | inference 需实际设备验证；NVML 每项独立读取，不支持的 API 保持缺失 |
| CPU topology、affinity | partial | guest 可见 CPU 与绑定范围，不证明裸机拓扑或独占 CPU |
| cold-start 分段时间 | partial | 当前环境生命周期观测，不是严格 native cold-start benchmark |
| RAPL package / DRAM energy、PMU/perf、cycles / ref-cycles / IPC、宿主机能耗、GPU energy | unsupported | CSV 数值为 nan，JSON 缺失值为 null 或明确 unavailable；禁止估算和替代来源 |
| packet latency、Torch/NCU/Advisor/Massif/Nsight Systems 独立 profiler | requires_native_validation | 第一阶段未开放；使用 Native Linux 验证，不自动降级 |

`supported/partial/unsupported/requires_native_validation` 是平台支持策略。
`available/verified/unavailable/permission_denied/error/not_requested` 等仍表示实际采集证据。
`partial` 指语义受环境限制，不是降低精度；`verified` 也不能把 partial 能力提升成 Native。
GPU utilization 不支持时，GPU memory 可以独立成功；真实数值 `0` 保留，缺失数据不能填 `0`。
普通异常仍然失败，不因 WSL 身份吞掉代码错误。

## 开发、运行与测试

在 WSL 的 Linux 文件系统中维护独立 `.venv`，不要复用 Windows 的 `.venv/Scripts/python.exe`。
使用项目已有 Python 3.10+ 安装与依赖流程，参见[安装说明](../Getting_Started.md)。
第一阶段使用如下明确选择，不自动将用户请求的 full 改为 basic：

```bash
acprof doctor --profiling-mode basic --gpus off
acprof run --model <model-id> --profiling-mode basic \
  --compute-profile-tool none --execution-profile-tool none --dram-energy off
```

GPU 配置另需 Windows NVIDIA driver、WSL CUDA 支持及发行版内 NVIDIA Container Toolkit；
使用 `--gpus on` 后仍需完成真实容器推理验证。doctor 只检查前置条件，不会下载模型或运行采集。
TUI 顶栏显示 `WSL2 / PARTIAL`；确认页列出计划采集、语义受限与缺失指标。

测试统一使用 pytest，按[测试指南](../Testing.md)安装开发依赖；WSL 默认命令为：

```bash
.venv/bin/python -m pytest -m "not native_linux"
```

Native Linux 无需上述 WSL 平台筛选；具体测试范围按[验证范围](../Testing.md#验证范围)选择。
平台与硬件相关 markers 包括 `unit`、`wsl`、`native_linux`、`hardware`。
只对显式 `native_linux` / `wsl` 集成测试按环境 skip；普通单元测试和代码错误不因平台跳过。
模拟 RAPL、cgroup、NVML 的测试仍应在 WSL 运行。`hardware` 本身不会触发笼统 skip；
真实设备测试应逐项检查权限和数据源，并记录原因。

## 结果身份与比较隔离

新实验的 `static_meta.json`、`capability_report.json` 和 `run_state.json` 主机身份保存环境信息；
CSV 每行包含 `environment_class`。示例：

```json
{
  "platform": {"environment": "wsl2", "native": false},
  "collection_tier": "partial",
  "comparability_class": "wsl2",
  "environment_class": "wsl2"
}
```

`platform` 还包含 system、kernel、kernel_version、machine、WSL generation、发行版和检测证据。
`platform_runtime` 保存 Git commit、Docker server/kernel/cgroup/runtime 信息及 CUDA driver、NVML、
GPU driver 版本；单项探测失败保留 null 与错误原因。GPU 设备身份和容器包版本继续使用原有
`gpu_device`、`runtime_environment` 字段。所有这些探测只发生在准备阶段，不进入测量窗口。

缺少环境身份的旧结果读取为 `unknown`，不从当前主机、发行版名称或 Linux kernel 猜测。
`vm/cloud/container_host` 保留独立枚举；当前不自动推断 VM/cloud 类型，也不赋予 FULL。
`native_linux` 表示当前检测未发现 WSL/container 信号的 Linux；不能单凭该标签证明物理隔离。

`audit --compare` 在 same-hardware 和 cross-hardware 下都检查 comparability_class。
Native/WSL、未知身份或冲突身份不能产生性能比较结论；能耗一方缺失显示 `not comparable`。
独立实验统计拒绝为这些数据计算差值、比率与置信区间。报表按环境分组，图表放入
`plots/<environment_class>/...`；混合身份 CSV 明确拒绝，不能默认进入同一 benchmark 排名。
续跑、CSV 合并不接受混合环境，历史 unknown 目录也不能被当前环境重新盖章。

## 已知限制与 Native validation

WSL 虚拟 CPU、Windows 调度、GPU 共享、文件系统和内存回收会影响延迟、吞吐与冷启动。
这些影响不会通过修正系数抵消。CPU 频率 fallback 和估算 cycles 在 WSL 不输出。
同属 wsl2 也不表示两个实验自动可比：仍需通过资源、输入、测量协议与硬件条件审计。

涉及 RAPL、PMU/perf、cgroup、NVML、CPU topology、affinity、cold start、energy 的贡献必须报告
`Native validation: verified`、`required` 或 `not applicable`。WSL 通过和 mock 通过只能说明各自范围。

## 参考实现与复用取舍

- [is-wsl 源码](https://github.com/sindresorhus/is-wsl/blob/main/index.js)（MIT）：借鉴 kernel、proc 和自定义 kernel 信号组合，集中识别；不引入 Node.js 依赖。
- [Microsoft WSL PMU Issue #12836](https://github.com/microsoft/WSL/issues/12836) 与 [RAPL Issue #10160](https://github.com/microsoft/WSL/issues/10160)：作为能力限制的证据，不把 Issue 中某台机器的成功提升为 Native 测量语义。
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)：GPU inference 与 NVML 查询能力分开处理；沿用现有 NVIDIA Python binding，无新增运行时依赖。

平台层只使用标准库；不复制外部实现。环境识别及版本探测在测量窗口外完成，支持策略不会改变
Native 的计数器、归因公式、窗口或请求协议。
