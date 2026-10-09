# 实验运行指南

本指南集中保存 CPU/GPU/ONNX、资源矩阵、恢复和结果验收示例；主机准备见[安装与主机准备](../getting_started/installation.md)。

## 3. 跑一个最小 smoke test

首次尝试推荐 **basic CPU 单次实验**。只使用 1 个 CPU、4 GB 容器内存、一个输入规模和一次正式请求，主要用于验证采集流程，单次测量不足以得出性能结论。也可以直接在 TUI 中选择 Smoke 预设；以下是独立的 CLI 入门示例：

```bash
acprof run --model google-bert/bert-base-uncased \
  --profiling-mode basic \
  --cpus 1 --mems 4 --gpus off --input-scales 64 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none --execution-profile-tool none \
  --notify none --output-dir results/first-run
```

首次运行会先检查环境，再下载模型和依赖、构建镜像，准备阶段可能较久。此模式只采集基础指标，能耗、抓包和独立 profiler 的字段为 `nan` 属于预期结果。新实验需选择新的 `--output-dir`；中断后保留原参数并添加 `--resume`，详见[结果完整性与断点续跑](../profiling/protocol.md#结果完整性与断点续跑)。

需要验证完整采集链路时，再使用以下默认 **full CPU 示例**（同样只请求一次，不能作为性能结论）；它额外要求 RAPL、perf、抓包工具及主机权限：

```bash
acprof run --model google-bert/bert-base-uncased \
  --cpus 1 --mems 4 --gpus off \
  --input-scales 64 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none \
  --execution-profile-tool none \
  --output-dir results/smoke
```

Stable Diffusion 建议先做单 GPU、单分辨率 smoke test（同样使用 `full` 模式）：

```bash
acprof run --model stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --cpus 4 --mems 16 --gpus on --gpu-device 0 --input-scales 256 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none --execution-profile-tool none \
  --output-dir results/sd-smoke
```

`--gpu-device` 也可填写 `nvidia-smi -L` 显示的完整 GPU UUID。主容器、主机 NVML 监测和
独立 profiler 固定到同一物理设备；选择方式遵循
[NVIDIA Container Toolkit 的 UUID 参数](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)。

首次运行仍需下载模型并构建镜像。产物及字段归属见[结果阅读指南](../results/metrics.md#从结果目录开始)和[产物结构](../profiling/protocol.md#artifact-layout-v2)。

## 无 Torch 的 ONNX Runtime CPU 示例

`basic` 采集 application latency、吞吐、容器 CPU 和内存；支持 Native Linux 或 WSL2、本机 Docker、
cgroup v2。RAPL、perf、抓包不参与此模式，结果明确记录模式和能力状态。默认 `full` 保留原有严格条件。

```bash
acprof run --model Ritual-Net/iris-classification \
  --task tabular-classification --backend onnxruntime \
  --workload-spec examples/onnxruntime/iris.json --input-scales 1 \
  --cpus 1 --mems 1 --gpus off --batch-size 1 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --profiling-mode basic --compute-profile-tool none --execution-profile-tool none \
  --notify none --output-dir results/onnx-basic
```

此模型输入固定为 `[1,4]`，更大的输入或 batch 会明确拒绝。模型 revision 会解析并写入结果；
任务 sanity check 不等于分类准确率评测。接口与声明格式见[扩展声明](../models/runtime.md#扩展声明与按需加载)。

离线容器回归（含真实服务、CPU/内存采集、落盘和审计）：

```bash
.venv/bin/python scripts/check_runtime.py --profile onnxruntime-cpu --basic-e2e \
  --output-dir internal-testing/onnx-runtime
```

图像分类和多输入文本分类复用同一 ONNX 环境与现有任务生成器；
支持边界及预训练小模型的复现步骤见[无 Torch 验收](../development/testing.md#无-torch-运行时验收)。

## 查看结果

上面的 basic 示例将主要产物写入 `results/first-run/google-bert--bert-base-uncased/`；TUI 运行时请使用界面显示的实际输出路径。实验完成后生成 `result_layers.json` 及独立模块 CSV；详细文件树和字段归属见[产物结构](../profiling/protocol.md#artifact-layout-v2)与[结果阅读指南](../results/metrics.md#从结果目录开始)。

可先只读校验结果，再生成适用的图表和离线报告：

```bash
acprof audit results/first-run/google-bert--bert-base-uncased/ --require-complete --require-ok
acprof results verify results/first-run/google-bert--bert-base-uncased/
acprof plot results/first-run/google-bert--bert-base-uncased/result_layers.json
acprof report results/first-run/google-bert--bert-base-uncased/
```

用浏览器打开生成的 `report.html`，即可离线查看 Comparison Matrix、Pareto 和 Scaling 视图；报告不会默认覆盖旧文件，选项与分析口径见[交互式配置比较](../results/metrics.md#交互式配置比较报告)。正式分析只使用 `status=ok` 且 `warmup=0` 的行；没有采集的指标可为 `nan`。完整宽表不会自动生成，需要时显式运行 `acprof results export <结果目录> <新文件.csv>`。

恢复实验、模型预检、矩阵参数及其它 CLI 选项参见[正式实验](#运行正式实验)和[CLI 参考](cli.md)；TUI 操作、下载确认及镜像管理见[TUI 用户指南](tui.md)。

## 运行正式实验

建议先逐步扩大规模：最小 smoke test → 单个资源配置的全部 input scale → 不带 profiler 的目标资源矩阵 → 最后补采高开销 profiler。

中断后使用原命令加 `--resume`，或在 TUI 高级参数勾选“恢复未完成实验”。系统会核对原参数、
镜像和输入计划，保留完成的 case，并备份后重新测量中断的 case。新实验应选择新的输出目录；
已有产物不会被默认覆盖。符合当前产物协议的实验没有恢复状态文件时仍可绘图、补采，主实验恢复约定见
[结果完整性与断点续跑](../profiling/protocol.md#结果完整性与断点续跑)。

### 先探测最大输入

完整矩阵开始前，可先扫描候选内存上限，确认最大输入能否完成一次请求：

```bash
acprof probe --model google-bert/bert-base-uncased \
  --cpus 1,2,4 --mems 2,4,8 --gpus off,on --skip-build
```

这个例子固定使用最小 CPU `1`，优先选择 `GPU=off`，按 `2GB → 4GB → 8GB` 实测。
输入尺度留空时取自动规划结果的最大档；手动传入 `--input-scales` 时取其中最大值。
每档使用全新容器，最多发送一次 `/predict`，第一个成功值是这些候选中的最低可用内存。
明确的启动或运行期主机内存 OOM 会推进到下一档；CUDA OOM、超时、尺度不一致和其他
错误会停止，尚未验证的更大内存不会被报告为可行。

探测请求默认不设超时，需要限制时传入 `--timeout-seconds <正数>`。
这与正式矩阵默认 `--request-timeout-seconds 300` 相互独立。结果同时报告成功档的
容器冷启动、单次请求及两者合计耗时，写入独立的 `probes/` 目录；该流程不采集能耗、
PMU 或网络指标，也不写正式 CSV。字段见[探测输出](../profiling/protocol.md#最大输入探测结果)。
TUI 的“探测最大输入”调用同一入口。

### CPU-only 矩阵

```bash
acprof run --model google-bert/bert-base-uncased \
  --cpus 1,2,4 --mems 4,8 --gpus off \
  --compute-profile-tool none \
  --output-dir results/bert-cpu
```

### CPU / GPU 对比矩阵

```bash
acprof run --model google-bert/bert-base-uncased \
  --cpus 1,2,4 --mems 4,8 --gpus off,on \
  --compute-profile-tool none \
  --output-dir results/bert-cpu-gpu
```

上面两个例子使用默认 `full` 模式，先完成主矩阵，之后可用 `acprof profile` 补采计算指标。计算分析器现在默认关闭；若希望在矩阵开始前直接采集 Torch / NCU，请显式传入 `--compute-profile-tool both`。`--execution-profile-tool` 默认也是 `none`。

### 默认完整矩阵

```bash
acprof run --model google-bert/bert-base-uncased
```

默认配置如下：

| 维度 | 默认值 |
| --- | --- |
| CPU | `1,2,4,8` |
| 内存 | `2,4,8,16` GB |
| GPU mode | `off,on` |
| input scale | 自动规划，通常 6 档 |
| warmup / repeat | `2 / 5` |
| 每行 workload | 自动持续到累计 application latency 约 10 秒 |
| 单个 `/predict` 请求超时 | `300` 秒 |
| Idle 基线 / 前置冷却 | `20 / 5` 秒 |
| compute profiler | `none`（关闭；需要时显式启用或后续补采） |
| execution profiler | `none` |
| 企业微信通知 | 配置 Webhook 后自动启用；`--notify none` 可关闭 |

若实际规划出 6 档输入，完整矩阵包含 384 行 warmup 和 960 行正式测量。
行数、请求数与端到端耗时的区别见[时间成本估算](../profiling/protocol.md#结果行数和时间成本估算)。
默认每行约 10 秒 workload、20 秒 Idle 基线和 5 秒前置冷却，仅主测量窗口就约 13 小时；
模型下载、镜像构建、case 切换与显式启用的 profiler 还需额外时间。
大矩阵开始前也应检查 profiler artifacts 的磁盘占用。

### 常用变体

```bash
# 手动指定输入规模
acprof run --model google-bert/bert-base-uncased \
  --input-scales 64,128,256,512

# 时间序列模型
acprof run --model amazon/chronos-bolt-base \
  --task-family timeseries --backend chronos

# 复用已有镜像
acprof run --model google-bert/bert-base-uncased --skip-build

# 单个推理请求最多等待 30 分钟
acprof run --model stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --request-timeout-seconds 1800

# 查看全部参数
acprof run --help
```

### 镜像复用、超时与失败处理

- 模型文件筛选、依赖/模型/代码分层及缓存含义见[运行兼容说明](../models/runtime.md#模型文件选择规则)。
- `--skip-build` 核验匹配后复用，不存在则构建；镜像与独立推理验证的失败边界见[构建、复用和验证](../models/runtime.md#构建复用和验证)。
- 启动 OOM、剪枝推断及部分结果处理见[排障说明](troubleshooting.md#启动-oom-与剪枝占位)。
- 长请求可设置 `--request-timeout-seconds 1800`；它限制单次请求，不限制整个矩阵。默认值与适用阶段见[CLI 参数](cli.md#请求窗口与采样)。

通知配置见[企业微信通知](cli.md#企业微信通知)，补采流程见[补采已有结果](../profiling/profilers.md#补采已有结果)。
