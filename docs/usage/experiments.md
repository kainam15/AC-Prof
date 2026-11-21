# 实验运行指南

本指南集中保存 CPU/GPU/ONNX、资源矩阵、恢复和结果验收示例；主机准备见[安装与主机准备](../getting_started/installation.md)。

## 3. 跑一个最小 smoke test

第一次先跑一个最小 CPU 实验：1 个 CPU、4 GB 内存上限、一个输入规模、一次正式请求。它用来验证流程，不用来做性能排名。新实验采用 Best-effort Collection：按当前 Runtime、硬件和权限逐项采集；关闭深度 Profiler 不会关闭能耗、perf 或抓包。TUI 也可用 `/smoke` 填入最小配置。

```bash
acprof run --model google-bert/bert-base-uncased \
  --cpus 1 --mems 4 --gpus off --input-scales 64 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none --execution-profile-tool none \
  --notify none --output-dir results/first-run
```

首次运行先检查环境；模型或运行环境未准备好时，准备阶段可能较久。新实验使用新的 `--output-dir`；中断后保留原参数并添加 `--resume`，历史实验继续使用其冻结策略，详见[结果完整性与断点续跑](../profiling/artifacts.md#结果完整性与断点续跑)。

不用再跑一遍来检查 RAPL、perf 和抓包。先对同一份结果做只读审计：

```bash
acprof audit results/first-run/google-bert--bert-base-uncased/ --json
```

审计的 `missing_metrics` 解释缺值；`accepted` 不代表每个指标都采到了。打开具体任务结果目录的 `capability_report.json`，在 `field_outcomes` 查看 `cpu_energy_total_j`（RAPL）、`cpu_instructions_per_request`（perf）和 `latency_s`（抓包）的 `windows` 计数与 `reasons`。`complete` 表示该窗口有完整证据；`partial` 即使有数值也不是完整采集；没有记录则不能确认。`latency_app_s` 是应用延迟，不能替代抓包证据。

Stable Diffusion 建议先做单 GPU、单分辨率试跑：

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

首次运行仍需下载模型并构建镜像。产物及字段归属见[结果阅读指南](../results/metrics.md#从结果目录开始)和[产物结构](../profiling/artifacts.md#artifact-layout-v2)。

## 无 Torch 的 ONNX Runtime CPU 示例

下面的新实验同样按可用能力采集。可以先看应用延迟、吞吐、CPU 和内存，再从窗口证据确认 RAPL、perf 和抓包的实际状态。下文的 `--basic-e2e` 是开发容器回归选项，不是 `acprof run` 的采集模式。

```bash
acprof run --model Ritual-Net/iris-classification \
  --task tabular-classification --backend onnxruntime \
  --workload-spec examples/onnxruntime/iris.json --input-scales 1 \
  --cpus 1 --mems 1 --gpus off --batch-size 1 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none --execution-profile-tool none \
  --notify none --output-dir results/onnx-basic
```

此模型输入固定为 `[1,4]`，更大的输入或 batch 会明确拒绝。模型 revision 会解析并写入结果；
任务 sanity check 不等于分类准确率评测。接口与声明格式见[扩展声明](../models/extensions.md#扩展声明与按需加载)。

离线容器回归（含真实服务、CPU/内存采集、落盘和审计）：

```bash
uv run python scripts/check_runtime.py --profile onnxruntime-cpu --basic-e2e \
  --output-dir internal-testing/shared/onnx-runtime
```

图像分类和多输入文本分类复用同一 ONNX 环境与现有任务生成器；
支持边界及预训练小模型的复现步骤见[无 Torch 验收](../development/ci.md#无-torch-运行时验收)。

## 查看结果

上面的最小 CPU 示例将任务集合写入 `results/first-run/google-bert--bert-base-uncased/`（单任务结果位于其下的 `tasks/<task_id>/<run_id>/`）；TUI 运行时请使用界面显示的实际输出路径。准备阶段就创建 `result_layers.json` 和全部模块表头，每个完成窗口同步追加数据，完成实验时冻结快照；详细文件树和字段归属见[产物结构](../profiling/artifacts.md#artifact-layout-v2)与[结果阅读指南](../results/metrics.md#从结果目录开始)。

可先只读校验结果，再生成适用的图表和离线报告（单任务集合可直接传入 `<model--slug>` 目录或上层 `--output-dir`，多任务集合在 `report` / `compare` 中会自动展开全部任务，单任务命令需指定具体 `tasks/<task_id>/<run_id>/`）：

```bash
acprof audit results/first-run/google-bert--bert-base-uncased/ --require-complete --require-ok
acprof results verify results/first-run/google-bert--bert-base-uncased/
acprof plot results/first-run/google-bert--bert-base-uncased/result_layers.json
acprof report results/first-run/google-bert--bert-base-uncased/
```

用浏览器打开生成的 `report.html`，即可离线查看 Comparison Matrix、Pareto 和 Scaling 视图；报告不会默认覆盖旧文件，选项与分析口径见[交互式配置比较](../results/reports.md#交互式配置比较报告)。正式分析只使用 `status=ok` 且 `warmup=0` 的行；没有采集的指标可为 `nan`。完整宽表不会自动生成，需要时显式运行 `acprof results export <结果目录> <新文件.csv>`。

恢复实验、模型预检、矩阵参数及其它 CLI 选项参见[正式实验](#运行正式实验)和[CLI 参考](cli.md)；TUI 操作、下载确认及镜像管理见[TUI 用户指南](tui.md)。

## 运行正式实验

建议先逐步扩大规模：最小 smoke test → 单个资源配置的全部 input scale → 不带 profiler 的目标资源矩阵 → 最后补采高开销 profiler。

中断后使用原命令加 `--resume`，或在 TUI 的“恢复 / 重试”检查页选择“继续实验”（准备阶段为“重试准备”）。
从实验选择器、失败监控页或对原输出目录点击“开始采集”均可进入检查页。系统会核对原参数、
镜像和输入计划，保留完成的 case；新格式从可信 receipt 恢复中断 case，仅采集剩余窗口并记录新的 attempt。新实验应选择新的输出目录；
已有产物不会被默认覆盖。符合当前产物协议的实验没有恢复状态文件时仍可绘图、补采，主实验恢复约定见
[结果完整性与断点续跑](../profiling/artifacts.md#结果完整性与断点续跑)。

### 先探测最大输入

完整矩阵开始前，可先扫描候选内存上限，确认最大输入能否完成一次请求：

```bash
acprof probe --model google-bert/bert-base-uncased \
  --cpus 1,2,4 --mems 2,4,8 --gpus off,on
```

这个例子固定使用最小 CPU `1`，优先选择 `GPU=off`，按 `2GB → 4GB → 8GB` 实测。
只要候选中包含 `off`，镜像准备、输入规划、运行验证和最终请求都按 CPU 执行。要探测 GPU，请单独传入 `--gpus on`。
输入规划和运行验证可使用最大的 CPU 和内存配额。最终请求使用最小 CPU 和当前候选内存。
输入尺度留空时取自动规划结果的最大档；手动传入 `--input-scales` 时取其中最大值。
每档使用全新容器，最多发送一次 `/predict`，第一个成功值是这些候选中的最低可用内存。
明确的启动或运行期主机内存 OOM 会推进到下一档；CUDA OOM、超时、尺度不一致和其他
错误会停止，尚未验证的更大内存不会被报告为可行。

探测请求默认不设超时，需要限制时传入 `--timeout-seconds <正数>`。
这与正式矩阵默认 `--request-timeout-seconds 300` 相互独立。结果同时报告成功档的
容器冷启动、单次请求及两者合计耗时，写入独立的 `probes/` 目录；该流程不采集能耗、
PMU 或网络指标，也不写正式 CSV。字段见[探测输出](../profiling/metadata.md#最大输入探测结果)。
TUI 的“探测最大输入”调用同一入口。

最大输入探测与同一主机、同一用户的正式采集和补采共用测量锁。从 Docker 预检到写入探测摘要期间，其他采集不能开始。已有任务持锁时，`acprof probe` 会在创建探测目录之前报告原因并退出，待原任务结束后可重试。[主机测量锁](../profiling/artifacts.md)

对同一个 T5 模型，可以先检查语义任务，再用相同的 `--tasks` 探测指定方向的翻译。

```bash
acprof inspect google-t5/t5-small --revision df1b051c49625cf57a3d0d8d3863ed4d13564fe4 --tasks translation_en_to_fr
acprof probe --model google-t5/t5-small --revision df1b051c49625cf57a3d0d8d3863ed4d13564fe4 \
  --tasks translation_en_to_fr --gpus off --input-scales 32
```

探测使用英语到法语的提示前缀，并复用正式采集的任务分组和生成参数规则。输入计划的 `task_workload` 保存所选语义任务和指纹。[固定版本的 T5 配置](https://huggingface.co/google-t5/t5-small/blob/df1b051c49625cf57a3d0d8d3863ed4d13564fe4/config.json)

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

两个例子均先按 Best-effort 策略完成主矩阵，之后可用 `acprof profile` 补采计算指标。计算分析器默认关闭；若希望随矩阵安排 Torch / NCU，请显式传入 `--compute-profile-tool both`。`--execution-profile-tool` 默认也是 `none`。

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
行数、请求数与端到端耗时的区别见[时间成本估算](../profiling/sizing.md#结果行数和时间成本估算)。
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

# 默认核验并复用已有镜像，缺失时自动构建
acprof run --model google-bert/bert-base-uncased

# 单个推理请求最多等待 30 分钟
acprof run --model stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --request-timeout-seconds 1800

# 查看全部参数
acprof run --help
```

### 镜像复用、超时与失败处理

- 模型文件筛选、依赖/模型/代码分层及缓存含义见[运行兼容说明](../models/adaptation.md#模型文件选择规则)。
- 默认核验匹配后复用，不存在则构建；`--rebuild-image` 显式重建，仍可使用 Docker 缓存；镜像与独立推理验证的失败边界见[构建、复用和验证](../models/images.md#构建复用和验证)。
- 启动 OOM、剪枝推断及部分结果处理见[排障说明](troubleshooting.md#启动-oom-与剪枝占位)。
- 长请求可设置 `--request-timeout-seconds 1800`；它限制单次请求，不限制整个矩阵。默认值与适用阶段见[CLI 参数](cli-run.md#请求窗口与采样)。

通知配置见[企业微信通知](configuration.md#企业微信通知)，补采流程见[补采已有结果](../profiling/profilers.md#补采已有结果)。
