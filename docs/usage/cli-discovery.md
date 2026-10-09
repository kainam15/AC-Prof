# CLI 检查与模型发现

[← 返回专题目录](cli.md)

## CLI 参数

源码开发与安装环境的唯一公开入口为 `acprof <command>`；公共子命令有 `run`、`tui`、`probe`、`plot`、
`doctor`、`profile`、`audit`、`stats`、`inspect`、`auto`、`coverage`、`compare`、`load`、`model-store`、`report`。
`acprof --version` 查看版本，`acprof <command> --help` 查看对应帮助。

### `acprof doctor`

`doctor` 只检查环境，不下载模型、启动容器、安装工具或修改权限。
检查覆盖 Native Linux / WSL2 x86_64、cgroup v2、本机 Docker、Buildx、安装资源与输出目录。
输出 Environment、Collection tier、Native benchmark 和按指标的支持策略；WSL2 使用 `--profiling-mode basic`。
`full` 另外检查 RAPL、perf instructions、抓包工具和网卡；`--gpus on` 检查 NVIDIA driver
和 Docker NVIDIA runtime。模型下载、容器 GPU、任务输出与正式测量仍由实际运行验证。

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--profiling-mode basic/full` | `full` | 与主流程保持相同的必需能力范围 |
| `--gpus off/on` | `off` | 是否检查 NVIDIA 主机和 runtime |
| `--sniff-iface` | `docker0` | full 模式抓包网卡 |
| `--output-dir` | 当前工作目录 | 检查输出路径可写性与所在磁盘余量 |
| `--json` | 关闭 | 输出包含 `ready`、`checks`、`scope` 的 JSON |

退出码为 `0`（必要检查通过）、`1`（必要条件缺失）、`2`（参数错误）。
检查状态是 `available`、`unavailable`、`not_requested` 或 `warning`；
`available` 不等于真实 workload 的 `verified`。低磁盘余量为提示，不自动删除镜像。
`doctor` 不检测所有模型的联网和容量需求，`run` 仍在正式采集前执行权威 preflight。

以下参数表对应 `acprof run`。示例命令见[运行指南](experiments.md#运行正式实验)，
默认值与实际选项以当前入口的 `--help` 和 [acprof/config.py](../../acprof/config.py) 为准。

[acprof run](cli-run.md#acprof-run) · [acprof probe](cli-tools.md#acprof-probe) · [acprof profile](cli-tools.md#acprof-profile) · [其他入口](cli-tools.md#其他入口) · [输入规模与音频清单](cli-run.md#输入规模与音频清单)

### `acprof inspect`

`acprof inspect MODEL` 只进行静态解析；`--explain` 显示固定 revision、字段来源和未决项。
`--output-dir DIR` 导出 `model_resolution.json`。静态缺口退出 2，保留 draft；模型访问或网络错误退出 1，
分类见[共享接口解析](../models/routing.md#共享接口解析)。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model-spec`、`--task`、`--backend` | 自动解析 | 声明或选择覆盖，仍检查冲突 |
| `--expected-revision` | 空 | 要求模型 SHA 与已审阅 SHA 相同 |
| `--revision` | Hub 默认分支 | 解析 branch、tag 或完整 SHA，文件固定到解析后的 SHA |
| `--probe-interface` | 关闭 | 使用纯源码 bundle，在依赖镜像内检查 import 和 method signature |
| `--cpus`、`--mems` | `2`、`4` | 接口检查的 CPU 核数和 GiB 内存上限 |
| `--timeout-seconds` | `300` | 单次接口检查容器超时，不包括依赖镜像准备 |

```bash
acprof inspect MODEL --explain
acprof inspect MODEL --probe-interface --output-dir results/inspection/model-interface
```

接口检查不调用 `prepare_image()`、`prepare_model()` 或 Model Store，不下载模型权重、不执行推理。
首次可能准备依赖基础镜像。未指定输出目录时使用独立 `results/inspection/` 子目录，已有接口报告的
目录不能复用；写入 `interface_validation.json` 和 `logs/interface_validation.log`，不生成正式 CSV。
源码图缺失直接失败，禁止整仓下载回退。容器使用断网、只读文件系统和只读源码挂载。
旧 `--probe basic/full`、`--gpus`、`--skip-build` 不再用于 inspect；完整运行验证由采集流程自动执行，
批量独立验证可用 `coverage run --validate-runtime`。接口与运行证据边界见[运行兼容](../models/contracts.md#自动生成模型契约m1m6)。

### `acprof auto`

`acprof auto MODEL` 接受精确 Hub ID 或 Hub 本身支持的别名，检查主模型及声明依赖的访问权限，
导出静态裁决，检查主机，然后复用 `run` 的镜像准备、输入规划、独立 runtime validation、
正式矩阵和报告。模糊名称不按下载量替换，访问失败不改选其他模型；语义冲突保留解释和 draft 后退出。

除模型改为位置参数外，其余资源、输入、窗口和 profiler 参数与 `run` 相同，默认矩阵也相同。
首次验证建议显式限制矩阵：

```bash
acprof auto google-bert/bert-base-uncased --profiling-mode basic \
  --cpus 1 --mems 4 --gpus off --input-scales 32 --warmup 0 --repeat 1 \
  --repeat-in-window 1 --output-dir results/auto-first
```

默认仍为 `--profiling-mode full`。只有显式选择 `--profiling-mode auto`，才允许因
RAPL/perf/packet 不可用而选择 basic；Docker、Linux/cgroup、安装资源或所选 GPU 不可用时仍停止。
`--dram-energy required` 等显式要求仍由正式入口检查，不能通过 auto 绕过。
`profiling-mode basic/full` 只决定性能指标范围；两种模式都必须通过完整运行验证。

`auto_report.json` 位于 `OUTPUT/MODEL--NAME/`，保存请求模式、实际模式、来源身份、预检和最终状态。
只有主采集与所需指标均成功才退出 0；冲突、缺条件或部分失败退出 2。已有实验不会覆盖；
`--resume` 需要原资源/输入/模式选项，并复用记录的模型 SHA、镜像和计划，不重新解析浮动分支。
恢复时显式指定 `--revision` 必须与保存的完整 SHA 一致。换模式、依赖或输入计划需使用新目录。
当前不做环境修复、pad token 替换、缩小尺度或预算重规划。

### `acprof coverage`

固定回归集与滚动样本使用同一个 schema v1 manifest。每个条目需有 `model_id`、完整
`revision` 和非负 `weight`，顶层需说明 `sampling`、`weight_basis`。样本、权重和 SHA 一经
冻结不在运行中刷新；重新 snapshot 才产生新一轮滚动样本。

```bash
acprof coverage run examples/coverage/regression.json \
  --output-dir internal-testing/coverage-fixed
acprof coverage snapshot --stratum fill-mask:transformers \
  --stratum image-classification:transformers --limit 5 \
  --output internal-testing/coverage-sample.json
acprof coverage run internal-testing/coverage-sample.json \
  --output-dir internal-testing/coverage-static
# 显式运行容器验证；可能构建镜像和下载权重
acprof coverage run internal-testing/coverage-sample.json --validate-runtime \
  --cpus 2 --mems 4 --gpus off --timeout-seconds 300 \
  --output-dir internal-testing/coverage-runtime
# 在下载前应用 conservative 预算；大小来自选中的制品而非整个仓库
acprof coverage run internal-testing/coverage-sample.json --validate-runtime \
  --max-parameters 1000000000 --max-download-bytes 4294967296 \
  --output-dir internal-testing/coverage-budgeted
# 中断后使用原参数继续；已经完成的模型不重复执行
acprof coverage run internal-testing/coverage-sample.json --validate-runtime \
  --cpus 2 --mems 4 --gpus off --timeout-seconds 300 --resume \
  --output-dir internal-testing/coverage-runtime
# 只重试超时失败项；新 attempt 保留提高预算前的证据
acprof coverage run internal-testing/coverage-sample.json --validate-runtime \
  --timeout-seconds 600 --resume --retry-reason request_timeout \
  --output-dir internal-testing/coverage-runtime
# 只读取已有结果，输出统一的 CSV、JSON 和 Markdown 报告
acprof coverage report results/model-a results/model-b \
  --output-dir internal-testing/coverage-recorded
```

snapshot 按 `TASK:LIBRARY` 各取下载量前 N 个，属于所选样本统计，不代表全 Hub 或随机长尾。
run 默认只做静态检查；`--validate-runtime` 的验证时间限制不包含构建和下载。运行时验证会取得同机同用户的
测量锁，与正式采集和 post-hoc profiling 串行，避免额外容器/推理污染性能与能耗窗口；静态 coverage 不取得该锁。
`coverage.json` 的分母始终是冻结样本总权重，分别报告解析、适配、运行、拒绝、权限与资源限制；静态检查不检查权重读取权限，
运行成功率、权限拒绝率与资源限制率均为 null。
独立审阅的 `semantic_reference: {"task": "...", "source": "..."}` 才用于语义正确率；
snapshot 不把 Hub 标签自动当成正确答案。零总权重和没有审阅样本的比率为 null。
报告生成成功退出 0 不表示所有模型成功；逐模型失败保留在 rows 中，不生成正式性能 CSV。

样本条目的 `source` 可为 `huggingface` 或 `modelscope`；缺失时沿用历史 Hugging Face 语义，
不会被 `ACPROF_MODEL_SOURCE` 覆盖。模型身份为 `source + model_id + revision`，不同来源的同名
仓库不合并。解析结果必须匹配样本来源与固定 revision；访问预检显式传入这两个条件，
模型规范中声明的 Hugging Face 依赖仍使用自己的来源与 revision。新 snapshot 明确记录来源。
恢复时逐行核对来源；历史缺来源的 HF attempt 保持原字节，只在新概览中展开默认值。
导出 CSV/Markdown 同时展示来源和版本，CSV 保留已有列顺序并追加来源列；来源证据损坏或冲突时记录 `unknown` 与
`recorded_evidence_invalid`，不猜测或重写原始实验。

run/report 同时输出 `coverage.json`、`models.csv` 和 `REPORT.md`。失败列保留稳定的
`reason_code` 及 evidence，质量警告单独保留在 `quality_checks`；TUI 的 `/report <coverage.json>`
可读取两种报告。report 不执行模型，不修改源结果，也不从旧日志猜测缺少的原因。
Runtime Validation 成功只表示独立推理验证通过；已有采集是否完成由记录的 `full_profile_complete` 决定。

`--max-parameters` 和 `--max-download-bytes` 默认不设置，传入时须为正整数。
超预算或无法确定所需大小时保存 `resource_limit` 与 `unverified`，不下载权重、不宣称实测 OOM。
selected artifact size 包括显式模型依赖；参数量不是峰值 RAM/VRAM 预测。
`--timeout-seconds` 默认保持 300；例如复核 60 秒耗尽的条目，可用同一固定 manifest，显式传入
`--timeout-seconds 600 --resume --retry-reason request_timeout`。每个选中模型在本次调用中只执行
一次 probe，不自动无限重试。
timeout 展示为 `inconclusive`，与 `inference_failed` 分开；详细定义见
[质量与失败产物](../profiling/metadata.md#质量与失败产物)。

新的 `coverage.json` 使用 schema v2；manifest 仍为 schema v1。`--resume` 核对原始
`sample.json` 与传入 manifest 的完整摘要、各模型 source/revision、probe 模式、设备（GPU 使用物理 UUID）、
资源、预算、有效下载策略与 Hub endpoints、运行环境及主机/源码身份。
有效下载条件同时记录到 attempt 配置。普通续跑只能沿用原参数，继续没有完成记录的模型，
不会自动重试已记录的失败；中断的重试则沿用该次 attempt 的条件和剩余选择。
历史 schema v1 缺少完整预算与 attempt 证据，保持可读，恢复时明确要求新目录，不猜测旧条件。

`--retry-failed` 重试所有未验证成功的已记录模型；`--retry-stage STAGE` 和
`--retry-reason REASON` 可分别重复指定，同类选择取并集、阶段与原因之间取交集。
这些选项都要求 `--resume`，没有匹配项时明确报错；已经验证成功的模型始终跳过。
显式重试允许调整资源、设备、时间和下载预算等配置，但不允许替换冻结样本或切换静态/full 模式。
配置变化逐项保存在 `configuration_changes`，不同条件的结果同时存在时标记
`mixed_configurations=true`；汇总只描述各模型最后一次观测，不代表同一配置下的统一验证。

每次执行写入独立的 `attempts/attempt-NNNNNN/attempt.json` 和逐模型目录。
完成记录先原子落盘，再更新可重建的 `coverage.json`、CSV 和 Markdown；进程被强杀后仍可从
attempt 恢复已完成模型，进行中的模型使用新目录重新验证。旧 attempt 的预算、失败和部分产物不覆盖，
顶层 `resources`/`budgets` 保留第一次条件，每行 `attempt_id` 指向实际验证配置。

清理状态为 `incomplete` 时，先保存当前 model/attempt、原始运行失败和清理证据，再中止批次；
后续必须显式重试并包含所有清理未完成的模型（可用 `--retry-stage cleanup` 选择），
优先执行既有 owner 标签和不可变容器 ID 的清理核验，再继续其他模型。
清理问题独立于最新解析或推理结论保留。重试先核对原主机、原 owner UID 与原 Docker endpoint/context，再取得测量锁，
确认原容器已不存在后才重新解析模型；失败、超时或无法确认时不执行后续模型。
`cleanup_recovery` 保存核验结果，原始 attempt 不改写；旧重试记录缺少明确清理证明时继续阻断。
冻结条件还包括 `ACPROF_NLP_TORCH_INDEX_URL`、`ACPROF_NLP_TORCH_SPEC`、`ACPROF_HOST_CUDA_VERSION`；
修改这些运行依赖选型参数须显式重试，attempt 会记录前后差异，普通续跑不能静默混用锁定环境。
同一报告目录通过现有目录锁排除并发写入，运行验证继续使用原有测量锁。

恢复语义参考 [Ray Tune 的 `Tuner.restore`](https://github.com/ray-project/ray/blob/f8a314bf077c9772fee2a8a1073368ca2ef9360b/python/ray/tune/tuner.py)
对已完成、未完成和失败任务的区分（Apache-2.0，维护中的实现）。AC-Prof 沿用已有 JSON 原子写入和目录锁，
只借鉴选择与证据保留原则，不引入 Ray、训练 checkpoint 或 pickle 依赖；记录发生在独立验证之外，不进入正式测量窗口。
