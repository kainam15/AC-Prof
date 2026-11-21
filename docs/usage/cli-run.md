# CLI 采集与输入清单

[← 返回专题目录](cli.md)

## `acprof run`

同一模型可声明多个语义用途，CLI 默认按推理接口和关键解码配置归并，避免为翻译、摘要等
提示词重复跑完整资源矩阵。只有发现多个独立推理组（或存在待确认冲突）时，
非交互 CLI 才要求 `--tasks`。它接受语义任务 ID 或组 ID；`--task` 是手动推理接口。
例如 T5 的翻译与摘要通常属于同一 Seq2Seq 推理组：

```bash
acprof run --model google-t5/t5-small --tasks translation_en_to_de,summarization \
  --cpus 1 --mems 4 --gpus off --input-scales 64 \
  --warmup 0 --repeat 1 --repeat-in-window 1 --notify none --output-dir results/t5-tasks
```

`--tasks all` 选择所有已识别用途，再自动合并为不同推理组；省略它时，
单组自动继续，多组才询问用户。即使选择两个同组任务也只运行一次标准测量，
应用任务列表和输入代表任务仍在元数据中。
恢复使用原参数及 `--resume --task-run-id <运行时打印的 ID>`；`acprof auto` 遵循同一规则。
每组默认统一最大输出 64 Token（实际输出依模型 EOS 而定），可由用户 workload 覆盖。
参见[推理分组](../models/routing.md#任务发现与选择)与[结果恢复](../profiling/artifacts.md#按任务保存与恢复)。

### 推理配置：先审阅再冻结

新实验的非交互 CLI 必须确认具体的推理配置。第一次执行上面的命令时，系统解析模型、
所选任务、后端、Runtime Profile、各设备的加载参数与生成参数，在模型下载和运行前停止，
打印候选配置文件、请求文件和 `Configuration SHA-256`。文件位于
`<output-dir>/.inference-configurations/`，候选文件的 `state` 为 `resolved`；
这一步没有授权正式采集。

打开候选 JSON，检查最终值、参数来源、Auto 决策和验证范围，再保留原命令的全部参数，
添加打印出的请求文件和 SHA：

```bash
acprof run --model google-t5/t5-small --tasks translation_en_to_de,summarization \
  --cpus 1 --mems 4 --gpus off --input-scales 64 \
  --warmup 0 --repeat 1 --repeat-in-window 1 --notify none --output-dir results/t5-tasks \
  --inference-config '<printed-requests-file>' --confirm-inference '<printed-sha256>'
```

确认摘要必须与这次重新解析的完整配置完全一致。模型 revision、后端、请求、参数来源或
影响执行的配置改变时，需要重新审阅新的候选；旧 SHA 不会授权新配置。确认通过后才冻结配置、
进入模型准备和运行时验证；实际加载值不满足冻结值时终止，不自动替换用户选择。

`--inference-config` 接受有 `schema_version: 1` 的 JSON 请求。参数请求保留三种模式：

| 请求 | 含义 |
| --- | --- |
| `{"mode":"explicit","value":...}` | 必须使用明确值，即使它恰好等于默认值；可空参数的 `null` 也必须显式给出。 |
| `{"mode":"auto"}` | 授权该字段使用已登记的选择规则；最终选择、依据和规则身份仍须审阅。 |
| `{"mode":"unset"}` | 未设置，不授权自动选择。活动字段没有可执行值时要求补充；因依赖而不活动的字段保留该状态。 |

`backend`、`runtime_profile` 为独立请求；加载参数位于 `loading.off`（CPU）和
`loading.on`（GPU）；任务参数位于 `tasks.<初始所选任务 ID>`。请从生成的请求文件开始修改，
保留完整任务 ID，包括其中的点；改变 Beam 或后端导致最终任务重新分组时，系统保存原选择到
最终任务的映射。不要把模型架构字段写入可调参数，也不要把候选配置 JSON 当成请求文件。

请求也支持 `preset: custom/auto/author/benchmark`。例如新请求
`{"schema_version":1,"preset":"benchmark"}` 采用 AC-Prof Benchmark v1 的固定任务参数。
仅修改已有文件的 preset 名称会保留其显式字段覆盖；要应用全部基线值，应从上述最小请求开始，
再添加需要保留的 backend、加载和 workload 请求。作者预设只应用有固定来源的、当前已开放的
`task_specific_params`；缺证据或冲突时拒绝。新预设仍需审阅完整候选和确认摘要。

消费核验除 native NLP 外，还登记了 Whisper、Chronos、多模态生成和 SDXL 的指定参数。
新增路线的开发验收仅有离线 Fixture／Mock；静态声明不能证明真实 checkpoint 已执行。
协议、来源和支持边界见[推理配置契约](../models/inference-configuration.md)。

恢复时沿用已冻结配置，Auto 不重新运行。可省略新的请求文件；若仍提供
`--inference-config`，每个明确给出的请求必须与冻结记录的模式、值和 JSON 类型一致，
未知字段、不同预设或不同确认摘要都会拒绝恢复。文件路径与确认凭据本身不构成实验内容身份。

### 模型与资源矩阵

GPU mode 在配置、runtime、profiling 和分析入口共用 `acprof.gpu_mode.normalize_gpu_mode`：
`on` 接受 `1`、`1.0`、`true`、`yes`、`gpu`，`off` 接受 `0`、`0.0`、`false`、`no`、`cpu`，
忽略大小写和首尾空白；Python bool 与数值 0/1 使用同一契约。其他值明确报错，
持久化的新实验身份统一为 `on/off`。历史分析表仅在缺少 mode 时保留已有 `device` 标签。
解析方式参考 [CPython ConfigParser 的显式值表和错误处理](https://github.com/python/cpython/blob/main/Lib/configparser.py)
（PSF），扩展本项目已有别名，不添加依赖或兼容 shim。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | required | Hugging Face model ID，例如 `google-bert/bert-base-uncased`。 |
| `--revision` | Hub 默认分支 | 指定模型分支、tag 或完整 commit SHA；运行时固定解析后的 SHA。 |
| `--task` | auto | 覆盖 `pipeline_tag`，例如 `fill-mask`、`text-generation`。 |
| `--tasks` | 单个推理组自动继续 | 用语义任务 ID 或推理组 ID 选择输入/执行范围，逗号分隔；`all` 选择全部用途并按组去重，只有多个推理组时必须选择。 |
| `--task-run-id` | 唯一既有集合 | 配合 `--resume` 指定已冻结的推理组集合，不用于为新实验命名。 |
| `--task-family` | auto | 覆盖任务族：`nlp`、`cv`、`audio`、`timeseries`、`diffusion`、`multimodal`、`structured`。 |
| `--backend` | auto | 覆盖声明清单中的 runtime backend，例如 `transformers_pipeline`、`chronos`、`diffusers`、`onnxruntime`。 |
| `--model-spec` | 无 | 本地 `acprof_model.json` 格式的模型接口声明，优先于仓库声明，固化到服务镜像并参与恢复身份。用于缺少任务元数据、制品选择、custom pipeline 输入映射与固定离线依赖；TUI 对应“高级参数 → 识别覆盖 → 模型接口声明”，见[模型声明](../models/pipelines.md#本地模型声明与自定义-pipeline)。 |
| `--inference-config` | 无 | 推理请求 JSON；未提供时先生成可审阅的具体候选。与 `--workload-spec` 的输入素材职责分开。 |
| `--confirm-inference` | 无 | 已审阅候选的完整小写 SHA-256；新实验没有匹配确认时在模型准备前停止。 |
| `--runtime` | `auto` | 优先选择可用的本机 Docker；Linux 在 Docker 不可用时选择 Process。可显式覆盖，完整规则见 [Linux Runtime](../platforms/linux-runtimes.md#选择与准备)。 |
| `--profiling-mode` | `best_effort` | 新实验自动生成逐采集器计划。兼容旧参数 `basic` 和 `full`，Resume 按原实验冻结的策略执行。 |
| `--cpus` | `auto` | Docker 展开为原有 `1,2,4,8`；Process 继承当前限制，数值列表需要可验证的 cgroup 配额。 |
| `--cpuset-cpus` | 空 | 限定工作负载可用的 CPU ID，如 `0-3,8`。输入规划启动的服务、隔离运行验证、startup probe 和正式采集均使用该集合。规范化集合参与恢复身份，正式采样前核验线程 affinity。 |
| `--mems` | `auto` | Docker 展开为原有 `2,4,8,16`；Process 继承当前限制，数值列表需要可验证的 cgroup 内存限制，单位 GiB。 |
| `--gpus` | `off,on` | GPU mode 列表。`on` 只向容器暴露选定的物理 GPU。 |
| `--gpu-device` | 环境变量或 `0` | 单个主机 GPU index 或完整 UUID，优先级为此参数、`ACPROF_GPU_DEVICE`、`DEVICE_INDEX`、`0`。运行前解析并固定 UUID；不接受 `all`、设备列表或 MIG。`acprof probe` 使用同样的环境变量，post-hoc GPU 补采使用原实验记录的 UUID。 |
| `--prune-startup-oom` / `--no-prune-startup-oom` | enabled | 正式矩阵前，用最低选中 CPU、内存升序执行独立 startup probe，只启动并等待 `/ready`，不产生性能结果。仅 Docker 确认启动 OOM 的连续低内存前缀用于剪枝；遇到 ready、timeout、CUDA OOM 或普通错误即停止扩展。所有被剪枝 case 标为 `inferred_not_measured`。禁用后逐格正式尝试。 |
| `--matrix-order` | `seeded` | `seeded` 用版本化的确定性 hash 排序资源 case，并用每个 case 的独立派生 seed 排列 input scale；`declared` 保持资源参数与物化输入尺度的顺序。probe 完成后将实际顺序冻结到 `metadata/matrix_plan.json`。 |
| `--matrix-seed` | `0` | 整数 seed；相同实验身份、probe 结论、算法版本和 seed 生成相同计划。resume 校验并复用已冻结顺序，不重新排序；不能在原目录改变 seed 或 order。 |
| `--dram-energy` | `auto` | 按可读性采集 DRAM RAPL 域，缺失时保留 `nan`。`off` 不采集，`required` 要求所有选中 package 的 DRAM 都可用且取得有效测量；自动采集需正的 idle 窗口。DRAM 不加入 container-attributed energy。 |

GPU device 表格中的主机序号规则适用于 Docker。Process 的数字为 `CUDA_VISIBLE_DEVICES` 内的可见序号，
不会扩大可见设备范围；记录 CUDA UUID 与共享状态，NVML 指标不可用不等于 GPU 推理不可用。
Process 自动关闭 Docker startup OOM pruning 并逐格执行；`--rebuild-image`、独立 profiler、
`probe` 最小内存诊断和已有服务负载重放仍要求 Docker。

### 请求窗口与采样

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--batch-size` | `1` | 每个 request 的 batch size。 |
| `--warmup` | `2` | 每个资源配置、每个 input scale 的 warmup 行数。 |
| `--repeat` | `5` | 每个资源配置、每个 input scale 的正式测量行数。 |
| `--repeat-in-window` | `0` | 每一行内部连续发送的 `/predict` request 数量。`0` 表示 auto 模式：每行至少发送 1 个请求，并持续到累计 `latency_app_s` 达到 `--repeat-window-seconds`。 |
| `--repeat-window-seconds` | `10.0` | `--repeat-in-window 0` 时的目标 workload window 秒数。auto 模式不再额外跑一个 10 秒校准窗口。 |
| `--request-timeout-seconds` | `300.0` | 正式矩阵中每个 `/predict` 请求的连接等待和读取无进展上限（分别应用），必须是大于 0 的有限值；它适用于 warmup、auto-window warmup 和正式请求，不是请求的严格总截止时间，也不限制整行、整个 case 或整条命令的总运行时间；持续有数据到达可使总耗时超过此值，不自动重试。超时后保留已完成行，并将触发请求及后续未测计划行分别写成可诊断的 error 占位。 |
| `--sample-hz` | `20.0` | GPU power sampling rate，单位 Hz；CPU workload 和 matched control window 期间也用它控制 RAPL、container cgroup、CPU frequency 和 GPU/resource usage 的采样间隔，以估计 average/peak power、vCPU share、CPU utilization 和 CPU cycles。perf MIPS 使用独立的 `perf stat` 窗口，不受该采样率影响。 |
| `--idle-seconds` | `20.0` | 启用 CPU 或 GPU 能耗采集时，每个工作负载窗口前采集一次空闲基线。控制窗口不发送推理请求。新实验显式设为 `0` 会跳过冷却和基线，仅记录可用的总能耗。Resume 沿用实验冻结的基线时长。 |
| `--idle-cooldown-seconds` | `5.0` | 每个 workload window 采集 idle baseline 前的统一冷却等待时间。CPU-only 和 GPU+CPU case 都使用同一个值，避免上一轮推理刚结束后的短时热状态、Docker/server 收尾或 GPU clock/power 瞬态直接进入 idle baseline。 |
| `--idle-debug` | false | 开启 baseline 调试输出。主 CSV 会填充 GPU 的 `gpu_idle_measured_at` / `gpu_idle_rel_range_so_far` 和 CPU 的 `cpu_idle_measured_at` / `cpu_idle_rel_range_so_far`，并写出 `debug/idle/<case-id>.jsonl`。诊断文件记录 matched control window 的 GPU NVML trace、CPU RAPL 子窗口、host/container CPU delta，以及 control 结束后的 `nvidia-smi`、loadavg、top CPU processes、Docker 容器和 `docker stats` 快照。为避免诊断本身污染 baseline，逐进程 `/proc` 快照移到 control window 外，不再归入 RAPL control 能量。 |

### 输入

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input-scales` | auto | 手动覆盖 input scale 列表；未提供时通常自动规划 6 档，自定义音频清单按声明档数。 |
| `--workload-spec` | task default | NLP、cv、audio、multimodal、diffusion 或 structured 的 workload 清单 JSON。NLP 可声明 embedding／生成参数；读取音频的任务默认复用内置 LibriSpeech 语音；结构化清单声明输入宽度、尺度与种子。各任务须使用对应的清单格式。 |

### 计算分析器

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--compute-profile-tool` | `none` | 默认跳过全部 compute probe；`both` 独立采集 `torch_profiler_eager` 逻辑 FLOP，并在 `gpu_mode=on` 时采集 NCU GPU 实际执行 FLOP。`torch`、`ncu` 用于单工具诊断，`vendor` 用于 CPU Advisor 与 GPU NCU；`auto` 不再接受。 |
| `--advisor-root` | auto | Host Intel Advisor install root or executable；显式值优先于自动检测。 |
| `--ncu-root` | auto | Host Nsight Compute install root or `ncu` executable；显式值优先于自动检测。 |
| `--advisor-repeat` | `20` | `vendor` CPU Advisor probe 的推理重复次数；最终 FLOP 会除回单 request。 |
| `--torch-profiler-repeat` | `1` | `torch_profiler_eager` probe 的推理重复次数；CPU/GPU 结果分别除回单 request。 |
| `--ncu-repeat` | `1` | NCU GPU probe 的推理重复次数；FLOP、kernel 数和 kernel 时间最终都除回单 request。 |
| `--compute-profile-cpus` | host logical CPUs | 临时 compute profiler container 的 CPU core cap。 |
| `--compute-profile-mem` | 75% host memory | 临时 compute profiler container 的 memory cap，单位 GB。 |
| `--keep-compute-profiles` | true | 保留 raw profiler artifacts；这是默认行为。artifact 位于模型结果目录的 `raw/compute_profiles/`，路径不写入结果行。 |
| `--discard-compute-profiles` | false | 汇总完成后删除 raw profiler artifacts。 |

### 执行分析器

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--execution-profile-tool` | `none` | 显式启用高开销 execution profiler：`massif` 用于 CPU-only、`nsys` 用于 GPU，`both` 同时选择两者；默认 `none` 不运行。 |
| `--massif-sampling` | `per-scale` | `per-scale` 使用一个代表 CPU/内存逐 input scale 采集并复用；`full` 采完整 CPU × memory 矩阵。 |
| `--massif-reference-cpu` / `--massif-reference-mem` | 最大选中值 | Massif `per-scale` 的代表 CPU 与内存；必须存在于本次 `--cpus` / `--mems` 中。 |
| `--massif-repeat` | `1` | 每个 Massif probe 内执行的 inference 次数。Massif peak 仍是包含加载和预热的 process-lifetime peak，不按此值归一化。 |
| `--nsys-sampling` | `per-cpu-scale` | `per-cpu-scale` 保留全部 CPU、只用一个代表内存；`per-scale` 只用一个代表 CPU/内存；`full` 采完整矩阵。 |
| `--nsys-reference-cpu` / `--nsys-reference-mem` | 最大选中值 | Nsys 缩减采样的代表资源；`per-cpu-scale` 只使用代表内存，`per-scale` 同时使用两者。 |
| `--nsys-repeat` | `1` | 每个 Nsight Systems `acprof_compute` NVTX range 内的 inference 次数；time、count 和 bytes 汇总会除回单 request。 |
| `--nsys-root` | auto | Host Nsight Systems install root 或 `nsys` executable；显式值优先于自动检测。 |
| `--keep-execution-profiles` | true | 保留 `raw/execution_profiles/` 下的 raw Massif `.out` 与 Nsight Systems `.nsys-rep`；这是默认行为。stats 导出的 `.sqlite` 缓存会自动删除。 |
| `--discard-execution-profiles` | false | 汇总成功后删除 raw execution-profiler artifacts，保留 plan、CSV 数值与错误诊断。 |

### 输出与运行环境

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sniff-iface` | `docker0` | 本机 Docker 默认 bridge 对应的 `tcpdump` 抓包网卡。只有 daemon 改过 bridge 名时才覆盖。 |
| `--output-dir` | `results` | 输出根目录。采集组保存在 `<model-dir>/tasks/<profile-id>/<collection-id>`，同一推理组内的应用用途共享一次测量。 |
| `--resume` | false | 使用原参数和目录恢复实验；逐项报告身份差异，保留完成 case，备份后重测中断 case；准备未完成时先归档准备证据再重试。已完成实验不重测。TUI 的“恢复 / 重试”也可自动分配新实验目录。 |
| `--rebuild-image` | false | 显式重建模型镜像，仍允许 Docker 缓存；默认核验并复用已有镜像，不存在时自动构建，校验异常时退出。不能与 `--resume` 同用。 |
| `--model-source` | `huggingface` | `huggingface`、`modelscope`；不同来源独立记录身份，不静默替换。 |
| `--download-mode` | `auto` | 高级/兼容参数：`auto`、`mirror-only`、`mirror-preferred`、`official`；显式参数优先于 `HF_DOWNLOAD_MODE`，TUI 固定使用 `auto`。 |
| `--max-download` | 不设上限 | 下载前核验全部批量 payload 预算，例如 `5GB`、`5GiB`；`0` 只允许缓存命中。任何来源大小未知或总量超限时，在 pull/build/权重下载前停止。环境变量为 `ACPROF_MAX_DOWNLOAD`。 |
| `ACPROF_MODEL_MAX_DOWNLOAD` | 不设上限 | 独立的模型权重剩余下载预算（如 `4GB`），在预检和 Model Store 实际下载前均验证；不限制 OCI 镜像、Python/Debian 依赖或元数据流量。与总预算同时设置时两个限制均须满足；此项仅为环境变量。 |
| `--model-store` | `~/.cache/acprof/model-store` | 单份主机模型目录，对应 `ACPROF_MODEL_STORE`；运行容器只读挂载。 |
| `--model-store-max` | 不设上限 | Model Store 容量上限，对应 `ACPROF_MODEL_STORE_MAX`；超限须先显式清理。 |
| `--model-download-policy` | `auto` | `auto` 按已覆盖的加载器规则筛选文件，未知结构保留完整快照并记录原因；`full` 下载固定 commit 的完整仓库。策略进入镜像指纹，不能相互误复用。采集和探测入口均支持。 |
| `--notify` | `auto` | `auto` 在配置 Webhook 后启用企业微信；`none` 关闭，`wecom` 显式选择企业微信。配置见[企业微信通知](configuration.md#企业微信通知)。 |
| `--help` | — | 显示此入口的全部公开参数后退出。 |

`--allow-cgroup-v1`、`--no-compute-profile` 和 `--compute-profile-tool auto` 已删除，使用它们会在参数解析时退出。关闭计算分析使用 `--compute-profile-tool none`。

结果目录存在异常中断留下的 `result_case_*.csv` 时，`acprof run` 会先读取同目录 `static_meta.json/cgroup_version`。只有版本与当前 host 一致才允许续写；版本不同、缺失或元数据不可读时会退出，避免把 v1/v2 窗口合并到同一结果文件。

## 输入规模与音频清单

`--input-scale-policy auto` 是普通 run 的默认范围规划；`minimal` 在任务解析后从对应 workload 的默认尺度
（或同一任务族既有范围）选取最小单一尺度，仍复用原有 tokenizer、音频采样率和模型上限校验。
TUI Smoke 使用 `minimal`；显式 `--input-scales` 优先且严格校验，不被该策略改小。
未显式使用 `minimal` 的历史恢复参数保持原记录语义；更改规划策略或输入尺度属于新实验。

`input_scale` 是每个任务族的主输入尺度，语义由 `static_meta.json` 的 `input_scale_type` 决定：

| task family | `input_scale_type` | 含义 |
| --- | --- | --- |
| `nlp` 文本任务 | `seq_length` | 输入 token length；问答取 context，检索／排序取候选文本 token 数的最大值，固定 query。 |
| `nlp` 表格问答 | `table_rows` | 每张表的行数；query 和列结构固定。 |
| `cv` | `resolution_scale` | 图像／视频帧基础边长 224 像素的缩放倍率；视频帧数固定。 |
| `audio` 读取音频 | `duration_s` | 输入音频时长，单位秒。 |
| `audio` Transformers 文本到语音／音频 | `seq_length` | tokenizer 实测输入 token 数；不是生成音频的秒数。 |
| `audio` Kokoro 原生文本到语音 | `phoneme_ids` | 全部已处理段的 KModel 音素 ID 总数，不含每段两个边界 ID；不是单词数或 tokenizer Token 数。 |
| `timeseries` | `context_length` | 时间序列 context length。 |
| `diffusion` | `resolution_px` 或 `denoising_steps` | 图像／视频生成是方形输出边长（像素），帧数固定；无条件图像与 Shap-E 3D 为去噪步数，分辨率／网格解码设置固定。以输入计划为准。 |
| `multimodal` 图像理解／问答／文档检索 | `resolution_px` | 输入图像边长，单位像素；模型 processor 可能重新缩放、切块或固定尺寸。 |
| `multimodal` 音频理解 | `duration_s` | 输入 WAV 的实际样本数 / 采样率，单位秒。 |
| `multimodal` 视频理解 | `frame_count` | 输入 PNG 帧数；FPS 和帧分辨率固定并写入计划。 |
| `multimodal` Any-to-Any | 由 `scale_modality` 决定 | 单一输入模态随尺度变化，其余固定；默认改变音频秒数。 |
| `structured` 表格 | `table_rows` | 每个 batch 项的表格行数，特征宽度固定。 |
| `structured` 策略 | `observation_count` | 每个 batch 项的独立向量观测数，观测宽度固定；不是交互时间步。 |
| `structured` 图 | `node_count` | 每张图的节点数，特征宽度固定，默认双向环有 2 × 节点数条边。 |

未提供 `--input-scales` 时，当前内置 workload 配置通常会为一次 profiling run 规划 6 档 input scale；自定义音频清单则使用清单中声明的档数：

- `nlp` 文本任务会启动容器读取 tokenizer / handler 的可用最大输入长度，最后一档尽量贴近有效上限。Decoder-only 使用 `max_new_tokens` 时从上下文预留相应的输出空间。使用 `max_length` 时，提示词（含特殊 token）与输出共用总长度，至少为输出保留一个 token 或显式指定的 `min_new_tokens`，取较大值。总长度超过已知可用上下文会在准备阶段报错。Encoder–Decoder 的输入预算独立于 decoder 输出长度。表格问答按 `1,2,4,8,16,32` 行规划，不进入 token 二分搜索；超出模型容量时明确失败。
- 读取音频的任务从 workload 清单读取默认尺度；内置英文语音清单为 `1,2,5,10,20,30` 秒。文本到语音／音频进入同一 token 规划器；tokenizer 没有有限上限时采用显式 512-token 采集上限，该值不是模型最大容量，Bark 使用自身 semantic 输入限制。`cv` 使用 generator 最大尺度；`timeseries` 读取已加载 Chronos 的 context limit，并与 workload 上限取较小值，拒绝静默截断。
- `structured` 表格／策略默认 `1,8,32,128` 行／观测，图默认 `8,32,128,512` 节点；可用清单或 CLI 覆盖，当前生成器限制非图不超过 4096、图不超过 2048。该限制是 workload 的输入大小限制，不是模型容量。
- `diffusion` 图像／视频任务默认使用 `128,192,256,320,384,512` 像素输出边长；提示词、随机种子、guidance scale 和去噪步数在各尺度间保持不变。`unconditional-image-generation`、`text-to-3d`、`image-to-3d` 改为扫描 `1,2,4,8,16,20` 个去噪步；Shap-E 使用真实 mesh 解码，渲染图片尺寸不作为网格计算规模。
- `multimodal` 从任务 workload 读取默认尺度：图像边长 `224,336,448`；音频 `1,2,5,10` 秒；视频 `2,4,8` 帧。清单 `input_scales` 可覆盖默认值，CLI `--input-scales` 优先。`static_meta.input_scale_type` 在计划完成后取实际 workload 的单位，不能将所有多模态任务统一解释为 token 数。
- 固定提示词、分词或音素转换可能让多个文本尺度目标对应同一有效长度。自动规划会尝试补足六档，只写入实际探测过、未截断且在规划上限内的请求。若仅找到两档，就显示并使用这两档。找不到可用长度或探测出错时停止准备。
- 同一次 run 的所有资源配置共用同一组 scale。
- 所有任务族都会把已确定尺度的 payload 写入唯一的 `input_scale_plan.json`；主采集与 compute profiler 共同读取该文件，保证实际执行 payload、FLOP profiling 和 CSV 中记录的 `input_scale` 一致。
- 手动传入 `--input-scales` 时以手动值为准；workload 会在 sweep 前验证合法性（图像／视频生成分辨率至少为 64 且必须是 8 的倍数；去噪步数是正整数；CV 倍率为有限正数）。手动文本尺度按请求词数生成输入。若两个请求经分词或音素转换得到相同的有效长度，准备阶段会报告两个请求值和冲突长度。可调整其中一档或省略 `--input-scales` 启用自动规划。已有计划若为同一有效尺度保存了多条请求，采集、重放和 Profiler 都会拒绝使用它。

Kokoro 继续复用文本尺度规划器和 `AudioWorkloadGenerator`：手动 `--input-scales` 表示用于构造
固定语料前缀的请求单词数，经过实际 G2P／词表检查后，计划和结果中的 `input_scale` 改记音素 ID 数。
自动规划以 510 音素 ID 为目标上限，通过 `/probe` 的实际预处理结果搜索；这不是整个请求的最大容量。
英语使用官方 `en_tokenize` 分段，其它已声明语言按非空换行段处理；每段最多 510 个可映射字符，
连同两个边界 ID 不超过模型 context。多段总量可超过 510，超长单段或未知词表字符明确拒绝，
不静默截断。原生 `phonemes` 列表保留用户给出的分段；当前 CLI 自动 workload 生成文本。
`predict()` 在返回前完整消费全部官方生成器；任何一段失败都使请求失败。
文本字符数、单词数、音素 ID 数和分段数分别记录，字段来源见[结果字段](../results/fields.md#资源配置输入与网络)。

### NLP workload 参数

NLP 清单使用 `schema_version=1`、与检测结果一致的 `task`、`params` 和可选 `input_budget`，不接受其他字段。

NLP、CV、Multimodal 与 Diffusion 共用输入构造预算：默认 `max_batch_items=4096`、
`max_text_characters=16777216`、`max_pixels=67108864`。batch 条目计入 NLP 候选展开，
像素按 `width × height × frames × batch` 校验；文本按展开后的 Unicode 字符数校验，
不另做 tokenizer 推理或估算显存。host 在文本、图像和视频帧分配／resize 前检查，
container 在 tokenizer／decode／batch 展开前检查。图像先读取全部源文件头，按单次请求累计所有图像与视频帧的像素，再解码；多模态的图像与视频共享该总量。源图像与 resize 后的目标尺寸分别受同一上限约束，缩小输出尺寸不会绕过源图像预算。CV 的 ONNX 图像路径也执行相同检查。

可在上述任务族的 workload JSON 中设置例如
`"input_budget": {"max_batch_items": 8192, "max_pixels": 134217728}`，每个值必须为
正整数，省略的项使用默认值。解析后的完整预算随 payload 和 input plan metadata 保存，
纳入输入计划身份；适用于正式画像和容量探索，不限于 probe。该预算限制输入构造，
不承诺模型能在给定 CPU／内存／GPU 下运行；预算拒绝使用 `input_budget_exceeded`，
与实际推理中的 `resource_limit`／OOM 分开记录。QA 在有效 context token 为零时即拒绝。
例如 `feature-extraction` 的句向量归一化：

```json
{"schema_version": 1, "task": "feature-extraction", "params": {"normalize_embeddings": true}}
```

`feature-extraction`／`sentence-similarity` 可指定字符串 `prompt` 或 `prompt_name`（二选一）以及布尔
`normalize_embeddings`；命名 prompt 必须由模型声明。特征提取的这些参数要求
`sentence_transformers` backend，普通 Transformers token 特征不能忽略参数继续运行。
未指定时保留模型默认 prompt 和模块图的归一化行为。`normalize_embeddings=false` 不会移除仓库已有的
Normalize 模块，遵循原生 `encode` 语义。尺度表示正文 token 数；容量预算额外扣除
prompt 和 special tokens，预处理还验证拼接后的实际长度，防止编码器隐式截断。
正式采集和经审阅的 `acprof probe` 都会在每档请求中保留清单指定的句向量选项。
恢复时的检查见[推理配置冻结](../models/inference-configuration.md#解析确认与冻结)。

生成任务接受正整数 `max_new_tokens`，例如 `{"schema_version":1,"task":"text-generation","params":{"max_new_tokens":16}}`。
清单通过 `--workload-spec /path/to/workload.json` 传入；参数、清单 SHA256 及每档实际 payload
写入输入计划，主采集与 profiler 重放同一参数。本接口不加载用户执行代码，也未增加任意 chat 模板或非对称检索任务。

### 真实音频 workload

`automatic-speech-recognition` 默认使用 `assets/audio/librispeech-clean-test-en-30s/source.json`。该清单引用 LibriSpeech `clean/test` 中同一说话人、同一章节的三条连续语音，按固定顺序拼接后截取前 30 秒；素材是单声道 16 kHz PCM16 WAV，许可证为 CC BY 4.0。每一档输入都从同一个 30 秒基准音频取前缀，不做逐档归一化、补全或循环。

音频分类、Encodec／DAC 音频重建和 Silero VAD 默认复用相同语音前缀。codec 按模型需要在预处理阶段重采样，输入规模仍按源音频时长记录。生成／重建响应的 `audio_num_samples`、`audio_sample_rate`、`audio_duration_s` 描述输出波形，不能写入文字 token 计数；VAD 的 `segments` 是秒为单位的连续阈值帧区间，阈值与分帧策略随响应记录，不是识别文本。Silero 每个请求重置状态，profiler 重复调用也不会延续上一请求的隐藏状态。

新增任务使用当前 CSV 字段、静态 schema v7 与输入计划 schema v2；旧 schema 会被拒绝。`input_units_per_request = effective_input_scale × batch_size`：表格／策略是总行数／观测数，图是总节点数。结构化 `input_num_samples` 对表格／策略记总行数／观测数，对图记图数量，另外在计划记录总节点数与边数。NLP 检索／排序的单位仍为候选文本尺度乘 batch，不再乘候选数量；固定 query、候选数及重复编码成本属于该请求，比较实验时必须保持一致。零样本 NLI 的候选标签推理成本同样包含在请求中。

结构化输入为固定种子的合成矩阵／环图，保存特征宽度、种子、结构、清单 SHA256 及实际 payload；各资源组合和 profiler 使用同一计划。模型缺少 SafeTensors 元数据时，参数量／权重字节数保持 `null`，不能以输入大小代替。skops 与 Silero 的 GPU 配置失败不会生成虚假的 GPU 指标；TorchScript 不支持加载时更换 attention implementation，eager FLOP 采集明确失败并保留工具状态，不能把缺失 FLOP 当作 0。

音频请求采用 JSON 内的 `audio_base64` 单声道 PCM16 WAV，并携带 `audio_format="wav"` 和 `sample_rate`；`audio_samples` 浮点数组会被明确拒绝。短音频模式会读取模型 feature extractor 的约束并拒绝超过 receptive field 的尺度。对于 Whisper，30 秒是音频 receptive field；当前 feature extractor 会把接受的短音频补齐为固定的 480,000 samples / 3,000 frames，`/scale_meta` 会显式记录这一点。`max_target_positions=448` 是解码器输出 token 上限，不是音频输入上限，因此框架不会把 latency 必须随 `duration_s` 单调增加作为正确性条件。

自定义素材时可复制内置 `source.json`，设置新的 `workload_id`，再修改相对素材路径、SHA256、provenance 和 inference 字段。自定义 provenance 可以描述单条录音或既有重采样/增益流程；运行时仍会严格验证派生 WAV 本身是单声道、16 kHz、PCM16 且哈希匹配。然后传入：

```bash
acprof run --model openai/whisper-large-v3 \
  --workload-spec /path/to/source.json
```

当前音频 request 只实现 `batch_size=1` 和 `short_form`。清单会拒绝非空的 `chunk_length_s` / `stride_length_s`；长音频 sequential/chunked 应使用独立 workload，不能通过把本清单尺度直接扩展到 30 秒以上来混测。

视觉、多模态与图像条件生成也接受 `--workload-spec`，清单和支持边界见 [README 视觉任务](../models/supported-tasks.md#视觉任务) 与 [多模态任务](../models/supported-tasks.md#多模态任务)。输入计划沿用 schema v2，新增信息写在扩展的 `workload`、`input_metadata` 和 `payload` object 内；CSV 未增加列，当前 schema 的可选字段允许缺失。`workload` 保存素材路径／SHA256、清单 SHA256、提示词、参数、尺度单位和固定条件；实际序列化 payload 及计划 SHA256 是重放依据。

CV 每请求一个图片／视频样本，`input_num_samples=1`；视频帧数及每帧 SHA256 单独记录在 `input_metadata`。零样本标签、VitPose 的人物框和参数也随 payload 重放，不增加隐藏的人物检测请求。CV 响应按任务区分 classification、detection、caption、depth、segmentation、masks、features、keypoints；大张量／掩码／深度图只返回摘要。未产生文本的任务不填写输出 token 数，相关 CSV 指标保持 `NaN`。

无条件图像与 3D 的 `input_units_per_request` 单位为去噪步；其 `task_param.num_inference_steps` 随输入尺度变化。Shap-E 响应 `output_type="mesh"`，输出长度表示网格数量，顶点和三角面数量另行记录，不冒充文本长度、像素数或 token 数。该任务的 profiler 覆盖去噪及真实网格解码；图像条件预处理仍属于请求应用延迟。此扩展不改变历史图像生成的分辨率单位，读取时应依据每次实验的 `input_scale_type`，不能只按 diffusion 任务族判断单位。

多模态 `input_num_samples=1` 表示一个请求样本；其中可含图像、音频、视频或 query＋文档。音频 PCM 样本数与视频帧数分别保存在 `input_metadata.audio_num_samples` / `video_num_frames`。`input_units_per_request` 继续等于有效 `input_scale × batch_size`，因此其单位取决于上表，不可横跨不同尺度类型直接比较每单位延迟。

文字生成／问答的 `output_length_avg` 是返回文字的字符数窗口均值；`output_token_count_avg` 是对输出文字重新分词的 token 数，不代表所有解码步或 Omni 音频 token。Omni 音频摘要另含 24000 Hz 采样率、PCM 样本数和秒数，不混入文字长度。图像生成 `output_length` 为图像数，视频生成为总帧数；检索仅返回 query-by-document 分数矩阵，不产生文字长度／token 字段，其对应 CSV 值保持 `NaN`。

主请求延迟包含输入预处理、模型推理和结果摘要。后置 profiler 测量 handler 的 `predict()`：多模态理解、视频分类和关键点的直接模型路径不包含在 `preprocess()` 中运行的 processor；使用 Transformers pipeline 的 CV 路径仍包含 pipeline 内部预处理和后处理。Base64 解码与 handler 的摘要处理均在 profiler 的预测段之外；Diffusers 的完整生成／媒体解码、检索的两个编码器 forward 与 MaxSim 则在预测段内。Omni GPU 采用 thinker/talker FP16 和 Token2Wav FP32；其 eager FLOP 请求明确不支持，NCU/Nsys 仍可测完整推理。Shap-E 及 Diffusers Transformer 视频架构在没有可验证 eager 替换时同样报告工具错误，不伪填 FLOP。生成媒体不会作为图片／音频／视频／网格响应体返回，因此网络指标描述当前摘要服务协议。
