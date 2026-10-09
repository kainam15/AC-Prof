# CLI 采集与输入清单

[← 返回专题目录](cli.md)

## `acprof run`
### 模型与资源矩阵
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | required | Hugging Face model ID，例如 `google-bert/bert-base-uncased`。 |
| `--revision` | Hub 默认分支 | 指定模型分支、tag 或完整 commit SHA；运行时固定解析后的 SHA。 |
| `--task` | auto | 覆盖 `pipeline_tag`，例如 `fill-mask`、`text-generation`。 |
| `--task-family` | auto | 覆盖任务族：`nlp`、`cv`、`audio`、`timeseries`、`diffusion`、`multimodal`、`structured`。 |
| `--backend` | auto | 覆盖声明清单中的 runtime backend，例如 `transformers_pipeline`、`chronos`、`diffusers`、`onnxruntime`。 |
| `--model-spec` | 无 | 本地 `acprof_model.json` 格式的模型接口声明，优先于仓库声明，固化到服务镜像并参与恢复身份。用于缺少任务元数据、制品选择、custom pipeline 输入映射与固定离线依赖；TUI 对应“高级参数 → 识别覆盖 → 模型接口声明”，见[模型声明](../models/pipelines.md#本地模型声明与自定义-pipeline)。 |
| `--profiling-mode` | `full` | `full` 保留 Native Linux 的 RAPL、perf 和 packet latency 必需条件；`basic` 要求 application latency、吞吐、容器 CPU/内存，跳过能耗、PMU、抓包，允许 WSL2 PARTIAL。两者均要求本机 Docker 和 cgroup v2；不自动降级，见 [WSL2](../platforms/wsl2.md)。 |
| `--cpus` | `1,2,4,8` | CPU core 限制列表。 |
| `--cpuset-cpus` | 空 | 可选固定 CPU ID/范围，如 `0-3,8`，应用于正式采集和 startup probe；留空保留原有配额调度。规范化集合参与恢复身份，采样前核验实际 affinity。 |
| `--mems` | `2,4,8,16` | Memory cap GB 列表。 |
| `--gpus` | `off,on` | GPU mode 列表。`on` 只向容器暴露选定的物理 GPU。 |
| `--gpu-device` | 环境变量或 `0` | 单个主机 GPU index 或完整 UUID，优先级为此参数、`ACPROF_GPU_DEVICE`、`DEVICE_INDEX`、`0`。运行前解析并固定 UUID；不接受 `all`、设备列表或 MIG。`acprof probe` 使用同样的环境变量，post-hoc GPU 补采使用原实验记录的 UUID。 |
| `--prune-startup-oom` / `--no-prune-startup-oom` | enabled | 正式矩阵前，用最低选中 CPU、内存升序执行独立 startup probe，只启动并等待 `/ready`，不产生性能结果。仅 Docker 确认启动 OOM 的连续低内存前缀用于剪枝；遇到 ready、timeout、CUDA OOM 或普通错误即停止扩展。所有被剪枝 case 标为 `inferred_not_measured`。禁用后逐格正式尝试。 |
| `--matrix-order` | `seeded` | `seeded` 用版本化的确定性 hash 排序资源 case，并用每个 case 的独立派生 seed 排列 input scale；`declared` 保持资源参数与物化输入尺度的顺序。probe 完成后将实际顺序冻结到 `metadata/matrix_plan.json`。 |
| `--matrix-seed` | `0` | 整数 seed；相同实验身份、probe 结论、算法版本和 seed 生成相同计划。resume 校验并复用已冻结顺序，不重新排序；不能在原目录改变 seed 或 order。 |
| `--dram-energy` | `auto` | `full` 中独立采集可用的 DRAM RAPL 域；缺失或不可读时保持 `nan`，不因此使 full 失败。`off` 不采集，`required` 要求所有选中 package 的 DRAM 都可用且取得有效测量；不能与 `basic` 同用。DRAM 不加入 container-attributed energy。 |

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
| `--idle-seconds` | `20.0` | 每个 workload window 前 matched control window 的目标时长。CPU、GPU、resource usage 以及启用时的 perf MIPS monitor 会按与 workload 相同的 `start()` / `stop()` 生命周期同时运行，但 control window 内不发送 `/predict` 请求。CPU baseline 为整段 RAPL 能耗 / 实际 duration；GPU baseline 为 NVML samples 的时间加权平均功率。case 结束后会复查该 case CSV 中所有有效 CPU/GPU baseline 的相对极差，达到或超过 5% 会输出 warning，实验继续运行。 |
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
| `--output-dir` | `results` | 输出根目录。最终还会追加 model name 子目录。 |
| `--resume` | false | 使用原参数和目录恢复实验；逐项报告身份差异，保留完成 case，备份后重测中断 case；准备未完成时先归档准备证据再重试。已完成实验不重测。TUI 的“恢复 / 重试”也可自动分配新实验目录。 |
| `--skip-build` | false | 核验构建指纹和环境清单后复用镜像；不存在时自动构建，不匹配时退出。 |
| `--model-source` | `huggingface` | `huggingface`、`modelscope`；不同来源独立记录身份，不静默替换。 |
| `--download-mode` | `auto` | 高级/兼容参数：`auto`、`mirror-only`、`mirror-preferred`、`official`；显式参数优先于 `HF_DOWNLOAD_MODE`，TUI 固定使用 `auto`。 |
| `--max-download` | 不设上限 | 下载前核验全部批量 payload 预算，例如 `5GB`、`5GiB`；`0` 只允许缓存命中。任何来源大小未知或总量超限时，在 pull/build/权重下载前停止。环境变量为 `ACPROF_MAX_DOWNLOAD`。 |
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
| `audio` 文本到语音／音频 | `seq_length` | tokenizer 实测输入 token 数；不是生成音频的秒数。 |
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

- `nlp` 文本任务会启动容器读取 tokenizer / handler 的可用最大输入长度，最后一档尽量贴近有效上限。Decoder-only 生成额外预留 `max_new_tokens`；encoder-decoder 不从 encoder 输入预算扣除 decoder 输出长度。表格问答按 `1,2,4,8,16,32` 行规划，不进入 token 二分搜索；超出模型容量时明确失败。
- 读取音频的任务从 workload 清单读取默认尺度；内置英文语音清单为 `1,2,5,10,20,30` 秒。文本到语音／音频进入同一 token 规划器；tokenizer 没有有限上限时采用显式 512-token 采集上限，该值不是模型最大容量，Bark 使用自身 semantic 输入限制。`cv` 使用 generator 最大尺度；`timeseries` 读取已加载 Chronos 的 context limit，并与 workload 上限取较小值，拒绝静默截断。
- `structured` 表格／策略默认 `1,8,32,128` 行／观测，图默认 `8,32,128,512` 节点；可用清单或 CLI 覆盖，当前生成器限制非图不超过 4096、图不超过 2048。该限制是 workload 的输入大小限制，不是模型容量。
- `diffusion` 图像／视频任务默认使用 `128,192,256,320,384,512` 像素输出边长；提示词、随机种子、guidance scale 和去噪步数在各尺度间保持不变。`unconditional-image-generation`、`text-to-3d`、`image-to-3d` 改为扫描 `1,2,4,8,16,20` 个去噪步；Shap-E 使用真实 mesh 解码，渲染图片尺寸不作为网格计算规模。
- `multimodal` 从任务 workload 读取默认尺度：图像边长 `224,336,448`；音频 `1,2,5,10` 秒；视频 `2,4,8` 帧。清单 `input_scales` 可覆盖默认值，CLI `--input-scales` 优先。`static_meta.input_scale_type` 在计划完成后取实际 workload 的单位，不能将所有多模态任务统一解释为 token 数。
- 同一次 run 的所有资源配置共用同一组 scale。
- 所有任务族都会把已确定尺度的 payload 写入唯一的 `input_scale_plan.json`；主采集与 compute profiler 共同读取该文件，保证实际执行 payload、FLOP profiling 和 CSV 中记录的 `input_scale` 一致。
- 手动传入 `--input-scales` 时以手动值为准；workload 会在 sweep 前验证合法性（图像／视频生成分辨率至少为 64 且必须是 8 的倍数；去噪步数是正整数；CV 倍率为有限正数）。

### NLP workload 参数
NLP 清单使用 `schema_version=1`、与检测结果一致的 `task` 和 `params`，不接受额外字段。
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

生成任务接受正整数 `max_new_tokens`，例如 `{"schema_version":1,"task":"text-generation","params":{"max_new_tokens":16}}`。
清单通过 `--workload-spec /path/to/workload.json` 传入；参数、清单 SHA256 及每档实际 payload
写入输入计划，主采集与 profiler 重放同一参数。本接口不加载用户执行代码，也未增加任意 chat 模板或非对称检索任务。

### 真实音频 workload
`automatic-speech-recognition` 默认使用 `assets/audio/librispeech-clean-test-en-30s/source.json`。该清单引用 LibriSpeech `clean/test` 中同一说话人、同一章节的三条连续语音，按固定顺序拼接后截取前 30 秒；素材是单声道 16 kHz PCM16 WAV，许可证为 CC BY 4.0。每一档输入都从同一个 30 秒基准音频取前缀，不做逐档归一化、补全或循环。

音频分类、Encodec／DAC 音频重建和 Silero VAD 默认复用相同语音前缀。codec 按模型需要在预处理阶段重采样，输入规模仍按源音频时长记录。生成／重建响应的 `audio_num_samples`、`audio_sample_rate`、`audio_duration_s` 描述输出波形，不能写入文字 token 计数；VAD 的 `segments` 是秒为单位的连续阈值帧区间，阈值与分帧策略随响应记录，不是识别文本。Silero 每个请求重置状态，profiler 重复调用也不会延续上一请求的隐藏状态。

新增任务使用当前 CSV 字段、静态 schema v7 与输入计划 schema v2；旧 schema 会被拒绝。`input_units_per_request = effective_input_scale × batch_size`：表格／策略是总行数／观测数，图是总节点数。结构化 `input_num_samples` 对表格／策略记总行数／观测数，对图记图数量，另外在计划记录总节点数与边数。NLP 检索／排序的单位仍为候选文本尺度乘 batch，不再乘候选数量；固定 query、候选数及重复编码成本属于该请求，比较实验时必须保持一致。零样本 NLI 的候选标签推理成本同样包含在请求中。

结构化输入为固定种子的合成矩阵／环图，保存特征宽度、种子、结构、清单 SHA256 及实际 payload；各资源组合和 profiler 使用同一计划。模型缺少 SafeTensors 元数据时，参数量／权重字节数保持 `null`，不能以输入大小代替。skops 与 Silero 的 GPU 配置失败不会生成虚假的 GPU 指标；TorchScript 不支持加载时更换 attention implementation，eager FLOP 采集明确失败并保留工具状态，不能把缺失 FLOP 当作 0。

音频请求采用 JSON 内的 Base64 WAV；handler 仍能读取历史 `audio_samples` 浮点数组。短音频模式会读取模型 feature extractor 的约束并拒绝超过 receptive field 的尺度。对于 Whisper，30 秒是音频 receptive field；当前 feature extractor 会把接受的短音频补齐为固定的 480,000 samples / 3,000 frames，`/scale_meta` 会显式记录这一点。`max_target_positions=448` 是解码器输出 token 上限，不是音频输入上限，因此框架不会把 latency 必须随 `duration_s` 单调增加作为正确性条件。

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
