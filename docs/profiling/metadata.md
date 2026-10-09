# 实验元数据与失败证据

[← 返回专题目录](protocol.md)

## `static_meta.json` 字段
`static_meta.json` 是一个 model/image/run-level JSON object，只保存描述实验对象、环境和最终 profiling 配置/口径的相对稳定信息。补采、重试、回填与备份过程记录放在独立的 `collection_history.json`。数组、布尔值、数字和 `null` 均保留 JSON 原生类型，不再编码成 CSV 字符串。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | `static_meta.json` schema 版本；新增运行环境绑定与验证记录后的当前版本为 `7`。 |
| `model_name` | 所选来源的 model ID，例如 `google-bert/bert-base-uncased`。 |
| `model_source` | `huggingface` 或 `modelscope`；历史缺字段沿用 HF 语义。补采恢复使用原记录，不受当前来源环境变量影响；显式无效值会被拒绝。相同模型名不代表相同来源或 artifact。 |
| `model_revision` | 实际解析到的 model revision / commit hash。 |
| `parameter_count` | Hugging Face Hub SafeTensors metadata 的参数总数；Hub 未提供时为 `null`。 |
| `parameter_bytes` | 根据 `parameter_dtype_counts` 的各 dtype 元素数量与字节宽度精确求和得到的逻辑 tensor payload 大小，不含序列化 header；没有 dtype 统计或存在未知 dtype 时为 `null`。 |
| `precision_dtype` | SafeTensors 参数中数量占主导的权重精度，例如 `FP32`、`FP16`、`BF16`、`INT8`；无法确认时为 `null`。 |
| `parameter_dtype_counts` | 按 dtype 统计的参数/张量元素数量，保留混合精度与少量整型 buffer 信息。 |
| `inference_precision_by_device` | 当前 handler 明确请求的 CPU/GPU 推理精度。通常 Transformers NLP/CV/audio 为 CPU FP32、GPU FP16；MOSS adapter 为 CPU FP32、GPU BF16；Encodec/DAC 两者均 FP32。TorchScript 保留导出权重精度、输入 FP32；skops 保留 estimator 内部精度、输入 FP32。Silero 和 skops 仅声明 CPU。 |
| `static_flops` | Torch eager profiler 得到的逻辑 shape FLOPs，按 `input_scale` 保存 `flops_per_request`；未采集成功时为 `null`。 |
| `static_macs` | 静态 MACs。当前不做不可靠的 FLOPs/2 推断，因此未单独采集时为 `null`。 |
| `input_format` | 实际 `/predict` HTTP JSON 输入协议及其 JSON Schema。 |
| `output_format` | 实际 `/predict` HTTP JSON 响应协议及其 JSON Schema。 |
| `quantized` | 是否检测到量化配置、量化 tag 或量化权重 dtype；无法确认时为 `null`。 |
| `quantization_method` | 量化方法，例如 `gptq`、`awq`；不适用或未知时为 `null`。 |
| `quantization_config` | Hub model config 中的完整量化配置；没有时为空 object。 |
| `model_license` | Hugging Face model card 许可证，例如 `apache-2.0`、`mit`；无法确认时为 `null`。 |
| `model_metadata_source` | 参数量、参数 payload、精度、量化和许可证的元数据来源，当前在线 Hub 检测成功时为 `huggingface_hub`。 |
| `model_resolution` | 可选的接口解析 object（内部 schema v1）：任务、backend、library、制品格式、loader、operation、model type、固定 revision、元数据文件和 runtime profile。包含 `candidates/evidence`、`conflicts/missing`、`selection`、`interface_kind`、`pipeline_task`、`code_files/code_revision`、有效 `model_spec`；自动解析追加独立 `contract` provenance 和仅在无缺口时生成的 `generated_spec`。依赖固定、用户审阅、接口检查和真实运行的观察分别记录；`contract.runtime_validation` 从 `not_run` 变为包含 mode、image ID、payload／报告 hash 与设备结果的 object，详见[契约生成](../models/contracts.md#自动生成模型契约m1m6)。`candidate` 不是执行成功；`ambiguous/needs_configuration` 在镜像准备前拒绝。历史 v7 缺失字段按未知处理，不推算。无数值单位或测量窗口，不增加 CSV 列。 |
| `task_family` | 任务族：`nlp`、`cv`、`audio`、`timeseries`、`diffusion`、`multimodal`、`structured`。 |
| `pipeline_tag` | Hugging Face pipeline tag，例如 `fill-mask`、`image-classification`。 |
| `runtime_backend` | 容器内使用的 runtime backend，例如 `transformers_pipeline`、`chronos`、`diffusers`。 |
| `image_tag` | 本次传给 Docker 的镜像引用；v7 新采集使用不可变 `sha256:` image ID，历史文件可能为可变 tag。 |
| `image_id` | 经 Docker inspect 核验的不可变镜像 ID；补采优先使用该字段。历史文件无法确认时不补造。 |
| `image_name` | 便于查看的构建标签，含模型名和构建请求指纹前缀；执行仍使用 `image_id`。 |
| `runtime_environment` | 镜像内生成的环境清单：profile、adapter、构建指纹、模型及实际 snapshot revision、Python 和已安装包版本、依赖锁与包清单 SHA256、自定义 Python 源码 SHA256，以及 `model_download` 文件清单。新构建追加平台/环境身份、各父镜像 ID、系统锁摘要与实际系统包集合，字段详见下文；历史缺失字段不推算。 |
| `runtime_validation` | 独立容器验证报告。保存实际 image ID、输入尺度／payload SHA256、每个设备的状态、dtype、attention 实现、输出摘要、有效 `model_spec` 和分阶段 `stages`。验证推理接口，不替代 profiler 兼容性检查，也不计入请求或性能测量。失败的完整报告另见 `runtime_validation.json`。 |
| `batch_size` | 本次 profiling 的 batch size。 |
| `input_scale_type` | `summary.csv/input_scale` 的语义名，例如 `seq_length`。 |
| `workload` | workload 清单的可复现元数据，包括素材 SHA256、来源、变换、推理模式以及模型侧输入约束。 |
| `input_scale_plan_sha256` | 本次实际执行的 `input_scale_plan.json` SHA256。 |
| `run_command` | 启动本次 profiling 的 `acprof run ...` 命令，便于复现实验参数。 |
| `model_download_url` | 所选 source 的规范模型页面 URL；不是镜像／存储 endpoint。 |
| `gpu` | 存在 GPU case 时为选定物理 GPU 的名称；仅 CPU 实验保留主机设备信息，没有可见 NVIDIA GPU 时为 `unknown`。 |
| `gpu_mem_total_bytes` | 对应上述设备的 total VRAM，单位 bytes；无法读取时为 `null`。 |
| `gpu_device` | schema v7 的新增可选 object：`uuid`、主机 `index`、`pci_bus_id`、`name`、`memory_total_bytes`。GPU case 运行前解析并固定 UUID，Docker、NVML 和独立 profiler 共用；容器内单卡编号为 `0`。CPU 实验为空对象；历史缺字段表示身份未知，重新执行 GPU post-hoc 采集时拒绝猜测设备。 |
| `host_mem_total_bytes` | Linux 测量环境可见 RAM 总量，单位 bytes；Native 沿用主机 RAM，WSL 为 guest 可见内存，不代表 Windows host；无法读取时为 `null`。 |
| `host_swap_total_bytes` | 实验启动时 host 已启用 swap 的总容量，单位 bytes；无法读取时为 `null`，未启用时为 `0`。 |
| `host_swap_used_bytes_at_start` | 静态元数据采集时 host 已使用的 swap 快照，单位 bytes；无法读取时为 `null`。 |
| `host_swap_type` | `/proc/swaps` 中 active swap 的 backing 类型：`none`、`file`、`partition`、`zram`、`mixed` 或无法识别时的 `unknown`。 |
| `host_vm_swappiness` | 实验启动时 `/proc/sys/vm/swappiness` 的整数值；无法读取时为 `null`。 |
| `model_cache_bytes` | mounted 模型为 `model_artifact_bytes` 的兼容别名；历史 baked 模型仍为镜像 `/models/hf` 下按 inode 去重的普通文件逻辑 bytes，不重写历史结果。 |
| `model_storage_mode` | `mounted` 表示主机 Model Store 只读挂载，`baked` 表示历史镜像内权重。历史缺字段时不推断具体缓存/制品大小。 |
| `model_artifact_bytes` | mounted 模型清单中主 snapshot 与离线依赖文件的逻辑 bytes；按清单路径求和，包含配置、tokenizer、代码，不等同于物理共享 blob 占用。历史 baked 结果为未知。 |
| `runtime_image_bytes` | mounted 模式下的 Docker image size，包含 runtime、代码和模型清单，不含挂载权重。历史 baked 镜像无法可靠拆分，保留未知。 |
| `total_deployment_bytes` | mounted 模式为 `model_artifact_bytes + runtime_image_bytes`；baked 模式为完整 `docker_image_bytes`，避免重复加权重。非 registry 压缩流量、非共享层实际占盘量。 |
| `docker_image_bytes` | `docker image inspect <image_tag> --format "{{.Size}}"` 返回的本地 image size，单位 bytes。 |
| `docker_storage_total_bytes` | Docker daemon `DockerRootDir` 所在文件系统的总容量，单位 bytes；无法访问 daemon 路径时为 `null`。 |
| `docker_storage_available_bytes_at_start` | 静态元数据采集时 `DockerRootDir` 所在文件系统对当前用户可用的容量快照，单位 bytes；该值会随磁盘使用变化。 |
| `docker_storage_filesystem` | `DockerRootDir` 所在文件系统类型，例如 `ext4`；无法识别时为 `unknown`。 |
| `docker_storage_device` | 承载 `DockerRootDir` 的 mount source，例如 `/dev/nvme0n1p2`；无法识别时为 `unknown`。 |
| `docker_storage_type` | 根据 `lsblk` transport/rotational 信息得到的 `nvme_ssd`、`ssd`、`hdd` 或内存文件系统 `memory`；证据不足时为 `unknown`。 |
| `environment` | 兼容保留的发行版标签，例如 `ubuntu24.04` 或 `ubuntu24.04+wsl`；不能用于判定采集等级或补全历史身份，环境边界以 `platform` / `comparability_class` 为准。 |
| `cgroup_version` | 本次 preflight 实际检测到的 hierarchy：仅支持 `v2`，其它 hierarchy 在预检退出。 |
| `cgroup_collection_mode` | 当前唯一采集策略为 `strict_v2`。分析正式数据集时应同时要求 `cgroup_version=v2` 和 `cgroup_collection_mode=strict_v2`。 |
| `cpu_power_source` | CPU package 功耗来源。`rapl` 表示使用 Linux RAPL powercap 真实计数器；`unavailable` 表示当前环境没有可用 RAPL。 |
| `rapl_topology` | static schema v7 可选对象。记录全部发现的 powercap 域、父域、真实路径、sysfs aliases、计数器范围、enabled、可读状态、选中来源及 DRAM 覆盖缺口。子域与 MSR/MMIO 重复接口保留元数据，只有明确选中的 package/DRAM 分别计量。历史缺失表示未知。 |
| `vcpu_power_method` | estimated vCPU 功耗计算方法。`rapl_cgroup_cpu_share` 表示对 RAPL package energy 逐采样区间按 container cgroup CPU share 归因；`unavailable` 表示无法估算。 |
| `cpu_governor` | Host CPU frequency governor 汇总值，例如 `performance`、`powersave`、`schedutil`；如果各 CPU policy 不一致，会写成 `mixed:<governor>=<count>,...`；无法读取时为 `unavailable`。 |
| `cpu_boost` | Host CPU boost / turbo 状态。`on` 表示 boost 可用，`off` 表示关闭；无法读取时为 `unavailable`。 |
| `compute_profile_tools` | 本次启用/实际记录的 profiler 列表，例如 `["torch_profiler_eager","ncu"]`。 |
| `torch_profiler_eager_flop_semantics` | Torch eager FLOP 的统计口径说明。 |
| `torch_profiler_eager_attention_implementation` | 独立 Torch probe 强制并验证的 attention 实现，当前为 `eager`。 |
| `torch_profiler_eager_repeat_cpu` | CPU Torch eager probe 的 repeat；未采 CPU profile 时为 `null`。 |
| `torch_profiler_eager_repeat_gpu` | GPU Torch eager probe 的 repeat；未采 GPU profile 时为 `null`。 |
| `ncu_flop_semantics` | NCU GPU 实际执行 FLOP 的计数器、Tensor/Scalar 分类口径说明。 |
| `ncu_repeat` | NCU probe repeat；所有 NCU per-request 指标据此归一化。 |
| `ncu_fma_flop_weight` | NCU FMA 指令的 FLOP 权重，当前为 `2`。 |
| `ncu_metrics` | 本次 NCU 实际请求/解析的 metric 列表。 |
| `torch_version` | Torch probe 使用的 PyTorch 版本。 |
| `transformers_version` | Torch probe 使用的 Transformers 版本。 |
| `ncu_version` | Host NCU 版本；历史回填无法可靠确认时为 `unknown`。 |
| `gpu_compute_capability` | profile 使用 GPU 的 compute capability。 |
| `gpu_sm_count` | profile 使用 GPU 的 SM 数。 |
| `compute_profiles_retained` | raw profiler artifact 是否保留。 |
| `compute_profile_provenance` | profile 来源，例如本次直接采集或历史 `posthoc_backfill`。 |
| `execution_profile_schema_version` | execution profile plan schema 版本。 |
| `execution_profile_tools` | 本次显式启用且适用于所选 GPU modes 的 execution profiler 列表，例如 `["massif","nsys"]`；工具缺失时仍列出，并通过对应 error 字段诊断；默认关闭时为空列表。 |
| `massif_peak_semantics` | Massif peak 的 process-lifetime 口径，明确包含模型加载与预热。 |
| `massif_repeat` | 每个 Massif probe 内的 inference repeat；peak bytes 不按 repeat 归一化。 |
| `massif_version` | 实际执行分析的 container image 中的 Valgrind/Massif 版本；使用共享运行依赖的原模型镜像，未启用或无法确认时为 `unknown`。 |
| `massif_sampling_strategy` | `representative_per_scale` 或 `full_resource_matrix`。 |
| `massif_reference_cpu_cores` / `massif_reference_mem_cap_gb` | 缩减采样实际使用的代表资源；完整矩阵时为 `null`。 |
| `massif_reused_across_resource_cases` | Massif entry 是否从代表资源复用到其他结果行。 |
| `nsys_timeline_semantics` | Nsight Systems timeline 的 NVTX range 与汇总口径。 |
| `nsys_repeat` | 每个 Nsight Systems NVTX range 内的 inference repeat；动态汇总据此归一化到单 request。 |
| `nsys_version` | Host Nsight Systems 版本；未启用或无法确认时为 `unknown`。 |
| `nsys_sampling_strategy` | `representative_per_cpu_scale`、`representative_per_scale` 或 `full_resource_matrix`。 |
| `nsys_reference_cpu_cores` / `nsys_reference_mem_cap_gb` | Nsys 缩减采样使用的代表资源；per-CPU 策略的代表 CPU 为 `null`。 |
| `nsys_reused_across_resource_cases` | Nsys entry 是否从采样资源复用到其他结果行。 |
| `execution_profiles_retained` | raw Massif / Nsight Systems artifacts 是否保留。 |
| `execution_profile_provenance` | execution profile 的来源；默认关闭时为 `disabled`。 |

`static_meta.json` 只接受 schema v7；旧版本或缺失版本直接报错。读取端最多接收 4 MiB 的
UTF-8 JSON object，并拒绝 `NaN`、`Infinity` 及溢出为无穷大的数字。可选读取只在文件不存在时
返回空对象；文件存在但超限、损坏或类型错误时仍严格报错且不改写原件。当前完整 cache artifacts
大小字段为 `model_cache_bytes`，不读取旧 `model_weight_bytes`。采集/修复记录独立写入
`collection_history.json`，不从旧静态元数据补造 swap、cgroup 或运行环境信息。

`runtime_environment` 继续使用 schema v1。新服务镜像构建时增加以下身份字段，保持原有包版本、
模型快照及 `build_fingerprint` 的语义；`static_meta.json` 仍为 v7，CSV 没有新增列。

| 字段 | 含义 |
| --- | --- |
| `platform_id` | 声明的平台键：`cpu`、`cu124` 或 `cu128`。 |
| `platform_definition_id` | 规范化 Python 基础镜像、架构、系统锁和 Torch 闭包的内容摘要。 |
| `environment_id` | 平台声明与完整 Python 依赖制品的内容摘要，独立于 profile、adapter、模型和代码。 |
| `platform_image_id` / `environment_image_id` / `model_image_id` | 本次实际继承的平台、环境与模型快照镜像的不可变 Docker ID。 |
| `platform_build_fingerprint` / `environment_build_fingerprint` | 包含安装配方的构建身份；环境构建还绑定实际平台 image ID。 |
| `request_fingerprint` | 服务构建声明的查找键；原有 `build_fingerprint` 另绑定不可变模型父镜像 ID，继续表示实际构建身份。 |
| `model_spec` | 本地 `--model-spec` 优先于仓库 `acprof_model.json` 的有效内容，无声明时为空 object。规范化内容参与服务构建指纹，加载器消费构建期固化的相同内容；不改变依赖环境或模型文件层身份。本地声明文件的绝对路径和原始字节 SHA256 另参与 run 恢复身份。 |
| `python_base_image` / `architecture` | 固定 OCI digest 的 Python 基础镜像及目标 `linux/amd64`。 |
| `system_lock_sha256` | 规范化系统锁的 SHA256，覆盖 snapshot 来源、索引摘要、系统包集合和 `.deb` 制品。 |
| `system_packages` | 构建时实测的完整已安装系统包集合，键为 `包名:架构`，值为包含 epoch 的版本。 |

新缓存必须同时核对标签、上述身份字段与内部完整包清单。历史 v7 结果仍按记录的原始 image ID
和原构建指纹补采；缺失这些新增字段不会触发推算、回填、升级依赖或重建替代镜像。
`dependency_lock_sha256` 仍是镜像内完整 Python 锁文件字节的摘要；可复用环境中的注释排版
不参与 `environment_id`，因此构建服务时读取实际父环境清单中的该值。

`runtime_environment.model_download` 是可选的独立 schema v1 清单，历史结果可缺失。`requested_policy` 保存 `auto/full`，`effective_policy` 保存实际 `selected/full`，`reason` 说明筛选或回退原因；`weights` 记录组件、格式、variant 和索引／分片文件。`files` 保存路径、实际逻辑大小和准备阶段计算的 SHA256，另保留 Hub 提供的 Git blob／LFS 标识；`excluded_files` 是未下载文件的远端元数据。`verification=sha256` 表示准备阶段已完成完整性检查，`plan_sha256` 校验规范化 JSON（不含自身字段）。`selected_bytes` 按清单路径求和，不对相同内容的多个路径去重，mounted 模式下其含依赖总量成为 `model_artifact_bytes`（及兼容别名 `model_cache_bytes`）；不等同于镜像大小或释放的磁盘空间。新增清单不改变 CSV 字段和历史指标定义。`runtime_environment.model_store` 记录 entry ID、计划 SHA256 和逻辑制品大小；主机结果额外记录 `host_path`，仅用于挂载定位，不参与便携镜像内容身份。

清单新增可选的 `source`（历史缺字段为 HF）、`requested_revision`、`repository_context` 与
`download_provenance`。身份由 source、model ID、固定 commit 和已验证文件决定，HF／ModelScope
同名模型保持不同来源；endpoint 不作为模型身份。`endpoint` 记录成功使用的 Hub 基地址，
依赖模型在 `dependencies[].download.endpoint` 分别记录。它参与清单完整性摘要，无单位，
不属于测量窗口，也不等同于最终 CDN 地址。镜像入口或备用列表不再独立参与模型请求指纹。

`download_provenance` 是可选 schema v1 传输记录，保存 Hub、最终 endpoint 类型、失败尝试与
重定向响应（URL 移除 query、fragment 和内嵌凭据）；有界记录最多 256 次响应，截断有明确标志。
应用层连接用 `direct-socket`／`explicit-proxy` 表示，不推断实际公网出口。
ModelScope SDK 未暴露的存储链、未观察到的本地缓存原始地址保留 `unknown`。
完整缓存命中保留原计划与 provenance，不新增虚构的网络记录。历史清单与已有实验不改写，
缺失传输字段不补成官方、镜像或零流量；新 source 字段不改变 CSV 测量指标及单位。
新的网络预检报告使用 schema v2，删除基于域名猜测的 DIRECT／PROXY 字节分摊，旧报告仍可读取。
下载失败不发布已验证清单或模型镜像。

`runtime_validation.json` 使用独立 schema v1：`devices.off/on` 分别保存 CPU／GPU 的 `ok`、`error`、`inconclusive` 或明确 cgroup OOM 的 `resource_limit`；总状态为 `ok`、`error`、`inconclusive` 或 `resource_limited`。每个模式只执行一次最小计划输入，资源上限为本次配置的最大 CPU／内存。load、preprocess、predict、postprocess、validate_output 五阶段都须成功；错误、验证预算耗尽和资源限制均阻止正式矩阵，CPU + GPU 必须两者通过。stdout/stderr 保存在 `runtime_validation_off/on.log`，超时也清理验证容器。它们不是 warmup、测量行或 profiler 结果。验证前已有的结果不因此变为本次成功结果。

每个设备的可选 `stages` 依次记录 `execution/load/preprocess/predict/completion/postprocess/validate_output/metadata`，
每项包含阶段名与 `verified/error`；错误保存原异常类型和消息。失败报告的 `failed_stage` 指向
失败步骤，读取输入或进入／退出推理上下文的异常为 `input_or_execution_context`。被终止或超时的
进程可能没有阶段回报，以外层状态和日志为准；未执行阶段和历史缺失字段不补为成功。
这些诊断没有时间单位，不用于比较阶段耗时，也不增加正式测量请求。

这些字段在 profiling 后原子补写，原始 `run_command` 保持不变。`static_flops` 只保存不依赖硬件计数器的 Torch 逻辑 shape FLOPs，并按 input scale 展开；NCU 实际执行 FLOPs、吞吐率以及 execution 数值分别保存在各自的 `profiling/*.csv`，execution 字段是否来自代表资源由上述 sampling metadata 和 plan entry provenance 说明。

## 质量与失败产物
`quality_checks.json` 为 `{"schema_version": 1, "checks": [...]}`。每个 check 包含
`code`、`severity`、`observed`、`threshold`、`detail`、`evidence`，与 Capability 分开：

| code | 观察值与边界 |
| --- | --- |
| `cpu_idle_baseline_unstable` / `gpu_idle_baseline_unstable` | 该 case 的空闲功率 `(max-min)/mean`，无量纲；保存原始 W 样本与现有阈值，不改变能耗公式或采样窗口 |
| `weights_reinitialized` | Transformers loading info 的 missing/mismatched keys；threshold 为 0 个 key |
| `unused_checkpoint_weights` | loading info 的 unexpected keys；保留 loader 来源，不根据告警文本归类 |
| `missing_cli_exit_code` | process return code 为 null；threshold 表示需要进程退出证据 |

加载质量在独立 probe 的 `devices.<device>.quality_checks` 中保存，case 能耗质量在窗口结束后
写入 case sidecar。资源矩阵退出时（包括中途失败或取消）保存根目录 `quality_checks.json`；失败的 probe 仍可由 audit 直接读取。
quality 数据不进入 capability evidence；普通 warning 不改变 `full_profile_complete=true`。
兼容性报告据此区分 `full_success` 与 `full_success_with_warnings`。
审计和质量消费者只读取不超过 4 MiB 的 UTF-8 JSON 对象，并拒绝 NaN、Infinity 和溢出为无穷值的数字。
损坏的运行元数据记为 `invalid_metadata` error；损坏的 canonical quality evidence 记为
`quality_evidence_invalid` warning，同时保持 `quality_status=unknown` 且禁止自动选择。只读检查不重写原始证据。

typed `failure` 包含 `stage`、`reason_code`、`detail`、`device`、`runtime_profile`、
`retryability`、`evidence`、`exception_type`。`reason_code` 包括 task、dependency、precision、
contract、artifact、processor、access、initialization、inference 与 timeout/resource 原因；
稳定代码定义在 [`failures.py`](../../acprof/failures.py)。原始异常链和日志用于诊断，不作为报告分类输入。
CLI 使用 `ACPROF_FAILURE=` JSON 记录，TUI 按该结构展示；HTTP 错误响应携带同一 failure，
client 在请求及 monitor 结束后写入 case sidecar，资源矩阵退出时汇总为根目录 `runtime_failures.json`。
非零 client 退出优先保留结构化原因；恢复运行时，旧 case 失败随该次尝试归档，根侧车只反映当前 case 证据。
两个新增根侧车不更改既有 Artifact Layout v2 manifest 身份或历史 CSV 的列和数值。

timeout evidence 包含 `timeout_seconds`（秒）、`request_phase`、`request_id`、`input_scale`、
`model_loaded`、`service_alive` 与 `timeout_scope`。正式 HTTP 请求沿用 connect/read inactivity
超时语义；兼容性验证沿用整个子进程预算。未观察的布尔字段为 null，不能补成 true 或 false。
`request_timeout`、`compatibility_budget_exhausted` 和只表示已有报告产物损坏的
`recorded_evidence_invalid` 在兼容性报告中为 `inconclusive`；后者不覆盖同目录已有的有效 typed failure。
预算筛查或实测资源限制为 `unverified`，用 `evidence.measured_oom` 区分是否有真实 OOM 证据。
提高预算需显式重试并使用新目录，不覆盖原始预算与失败证据。

历史结果缺少这些字段时保持 unknown；不能从 Capability 推导质量良好，也不回读自然语言日志
伪造结构化告警。`coverage report` 对已有 full 结果但缺少质量记录的条目展示
`full_success_quality_unknown`。2026-09-27 旧审计的 140 个 warning 项没有在本次改动中被重写或重新实测；
新运行会直接产生可消费的结构化原因。

## `collection_history.json` 字段
| 字段 | 含义 |
| --- | --- |
| `schema_version` | `collection_history.json` schema 版本，当前为 `1`。 |
| `posthoc_profile_history` | `acprof profile` 事后补采记录，包括工具、采样策略、完成时间与备份位置。 |
| `timeout_retry_history` | 请求超时后的重采/合并记录。当前仓库没有自动生成该记录的入口，保留独立历史文件中的已有记录。 |
| `quality_retry_history` | 质量检查后的定向重采/合并记录。当前仓库没有自动生成该记录的入口，保留独立历史文件中的已有记录。 |
| `static_meta_backfill_history` | 对历史结果补充静态元数据时的来源、字段、备份位置及无法回溯的字段。 |

不再重复保存 `posthoc_profile_last_run`、`timeout_retry_last_run` 或 `quality_retry_last_run`；需要最新记录时读取对应 `*_history[-1]`。`static_meta.json` 中的 history/last-run 字段会被拒绝，不再执行迁移。

`acprof profile` 读取 `input_scale_plan.json`、`collection_history.json` 及可复用 profiler plan 时，
每份最多接收 4 MiB 的 UTF-8 JSON object，并拒绝非有限数字。必需输入损坏时保留现有
`PosthocError` 严格失败；可选复用计划损坏时只视为不可复用，不改写历史文件，也不把损坏证据用于补采。

## 最大输入探测结果
`acprof probe` 每次生成独立的 `input_scale_plan.json` 和 `largest_scale_probe.json`。
后者当前为 schema v3，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `memory_probe.candidate_order_gb` | 去重、升序排列的候选内存上限。 |
| `memory_probe.attempts` | 每档的 `startup_oom`、`runtime_oom`、`cuda_oom`、`timeout`、`error` 或 `ok` 结果，以及错误与分段计时。 |
| `memory_probe.minimum_viable_mem_gb` | 完整返回且有效尺度与计划一致的第一个候选；没有成功档时为空。 |
| `timing.request_s` | 成功档 `/predict` 的 host 端到端耗时。 |
| `cold_start.total_s` | 成功档全新容器启动到 ready 的耗时。 |
| `timing.ready_plus_request_s` | 上述冷启动与请求耗时之和。 |
| `timing.command_s` | 包含预检、检测、可选构建、输入规划、失败候选和清理的整条命令耗时。 |
| `timing.request_timeout_s` | 默认无限等待时为 `null`；显式设置 `--timeout-seconds` 时为对应秒数。旧 schema v2 始终记录有限值。 |

探测不会写入或修改 case 中间 CSV、`result_layers.json`、`static_meta.json` 或
`collection_history.json`，结果不包含 idle、能耗或网络测量。用法见[最大输入探测](../usage/experiments.md#先探测最大输入)。
