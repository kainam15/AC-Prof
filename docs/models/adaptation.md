# 模型制品与适配

[← 返回专题目录](runtime.md)

## 模型文件选择规则

模型文件默认采用 `--model-download-policy auto`。程序在主机准备阶段读取固定 commit 的文件清单、配置和分片索引，按照目标 runtime 的加载器选择权重：标准 Transformers 优先默认 safetensors（含分片），否则保留默认 PyTorch `.bin`；Sentence Transformers 保留模块结构；已覆盖的 Stable Diffusion／SDXL／DDPM／DDIM pipeline 按组件选择；TorchScript／skops 遵循现有 artifact 清单。配置、tokenizer、processor 和其它未确认可省略的附属文件会保留。分片缺失直接报错，不静默换一套权重。

DDPM/DDIM 还支持原生根目录布局：组件目录不存在时，`UNet2DModel` 使用根目录的
`config.json` 和默认权重，`DDPMScheduler`/`DDIMScheduler` 使用 `scheduler_config.json`。
已有组件目录优先；缺少所需配置、权重或分片仍报错。筛选保留选中的根目录 safetensors，
不会将其当作冗余单文件删掉。此规则对齐锁定版本
[Diffusers 的组件加载](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/pipeline_loading_utils.py)，
不为未知 pipeline 猜测组件。离线随机 DDPM 已验证筛选后的根目录及子目录快照可加载并生成图像。

自定义 adapter、`auto_map`、量化配置、未知模型类型或未覆盖的 pipeline 使用完整快照，并打印回退原因。GPU 推理 dtype 不用于选择文件名中的 FP16／FP32 variant；不会自动转换、量化权重或切换 EMA checkpoint。需要完整仓库时，`acprof run` 和 `acprof probe` 均可传入 `--model-download-policy full`。TUI 使用默认 `auto`；两种策略具有不同的镜像指纹。

共享环境层不包含 AC-Prof 业务代码或模型。默认构建使用带完整依赖锁的 runtime 镜像，
再构建不含权重的模型清单和最终代码层。模型清单层不再包含权重，指纹绑定真实环境 image ID、模型 commit、backend、adapter、
下载策略、筛选器及已验证计划 SHA256；最终层复制 AC-Prof 代码。修改界面或 handler 可以复用依赖与模型层，
修改筛选规则只重建模型及最终层。Python 和系统依赖均消费锁，补采仍需保留原始 image ID。

主机 NVML 依赖直接使用 NVIDIA 的 `nvidia-ml-py`，Python 导入名仍是 `pynvml`。
已删除的同名 `pynvml` 发行包由[上游标记为弃用](https://github.com/gpuopenanalytics/pynvml#readme)；
这项替换不增加采集步骤或测量开销。

Model Store 的 `entries/<id>/model_download_plan.json` 保存所选文件、排除文件、选择原因、框架版本、文件 SHA256 和清单 SHA256。文件大小／内容检查在构建阶段执行，清单写入 `static_meta.json/runtime_environment/model_download`；正式 server 启动不会再次扫描、下载或校验全部权重。mounted 模型的 `model_cache_bytes` 是 `model_artifact_bytes` 的兼容别名；`docker_image_bytes` 只包含 runtime、代码与清单层；判断磁盘节省应查看 `docker system df -v` 的共享／独占占用。保留旧镜像时，它引用的大层仍会占用空间。

声明离线依赖时，下载计划另含 `dependencies[].download` 子清单和 `total_selected_bytes`；
原 `selected_bytes` 仍只统计主 snapshot，新增总量包括主模型和依赖。子清单分别固定 commit、
文件大小和 SHA256，并纳入父清单哈希；没有依赖的历史 v1 清单继续有效。

实现参考 [Hugging Face Hub 0.36.2 文件筛选](https://github.com/huggingface/huggingface_hub/blob/v0.36.2/src/huggingface_hub/_snapshot_download.py)、[Transformers 4.57.6 权重解析](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py)、[Diffusers 0.39.0 组件下载](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/pipeline_utils.py)（Apache-2.0）和 [Docker 分层缓存](https://docs.docker.com/build/cache/optimize/)。复用现有 Hub 下载和重试机制，以标准库实现有边界的文件规划；不绑定框架私有下载入口，不在主机新增推理框架依赖。

## 新增一个模型适配

本节维护扩展契约；执行步骤见[模型适配 Skill](../../.agents/skills/acprof-model-adaptation/SKILL.md)。

| 边界 | 契约与实现入口 |
| --- | --- |
| 环境路由 | `acprof/extensions/*/manifest.json` v2 声明 adapter、平台、依赖环境/profile、task、model_type 和 backend；[`runtime_profiles.py`](../../acprof/runtime_profiles.py) 从声明生成 `ENVIRONMENTS` / `PLATFORMS`，精确锁描述实际依赖。 |
| 任务支持 | `host/detect.py`、`host/task_support.py` 与 `config.py` 共用 manifest；新任务协议还须补充对应 workload 的物化与尺度处理，不能仅移除预检限制。 |
| 推理接口 | 已满足协议时使用 `family-default`；自定义实现由 manifest 的 `handler_entrypoint` 按需导入；程序注册 `register_adapter` 仍可用，重复 key 默认拒绝，覆盖必须显式 `override=True`。 |
| 输入输出 | `BaseHandler` 保留四阶段接口，增加仅在窗口外调用的 `validate_output`；模型提示词、参数和尺度经 workload/输入计划传递，输出与 `host/model_schema.py` 一致。 |
| 依赖和验证 | 精确锁对应目标 Python/CUDA 容器并通过 `pip check`；CPU、GPU、dtype 与 profiler 分别声明支持，普通推理验证不证明工具兼容。 |

`ARCHITECTURE_PROFILES` / `MODEL_PROFILES` 兼容视图已移除，旧导入明确失败。
开发扩展在 manifest 维护。运行时选择同时匹配 architecture、family、backend 与 task，
同一 architecture 的多个 backend 不再依靠全局单键覆盖决定。

当前 loader 的设备、模态和 profiler 边界在下方任务章节维护；新的行为须同时满足[采集协议](../profiling/measurement.md#协议不变量)。

## 参考实现与取舍

### 2026-09-27 冻结样本的 ecosystem 取舍

2026-10-02 只读复核 `internal-testing/hf-top10-all-tasks-20260927/models.csv`：520 项中
113 项为 library 未登记，涉及 50 种原始标签（区分大小写）。这是历史样本统计，不是本次重新实测。
主要集中度与实现取舍如下；成本为设计评估，尚未新增或宣称支持这些 backend。

| 原始 library 标签 | 条目数 | 取舍 |
| --- | ---: | --- |
| `gguf` | 11 | 覆盖最多，但须新增原生执行环境、量化语义及测量边界；不通过 generic Transformers fallback 加载 |
| `minimax-h3` | 9 | 先核实实际架构、权重选择与资源可测性；library 标签不足以确定共享接口 |
| `transformers.js` | 5 | 需要 JS/ONNX 执行与输入输出契约，当前 Python 路径不能直接替代 |
| `colpali` / `colpali_engine` | 5 / 1 | 可复用部分 Torch 依赖；有多模态检索科研价值，但须定义 late-interaction 输入、输出和计量边界 |
| `trellis` | 5 | 3D/GPU 依赖及专用预后处理成本较高，待明确实验问题后实现 |
| `depth-anything-3`、`anemoi`、`PaddleOCR`、`stable-baselines3`、`lerobot` | 各 4 | 各自需要模型或环境契约；Paddle、RL、robotics 不属于同一个通用 backend |
| `lightgbm` | 2 | 可作为较轻量的后续 structured adapter 候选；仍需固定特征契约、模型制品与精确依赖锁 |
| 其余标签 | 55 | 长尾逐项按科研价值和验证成本筛选；`mlx` 仅 1 项且与当前 Linux 测量平台不同 |

本轮决定不新增 ecosystem。后续每个新增项必须具备 extension manifest、locked environment、
handler contract 和独立真实推理证据，不能仅以提高通过率为目标。
同一旧样本中的 42 个资源筛查项继续为 `resource_unverified`；预算筛查不是实测 OOM。
新 coverage 的下载预算使用包含显式依赖的 selected artifact plan，参数量只作为 conservative
preflight evidence；超过预算或缺少必要大小证据时保持 `unverified`，并记录为什么没有测。

候选解析参考 [vLLM 模型 registry](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py)
（Apache-2.0）的延迟入口、显式选择、独立检查和有条件回退；
[worker 注册问题 #16228](https://github.com/vllm-project/vllm/issues/16228) 提醒解析结果必须传入执行进程。
AC-Prof 沿用现有 manifest／Handler 和独立 Docker 验证，将有效声明固化到服务镜像，
不引入 vLLM 的推理调度、CUDA 依赖或第二套 adapter 框架。自定义 pipeline 桥接直接使用
[Transformers pipeline 工厂](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/__init__.py)
的 `custom_pipelines/auto_map` 协议；上游维护中的接口仍以本项目锁定版本和实际验证为准。

共享模型接口直接复用官方 [Transformers Auto 注册表](https://github.com/huggingface/transformers/blob/v5.6.0/src/transformers/models/auto/modeling_auto.py)、
[timm wrapper](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/timm_wrapper/configuration_timm_wrapper.py)、
[Chronos 基类分派](https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/src/chronos/base.py) 与
[SentenceTransformer 模块加载和 encode](https://github.com/huggingface/sentence-transformers/blob/v5.1.2/sentence_transformers/SentenceTransformer.py)。
这些上游持续维护，许可均为 Apache-2.0；保留固定版本，模型权重另按仓库许可。
不复制各 checkpoint 的推理脚本；新增依赖为 CV 的 timm 和独立的 Transformers 5.6.0 共享环境。
新版共享环境还锁定 [OpenCV headless](https://github.com/opencv/opencv-python) 4.13.0.92，供原生
图像 processor 的轮廓／多边形处理使用；只安装 headless 包，不引入 GUI 依赖。OpenCV 采用
Apache-2.0，Python 打包工具为 MIT；目标 wheel 约 60 MB，沿用 NumPy 2.2.6，不升级其余锁定包。
已有采集器不按模型分支。元数据解析、环境选择与下载在测量前完成；prompt、Pooling、Normalize
和必要的预测输出整理属于实际请求工作，不从测量中扣除。

本次增量核查了 [Optimum Benchmark 的配置与依赖](https://github.com/huggingface/optimum-benchmark/blob/main/pyproject.toml)
（Apache-2.0、Python 3.10+）：采用后端配置与实验报告分工，不引入其 Transformers、Accelerate、
Hydra、datasets 等依赖；项目自述仍为 WIP，不能据此替代本仓库实测。
[Pluggy 注册实现](https://github.com/pytest-dev/pluggy/blob/main/src/pluggy/_manager.py)（MIT）用于参考
重复身份与冲突诊断，继续使用本地 manifest 和标准库 registry，无插件框架依赖。
[ORT Session/Provider API](https://onnxruntime.ai/docs/api/python/api_summary.html) 及
[线程约定](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)直接用于当前固定 ORT 版本，
任务处理只新增已锁定 Pillow/Tokenizers。发现、哈希、参考验证与比较均在测量窗口外；请求完成
等待属于必要执行时间，不把等待或后台失败排除以获得更短延迟。

本轮扩展机制参考 [pluggy](https://github.com/pytest-dev/pluggy) 的显式注册冲突检测（MIT）、
[vLLM](https://github.com/vllm-project/vllm/blob/main/docs/contributing/model/registration.md) 的字符串入口延迟加载
（Apache-2.0），使用 [ONNX Runtime 官方 CPU API](https://onnxruntime.ai/docs/api/python/api_summary.html)（MIT）。
这些上游有持续维护；这里只借鉴模式，不复制大块代码，不引入 pluggy/vLLM 运行依赖，也不扫描模型安装插件。
声明解析、导入和完整验证在测量前完成。每请求的实际工作量摘要属于当前响应协议，有小量固定序列化成本；
不把它当作零开销，也不重算或重新发送推理请求。

依赖解析继续使用 [uv](https://github.com/astral-sh/uv) 0.12.13（MIT / Apache-2.0），
复用其目标平台解析和制品哈希；分层缓存复用 [BuildKit](https://github.com/moby/buildkit)
（Apache-2.0）。两者持续维护，不新增服务运行依赖。系统来源采用
[Debian Snapshot](https://snapshot.debian.org/)，显式锁更新时由 APT 验证签名索引，普通构建
只消费锁定制品。新增逻辑限定于声明、编排和构建校验，不进入正式测量窗口。

构建配置参考 [Cog 的环境声明](https://github.com/replicate/cog/blob/main/docs/yaml.md) 和
[BentoML 的构建配置](https://github.com/bentoml/BentoML/blob/main/src/bentoml/_internal/bento/build_config.py)，
两者为 Apache-2.0。它们服务于各自的打包／服务框架；这里复用独立环境、依赖锁和内容缓存的
做法，沿用 AC-Prof 的 Docker 和四阶段 handler，无需引入新的服务框架或改变测量窗口。

MOSS 直接使用 [OpenMOSS 官方实现](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)，
默认提示词和调用约定参考其 `inference_utils.py`，许可为 Apache-2.0；模型代码与权重按
同一 snapshot 固定。该自定义接口与 Transformers 主版本相关，升级须重新锁依赖和验证。
所有安装、环境清单生成和接口验证都在正式测量窗口外完成。
