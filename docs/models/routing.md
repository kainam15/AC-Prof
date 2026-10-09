# 接口解析与 Runtime 预检

[← 返回专题目录](runtime.md)

## 共享接口解析
主机在同一个模型 commit 读取文件列表及 `config.json`、`model_index.json`、`modules.json`、
`adapter_config.json`、`acprof_model.json`、`generation_config.json` 和 tokenizer/processor 配置，不执行仓库 Python。
即使 Hub 没有 `pipeline_tag`，仍保留 revision、文件与元数据，并从 architecture、`auto_map`、
`custom_pipelines` 和制品格式汇总候选。选择优先级为显式任务、本地／仓库声明、Hub 任务，
最后才使用唯一推导候选；多候选为 `ambiguous`，信息不足为 `needs_configuration`。
Hub 与模型声明的任务冲突必须显式选择；底层 loader 标签按下节规则分类。
`--task`／`--backend` 不能与有效模型声明矛盾。
ONNX 文件本身只能证明格式，不能凭输入 shape 猜分类、回归或预处理；这些语义须显式补充。

`model_resolution` 记录候选、证据、冲突、缺失项、选择结果以及格式、loader、operation、
`model_type`、元数据文件名与所选 profile。`status=candidate` 只表示通过静态检查，成功加载、
输入 dtype/shape 和真实输出仍由独立 `runtime_validation` 判断。元数据读取失败单独报告，
不能归因为任务不支持。
模型 ID 先经过本地格式校验；`model` 和 `namespace/model` 均合法，存在性由 Hub 元数据查询确认。
主仓库查找失败通过 `ModelLookupError` 保留模型 ID、revision、错误类别、诊断及原始异常链，业务层不退出进程。
仓库未找到与私有仓库不可访问可能返回相同错误，因此提示“未找到或无权访问”；明确的 gated/权限拒绝、
revision 不存在、离线模式、网络连接/超时、服务限流/故障分别处理，不通过错误文字猜测原因。
明确的仓库、权限或 revision 错误不继续配置回退，也不能用手动任务覆盖；临时网络失败仍可使用固定 SHA
的缓存配置恢复。已读取元数据但缺少任务的模型继续保留未决证据，不归为仓库不存在。
分类复用当前锁定的 Hugging Face SDK [异常类型](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/errors.py)
与 [ID 校验](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_validators.py)
（官方维护，Apache-2.0），不复制下载实现、不增加依赖；所有检查仅在正式测量前执行。

镜像站 HEAD 缺少 Hub 元数据头时，客户端对同一 revision 回退到官方 Hub；认证、文件不存在和
离线缓存错误不会触发这项回退。

Transformers 版本选择使用 [`extensions/transformers`](../../acprof/extensions/transformers) 中从官方
固定版本导出的 Auto 注册表，按任务所需模型类与 `model_type` 匹配，不维护 checkpoint ID 白名单。
旧版本已经登记的架构继续使用旧环境；仅 `config.transformers_version` 较新不会强制升级。
CPU/CUDA 平台切换保留所选版本线。该候选选择目前面向 Transformers 原生 backend，
Sentence Transformers/CrossEncoder 继续使用其既有 4.57.6 环境；自定义 `auto_map` 由扩展与容器验证。
Auto 中存在类不保证 processor、pipeline、dtype 或所有 profiler 兼容。

## 解析证据与自动裁决
所有模型的 `model_resolution` schema v1 增加以下可选字段；历史报告缺字段表示未知，
不能补写为已验证。可执行 `acprof_model.json` 的 schema 和职责保持独立。

| 字段 | 含义 |
| --- | --- |
| `provenance` | schema v1 的来源图、观察项、resolver 版本、选择理由和 `identity_sha256`；只包含静态决策，不包含运行结果或时间戳 |
| `semantics` | `explicit/declared/inferred/conflict/unresolved`；表示所选 workload 的依据，不证明作者意图或模型质量 |
| `benchmark_kind` | 当前为 `task_pipeline`，沿用现有任务和四阶段 handler 协议；没有实现 `model_forward` 自动兜底 |
| `execution` | backend、handler 调用入口、执行阶段与计时协议；不把 pipeline 与其内部的 generation 当成互斥等级 |
| `adapter_origin` | 内置 handler 为 `builtin`，声明的 repository Pipeline 为 `repository`；生成声明不等于生成 adapter |
| `runtime_validation` | 初始为 `not_run`，之后记录实际 mode、镜像、payload hash、设备及验证结果；与静态语义分开 |

解析器读取 Hub `transformers_info` 的任务、Auto class 与 processor 提示。Hub 字段共享同一
source，Hub 与仓库配置保留共同 snapshot 的派生关系，不按字段数量投票，也不输出未经校准的
数值置信度。缺少已有架构匹配时，反查固定 4.57.6 / 5.6.0 Auto 注册表；多个任务仍保持歧义。
裸 `AutoModel` 不证明应执行哪一种任务。共享同一 Auto loader 的 translation/summarization
等任务不会仅因名称不同被判为结构冲突。

当 `transformers_info` 同时返回 `auto_model=AutoModel`、`pipeline_tag=feature-extraction`，
且 config 中的 custom Pipeline 声明使用该通用 Auto loader 时，这个库标签只记为
`loader_hint`，不作为任务候选或与 Hub 主任务产生冲突。原始字段保留在来源证据中，
`inspect --explain` 会说明其分类理由；其他库任务标签、缺少匹配 Pipeline 的情形仍按原规则检查。
唯一 custom Pipeline 可以继续进入静态契约分析，多个 Pipeline 仍须选择；
loader hint 本身不能补全缺失的任务语义，也不能使未解决的输入或依赖通过预检。

Hub 主任务存在时，`transformers_info.pipeline_tag` 若恰好是其 `auto_model` 对应的通用库任务，
且两个任务共享已登记的加载操作，则只记作 `loader_hint`。例如 summarization/translation 的
`text2text-generation + AutoModelForSeq2SeqLM`，以及 text-ranking/zero-shot-classification 的
`text-classification + AutoModelForSequenceClassification`。image-to-text 也可接受共享视觉生成
Auto loader 的 image-text-to-text 标签，包括 Hub 的 `AutoModelForMultimodalLM` 提示；
最终可用版本仍由所选 runtime 的注册表检查，不因此升级或借用其它环境的支持资格。
缺少匹配 Auto class、明确 head 不相容，以及 summarization 与 translation 这类两个具体任务
的冲突仍须选择。解析报告保留原标签、分类理由和来源哈希，不按榜单任务强制覆盖。

Sentence Transformers 的 feature-extraction/sentence-similarity 路径，若 `modules.json` 明确
声明从根目录加载 `sentence_transformers.models.Transformer`，实际使用的是 `AutoModel`。
此时底层 checkpoint 的 MLM/CausalLM head 与库标签只描述存储架构，不阻止声明的编码操作；
缺少该模块声明或选择了其它 backend 时不应用此规则。模块、processor、依赖与实际输出仍需验证。

声明、Hub task、具有任务语义的 `transformers_info.pipeline_tag` 冲突，或原生模型的明确 head 与任务操作
不相容时，保留候选并 abstain。显式 `--task` 可以解决元数据冲突，原冲突进入
`provenance.overridden_conflicts`；它仍不能违背有效的模型声明。
Interface Probe 或 Runtime Validation 成功不能修改这些静态裁决。静态来源摘要进入服务镜像 request fingerprint，
模型 SHA、声明与依赖仍沿用已有模型层和服务层身份规则。

实现复用 [Hugging Face Hub 的 ModelInfo/TransformersInfo](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py)，
借鉴 [Optimum TasksManager](https://github.com/huggingface/optimum/blob/main/optimum/exporters/tasks.py)
集中维护映射的方式。两者为 Apache-2.0 项目；使用现有 Hub 依赖和锁定静态注册表，不引入
Optimum 或主机端推理依赖，也不在正式测量窗口执行来源分析。
自定义 Pipeline 的选择参考锁定版本
[Transformers 4.57.6 的 pipeline 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/__init__.py)
（Apache-2.0）：借鉴唯一接口优先的规则，继续使用本项目已有的受限 AST 分析和依赖检查，
不在主机执行 `trust_remote_code`，也不增加依赖或测量开销。
通用 loader 标签的分类同时参考固定版本
[Transformers pipeline registry](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/__init__.py)
和 [Sentence Transformers Transformer 模块](https://github.com/huggingface/sentence-transformers/blob/v5.1.2/sentence_transformers/models/Transformer.py)。
两者为 Apache-2.0；复用现有执行接口，不新增 backend 或放宽未知 library 的限制。

`audio-text-to-text` 的预检与容器加载共用 `model_resolution.audio_text_loader`，依据相同版本的
Auto 注册表选择 `AutoModelForSeq2SeqLM` 或 `AutoModelForImageTextToText`。对于组合模型，
可选择 config 中唯一已注册的多模态文本子模型；不会抽取普通语言模型而丢失音频编码器。
例如 Voxtral、Qwen2 Audio 使用 4.57.6 的共享 Seq2Seq Auto 接口；Qwen2.5 Omni 的文本子模型
通过 5.6.0 的 Auto 接口加载，不实例化 Talker。没有匹配或子模型选择有歧义时拒绝，
不轮流尝试加载模型类。原生架构的候选资格不再由音频模型名称名单决定。

更新版本时，用 [`export_transformers_support.py`](../../scripts/export_transformers_support.py) 对固定 tag 的
`src/transformers/models/auto/modeling_auto.py` 和 `src/transformers/pipelines/__init__.py`
执行受限 AST 解析，分别记录源码 URL/SHA256；后者导出 task registry 和 aliases。
新导出的 dynamic-module capabilities 初始为 `null`（unknown），不能从 Auto 注册表推断。
仅重新导出相同 schema、version、source URL 和 SHA256 时保留已有 capability 评审；来源变化后重新验收。
`source_sha256` 仍只表示 Auto 注册表来源，不是 dynamic loader 的验证证明。

```bash
.venv/bin/python scripts/export_transformers_support.py \
  --source /path/to/modeling_auto.py --version 5.6.0 \
  --pipeline-source /path/to/pipelines/__init__.py \
  --output acprof/extensions/transformers/5.6.0.json
```

必须联动精确依赖锁和实际容器验证；注册表也参与服务构建指纹。未匹配到已登记环境的原生架构
提前拒绝。已知独立 adapter 缺少 base、GGUF 或缺少 `model_index.json` 的 Diffusers 单文件／组件
仓库也提前拒绝；本阶段没有增加这些制品的加载器。pyannote、SB3、LeRobot 等生态不能仅凭
Hub task 标签当作 Transformers 模型加载。

## Runtime 预检与失败证据
预检先确定逻辑 `runtime_profile`，再使用该 profile 的精确 Transformers lock 检查 Auto 和
pipeline registry。硬件选择 CPU/CUDA 平台后，会对最终 lock 再检查一次；一个版本支持不能
替另一个版本提供支持证据。GLM-OCR 的 `image-to-text` 在所选 5.6.0 registry 中不存在，
因此在下载权重、构建镜像、启动容器前返回 `runtime_task_unsupported`。
容器加载时还核对实际安装的 Transformers 版本与 profile lock，版本不一致返回
`runtime_dependency_incompatible`。宿主机预检从项目资源目录读取 profile lock；推理镜像则读取
`/opt/acprof/requirements.lock`（环境镜像构建时拷贝并在最终镜像构建时校验 SHA256）。
推理镜像不包含 `dockerfiles/locks/`，镜像锁缺失时必须报错，不回退读取项目源文件。
自定义 pipeline 走明确注册的策略，不借用标准 task 注册资格。

`precision_policy` 由 profile 与 extension 合并，包含 `supported_dtypes`、`preferred_dtype`、
`device_overrides`、`task_overrides` 和 `model_type_overrides`。应用顺序为基础策略、设备覆盖、
task 覆盖、model type 覆盖；task/model type 内部也可声明设备覆盖。支持集合还必须满足
extension 的 `dtypes`。显式 loader `dtype` 不能越过支持集合或有证据的排除规则。
带版本、任务、设备和模型类型约束的 `rules` 可声明 `preferred_dtype`，覆盖匹配范围内的默认值；
显式请求仍有最高优先级，并接受同一支持集合与排除规则的检查。
策略不满足时返回 `precision_mismatch`，不会遍历其它 dtype 重试。

SAM/SAM2 目前只排除 `mask-generation + GPU + FP16 + Transformers 4.57.6`，依据是
2026-09-27 冻结审计中 sam-vit-base、sam2.1-hiera-tiny 的 NMS dtype 失败。
同一范围的默认值为 FP32，因为锁定版本的
[SAM processor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/sam/image_processing_sam.py)
将 NMS boxes 转为 FP32，scores 必须匹配；显式 FP16 继续提前拒绝。
这不改变其它版本、任务或模型的默认策略。离线随机小型 SAM 已验证 CPU/CUDA FP32 的
四阶段接口，不能外推为 sam-vit-base/huge 或全部 SAM2 checkpoint 已完成真实推理验收。
容器 probe 的 `dtype` 取自实际加载模型的浮点参数；无法观察时为 `unknown`，不从 CPU/GPU 名称猜测。
静态元数据中的精度是策略选择，实际 probe dtype 才是运行证据。混合精度模型列出观察到的
浮点参数类型，如 `mixed[torch.float16, torch.float32]`；原生 adapter 的局部精度约束仍需单独核验。

`dependency_preflight` 只分析固定 commit 的源码：从 `auto_map/custom_pipelines` 出发遍历本地
Python imports，最多 64 个文件、每文件 512 KiB；结合 tokenizer 配置及根目录
`requirements.txt`、`requirements-inference.txt`、`requirements-runtime.txt` 比较最终 runtime lock。
开发 requirements 不进入推理需求。包缺失、版本冲突、无法安全判断分别使用
`runtime_dependency_missing`、`runtime_dependency_incompatible`、`runtime_dependency_unknown`。
源码与 requirements 的 SHA256、模型 revision、lock 路径和比较结果保存在解析报告中。
动态 import、未知 import/distribution 映射、未固定 revision、URL/extras 等不能自动证明可用，
须补充受审阅的锁定环境。分析不 import 仓库代码、不执行安装命令，也不修改基础镜像。
裸相对 from-list 先核对包 `__init__.py` 的明确顶层绑定或重导出；无法证明时仍要求同名
子模块存在。条件、动态绑定以及 initializer 自引用均保持 unknown，不执行包代码来猜测结果。
RMBG 的 `skimage` 映射为 `scikit-image`；manga-ocr 的 MeCab tokenizer 明确要求 `fugashi`。
未指定版本的旧 `typing` backport 在目标 Python 3.5+ 中记录为 `stdlib`，保留 requirements
来源与目标 Python 版本；带版本限制的 `typing` 和 `typing-extensions` 仍按发行包检查。
这不消除 RMBG 对 `scikit-image` 的真实依赖缺口。

`trust_remote_code` 的有效值是 profile 允许且 extension 没有显式禁止。loader options 可进一步
收紧为 false，不能将 false 提升为 true。普通 family-default 遵循 profile；只有明确注册的
custom-code profile/adapter 才能开启，Interface Probe 也遵守同一策略。
TorchScript/graph/structured extension 的 `requires_model_spec` 在 resolver 消费：缺少
`acprof_model.json` 时返回 `needs_configuration` 和 `model_contract_required`，不猜输入语义。
已有 ONNX/skops 的安全格式推断规则保持独立，不推广到任意 TorchScript。

失败由 [`Failure`](../../acprof/failures.py) 统一描述，CLI、audit、`models.csv`、TUI 和 `REPORT.md`
直接传递 `reason_code`；异常链与原始日志保留用于诊断。`request_timeout` 表示请求期限耗尽，
`compatibility_budget_exhausted` 表示独立验证整体预算耗尽，均为 `inconclusive`。
已有结果中的兼容性 JSON 按单文件 4 MiB 上限读取，并要求对象结构及有限数值；损坏证据记录为
`recorded_evidence_invalid` 和 `inconclusive`。同一结果目录中的有效 typed failure 继续优先展示，
不会因另一份产物损坏而被覆盖，也不会把损坏产物解释成模型本身不兼容。
真实推理异常使用 `inference_failed`，不能仅因为 60 秒未完成就认定不兼容。
默认 timeout 仍为 300 秒；显式重试使用更高 `--timeout-seconds` 和新的输出目录，不自动循环。
质量警告独立于 Capability，字段及历史结果边界见[质量与失败产物](../profiling/metadata.md#质量与失败产物)。
HTTP 响应为 401/403 的失败在原阶段记录 `access_denied` 和 `http_status`，包括文件计划阶段的
gated 配置读取；不会因已有缓存配置可通过静态检查而宣称获得权重权限，也不将 429 当作权限失败。

实现参考锁定的 [Transformers 4.57.6 pipeline registry](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/__init__.py)
和 [5.6.0 registry](https://github.com/huggingface/transformers/blob/v5.6.0/src/transformers/pipelines/__init__.py)，
并使用两版本 `PreTrainedModel.from_pretrained(output_loading_info=True)` 的结构化 loading info
捕捉权重初始化与未使用权重（[4.57.6 源码](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py)、
[5.6.0 源码](https://github.com/huggingface/transformers/blob/v5.6.0/src/transformers/modeling_utils.py)）。
Chronos 沿用其 [`from_pretrained` 参数协议](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos.py)。
这些上游采用 Apache-2.0；AC-Prof 复用现有库与静态快照，不新增推理依赖。
依赖比较使用已有 lock 中的 `packaging==26.3`，现将其列为直接主机依赖；安装包集合不变。
快照随锁版本评审，加载观察在初始化完成后恢复原方法；依赖分析和质量持久化均在正式测量窗口外。
