# 扩展与执行约定

[← 返回专题目录](runtime.md)

## 扩展声明与按需加载

`acprof/extensions/*/manifest.json` 是共享声明目录，标准库 JSON loader 在主机与容器读取，
不导入 Handler 或可选依赖。每条声明包含 ID、任务族、任务／架构匹配、backend/runtime、
profile/environment 引用、CPU/CUDA、dtype、输入模态、batch、streaming、measurement 状态、
Handler 和 validator 的 `module:callable` 入口。可选 `execution_entrypoint` 提供运行时上下文；
空值采用 CPU/nullcontext，Torch 声明指向 `container.torch_execution`。
声明只描述能力，实际执行仍由 `BaseHandler.load → preprocess → predict → postprocess` 完成。

内置 profile 的环境变体由声明的 `environments` 映射选择。已有架构的新 checkpoint 通常只需模型配置；
新的架构在已有任务协议下增加 Handler／manifest／validator；新 backend 再增加完整依赖环境和锁。
不同任务协议仍需相应 workload 与输出描述；声明不能让不兼容模型自动变为兼容。
`RuntimeProfile`、`DependencyEnvironment`、模型 snapshot 和执行 Handler 各有自己的身份。

注册默认拒绝重复 `(family, backend)`／adapter key；错误包含原、新实现和来源模块。
`register(..., override=True)` 才允许替换。`_auto_register()` 只登记入口字符串，选中时才导入和构造。
未选 backend 缺依赖不影响启动；选中的 backend 在导入或模型加载失败时保留原始 exception chain，
分别报告未注册、依赖缺失、模块导入失败、Handler 初始化失败与不支持。

内部 extension manifest 使用 `schema_version: 2`，v1 和未知字段直接报错；不加载外部插件目录。
类型化字段与校验位于 [`extensions/schema.py`](../../acprof/extensions/schema.py)。
`acprof_model.json`、workload 清单和结果文件各自的 schema 版本保持独立。

`ExtensionCatalog.resolve(task, family, library, config)` 是 backend/family 的共同解析入口，
返回选中的声明、设备精度和 IO 格式；`describe(task_info)` 用于已经选定 backend 的消费端。
显式 backend 优先，并校验 task/family/backend 路由；未知 library 不再默认进入 Transformers。
检测阶段保留未决元数据供 CLI 覆盖或契约审阅，候选解析将错误记录为 `needs_configuration`，
预检在构建或采集前拒绝未解决的组合。

`backend_rules` 的选择顺序为：制品布局证据（`artifact: true`）→ task 规则 →
library/config 规则 → family 规则，随后才可使用已声明的 `library_backends` 映射。
每个层级内按整数 `priority` 从高到低选择；同优先级命中不同 backend 时报歧义，文件顺序不决定结果。
`when_config` 的 `true` 表示字段存在且非空，其他值要求相等。Chronos 的配置标记及缺失 task
的推断由 `inferred_task` 声明；没有 task 时才使用 model type 和 architecture 推断，
不会用通用架构后缀覆盖显式任务。单用途 `model_type_tasks` 优先于后缀匹配，
多任务模型保留具体 architecture 证据；缺少 model type 的快照仍可使用架构映射。

重复的 family 行为集中在顶层 `families`，单条 extension 可覆盖；两遍加载使默认值不依赖文件顺序。

| 声明字段 | 消费行为 |
| --- | --- |
| `scaling` / `task_params` | 生成 `SCALING_DIMENSIONS` / `DEFAULT_TASK_PARAMS`；同 family 的共享值必须一致 |
| `io_format` / `task_io_format` | `/predict` 的输入输出模板及任务差异；对象递归合并，数组替换，`null` 删除键；返回副本 |
| `precision_policy` | 与 profile 合并的运行 dtype 策略；任务、模型类型、设备覆盖和有证据的排除规则共用于预检与 loader |
| `precision` / `backend_precision` / `task_precision` | 用于 ONNX/TorchScript/skops 等制品定义精度的描述；浮点模型加载由 `precision_policy` 决定，不再用静态 BF16 特例覆盖 |
| `trust_remote_code` | 可收紧 profile 的授权，false 不能被 loader options 提升；未声明时遵循 profile |
| `input_plan` / `task_input_plan` | workload 清单、文本/音频探测、上下文上限、workload 默认尺度、连续尺度及 feature dimension 传递能力 |
| `handler_options` / `backend_handler_options` / `task_handler_options` | 依次合并 handler 的 loader 选项、句向量模式、模型清单格式与张量输入方式；不改变四阶段执行协议 |
| `workload_defaults` / `profile_options` | adapter 的提示词、生成参数和 profile dtype/remote-code 策略；MOSS 特有值只在其声明中维护 |

`input_plan.requires_scale_meta` 使用模型返回的上下文长度上限；`max_scale_probe` 允许 generator
提供最大尺度。`text_payload`、`audio_payload` 选择已有的输入合法性探测；`workload_scales`
消费 generator 的默认尺度，`fractional_scales` 允许连续尺度。未声明的能力默认为关闭，
不能通过字段缺省绕过必须的输入验证。

平台与共享依赖分别在 `platforms`、`dependency_environments` 声明，逻辑配置在 `runtime_profiles`
或 extension 的 `profile` / `environment` / `dependency_environment` 声明。
[`runtime_profiles.py`](../../acprof/runtime_profiles.py) 使用既有 `PlatformSpec`、`DependencyEnvironment`、
`RuntimeSpec` 和 `RuntimeProfile` 实例化这些引用并继续检查完整锁。
新 runtime 可引用 `torch_version: null` 的平台以及独立的精确 lock，无需增加 runtime 名称分支。
这条声明路径的主机测试使用虚构 runtime；它不代表已实现或验证 OpenVINO 等额外推理后端。

实现参考 [Transformers PipelineRegistry](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py)
的集中任务注册、[任务别名去重讨论](https://github.com/huggingface/transformers/issues/7666)，以及
[vLLM ModelRegistry](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py)
的延迟解析与显式失败边界。采用这些组织方式，继续复用本项目的标准库 catalog 和 `BaseHandler`，
不复制上游执行框架，也不新增依赖或正式测量阶段的处理步骤。

Workload 也从同一 manifest 的可选 `workload_entrypoint: "module:Class"` 读取。
入口按 family 共享，同一任务换 backend 不复制样本生成器。声明发现和列表读取只使用标准库；
选中 family 时才导入实现。重复发现相同入口及其旧式模块自注册是幂等操作，不同实现争用同一
family 则报告双方来源；模块导入失败后不会留下可被误用的半注册结果。
`register_generator`、`get_generator` 和旧构造参数继续可用；实现可覆盖 `from_config` 消费
adapter 或任务参数，不再在 `get_generator` 增加任务名判断。未注册、依赖缺失、导入失败、
配置错误和初始化失败分别报告，原异常保留为 `__cause__`。

`onnxruntime-cpu` 支持单个 float32 `[rows, feature_dim]` 输入及一个 dense numeric tensor 输出，
任务为 `tabular-classification`／`tabular-regression`。可使用 snapshot 中唯一的 `*.onnx`，
或 `acprof_model.json` 指定 `schema_version=1`、`format=onnxruntime`、`task`、`model_file` 和可选 `feature_dim`。
表格适配仍拒绝多个输入／输出、非 float32 输入及不匹配的固定 batch，不逐行拆开请求冒充模型 batch。
ORT 固定为 1.23.2 以匹配 Python 3.10，运行环境不安装 Torch。完整 wheel 与系统制品仍用现有锁和严格包集校验。

`container.onnx_session` 共用 Session 配置、具名输入输出、dtype/shape 与 Provider 检查；
业务语义在各 Handler。新图像／文本适配要求 `acprof_model.json` 显式声明预处理与所选输出：

| 任务 | 已支持的输入与限制 | 声明 |
| --- | --- | --- |
| 图像分类 | float32 NCHW，RGB 或 L，batch=1；复用 CV 的原始图片与顺序 | `image_processing` 包含 `input_name`、`layout: "NCHW"`、`mode`、`rescale_factor`，可选逐通道 `mean/std`；改变尺寸必须显式提供 `resize: {width, height, resample}`，resample 为 nearest 或 bilinear。 |
| 文本分类 | int64 `[batch, sequence]`；`input_ids` 与可选 `attention_mask/token_type_ids`；显式等长样本列表可组成 batch，标量文本只接受 batch=1 | `tokenizer` 包含本地 `file`、`max_length`、`padding: "none"`、`truncation: "reject"`、布尔 `add_special_tokens`。 |

分类输出是 float32 `[batch, classes_or_scores]`。多输出模型必须用 `output_name` 明确选择分类
分数，所有输出仍参加独立签名验证；单列分数不推造标签、概率或阈值。不支持任意 ONNX 图、
外置 tensor data、GPU、隐式样本复制、padding 或 truncation。固定形状不匹配明确拒绝。
模型可增加 `artifact_sha256`；完整哈希核验在独立输出验证中完成，不计入服务启动或请求时间。

文本自动尺度规划复用既有 NLP 二分探测。超过已声明 token 上限时，预处理抛出携带实际长度的
`InputLimitError`；仅 `/probe` 将其转为 `limit_exceeded: true`，供规划器缩小候选范围。
`truncated_by_limit` 仍为 false，因为没有截断输入；正式 `/predict` 继续拒绝超限请求。
其它配置和预处理错误仍传播为失败，旧 probe 缺少新字段时按 false 读取。

本地可复现的预训练示例位于 [`real_models.py`](../../examples/onnxruntime/real_models.py)：
MNIST-12 为 26 KB、固定单灰度图输入；BERT-tiny-RAID 为 17.6 MB、三个具名整数输入及单分数输出。
准备阶段固定上游 revision、逐文件 SHA256 和来源，未在本机导出或量化，上游未提供的转换工具／
参数记为 unknown。MNIST 模型卡元数据标 Apache-2.0、正文标 MIT，示例原样记录两项；BERT 模型卡为 MIT。
运行检查使用同一任务的既有生成器；单样例与 ONNX ReferenceEvaluator 的数值比较不等于分类准确率评测。
执行步骤见[测试指南](../development/ci.md#无-torch-运行时验收)。

构建期与测量前的运行时验证继续独立于正式窗口。`validate_output` 可由 manifest 声明或 Handler override
提供，覆盖协议与少量任务 sanity；它不证明准确率或全部 profiler 兼容。实际工作量见
[Workload Contract](../profiling/measurement.md#workload-contract)。

### 运行参数与请求完成

`acprof.runtime_settings` 是标准库配置读取入口。CPU quota、CPU affinity 和推理线程数独立；
新增线程参数不会修改 `--cpus` 或 cpuset。现有 profiler 的 quota 派生线程默认值保留，显式请求
同时传入服务、独立验证与 profiler，并在执行前的恢复身份中记录。未设置新参数时不添加新的
空环境键，运行后观测值不参与恢复身份。

| 设置 | 优先级与默认值 |
| --- | --- |
| `ACPROF_RUNTIME_THREADS` | 通用线程请求，优先于旧 `TORCH_NUM_THREADS`；Torch 服务未设置时沿用运行时默认。 |
| `ACPROF_ONNX_INTRA_OP_THREADS` | ORT 专用，优先于通用设置、旧 Torch 名称和默认 1；显式值必须为正整数。通用或旧变量的 0 保持 ORT 原有的 1 线程语义。 |
| `ACPROF_ONNX_INTER_OP_THREADS` | ORT inter-op 线程数，默认 1，必须为正整数；大于 1 时启用 `ORT_PARALLEL`，否则使用 `ORT_SEQUENTIAL`。 |
| `ACPROF_ONNX_PROVIDERS` | 当前仅接受 `CPUExecutionProvider`；不允许隐式回退，CUDA 或混合列表明确报错。 |
| `ACPROF_REQUEST_TIMEOUT_S` | 显式设置优先；未设置时正式 case 继承 CLI 请求超时，独立验证继承验证时限。直接启动服务默认 300 秒；`none` 表示完成 hook 不设截止时间，其他值必须有限且大于零。 |

独立验证及 `/meta` 的 `runtime_parameters` 区分 requested、effective 和来源，实际启用的
Provider 从 Session 读取，线程数与执行模式从 `get_session_options()` 读取。CPU 路线不提供 GPU 算子位置证据；未来仅列出 CUDA Provider
也不能证明全图在 GPU 执行。旧结果缺参数时为 unknown，不推算线程、Provider 或制品哈希。

四阶段接口不变。已有 execution 模块可增加
`wait_for_completion(model_ctx, output, *, timeout_s)`，在 `predict` 之后、`postprocess`
之前返回已完成的原始输出；服务、独立验证和 profiler 共用此调用。同步 ORT 的 `run` 返回后
无需额外等待；没有 hook 的旧模块承诺同步返回，未解析的 Future/awaitable 会被拒绝。
异步实现负责等待本请求、传播后台失败并执行超时，不可只返回提交句柄。超时不会成为成功响应，
但不保证已取消后台计算；client 原有超时和失败处理继续适用。
原来允许无请求时限的最大尺度探测继续传递 `none`；该值不取消 client 自己的 HTTP 超时。
Compute/Execution Profiler 保持原来的无请求截止时间，避免分析器放大运行时间后意外触发
服务的 300 秒默认值；显式 `ACPROF_REQUEST_TIMEOUT_S` 仍可为 profiler 设置完成等待预算。

Torch CUDA hook 等待当前请求所在 stream 的 event，不增加全设备同步；使用额外 stream 的
Handler 必须先汇合到当前 stream，或声明自己的完成 hook。等待计入原推理窗口，输出验证仍
在独立进程、正式测量窗口之外。CPU 和可控异步替身测试不能代替真实 CUDA 验收。

## MOSS 的执行约定

MOSS 的固定 snapshot 包含官方自定义模型、processor 和配置，adapter 仅在容器中以
`local_files_only=True` 加载这些代码。使用官方提示词和音频参数；processor 自行分块，
保留 `audio_feature_lengths` 和 `audio_chunk_mapping`，不因单块长度而截断整个请求。
同架构的其它 checkpoint 可沿用此 adapter，输入计划会传递所选 adapter 的默认提示词。

CPU 使用 FP32，GPU 使用 BF16；常规推理使用 SDPA，Torch FLOPs 的独立分析按现有约定
请求 eager attention。音频解码、特征计算和张量搬运在 `preprocess`，生成在 `predict`，
文本解码在 `postprocess`。输出沿用文本响应 schema，保留时间戳与说话人标记；不额外宣称
分段准确率或说话人识别准确率。默认 `max_new_tokens=512`，可通过 workload 清单调整。
