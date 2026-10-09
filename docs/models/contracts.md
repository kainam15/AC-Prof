# 自动生成模型契约

[← 返回专题目录](runtime.md)

## 自动生成模型契约（M1～M6）
未提供本地或作者 `acprof_model.json` 时，`detect_task` 对声明了唯一 `custom_pipelines` 的
音频／图像／视频转文字任务收集证据，再生成兼容现有 schema v1 的 draft。
任务取自 Hub 或显式覆盖，Pipeline 取自 config；README 中 Python 示例的字典键仅作为补充证据。
已有作者声明保持优先，其任务／backend 与 Hub 或显式覆盖的冲突仍须解决。

主机只读取同一完整 commit SHA 下的 JSON、README 和 Python 文本。config fallback 必须取得
Hub cache 的固定 snapshot 后才分析内容。AST 沿声明文件和相对 import 读取，最多 32 个源文件、
总计 2 MiB、单文件 256 KiB、单个 AST 20,000 个节点；不会 import、eval 或执行模型代码。
结构化 JSON 的读取上限为 1 MiB；依赖索引保留 4 MiB 上限。已经固定完整 SHA 的静态文件
先通过 Hub `hf_hub_download(dry_run=True)` 查询大小和 commit，未知或超限大小、SHA 不一致
直接拒绝，不请求文件正文；实际下载和镜像回退继续使用同一 SHA。缓存命中也检查大小，
下载后再次核对文件大小，并在解析前执行有界读取。固定版本的 config、Diffusers
`model_index.json`、依赖索引和源码均使用此边界。

尚未固定 revision 的 config fallback 保留 SDK 的分支／tag 解析与 ref 写入，确保首次下载后
仍能按默认分支离线复用；返回路径仍须通过固定 snapshot 检查。此 fallback 在下载后、
解析前限制文件大小，首次网络下载量暂不具备同样的下载前上限；不把它描述为有界下载。

此边界采用现有 `huggingface_hub>=1.0` 的公开 API，参照
[Hub v1.0.0 下载实现](https://github.com/huggingface/huggingface_hub/blob/v1.0.0/src/huggingface_hub/file_download.py)
及 [dry-run PR #3407](https://github.com/huggingface/huggingface_hub/pull/3407)（Apache-2.0）。
不使用 `local_dir` 或上游私有下载实现，不增加依赖；额外预检只发生在准备阶段。

自动解析限于可确认的 Pipeline 子集：

- 本类直接定义的 `preprocess`、`_sanitize_parameters`、`_forward`、`postprocess`。
- `inputs["key"]`、`inputs.get("key", literal_default)`，保留必填性、默认值和来源行。
  媒体字段沿用 canonical 名称；字符串文字输入接受 `text` 或 `prompt` 的字符串默认值。
  需要 `messages`／`turns` 等嵌套结构时保留缺口，通过下述输入模板表达，不把字符串直接映射成对话列表。
- sanitize 的 literal key 集合与 forward 的明确参数签名。生成上限必须直接传入
  `generate(max_new_tokens=max_new_tokens)`；确定性来自显式 `do_sample` 参数，或源码中
  `temperature = temperature or None`、`do_sample = temperature is not None` 的直接赋值链。
  不能仅凭参数名推导 `temperature=0`，动态 kwargs、重写输入映射及不支持的控制路径需要声明／adapter。
- `register_pipeline` 的 literal 名称用于交叉核对。方法存在不证明 tensor shape、输出类型或真实推理成功。

`model_resolution.contract` 保存独立 provenance：字段的 `value/state/sources`、源文件 hash、
resolver 版本、锁定环境的 Transformers 版本、draft、依赖候选和未解决字段。状态为
`declared/derived/verified/ambiguous/unresolved`；静态分析只产生声明或推导，`verified` 来自实际 Probe。
`contract.status=resolved` 表示静态契约完整，外层仍为 `candidate`；真实执行证据继续保存在
独立 `runtime_validation`。冲突或缺口使外层成为 `ambiguous/needs_configuration`，并在构建前停止。
`contract.status=needs_confirmation` 汇总未决字段；TUI 正式采集只在有可裁决的未决项时暂停询问，
答案交回 resolver 重新解析，所有必填字段 resolved 后在原进程继续。已确定字段只读展示，修改与内部证据默认折叠。
输入映射和依赖选择写入 `reviews`，来源标为 `user.review`；多 Pipeline 选择会在同一 SHA 上重新分析。
动态源码、任务冲突等不能由当前字段编辑器解决的问题仍要求显式声明／adapter。

只有无缺口的 draft 才进入 `generated_spec`，由已有模型声明入口传给镜像指纹、通用 handler、
server 和 profiler；不会修改作者文件或 Hub snapshot。正式运行在准备阶段导出
`model_resolution.json`，同时保留 `static_meta.json.model_resolution`。单独查看失败 draft 可使用：

```bash
acprof inspect fixie-ai/ultravox-v0_5-llama-3_2-1b --explain \
  --output-dir internal-testing/model-resolution
```

`cache_key` 包含模型 ID、SHA、resolver 版本、Transformers 版本、完整文件清单和依赖控制流证据；AST 分析按源文本在
进程内缓存，Hub 文件复用其内容缓存。依赖 SHA、文件选择和用户决策也参与静态身份；没有跨进程的
解析结果缓存，也不复用旧运行验证。运行观察单独追加，不改变已经建立的静态身份。
JSON 元数据 hash 使用 canonical JSON；Python／README hash 使用所分析的 UTF-8 文本，报告会标明前者。

TUI 另有**用户确认选择缓存**，不是上述解析结果缓存：输出根目录 `.model-contracts/decisions/`
只保存答案和模型 ID、SHA、contract cache key、provenance identity、显式配置。下次仍读取当前证据，
身份一致才重新应用答案并校验支持性。显式 task／family／backend 不得被答案或推断替换；不兼容
声明明确拒绝。CPU Probe、历史验证或静态 resolved 都不能代替本次所选设备的完整运行验证。

采集准备阶段由 `host.collection_workflow` 连接静态解析、主机预检、镜像、输入和 runtime validation。
失败阶段保留其上游成功结果；普通“重新验证”复用原不可变镜像和输入，“重新准备环境”则使镜像及
下游输入／验证失效后重建。正式测量期间不询问用户、不轮询控制通道、不做阶段重试。
具体按钮和恢复边界见 [TUI 工作流](../usage/tui-interaction.md#模型契约解析与验证)。

外部 `from_pretrained` 调用按 tokenizer、processor、metadata、weights 等角色记录候选；
config 中的模型引用和动态表达式也会保留。明确的 repo／loader 通过 Hub 自动固定 SHA，按角色生成
兼容 v1 的 `dependencies/allow_patterns`：tokenizer／processor 只选根目录配置、词表和模板，以及
单层 `chat_templates/`；不会因名称前缀匹配而选中子目录快照或权重。metadata
只选 `config.json`；weights 选择一个标准 Transformers 权重格式及存在的 `generation_config.json`。
主模型与依赖共用 checkpoint 解析器：分片索引按 `weight_map` 的文件集合解析，支持合法自定义
分片名和子目录；任一必需分片缺失即报错，不靠标准编号文件名猜测，也不改用另一套权重。
解析只读取固定 SHA 的索引元数据（最多 4 MiB）；下载前按同一 SHA 查询该路径的大小，
大小未知或超限时直接拒绝，下载后再次有界读取。纯 tokenizer 解析不增加索引查询。
保留相对路径校验、十万条仓库清单和每依赖
128 个选择文件的上限。显式 `allow_patterns` 的依赖在 Model Store 下载权重和构建前也校验
选中 checkpoint 的完整性；纯 tokenizer／processor 依赖无需权重。
`AutoFeatureExtractor` 的 processor 只选 `config.json/preprocessor_config.json`，不附带 tokenizer 或权重。
实际下载继续使用镜像构建阶段的
既有 planner、文件 hash 和离线缓存，主机解析不下载权重。最多 16 个依赖，每个最多 128 个文件。
候选不等于运行时必需，`pinned` 也不代表文件已下载。依赖分析采用以下有界规则：

| 证据 | 处理 |
| --- | --- |
| `activation=active` | 在所分析加载入口和配置下可达，按 role 解析 SHA 与文件 |
| `activation=inactive` | 固定配置或已证明的选择路径排除了该调用；保留证据，不查询／下载 |
| `activation=unknown` | 动态条件或未能建立调用上下文，保留 review，不查询／下载该候选 |
| `dependency_kind=main_model` | 已声明主模型的 `super().from_pretrained` 转发，或已知 loader 对当前 snapshot 的自引用；不重复规划外部下载 |
| `alternative` | 保留同一 `try` 中的 primary/fallback 关系；未证明 primary 完整时，fallback 为 unknown |

固定 config 的 `None` 判断、布尔组合、简单赋值和同一 snapshot 内的函数调用可传递已知值。
普通参数、keyword-only 参数及已知 `*args/**kwargs` 会绑定到 loader；`call_chain` 保存来源。
动态 kwargs 不得抹去 revision；未知装饰器、动态训练条件、循环／match、递归或超出分析边界仍需 review。
本地调用最多 12 层、256 个调用上下文、512 个候选和 100,000 个分析步骤，不执行任意 Python。

主模型 tokenizer 的 primary 仅在固定文件清单含 `tokenizer.json/tokenizer_config.json`、
声明为 `PreTrainedTokenizer/PreTrainedTokenizerFast`、没有 custom tokenizer `auto_map`，且调用只使用
已知默认选项时，获得静态文件完整性证据。直接 loader 和仅转发／返回结果的 helper 可据此将 fallback
标为 inactive；额外处理、动态参数或缺文件仍为 unknown。文件完整性不保证内容有效或运行成功。

已登记的 Transformers 4.57.6 `PreTrainedModel` 使用版本限定的加载阶段摘要：构造期间关闭初始化，
随后分析模型的权重初始化 hook；候选的 `loader_context` 记录版本与阶段。这个摘要只用于已声明的
主模型及本地调用链；其他版本、任意 `dynamic_training_mode()` 和未知 framework 状态不会被猜成 true。
`active` 是加载规划中的可达性，不是某次实际运行已执行该分支的证据。

控制流按调用及 role 判断，再合并同 repo 的有效文件需求。`audio_model_id=None` 可排除 audio weights，
同时从 `audio_config._name_or_path` 激活 processor；不能按 repo 整体删除。多个 revision、非默认分支别名
及无法匹配的文件结构保持未决。TUI 依赖项只需给出 `repo_id/role`，
可用 `required:false` 明确排除未使用的候选；SHA 和 patterns 自动补齐，不推断任意 Python 依赖闭包。
作者已固定的依赖不会重新解析。`fixie-ai/ultravox-v0_5-llama-3_2-1b` 的固定 snapshot
`b95bec8ab291eeb04b5cd600dd473377f6b79026` 可自动生成完整静态契约，包括 Llama weights 和 Whisper
processor；主模型转发、音频权重分支及已证明的 tokenizer fallback 被排除。其任务、输入、生成参数、
依赖仓库／SHA 和所需角色与 `examples/multimodal/ultravox.model.json` 一致；文件按实际角色精确选择，
不照搬示例中的宽泛 `*.json`。该案例以固定源码 fixture 验收，生产逻辑没有 checkpoint 名称特判。

采集生命周期统一为：

```text
resolve → interface validation → prepare runtime → runtime validation → matrix measurement
```

Static Resolution 只读取固定 revision 的文本和 AST，不在主机执行仓库源码。
Interface Probe 使用 `source_bundle.py` 消费静态解析已有的源码图、`repository_sources` 和 metadata；
缓存存在时不重新联网，缺少已声明文件时只获取该文件。bundle 只包含已确认的 `.py`、`config.json`、
tokenizer／processor 的小型 JSON metadata 与 `acprof_model.json`；源码数量、大小和 SHA256 都有界，
并核对完整 commit SHA。相对导入的子模块与 package initializer 同样进入源码图。
权重后缀、路径越界、缺失依赖或源码身份不一致直接失败，缺图报告
`interface probe source graph incomplete`，不调用 snapshot 下载，也不扩展 `model_download_policy`。

`acprof inspect MODEL --probe-interface` 可独立运行相同的 Interface Probe。runner 只使用选定
profile 的 dependency base、AC-Prof 服务源码副本和 Source Bundle，完全脱离 `prepare_image()`、
`prepare_model()` 与 Model Store。它检查 import 和 `inspect.signature.bind`，不实例化权重、不做
preprocess 或 inference；默认 CPU 2 核、4 GiB，容器上限 300 秒，依赖镜像准备不在此超时内。
容器断网、read-only、挂载只读、移除 capabilities、禁止提升权限；临时缓存写入 `/tmp`，
不挂载主机凭据、Docker socket 或模型目录。结束或取消后按不可变 ID 清理容器和临时目录。

自定义 Pipeline 共用 [`load_local_pipeline_class`](../../acprof/container/local_pipeline.py) 和既有
remote-code 策略；`auto_map` 同样检查策略，导入均为 `local_files_only=True`。
Transformers 精确版本的 loader 选择与能力声明继续遵循
[dynamic-module 兼容生命周期](pipelines.md#transformers-dynamic-module-兼容生命周期)。
接口证据单独写入 `interface_validation.json`、`logs/interface_validation.log` 和
`model_resolution.interface_validation`，不改写静态裁决或冒充真实推理。

Runtime Validation 就是现有 `validate_runtime()`，按 `prepare_image → plan_input_scales →
validate_runtime → run_matrix` 顺序使用正式镜像、Model Store、adapter 和 workload。
每个选中设备执行一次最小合法输入，完整经过 load、preprocess、predict、postprocess、validate_output。
CPU + GPU 必须两者成功；失败、OOM、超时或缺少阶段证据均阻止正式矩阵。
报告仍为 `runtime_validation.json`，含每个设备的日志、输入 hash 和镜像身份；历史协议的 `mode=full`
表示端到端验证，新执行入口不再提供 basic/full 选择。它不产生正式 CSV row，不开启 profiler 测量窗口。

两类验证都不能代替所选 profiler 或正式矩阵实测。`profiling-mode basic/full` 只表示采集指标范围，
不表示模型检查深度。TUI 自动执行整个准备流程，交互见[模型确认](../usage/tui-interaction.md#模型契约解析与验证)。

实现参考 [Hugging Face 单文件缓存下载](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py)
与 [Transformers dynamic modules](https://github.com/huggingface/transformers/blob/main/src/transformers/dynamic_module_utils.py)
（Apache-2.0），以及 [Textual worker 取消讨论](https://github.com/Textualize/textual/discussions/4510)
（项目为 MIT）。复用现有依赖和版本能力表，只借鉴缓存、隔离与生命周期处理方式，不复制整套实现、
不新增下载模式或第三方依赖；新增准备工作均在正式测量窗口外。

输入 DSL 在 schema v1 中兼容旧字符串重命名，并增加 `from`、`literal` 和 `template`：

```json
{
  "turns": {"template": [{"role": "user", "content": {"from": "text"}}]},
  "audio": {"from": "audio"},
  "sampling_rate": "sampling_rate",
  "speaker": {"literal": "user"}
}
```

以上对象放在 `multimodal.inputs`。所有任务要求的 canonical 输入必须被引用；模板只允许 JSON
结构和已知输入引用，最多 8 层、256 个节点、每个容器 32 项、字符串 4096 字符，整份 spec 仍限
64 KiB。无 Python、eval、文件／环境访问或字符串插值；每次生成新容器结构，媒体数据保持原值。
转换在既有 preprocess 阶段完成，server 与 profiler 共用 handler。

实现借鉴 [Transformers 4.57.6 Pipeline](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/base.py)
和[动态模块加载边界](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/dynamic_module_utils.py)
（Apache-2.0），并以 [Ultravox Pipeline](https://github.com/fixie-ai/ultravox/blob/main/ultravox/model/ultravox_pipeline.py)
（MIT）核对模式。复用接口思想，以标准库 AST 实现受限分析，不复制 loader、不新增主机推理依赖；
所有下载、解析和报告写入均位于正式测量窗口外。

三态分支与合流参考 [Pyright 的代码流收窄](https://github.com/microsoft/pyright/blob/main/docs/type-concepts-advanced.md)
（MIT）；加载阶段和 tokenizer 文件规则分别核对
[Transformers 4.57.6 modeling_utils](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py)
与 [tokenization_utils_base](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/tokenization_utils_base.py)
（Apache-2.0），角色分离另对照
[Ultravox prefetch_weights](https://github.com/fixie-ai/ultravox/blob/main/ultravox/training/helpers/prefetch_weights.py)
（MIT）。只借鉴有界分析与 API 语义，保持标准库实现；升级锁定的 Transformers 版本须复核阶段摘要，
未登记版本继续保留 unknown。测试 fixture 保留上游源码许可，作为文本输入，不导入执行。

依赖规划另参考 [Hugging Face snapshot 下载器](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_snapshot_download.py)
（Apache-2.0）的固定 revision 与文件过滤；TUI 使用 [Textual Workers](https://github.com/Textualize/textual/blob/main/docs/guide/workers.md)
（MIT）的后台任务边界。均复用现有依赖，不复制上游 loader；DSL 是有界 JSON 解释，不引入模板执行引擎。
