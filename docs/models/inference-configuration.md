# 统一推理配置契约

[← 返回模型契约](contracts.md) · [文档导航](../README.md)

## 状态与范围

本页描述统一声明、静态发现及已登记的严格执行链。实现复用 Extension Manifest、现有任务审阅、
精度策略、Textual 和运行验证，没有新增独立插件系统。

**通用请求表单、Auto 解析、内容确认冻结、传输、静态元数据与恢复检查已经贯通。**
当前完整的 dtype／attention／生成参数消费核验范围是已登记的原生 NLP 生成路线：
`transformers_model`、`transformers_pipeline` 对应的 `NLPHandler`。
另外已接入 Whisper 语言、Chronos 预测长度、多模态生成长度、SDXL 步数／引导系数及 Scheduler，
字段与独立观察方法见下表。新增路线的开发证据限于离线 Fixture／Mock，尚未取得真实 checkpoint
执行证据；没有声明的加载／生成控制不开放，也不标为已核验。

现行模型解析和接口验证以[模型契约](contracts.md)为准，现行产物字段和 schema 以
[静态元数据](../profiling/metadata.md)为准。静态发现结果写入 inspect 的
`model_resolution.json.configuration_options`，继续与最终执行的 `inference_configuration` 分离。
新实验使用 v8 静态元数据；v7 历史数据保持原状，缺失的冻结配置保留未知，不补造证据。

新实验依次完成声明／请求意图、解析、最终审阅、确认冻结、准备、独立验证、正式运行和记录。
CLI 首次返回 `review_required`，TUI 使用独立最终确认页；未确认不得创建正式任务集合或开始下载／采集。
模型和任务扩展仍须同时登记消费者与验证方法。

## 当前已实现：声明与静态发现

- `extensions.parameters.ParameterSpec` 描述类型、作用域、默认值、范围、单位、消费者、来源绑定与有限依赖／冲突；
  `ParameterRequest` 保留 explicit、auto、unset，显式 null 与未提供值不同；解析由共享 resolver 执行。
- Manifest 的 `parameters`、`backend_parameters`、`task_parameters` 在既有 Catalog 内合并；
  当前六个文本生成控制（两种长度、do_sample、temperature、top_p、num_beams）及
  dtype、attention_implementation，以及下表的任务专属控制由此声明。CLI／TUI 与任务应用使用同一声明和值校验器。
  所有最终有效默认值也会物化到任务配置，不能只记录与默认值不同的 override。
- `models.configuration_sources` 从已取得的 Hub／JSON 元数据发现只读事实和逐参数默认候选。
  文件、JSON pointer、固定 revision、原始字节摘要与 canonical 摘要分别保留；不会联网补齐、
  应用默认值或从模型卡代码执行任意调用。未登记 processor 字段只作为事实，不开放为调参项。
- `models.backend_candidates` 枚举既有扩展路由，复用模型解析、RuntimeProfile、静态兼容性与已缓存源码的依赖检查。
  `selectable` 只表示当前静态证据允许继续准备；unknown、未登记实现或已知阻断不能成为可用选项，
  所有候选仍标记 `runtime_validation.status=not_run`。缺少已核验的作者推荐证据时保留 unknown。
- `inspect --explain` 展示候选、参数类型／作用域、消费者、默认候选与来源诊断；导出结果保留完整来源身份。
  架构事实不加入可调参数，多个语义任务的默认值不被压成一个全局默认。

声明使用标准库，不导入 Transformers、Diffusers、Torch 或 Textual。没有声明消费者的字段不能开放，
Pipeline 的任意 Python 签名或 `**kwargs` 也不会自动变成支持列表。更多任务、加载选项、版本／设备
条件及作者推荐规则需按实际 Handler 继续登记和验证；本轮不声称覆盖所有 Pipeline 参数。

元数据和请求 JSON 的支持深度为 256 层（根深度为 0）。超过上限时保留来源与字段路径并拒绝，
不截断原数据，也不让后续对象复制抛出未处理递归异常。已选任务组只沿同一 backend 保持身份；
换 backend 后须重新绑定任务组，不能把旧组的执行指纹当作新实现的证明。

通用 Runtime Profile 的 backend 清单已按实际 NLP、Audio、Diffusion、Structured、
Time Series 路由补齐。显式、默认和平台变体统一校验 family、task、backend、adapter 与已知 model type。
静态后端选项仍须通过既有制品、依赖及接口检查；冻结后禁止按平台静默替换 profile。

加载 dtype 按现有精度策略和设备过滤；attention 仅开放 Handler 已消费的 eager／sdpa。
这说明接口允许传入该选项，checkpoint 是否接受、加载后是否一致仍须运行验证；不存在未经核验的
Flash Attention、量化或任意 Python kwargs 选项。

### 任务专属参数与观察边界

| 静态适用范围 | 参数 | 独立验证的消费／观察位置 |
| --- | --- | --- |
| Whisper，`automatic-speech-recognition`，原生 Transformers 路线 | `language`、`asr_task=transcribe` | Pipeline 后的 `model.generate(language=…, task=…)`；不开放翻译任务 |
| Chronos，`time-series-forecasting` | `prediction_length`，正整数 | `pipeline.predict(..., prediction_length=…)` |
| 原生多模态 `image-text-to-text`、`video-text-to-text`、`audio-text-to-text` | `max_new_tokens`、`do_sample=false` | `model.generate`；仍只支持确定性生成，不扩展 `any-to-any` |
| `StableDiffusionXLPipeline`，`text-to-image` | `num_inference_steps`、`guidance_scale` | Pipeline 的实际 `__call__` kwargs |
| 同一 SDXL 路线的加载配置 | `scheduler`：EulerDiscreteScheduler、DDIMScheduler、DPMSolverMultistepScheduler | 加载后的 Scheduler 类名；切换先核对 `scheduler.compatibles`，再用当前 config 构造 |

Whisper 的 `model_type` 和 SDXL 的 Pipeline 类约束同时用于静态表单与运行观察。
Scheduler 切换使用 `from_config`，不触发第二次 checkpoint 加载；不匹配或无法观察即失败。
这些控制通过任务指纹、冻结对象、输入计划和实际 Handler 传递。自定义 workload 中已登记的
参数先显示为来源默认值，用户显式修改或应用预设后才覆盖；源文件摘要与最终有效值分别记录。
SDXL 的 Custom 默认仍为 20 步，Benchmark v1 为 25 步。

## 复用边界与参数分类

能力由 `ExtensionCatalog` 中的扩展声明、`RuntimeProfile` 的版本／设备限制及实际 Handler
共同约束。模型元数据提供默认值和身份事实，不凭字段存在、Hub 标签或 Python 签名证明支持。
Host 侧声明与静态解析保持轻量，不导入推理实现或执行任意模型代码。

| 分类 | 例子 | 使用边界 |
| --- | --- | --- |
| 模型固有属性 | 架构、层数、参数量、训练来源 | 只读；不能作为加载参数改变 checkpoint 含义 |
| 加载配置 | backend、runtime profile、dtype、attention、量化加载选项 | 每个加载实例固定；只能使用已登记且满足环境条件的值 |
| 推理配置 | 生成长度、sampling、beam、步数、语言、预测长度 | 按任务或请求声明；由对应 Handler 消费 |
| 工作负载 | 输入模板、输入规模、batch、seed、processor／tokenizer | 进入既有输入计划与实验身份 |
| 资源及测量配置 | CPU、内存、GPU、重复次数、Profiler | 沿用既有实验配置；不能作为模型推理 kwargs 透传 |

`Backend` 不是单一库名。身份须能区分接口／调用实现（如 Pipeline 或直接模型接口）、
推理引擎（如 PyTorch 或 ONNX Runtime）、已登记 adapter，以及固定依赖环境的 runtime profile。
模型来源仓库、实际权重制品仓库、代码仓库也分别记录，不能因名称相近而合并。
后端候选可展示作者推荐、已登记兼容实现及待适配项，但只有已登记且通过当前层级检查的候选可继续准备；
静态候选仍须在所选环境完成运行验证。

## 参数声明：ParameterSpec

统一参数描述接入 inspect、通用 CLI／TUI 配置和 Handler 严格执行，不在 TUI 维护第二份模型规则。
通用控件支持布尔、整数、有限浮点、枚举、字符串、列表与 nullable；实际显示的字段由选中声明决定。
当前依赖规则只支持字段存在或严格 JSON 标量匹配，不引入可执行表达式或任意 DSL。

| 信息 | 必需语义 |
| --- | --- |
| 标识与作用域 | 稳定字段名；模型、加载、推理、workload 或测量范围；是否按 task／device／case 变化 |
| 类型与约束 | boolean、integer、finite float、enum、string、list 与 nullable；元素类型、范围、长度、单位 |
| 默认值 | 固定值或已登记解析规则；与用户请求分离，保留来源 |
| 依赖与冲突 | 例如 sampling 关闭时 temperature 不生效；不支持值应禁用并给原因 |
| 消费位置 | loader、processor、predict、workload generator 或测量层；后端参数名映射及版本条件 |
| 能力与证据 | task、backend、版本、设备约束；静态支持与运行观察的范围 |
| 验证与身份 | 可使用的验证方法、是否为关键约束、参与的既有指纹层级 |

字段有 UI 控件但没有消费者，或后端仅接受 `**kwargs`，都不能标为已支持。
静态参数解析应报告未知字段和带路径的类型／范围错误；明确提供的未知字段不得被过滤后继续执行。

## 用户请求、默认值与 Auto

请求保存 `mode`、`value`、来源和作用域。模式与值分开，避免将空值、缺失和自动选择混在一起。

| 请求 | 含义 | 冻结前要求 |
| --- | --- | --- |
| `explicit` + 非空值 | 用户、CLI 或用户选定的预设要求该值 | 值必须满足约束，不能静默替换 |
| `explicit` + `null` | 用户要求合法的可选空值 | 仅在参数声明允许 nullable 时接受；不等于 Auto |
| `auto` | 用户授权已登记规则决定该字段 | 展示解析值、依据、规则版本及验证状态，再确认 |
| `unset` | 尚无选择；草稿或缺失字段 | 不自动视为 Auto；必需字段未解决时不能冻结 |

显式选择恰好等于默认值，仍须保留 explicit 意图。现有仅记录“与默认不同的 override”的机制不能单独承担此契约。
选择预设是一次可追溯的用户操作；“作者推荐”必须有固定来源，否则只能称为模型默认或 AC-Prof 默认。
确认采用显示的固定默认值也须保存其来源，不把它改称为自动决策。

基础规则 `acprof-inference-v1` 对生成参数使用已经选定的固定任务快照；快照没有该字段时，
使用统一声明默认值。加载精度使用现有登记策略，attention 默认 eager，后端／profile 默认采用
已经通过静态检查的平台路由。Auto 并不进行性能搜索或失败重试。
每个决策保存原请求、最终值、来源／候选、选择原因、规则版本及静态／运行验证状态。

初始表单显示具体默认值，模式为 explicit；只有用户选择 Auto 或明确应用 Auto 预设才授权解析。
提供 Custom、Auto，以及能力适用时的作者任务预设和 AC-Prof Benchmark v1。
`unset` 不授权默认选择；已禁用的依赖字段记录 inactive，不作为有效生成参数透传。
模型只读事实不参与这套覆盖；backend、设备、依赖版本的能力限制是验证条件，也不能被高优先级值覆盖。

Auto 解析必须限定到已登记、可解释的规则。候选缺少证据、存在冲突或无可行值时返回明确诊断，
不尝试未登记实现、不改换模型来源、不降精度重试。用户修改约束后才能生成新的配置。

### 作者任务预设与 AC-Prof Benchmark v1

作者任务预设只采用固定 revision 下、带原始字节 SHA256 的 `config.json.task_specific_params`，
而且只能应用当前声明已开放的字段。普通 `generation_config.json` 默认值、AC-Prof 默认值或
无来源的模型卡建议不算作者推荐。候选冲突时不展示该预设，显式请求 `preset: author` 则报错；
没有对应作者值的字段保留当前选择。来源保留文件、JSON pointer、revision、raw SHA256 和
`author-task-v1` 版本。它不是对整个模型或后端的作者背书。

`preset: benchmark` 使用 `acprof-benchmark-v1`，仅固定当前声明支持的任务参数：

| 任务 | 固定值 |
| --- | --- |
| 文本生成 | 当前长度字段为 64；`do_sample=false`、`num_beams=1`；sampling 关闭时 temperature/top_p 不参与执行 |
| Whisper ASR | `language=en`、`asr_task=transcribe` |
| Chronos | `prediction_length=64` |
| 多模态生成 | `max_new_tokens=64`、`do_sample=false` |
| SDXL | `num_inference_steps=25`、`guidance_scale=7.5` |

这是项目的任务参数基线，不宣称符合 MLPerf 或其他外部 Benchmark 标准，也不是质量评分。
应用预设保留已审阅的 backend、Runtime Profile、加载参数（含 Scheduler）、batch 和输入规模；
资源、输入内容、seed 和测量设置继续使用当前 workload。比较实验仍须核对这些条件。
随后编辑的显式请求可以覆盖预设值，记录同时保留预设版本和最终请求，因此仅看预设名称不能判断配置相同。
选择预设本身不提交修改，须点击应用并重新审阅、确认完整 SHA。

## 解析、确认与冻结

请求经过类型校验、来源合并、能力校验和 Auto 解析后，形成一个待确认的 resolved configuration：

- 模型来源、固定 revision、实际制品与必要代码身份。
- backend、adapter、runtime profile、依赖身份。
- 加载、推理与 workload 的最终值，并保留请求模式、原始请求及来源。
- 每项 Auto 的所选值、选择依据、规则版本和验证状态。
- task／device／resource case 到配置的映射，以及稳定内容身份。

确认页展示这些最终值和仍未运行验证的项。确认绑定完整配置内容，而不是“用户点击过继续”的布尔值。
backend、revision、任务、设备、参数、预设、输入计划或解析规则导致内容变化时，旧确认失效。
运行验证只能核对该冻结配置；发现不匹配后终止，不改写预期值以迎合观察结果。
重新选择或重新解析应生成新的确认对象。

冻结前的静态检查只证明声明与约束可满足；不能在确认页提前标成真实推理通过。
非交互式 CLI 使用 `--inference-config FILE` 读取有限 JSON 请求，使用
`--confirm-inference SHA` 确认同一候选。首次或旧摘要不匹配时，将完整候选和请求分别保存到
输出目录的 `.inference-configurations/<SHA>.json` 与 `<SHA>.requests.json`，正常返回等待确认。
用户审阅后以相同命令、生成的请求文件和摘要重试；路径本身不属于实验身份。
具体操作见 [CLI run](../usage/cli-run.md) 和 [TUI](../usage/tui.md)。

`acprof probe` 可通过同一对参数启用审阅。确认范围包括单个执行组的生成或推理参数、任务级批大小与输入尺度，以及本次 Docker 探测的有效资源和超时。探测摘要可用于查找实际发送请求所用的冻结配置。[最大输入探测示例](../usage/experiments.md#先探测最大输入)

请求按 `loading.off/on` 和 `tasks.<原选择任务 ID>` 保存。改变 Beam／后端可能重新生成最终任务组 ID；
最终 `tasks` 中保留 `source_task_ids` 与完整 snapshot。确认摘要涵盖模型、后端、加载、任务、
请求模式、来源、规则和 workload 定义，仅排除自身摘要及 resolved／frozen 生命周期字段。
每个配置 JSON 最多 65,536 字节，拒绝超限或非有限数值。
当前一个任务集合共用按设备的加载合同。所选任务的加载声明不一致时，在确认前拒绝并提示分开集合；
不能根据第一个任务漏掉后续生成任务的加载约束，也不隐式取空交集。

工作负载定义与其后物化的输入计划分别留痕：最终 payload、实际输入规模和 case 使用既有
`input_scale_plan.json`，绑定冻结配置 SHA 并进入原有 plan hash／恢复检查；不把未物化的自动规模说成已验证值。

## 参数消费与运行时验证

`RunConfig`、CLI／TUI、输入计划、`ModelLoadConfig` 和四阶段 `BaseHandler` 传递同一配置身份。
加载参数通过既有加载入口进入 Handler，推理参数通过任务／请求进入实际调用；
不把所有参数扁平化为一份任意 kwargs，也不把测量配置混入模型参数。

验证记录独立于冻结配置，至少保存字段路径、expected、observed、检查方法、状态、作用域和证据引用。
必须区分下列证据，不能用上游未抛异常替代逐项核验：

| 层级 | 证明内容 | 不证明的内容 |
| --- | --- | --- |
| 静态声明校验 | 类型、范围、已登记能力、版本与依赖关系满足 | 某个 checkpoint 已加载或执行 |
| 参数消费检查 | 参数到达指定消费者，未知／未使用值已处理 | 后端真实按预期执行该选项 |
| 加载后观察 | 可观测的 dtype、attention、backend 等与冻结值一致 | 所有输入形状与 kernel 都使用同一路径 |
| 请求验证 | 指定任务、设备和输入下的接口／输出契约通过 | 未验证 case、任务质量或完整 Profiler 支持 |

状态区分 verified、mismatch、unverified、not_applicable 和 not_run，并记录可验证的字段范围。
原生 NLP 独立验证临时观察实际 `model.generate` 边界的 kwargs／GenerationConfig；
没有调用记录、吞掉不匹配后重试或只回显请求，都不能生成通过的 consumer receipt。
未登记消费者的 receipt 为 not_applicable／consumer_not_declared，不能标为 verified。
关键显式约束没有可用验证方法时，应明确阻断，不能标为 verified；非关键观察缺口按参数声明保留未知。
未消费的显式参数和不一致的关键参数都阻止正式采集。上游 API 若对未知参数仅警告或忽略，适配层须补充检查。

最小输入验证不是整个资源矩阵的通用证明。不同 backend、加载配置、设备或重新启动的实例须绑定
自己的配置身份与必要检查；按输入条件才激活的选项，记录已验证条件。
加载核验位于加载计时结束之后、ready 之前；正式 HTTP 请求只做轻量冻结值一致性检查，
不会扫描模型参数或包装 generate。独立消费观察只在 runtime_validate 中执行。
Profiler 的核验位于 warmup／显式测量窗口之前；whole-process 工具仍可能观察到这部分准备开销，
不能把它描述为完全没有开销。

## 精度、作用域与 Profiler 变体

精度至少区分权重存储精度、用户请求、Auto 解析值、加载后观察到的参数 dtype。
量化方式与量化配置独立记录；参数 dtype 不等同于运算、中间值或累加精度。
严格加载核验扫描所有浮点 parameter 的 dtype，混合类型按实际集合记录；
attention 读取模型配置中的实际实现声明。两者都不是计算精度或实际 kernel 执行的完整证明。
不可观测的计算细节保持 unknown。

一个实验可能同时包含 CPU 与 GPU，也可能有多任务、输入规模和资源 case。
加载配置按实例固定；推理与 workload 参数按明确的 task／request／case 记录。
例如 Auto 为 CPU 解析 FP32、为 GPU 解析 FP16，必须在确认时展示并冻结两个值，
不能只写一个全局 dtype，再在运行时依设备替换。比较工具应识别资源变化之外的精度或实现差异。

独立 Profiler 为计数采用 eager 等特殊实现时，生成与正式配置关联的派生配置，
记录变更字段、原因和独立身份。不能覆盖正式推理配置，也不能用派生配置验证结果证明正式实现已验证。
Profiler 的权威测量语义仍见[独立 Profiler](../profiling/profilers.md)。

## 元数据、指纹与恢复

v8 `static_meta.json.inference_configuration` 保存冻结配置原对象。
运行观察沿用独立 runtime validation 证据并以 `inference_configuration_sha256` 关联，
不得把 observed 值写回 frozen。完整字段定义见[元数据规范](../profiling/metadata.md)。

内部 schema 仍为 1；新增任务的有效参数保存在 `tasks[id].inference`，NLP 继续使用
`tasks[id].generation`。包含这些任务参数、Scheduler 或新预设的配置使用规则
`acprof-inference-v2`，对应决策也记录该版本。旧 v1 配置原样可读；禁止将新字段写入 v1，
旧运行时也会拒绝未知规则，避免忽略参数继续执行。历史文件不自动补字段或重算身份。

后端身份包含锁定依赖版本、environment 身份、AC-Prof 版本及既有源码身份：
measurement-source-v2、Docker service-source-v2、process-service-v1 分别保留范围与摘要。
模型保存 revision、配置和仓库文件清单摘要、已知 base_model、许可证、量化与存储精度；
下载后的实际制品文件／字节沿用 Model Store receipt 与 runtime manifest，不能在确认前伪装成已下载验证。

模型身份需能回答使用了哪个来源／模型／固定 revision、实际制品仓库及选中文件，
以及能够确认的 base model、权重格式、量化和许可证。tokenizer、processor、chat template、
输入模板、seed、stopping policy 等影响结果的设置也应进入对应身份；缺失事实不推测补齐。

扩展既有 task fingerprint、输入计划 hash、构建身份和 run state 恢复检查，不建立互不相认的平行指纹。
规范化保留 JSON 原生类型、缺失与 null 区别，拒绝非有限执行数值。影响执行的 backend／版本、
dtype、attention、生成参数或 workload 变更必须影响相应执行身份。
用户意图、来源与 Auto 规则另作为审计和确认依据；相同最终数值不允许悄悄丢失 explicit 与 Auto 的区别。

运行观察、耗时、日志位置等追加证据不反向修改冻结身份。恢复必须核对原冻结配置、制品、环境及输入身份，
拒绝不匹配结果的无提示复用；不以当前默认值重建缺失的历史配置。
比较使用明确的配置作用域；不能把多个 task／device／case 的差异压缩成一个全局值。

## 固定静态回归证据

固定样本沿用 [coverage manifest](../usage/cli-discovery.md#acprof-coverage)：
source、model ID、完整 revision、固定 weight、sampling 与 weight basis。排名查询方式、时间和分母
另保留为取样证据，snapshot 后不随运行更新，也不将 Hub 标签直接当正确答案。

离线元数据回归应在同一 manifest 旁记录以下证据，不复制 coverage 的运行和汇总逻辑：

- 实际仓库文件清单与所读取的 JSON／文本；每份原始字节的 SHA256、相对路径、大小、来源 revision 和许可。
- AC-Prof commit、解析器／规则版本、依赖能力版本，以及明确的任务、backend 等检查输入。
- 原始观察结果与独立审阅的预期结果分别保存；预期至少含状态、reason code、证据层级和下一步。
- 合成 fixture 与真实固定快照明确标识；解析失败样本保留原始内容，不能先替换非法数值再声称复现。
- 重放验证原始字节 hash，再调用既有解析与汇总入口；禁止联网补齐缺文件或执行任意模型代码。

本轮开发验证使用离线环境变量、阻止 Python Internet socket 的验证入口、合成 metadata 与 Mock
消费者；没有下载权重、加载真实 checkpoint、执行 GPU 推理或完整 Profiler。未取得原始 Top 100
逐项模型／revision 清单时，不能用合成案例代替固定整集回归。

实现参考 [Transformers Whisper generation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/generation_whisper.py)、
[Diffusers Scheduler 切换](https://github.com/huggingface/diffusers/blob/v0.35.1/docs/source/en/using-diffusers/schedulers.md)
和 [Chronos predict](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos.py)
的实际调用边界，沿用本项目锁定的运行环境；只借鉴接口与配置复用方式，没有引入新依赖或复制其实现。
[Transformers 离线测试](https://github.com/huggingface/transformers/blob/main/tests/utils/test_offline.py)
提供了在导入前设置离线变量并阻断 socket 的参考；本轮没有执行其下载预热步骤。

- 比较保留历史结果，不以刷新 snapshot、缩小分母或放宽检查提高通过数量。

可复用 `tests/fixtures/t5_small_metadata/source.json` 及已有 raw-bytes hash 校验方式。
该 fixture 不等于 HF Top 100。未取得原始 Top 100 清单、revision 和元数据快照时，
只能报告已覆盖的固定／合成案例，不能声称完成 100 模型重放或重现原始数量。

## 实施验收

首个受支持 Handler 的贯通验收至少覆盖：

1. explicit、Auto、unset 和显式 null；显式默认值不会丢失用户意图。
2. 类型、范围、依赖、冲突、未知字段，以及参数到达真实消费者的证据。
3. task／backend／设备切换后的能力重算和确认失效。
4. 冻结值与观察值不一致时不开始正式采集；不可观测项不会误报 verified。
5. 元数据能还原来源、Auto 原因、所用实现与验证范围，修改关键配置后恢复被拒绝。
6. task／device／case 与 Profiler 派生配置独立留痕，CLI 和 TUI 使用同一配置身份。

默认使用定向 Mock／Fixture 与必要的轻量验证。真实模型下载、推理和完整 Profiler 验收仍按
[验证边界](../development/validation-scope.md)由用户明确授权；静态或 Mock 通过不升级为实测证据。

## 设计参考

- [Transformers v5.12.0 GenerationConfig](https://huggingface.co/docs/transformers/v5.12.0/en/main_classes/text_generation)
  与[配置实现](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/generation/configuration_utils.py)：
  借鉴配置来源、unused kwargs 与严格校验的边界；版本签名不能套用到项目锁定的 4.x 环境。
- [vLLM 模型解析](https://docs.vllm.ai/en/stable/configuration/model_resolution/)与
  [engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)：
  借鉴模型实现选择与生成配置来源的显式表达；不因此将 vLLM 宣称为当前已支持后端。
- [Diffusers 制品格式与加载](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/other-formats.md)：
  借鉴文件布局、序列化格式与 loader 能力分离，不因相同权重后缀推断通用兼容。

- [Transformers v4.57.1 dtype 实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py)
  的便捷属性只返回首个浮点 dtype；本系统因此单独核对全部浮点参数集合。
- [Transformers v4.57.1 Attention Interface](https://huggingface.co/docs/transformers/v4.57.1/attention_interface)
  说明 attention 可改变且可按 backbone 不同；请求值与加载后的配置观察须分开。
- [vLLM v0.18.1 backend selection](https://docs.vllm.ai/en/v0.18.1/design/attention_backends/)
  借鉴显式选择不兼容就失败、自动选择必须校验的边界；没有将 vLLM 新增为可执行选项。

上述资料用于设计对照，不复制上游实现、不引入新依赖，也不要求本轮升级推理库。
实施时以项目所选 runtime profile 的固定版本核验能力。
