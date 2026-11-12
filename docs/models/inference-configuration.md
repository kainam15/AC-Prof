# 统一推理配置契约（设计，待实现）

[← 返回模型契约](contracts.md) · [文档导航](../README.md)

## 状态与范围

本页定义 P1 统一推理配置的设计边界，**不表示通用动态表单、配置冻结或逐字段运行时核验已经实现**。
设计基线为 `3e4bde93`。当前已有任务参数审阅、部分生成参数传递、精度策略、任务指纹与运行验证；
本设计复用这些入口，不另建插件注册器、执行流程或 coverage runner。不新增或宣称已经可用的 CLI 参数。

现行模型解析和接口验证以[模型契约](contracts.md)为准，现行产物字段和 schema 以
[静态元数据](../profiling/metadata.md)为准。本页中的对象与字段是待实施的数据契约；
实施时须同步其生产者、消费者及对应权威文档，不能仅凭本页向现有 v7 元数据写入新字段。

第一条完整链路应同时交付：声明、请求意图、解析、确认冻结、参数消费、运行观察、元数据与恢复检查。
表单覆盖和更多模型随后扩展，不能先把未经 Handler 支持的参数显示成可执行选项。

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

在现有 Extension Manifest／任务声明中增加统一参数描述，并由 TUI、CLI、配置解析和 Handler 共用。
不在 TUI 维护第二份模型规则。第一版只支持有限、可校验的依赖关系，不引入可执行表达式或任意 DSL。

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

对于同一可调参数，建议采用以下来源顺序：显式字段覆盖 → 用户选定预设中的值 → 固定 revision
中的模型／任务默认 → 已登记后端默认 → AC-Prof 基准默认。后几层只能在已授权的 Auto 或已确认默认策略内参与；
`unset` 本身不授权这一选择链。记录覆盖过程，不能仅存最后一个值。
模型只读事实不参与这套覆盖；backend、设备、依赖版本的能力限制是验证条件，也不能被高优先级值覆盖。

Auto 解析必须限定到已登记、可解释的规则。候选缺少证据、存在冲突或无可行值时返回明确诊断，
不尝试未登记实现、不改换模型来源、不降精度重试。用户修改约束后才能生成新的配置。

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
非交互式 CLI 应使用同一声明和解析器，具备可审阅的解析结果及明确的确认策略；
具体 CLI 语法留待实施，不能通过跳过 TUI 绕过冻结和验证。

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

状态至少区分 verified、mismatch、not_observable 和 not_run，并记录可验证的字段范围。
关键显式约束没有可用验证方法时，应明确阻断，不能标为 verified；非关键观察缺口按参数声明保留未知。
未消费的显式参数和不一致的关键参数都阻止正式采集。上游 API 若对未知参数仅警告或忽略，适配层须补充检查。

最小输入验证不是整个资源矩阵的通用证明。不同 backend、加载配置、设备或重新启动的实例须绑定
自己的配置身份与必要检查；按输入条件才激活的选项，记录已验证条件。
检查位于准备或规定的验证阶段，不向正式计时窗口追加配置探测，也不改变现有测量口径。

## 精度、作用域与 Profiler 变体

精度至少区分权重存储精度、用户请求、Auto 解析值、加载后观察到的参数 dtype。
量化方式与量化配置独立记录；参数 dtype 不等同于运算、中间值或累加精度。
现有 `actual_dtype()` 观察浮点参数类型，混合类型按实际集合记录，不能将其包装成完整计算精度证明。
不可观测的计算细节保持 unknown。

一个实验可能同时包含 CPU 与 GPU，也可能有多任务、输入规模和资源 case。
加载配置按实例固定；推理与 workload 参数按明确的 task／request／case 记录。
例如 Auto 为 CPU 解析 FP32、为 GPU 解析 FP16，必须在确认时展示并冻结两个值，
不能只写一个全局 dtype，再在运行时依设备替换。比较工具应识别资源变化之外的精度或实现差异。

独立 Profiler 为计数采用 eager 等特殊实现时，生成与正式配置关联的派生配置，
记录变更字段、原因和独立身份。不能覆盖正式推理配置，也不能用派生配置验证结果证明正式实现已验证。
Profiler 的权威测量语义仍见[独立 Profiler](../profiling/profilers.md)。

## 元数据、指纹与恢复

拟在 `static_meta.json.inference_configuration` 保存冻结的执行配置及来源，运行观察沿用独立
runtime validation 证据并关联该配置。具体产物 schema 变更须按[元数据规范](../profiling/metadata.md)
实施，不在本轮提前改变当前 schema 或写入已有实验。

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

上述资料用于设计对照，不复制上游实现、不引入新依赖，也不要求本轮升级推理库。
实施时以项目所选 runtime profile 的固定版本核验能力。
