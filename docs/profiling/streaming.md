# OpenAI-compatible 流式测量

[← 返回采集协议](protocol.md)

## 范围与入口

`acprof load` 可对已经启动的 OpenAI-compatible 服务执行独立 HTTP 测量。
`openai-chat` 使用 `/v1/chat/completions`，`openai-completions` 使用
`/v1/completions`；默认的源实验 `/predict` 重放不变。两种协议复用
`measurement.load` 的串行／并发／固定或 Poisson 到达调度和每工作线程连接管理。
本入口不启动或管理模型服务，不声明 Docker／Process 硬件采集能力，不产生正式能耗 CSV。

`arrival-rate` 的 `rate` 必须为有限正数，生成的每个相对到达时刻也必须有限；
极小速率或累计间隔导致浮点溢出时，CLI 在创建输出目录、warmup 和 Runtime 准备前报错，
直接 `run_load()` 在创建 transport／worker 前拒绝。可提高速率或减少请求数后重新运行。
固定与 Poisson 计划仍在负载计时前生成，保留既有 seed 和到达分布；不截断或重标定计划。
设计核对了 [vLLM 的计时前到达计划生成](https://github.com/vllm-project/vllm/blob/831dd92674c818d7e440e750cbdcc7c9e8f4a97b/vllm/benchmarks/serve.py#L431)
（Apache-2.0），沿用标准库随机源，不复制实现或引入 vLLM 依赖。

```bash
acprof load --base-url http://127.0.0.1:8000 --protocol openai-chat \
  --model Qwen/Qwen3-0.6B --prompt "Explain what a token is in one sentence." \
  --max-tokens 64 --temperature 0 --requests 5 --warmup 1 \
  --output-dir results/qwen-streaming

acprof load --base-url http://127.0.0.1:8000 --protocol openai-completions \
  --request-json request.json --scenario concurrent --concurrency 2 \
  --connections reuse --requests 8 --warmup 1 \
  --output-dir results/qwen-completion-streaming

acprof load --base-url http://127.0.0.1:8000 --protocol openai-chat \
  --model Qwen/Qwen3-0.6B --prompt "Explain throughput in one sentence." \
  --max-tokens 64 --requests 20 --scenario concurrent --concurrency 4 \
  --slo-ttft-s 0.5 --slo-tpot-s 0.05 --slo-e2e-s 3 \
  --service-backend vllm --service-version 0.11.2 --service-probe vllm \
  --output-dir results/qwen-capacity
acprof report results/qwen-capacity/load.json \
  --output results/qwen-capacity/capacity.html
```

`--request-json` 接受完整请求对象，例如：

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Write the next three integers after 1, 2, 3:",
  "max_tokens": 32,
  "temperature": 0,
  "seed": 7
}
```

可直接使用 [Chat 示例](../../examples/streaming/qwen3-chat.json) 或
[Completions 示例](../../examples/streaming/qwen3-completions.json)作为 `--request-json`。

Chat 的确定性对照可显式传入 `chat_template_kwargs.enable_thinking=false`。
不要假定所有 OpenAI-compatible 后端支持此扩展；使用者负责核验服务端实际配置。
端点模式需要 HTTP(S) origin；不在 URL 中放入凭据。鉴权可用
`--api-key-env VARIABLE_NAME`，只读取指定环境变量，报告不保存 Authorization。
端点模式拒绝与源实验、`--gpu`、`--cpu`、`--mem`、`--input-scale` 或
`--capture` 混用。请求超时采用 `--request-timeout` 的连接／读取无进展秒数，
不是整个请求的总 deadline。预热报告与正式负载分开；不自动重试。
`--seed` 控制到达调度的随机序列；prompt 模式用 `--generation-seed` 指定模型采样 seed，
或在 `--request-json` 中显式传入；JSON 请求不允许叠加生成参数覆盖。
`--slo-ttft-s`、`--slo-tpot-s`、`--slo-e2e-s` 可独立设置，必须是有限正数；
SLO 违反不改变请求的协议成功状态。`--service-backend` 与 `--service-version` 保存用户声明，
`--service-probe vllm` 在预热前进行有限身份观测，默认 `none` 不发送额外请求。
这些选项仅适用于 endpoint 模式，source replay 会在读取源实验、联网或创建输出前拒绝。

协议只接受单个文本生成选择（`n=1`）；tool/function、多选择等尚未支持的输出明确失败。
Streaming Adapter 强制请求 SSE 与 usage。它处理 UTF-8 跨网络读取、BOM、
LF／CRLF／CR、注释心跳、多行 data、JSON 错误、finish reason 和 `[DONE]`。
网络读取块、SSE 事件、模型 Token 是三种不同单位。

## 本地请求集

`--request-jsonl` 接受 UTF-8 JSONL，每个非空行是一个完整的 Chat 或 Completions
请求对象；可在同一次运行中使用不同提示词、文本消息历史和生成参数。
所有请求必须使用同一个 served model，显式 `--model` 必须与每行一致。
它与 `--prompt`、`--request-json`、生成参数覆盖互斥；仍拒绝正文中的凭据与 headers，
鉴权沿用 `--api-key-env`。不自动生成内容、加载数据集、套用 chat template 或下载 tokenizer。

```bash
acprof load --base-url http://127.0.0.1:8000 --protocol openai-chat \
  --request-jsonl examples/streaming/qwen3-requests.jsonl \
  --requests 3 --warmup 1 --scenario concurrent --concurrency 2 \
  --output-dir results/qwen-request-dataset
acprof report results/qwen-request-dataset
```

[三请求示例](../../examples/streaming/qwen3-requests.jsonl)分别指定 32、64、96 的输出上限；
实际输出 Token 数仍来自响应证据，不能用上限代替。JSON 字符串 `"00123"` 保持文本，
空行不计作请求，诊断与来源保留物理行号。文件最多 64 MiB、100000 条请求，每个物理行最多
4 MiB（含行结束字节）；目录、非法 UTF-8／JSON、非有限数值和不支持的请求形态明确报错。
整个文件在创建输出、身份 probe 和 warmup 前读一次并校验，未选中的行也必须有效。

正式负载取前 `--requests` 条，warmup 单独取前 `--warmup` 条；每个 sweep 点重新从索引 0 开始。
文件必须至少有 `max(requests, warmup)` 条请求；保留既有 CLI 默认值 `100`／`5`，小文件应显式设置。
不自动打乱、补采样、循环、过滤失败行或改变到达 seed。重复输入只能由文件显式提供。
准备后的输入在本次调用内冻结，执行途中原文件变化不影响已准备的点。
warmup 与正式输入可能重叠，服务缓存也不清空，不能将此运行解释为冷缓存或真实生产流量。

请求集实验及阶段结果使用 schema v3；单请求模式继续使用 v2。v3 的 `identity.request_dataset`
保存 `selection=listed_prefix`、所需前缀的规范化 `bodies` 与有序 SHA256，`source` 保存原文件路径、
原始字节 SHA256、非空行数 `rows` 及所保留请求的物理 `lines`。原始提示词随输入身份保存在 JSON，
不在 sweep 汇总中复制。它不同时保存误导性的单个 `request_body`。
每阶段 `protocol.request_inputs_sha256` 绑定其实际前缀；每条记录的零基 `input_index`
按调度序号选取输入，独立于线程完成顺序，失败、丢弃和取消不移动后续输入。
文件读取、规范化、序列化和输入 hash 均在负载计时前完成；索引注记在计时后添加。
现有 TTFT／TPOT／ITL／E2E、SLO 与吞吐公式保持不变，原文件删除后仍可离线校验和生成报告。

输入准备参考 [vLLM CustomDataset](https://github.com/vllm-project/vllm/blob/4bfcd5d65fa8ea24e33d3b8296d204dea2f5bfc3/vllm/benchmarks/datasets/datasets.py)
（Apache-2.0）的逐请求输入与输出预算，以及[请求 ID 唯一性修复](https://github.com/vllm-project/vllm/pull/64537)
和[结果保存数据集身份的讨论](https://github.com/vllm-project/vllm/pull/62089)。沿用本项目调度器、
SSE 校验和有限 JSON 解析，不复制其采样实现，也不增加 pandas／datasets／tokenizer 依赖。

## 多负载点编排

端点模式可用 `--sweep-concurrency` 指定并发数列表，用 `--sweep-rate` 指定到达率列表
（单位 requests/s）。两者均为逗号分隔的正数，保持输入顺序，拒绝空项与重复值；
并发数必须是整数，到达率必须有限。并发列表要求 `--scenario concurrent` 或
`arrival-rate`，到达率列表要求 `arrival-rate`。列表覆盖对应的单值
`--concurrency`／`--rate`；未提供列表的轴沿用单值。两个列表同时使用时，
按并发数外层、到达率内层枚举笛卡尔积，最多 256 点。

```bash
acprof load --base-url http://127.0.0.1:8000 --protocol openai-chat \
  --model Qwen/Qwen3-0.6B --prompt "Explain throughput in one sentence." \
  --scenario arrival-rate --sweep-concurrency 1,2 --sweep-rate 1,5 \
  --requests 20 --warmup 1 --max-tokens 64 \
  --output-dir results/qwen-sweep --dry-run
```

`--dry-run` 必须与至少一个列表搭配：校验全部负载点及请求配置，并输出 JSON 计划，
不创建目录、不获取测量锁、不访问服务（包括身份 probe）。它仍会读取请求文件、所需鉴权
环境变量及可选本地 tokenizer；不证明服务可达或目标输出目录可写、为空。
检查计划后，移除 `--dry-run` 才会执行；实际执行要求输出目录为空。
全部点的 `LoadConfig` 和到达计划会在创建输出、发送任何请求之前校验。

所有点串行执行，共享一次测量锁。请求体、鉴权、tokenizer 和 SLO 配置在开始前准备一次，
每点独立执行所选身份 probe、串行关闭连接的 warmup 和正式负载；
`--requests`、`--warmup`、运行级字节／事件预算均按点生效。
各点沿用同一到达 seed 和输入，不重启服务、不清空服务缓存，也不自动添加重复实验。
顺序运行的不同负载点不构成同一条件下的独立重复，不能据此计算跨独立实验区间。

根目录的 `sweep.json` 是 schema v1、kind `stream_load_sweep` 的执行清单；
每点写 schema v2（单请求）或 v3（请求集）单点报告 `points/0001/load.json`、`points/0002/load.json` 等。
单点测量值、请求身份、服务观测和阶段错误以各自 `load.json` 为准，清单不复制请求记录。

| 字段 | 含义 |
| --- | --- |
| `run_id` | 本次 sweep 的唯一 ID；与各点的 ID 分开 |
| `execution_order` | `listed_concurrency_then_rate`，按列表顺序先并发、后到达率 |
| `status` | `running`、`complete`、`failed` 或 `cancelled`；`complete` 表示全部点正常返回，可含请求失败 |
| `successful` | 仅全部点成功且编排正常结束时为 `true` |
| `points[]` | 按执行顺序记录 `point_id`、相对 `report_path`、完整 `load_config`、`status`、`successful`、`run_id` |
| 点状态 | 初始 `not_run`，然后 `running`，最终 `complete`／`failed`／`cancelled`；未返回前 `run_id` 为 `null`，未运行及运行中 `successful` 为 `null` |
| `error` | 根或点异常时的可选类型与诊断，含异常 notes；正常请求失败的明细在单点报告内 |

清单在开始、每点前后和退出时原子发布。普通请求失败会使该点及整个 sweep 不成功，
但继续下一点；预热失败、抛出异常或取消会停止，后续点保持 `not_run`，不创建其目录。
`report_path` 是计划写入位置，不保证文件已发布。清单写失败会传播错误；
已有异常仍保留为主异常，写失败追加到 notes，磁盘清单可能停在上次成功发布的状态，
已发布的分点报告保持可读。该入口没有 sweep 恢复功能。

`acprof report results/qwen-sweep/` 可生成离线 `load_sweep_report.html`，切换并发／到达率轴与
吞吐、Goodput、P99 时延指标，并保留失败、未运行和部分证据；不同条件分组、空值断线。
口径与限制见[多负载点容量报告](../results/reports.md#多负载点流式容量报告)。
`acprof report results/qwen-sweep/points/0001/load.json` 仍可生成对应单点容量 HTML；
根清单不作为正式 CSV 或多实验比较输入。
设计参考了 [vLLM sweeps](https://github.com/vllm-project/vllm/blob/0f0172d90329d0dc3d867c0881d156f25c16b8bf/docs/benchmarking/sweeps.md)
的显式参数组合、预览与分点产物（Apache-2.0），复用本项目单点测量与原子写入实现，
不复制源码或增加 vLLM 依赖。

## 采样点与单位

所有时刻是客户端 `perf_counter` 相对本次负载起点的秒数，不是服务端调度时刻、
网卡时刻或 GPU 内部 Token 生成时刻。同一次网络读取中完整解析出的事件共用接收时刻，
不能用解析循环执行时间人为制造输出间隔。

| 字段 | 定义 |
| --- | --- |
| `planned_s` | Scheduler 计划提交时刻 |
| `sent_s` | 调用 HTTP 发送前；包含连接建立、网络、服务端排队等后续耗时 |
| `first_output_s` | 首个有效非空 content／reasoning／reasoning_content 或有明确 Token 证据的输出 |
| `last_output_s` | 最后一个有效输出事件到达 |
| `stream_completed_s` | 完整 `[DONE]` SSE 事件到达 |
| `completed_s` | 正常流的逻辑完成时刻；失败请求为可保留的结束观测，不代表成功 |
| `transport_completed_s` | HTTP 响应正文读取完成；独立于逻辑结束标记 |
| `output_events` | 有效输出事件、接收时刻、可用 Token 证据与来源 |
| `finish_reason` | 服务端终止原因；`length` 表示达到生成上限，不能解释成完整自然回答 |

role-only、usage-only、心跳、空文本元数据、结束标记不产生虚假输出时间点。
读到逻辑结束后仍完成 HTTP 响应读取，连接可复用性经过检查；
后续错误、正文不完整或要求复用但服务端关闭连接仍会使该请求失败。

正式 `duration_s` 在全部请求传输完成、worker 和连接收尾后固定。事件 Token 求和、全文拼接、
ITL 列表复制、可选 tokenizer 计数和指标派生随后统一执行一次，不在 worker 内重复聚合。
窗口外发现的协议错误仍令请求失败；可选 tokenizer 错误只记录 `token_count_error`，保留已观测时延，
不伪装成服务端失败。tokenizer 与模型 ID／revision 不匹配时明确拒绝，此计数不等同于服务端生成 Token 数。

### 运行级资源上限

单响应仍受 1 MiB SSE 事件、64 MiB 正文和 100000 输出事件限制。每次 warmup 或正式 run 另有
独立、跨 worker 原子累计的硬预算：默认最多 256 MiB 原始 SSE 字节和 1000000 输出事件。
可用 `--max-run-response-bytes` 与 `--max-run-output-events` 设置正整数；二者进入 adapter 与
identity。预算包括失败请求已接收的数据，避免仅凭并发或单响应限制造成全实验内存无界增长。
超限分别以 `run_response_limit`／`run_output_event_limit` 失败，实验不再成功，后续请求不再发送；
不静默截断原始证据。原始事件仍在预算内保留，报告生成也不应被解释为流式存储。

## 指标约定

设 `s` 为发送时刻，`f` 为首输出时刻，`l` 为末输出时刻，
`d` 为逻辑流完成时刻，`N` 为可信输出 Token 数：

| 字段 | 公式及有效范围 | 聚合 |
| --- | --- | --- |
| `ttft_s` | `f - s`；完成且有有效输出 | 每成功请求一个值 |
| `output_e2e_s` | `l - s`；截至最后输出 | 每成功请求一个值 |
| `e2e_s` | `d - s`；截至逻辑流结束，包含末输出后的 usage／finish 等等待 | 每成功请求一个值 |
| `tpot_s` | `(l - f) / (N - 1)`；完成、`N > 1` 且至少两个有效输出事件 | 每成功请求一个值 |
| `itl_s` | 有明确逐 Token 接收事件证据时，相邻 Token 事件时刻之差 | 汇集全部有效 Token 间隔 |
| `output_event_intervals_s` | 相邻有效输出事件时刻之差；可含多 Token 事件 | 汇集全部有效输出事件间隔 |

`tpot_s` 使用输出事件边界与总 Token 数，是客户端观测下的请求级摊销值；
`tpot_precision` 保留精度说明，不能外推成服务端逐 Token Decode 时间。
只有一个合并输出事件且 `N>1` 时，该值缺失，不报告虚假的零 TPOT。
对于单 Token，TTFT 可以有效，TPOT／ITL 缺失；零 Token 或无有效输出时首 Token
相关指标缺失。完整空响应与中途断流分别记录，不把未知 Token 数改为零。

文本 SSE 的事件数即便恰好等于 usage Token 数，也不能证明逐 Token 接收。
只有每个输出事件包含明确的单 Token 证据，且与输出总数一致时才提供 `itl_s`。
多个 Token 合并到一个事件时不补零、不平均摊造时间戳；
`output_event_intervals_s` 仍可用，但不能改名成精确 Token ITL。
这些接收事件始终是客户端观测，不等于模型内部生成时间戳。

汇总位于 `load.json.result.streaming`，各分布包含
`mean / p50 / p90 / p95 / p99 / sample_count`，分位数使用线性插值
`position=(n-1)*p`。TTFT、TPOT、E2E 按请求等权；ITL 与输出事件间隔分别池化，
因此长回答可能贡献更多间隔样本。`sample_unit`、`missing_requests`、
`requests_with_samples` 及错误分组说明分母；无有效样本是 JSON `null`。
失败、取消、丢弃请求不进入成功指标分布，原始观测留作诊断。

成功请求的延迟及区间样本必须为有限非负数，布尔值、数字字符串及无法表示的数值均拒绝；
`itl_s`／`output_event_intervals_s` 必须为数组或 `null`，数组内不能含 `null`。
合法零值保留，缺失字段／`null` 和空区间数组沿用缺失计数，不把错误类型替换成缺失值。
汇总和 SLO 评估共用数值校验；未配置 SLO 时，v2 报告仍检查全部成功请求的延迟证据。

## SLO Goodput 与单点容量

schema v2/v3 的 `result.capacity` 以同一个 `result.duration_s`（记为 `D`）汇总单次正式负载，
预热和 tokenizer 后处理不进入该窗口。`counts` 分别保存 offered、sent、succeeded、failed、
dropped、cancelled；`client_concurrency` 是客户端 Scheduler worker 上限，不是服务端实际并发。
`successful_requests_per_s` 为成功请求数除以 `D`，目标到达率仍由原有字段单独保存。
`success_rate` 为成功请求数除以 offered 请求数；`sent_requests_per_s` 以非空
`sent_s` 计数，`completed_requests_per_s` 以非空 `stream_completed_s` 计数，再各自除以 `D`。
流完成后仍可能因协议、连接收尾等原因失败，因此完成吞吐不等于成功吞吐，也不按 HTTP 状态码计数。
单点与扫描的离线读取共享这些摘要的一致性检查，具体失败及历史兼容行为见[容量报告](../results/reports.md#单负载点流式容量报告)。

对每个成功请求，已配置的 TTFT／TPOT／E2E 指标按以下三值规则评价：任一已知值严格大于阈值即
`violated`；没有已知违反但至少一个必需指标为 `null` 即 `unevaluable`；全部指标存在且不超过阈值
才为 `satisfied`。阈值相等视为满足。单 Token、合并输出事件或未知 Token 导致的 TPOT 缺失不能按零计算。
失败、取消和丢弃请求不进入 satisfied／violated／unevaluable 计数，但也不会计入满足数。

设满足请求数为 `A`，则 `goodput_requests_per_s = A / D`。未配置任何 SLO 时 goodput 为 `null`，
不以成功吞吐替代；配置后没有满足请求时 goodput 是有效的 `0`。`slo` 保存阈值、三类请求计数和
逐指标计数；报告可由原始请求、时长和声明阈值重新计算。单次 `load` 只有一个负载点，不能据此宣称
容量曲线、拐点或最大容量。

## Token 计数与可复现信息

优先使用通过类型、范围及一致性校验的服务端
`usage.completion_tokens`／`usage.prompt_tokens`，明确零值不触发回退。
累计 usage 必须覆盖最后一个输出事件；中途计数后仍有新增输出但没有更新计数时，
`output_usage_status=stale_before_last_output`，保留原始 usage 并改用可信回退或缺失值。
同一 SSE 事件中的 usage 覆盖该事件的输出；每份同时提供三个计数的快照立即校验
`total=prompt+completion`，不跨快照拼接字段进行算术校验。
输出缺少 usage 时，若每个输出事件都提供明确 token_ids／logprobs 证据，可使用
`server_event_tokens` 计数；仍未知时可选匹配模型和 revision 的本地 tokenizer。
完整输出只编码一次，不能逐 delta 编码后相加；回退在所有负载请求结束、duration
固定后执行，避免计数 CPU 开销干扰发送节奏。计数失败保留成功请求的 TTFT／E2E，
输出 Token／TPOT／ITL 为 `null`，并记录 `token_count_error` 和缺失原因。
无法可靠计数时保留 `null` 与 `unavailable`。

本地 tokenizer 回退为可选依赖：

```bash
uv sync --locked --extra streaming-tokenizer
acprof load --base-url http://127.0.0.1:8000 --protocol openai-chat \
  --model Qwen/Qwen3-0.6B --prompt "Hello" \
  --model-revision EXACT_REVISION \
  --tokenizer-json /path/to/tokenizer.json \
  --tokenizer-model Qwen/Qwen3-0.6B --tokenizer-revision EXACT_REVISION \
  --requests 3 --output-dir results/qwen-tokenizer-fallback
```

本地文件加载不访问 Hub、不执行远程代码；身份参数必须匹配，文件 SHA256 随报告保存。
这验证声明与本地文件身份，不证明服务端真的加载了同一 revision；
服务端模型版本仍需验收证据。回退统计的是可见输出文本，可能不包括隐藏 reasoning、
特殊 Token 或其他服务端内部生成内容，不能冒充服务端真实总 Token 数。
`output_token_count_source`、`input_token_count_source`、`token_count_scope`
和 tokenizer 元数据一起解释数值。没有匹配的输入模板证据时，不从 prompt 字符数推测输入 Token 数。

端点实验保存请求参数／请求体身份、模型声明、协议、预热、连接和调度设置。

### 系统 Token 吞吐

`capacity.token_throughput` 分别汇总 input、output 和 total。每个分量的 `tokens` 与 `tokens_per_s`
只在全部成功请求的对应分量都已知时提供；任何一个成功请求缺失，完整总量和完整 Token/s 都为 `null`。
`observed_tokens` 与 `observed_tokens_per_s` 可描述已知子集，但必须同时读取 `covered_requests`、
`successful_requests` 和 `coverage_ratio`；没有观测时仍为 `null`，已知零 Token 是有效观测。
total 只累加同一请求 input 与 output 均已知的样本，不拼接不同请求的部分值。

输入 scope 由 `input_token_count_scope` 记录，服务端 prompt usage 为 `server_prompt_tokens`；
原有 `token_count_scope` 继续表示输出 scope。`scope_request_counts` 分别保存 input、output 及
total 的 scope 组合请求数，total 以 `input_scope+output_scope` 命名；`source_request_counts`
分别保存输入和输出的计数来源。这样不会把服务端输入与本地可见文本输出的组合误称为纯服务端总 Token。
这些速率是本次客户端成功请求集合在整个 `D` 上的聚合，不是服务端全局吞吐、其他租户流量、
Decode 阶段吞吐、`1 / TPOT` 或每秒峰值。

### 服务身份

`service_identity` schema v1 分开保存 `declared`、`observed`、`probe`、`matches` 和 `mismatches`。
声明的 backend、版本、模型、revision 与 SLO 进入 `identity_sha256`；运行中观测不进入该哈希。
未观测或无法取得的字段保持 `null`，声明与观测冲突时双方保留并记录 mismatch，不互相覆盖。

`--service-probe vllm` 在预热前请求 `/version` 和 `/v1/models`，复用现有 Authorization 且不落盘，
连接关闭、响应限制为 1 MiB。模型列表仅保留 `id/root/max_model_len`，按请求模型 ID 精确且唯一匹配；
未找到或重复匹配时选中模型为 `null`。`root` 不能解释为 checkpoint revision。只有取得有效
`/version` 才能把观测 backend 记为 `vllm`；通用 `/v1/models` 响应不能证明后端身份。
单个端点失败时身份为 `partial`，两者均失败为 `unavailable`，均成功为 `available`；探测或 mismatch
不改变推理请求状态。
仅靠标准 OpenAI 端点无法确定缓存策略、dtype、硬件、调度上限及实际 revision，不能从模型名猜测。

`--service-probe vllm-config` 在上述两个身份请求之后，额外读取一次
`/server_info?config_format=json`，仅接受对象形式的 `vllm_config`；不解析字符串配置。
这是显式选择的扩展模式，原有 `vllm` 仍只发出两个请求。上游将 server-info 放在
[开发路由](https://github.com/vllm-project/vllm/blob/fce61a05a76ccf34e1af6511ff7071affc306424/vllm/entrypoints/serve/__init__.py)；
服务需要已开放该端点，AC-Prof 不改变服务启动设置。未开放、拒绝访问或解析失败时，
保留错误和可用的身份观测，不能把配置未知解释为配置相同。

```bash
acprof load --base-url http://127.0.0.1:8000 --protocol openai-chat \
  --model Qwen/Qwen3-0.6B --prompt "Explain throughput in one sentence." \
  --requests 5 --warmup 1 --service-backend vllm --service-probe vllm-config \
  --output-dir results/qwen-vllm-config
```

配置沿用 `observed.configuration`：`source=/server_info?config_format=json`、
`scope=reported_engine_configuration`，`values` 使用下面的点分路径作为键。
原始响应中的 `vllm_env`、`system_env`、完整配置、凭据和其他对象不会保存。

| 路径前缀 | 白名单字段 | 类型与含义 |
| --- | --- | --- |
| `model_config` | `model`、`revision`、`tokenizer`、`tokenizer_revision`、`dtype`、`quantization` | 非空文本或 `null`；服务报告的引擎配置，保留 `auto` 等原值 |
| `model_config` | `max_model_len` | 正整数或 `null`；引擎配置上限，单独保留，不覆盖模型列表的同名值 |
| `parallel_config` | `tensor_parallel_size`、`pipeline_parallel_size`、`data_parallel_size` | 正整数或 `null`；配置值，不相乘推导实际设备数或客户端并发 |
| `parallel_config` | `distributed_executor_backend` | 非空文本或 `null` |
| `scheduler_config` | `max_num_seqs`、`max_num_batched_tokens` | 正整数或 `null`；调度上限，不是本次实际使用量 |
| `scheduler_config` | `enable_chunked_prefill` | 布尔值或 `null`；不能替代 Prefill 阶段计时 |
| `cache_config` | `cache_dtype`、`block_size`、`enable_prefix_caching`、`gpu_memory_utilization` | 分别为非空文本、正整数、布尔值、`(0,1]` 内数值，均可为 `null`；最后一项是内存预算比例，不是 GPU 利用率或能耗 |

缺失和显式 `null` 保持未知。字段或容器形状错误记入 `probe.errors["configuration.<path>"]`，
保留其他有效字段；非有限 JSON、无效响应或 `vllm_config` 形状错误使整个配置不可用。
新模式的 `probe.requests` 记录三个请求的路径、HTTP 状态与访问错误；沿用每响应 1 MiB 上限、
关闭连接、鉴权不落盘及取消传播。没有静默跳转、重试或其他路径回退。
有错误但仍取得观测为 `partial`，全部不可用为 `unavailable`，无错误为 `available`；
`available` 不保证所有可选字段都有值。

引擎配置中的模型路径和 revision 不能证明当前请求所用的模型或 LoRA 身份。
`observed.model` 仍由 `/v1/models` 精确且唯一匹配，`observed.revision` 仍保持 `null`；
版本端点无效时不借助配置猜测 backend。配置不进入声明哈希，但进入 sweep 条件分组，
配置变化或观测缺失时不与其他条件连接成同一曲线。所有请求仍在每点 warmup 之前完成。
接口与类型核对了上游 [JSON server-info](https://github.com/vllm-project/vllm/blob/fce61a05a76ccf34e1af6511ff7071affc306424/vllm/entrypoints/serve/dev/server_info/api_router.py)
及 [CacheConfig](https://github.com/vllm-project/vllm/blob/fce61a05a76ccf34e1af6511ff7071affc306424/vllm/config/cache.py)
（Apache-2.0）；复用本项目 HTTP、字段校验和报告实现，没有引入 vLLM 依赖。

`--service-probe sglang` 先请求 `/server_info`，再请求 `/v1/models`，沿用相同的鉴权、
连接关闭与 1 MiB 响应限制。仅当 `/server_info` 明确返回 HTTP 404 时，才请求旧版
`/get_server_info`；401／403、其他 HTTP 错误、超时或无效 JSON 均不会触发旧路由。
`probe.requests` 按顺序保存请求路径、可确认的 `http_status` 和可选错误；
状态无法确认时为 `null`。旧路由成功可得到 `available`，最初的 404 访问记录仍保留。
默认 `--service-probe none` 不发送身份请求，source replay 不接受这些端点选项。

```bash
acprof load --base-url http://127.0.0.1:30000 --protocol openai-chat \
  --model Qwen/Qwen3-0.6B --prompt "Explain throughput in one sentence." \
  --requests 5 --warmup 1 --service-backend sglang --service-probe sglang \
  --output-dir results/qwen-sglang
acprof report results/qwen-sglang/load.json --output results/qwen-sglang/capacity.html
```

有效的 server-info `version` 建立 SGLang backend／版本观测；
当前请求模型仍必须在 `/v1/models` 中精确且唯一匹配，不能拿启动配置中的
`served_model_name` 替代。模型列表中的 `root` 可能只是服务别名，不等同于模型文件路径。
server-info 返回的白名单字段保存在可选的 `observed.configuration` 中：
`source` 是实际使用的路径，`scope=reported_startup_configuration`，
`values` 中字段规则如下。

| 字段 | 类型与含义 |
| --- | --- |
| `model_path`、`served_model_name`、`revision` | 非空文本；服务报告的启动模型配置，不能证明热更新后的 checkpoint 身份 |
| `dtype`、`kv_cache_dtype`、`quantization`、`schedule_policy` | 非空文本；按服务返回保留，包括 `auto`，不推断最终精度或调度行为 |
| `tp_size`、`pp_size`、`dp_size`、`attn_dp_size` | 正整数；保留上游并行配置字段，不相乘推导设备数或实时并发 |
| `context_length`、`max_running_requests`、`max_total_tokens`、`max_prefill_tokens` | 正整数；服务端启动配置的上下文、请求及 Token 限制，不是本次实际使用量 |
| `disable_radix_cache` | 布尔值；原样保留 Radix Cache 禁用配置，不替代缓存命中测量 |

缺失或显式 `null` 均保持未知，不填上游默认值。字段类型或范围错误时只将该字段置为
`null`，在 `probe.errors["configuration.<field>"]` 记录原因，
整体状态为 `partial`，其余有效观测保留。
任一端点失败也会保留另一端点证据；全部不可用时 `observed=null`、状态为 `unavailable`。
取消会向外传播；普通身份探测失败与 mismatch 不改变推理请求成功状态。
实际加载的 `observed.revision` 仍为 `null`，启动配置的 revision 不补填到该字段。
服务启动配置和其他运行观测均不进入声明的 `identity_sha256`。

容量 HTML 单独展示配置来源、scope、字段值及 probe 错误；历史文件缺少配置时不补造数据。
只保留上述字段，不保存返回对象中的 API key、启动命令、环境变量或内部调度状态。
每个 sweep 点仍在自己的 warmup 前重新 probe。
协议核对了 [SGLang v0.5.5 HTTP server](https://github.com/sgl-project/sglang/blob/v0.5.5/python/sglang/srt/entrypoints/http_server.py#L498)、
[当前 server-info 与模型列表](https://github.com/sgl-project/sglang/blob/93cfea6b12a908161719edf0e3a732e771ac9fad/python/sglang/srt/entrypoints/http_server.py#L859)
及上游 [404 旧路由兼容逻辑](https://github.com/sgl-project/sglang/blob/93cfea6b12a908161719edf0e3a732e771ac9fad/sgl-model-gateway/src/core/steps/worker/local/discover_metadata.rs#L128)
（Apache-2.0）；复用 AC-Prof HTTP 与单点报告，不引入 SGLang 依赖。

## 服务端阶段统计

可显式添加 `--server-metrics vllm`，从同一 origin 的 `/metrics` 读取服务级阶段直方图：

```bash
acprof load --base-url http://127.0.0.1:8000 --protocol openai-chat \
  --model Qwen/Qwen3-0.6B --prompt Hello --requests 10 --warmup 1 \
  --server-metrics vllm --output-dir results/qwen-server-phases
acprof report results/qwen-server-phases
```

默认关闭，不增加请求。启用后，每个负载点在 warmup 完成后、`run_load()` 之前抓取一次，
并在负载返回或抛出异常后再抓取一次；均位于现有锁范围内、正式负载 `duration_s` 之外。
不轮询、不刷新服务缓存、不等待 exporter 补齐，也不启动或更改服务。沿用环境变量鉴权；
关闭连接、不跟随重定向、不重试。每次最多读取 4 MiB，连接／读取无进展超时为
`min(--request-timeout, 10)` 秒；只接受请求的 Prometheus `text/plain` 格式。

精确选择 `model_name` 等于请求模型名的序列，保留全部标签（包括 `engine`）。
不匹配的模型不纳入；若服务用另一个别名导出指标，会明确失败，不猜测映射。
仅保留以下 histogram 的 `_count`、`_sum` 和可用的 `_created`：

| 阶段 | vLLM metric |
| --- | --- |
| Queue | `vllm:request_queue_time_seconds` |
| Prefill | `vllm:request_prefill_time_seconds` |
| Decode | `vllm:request_decode_time_seconds` |
| Inference | `vllm:request_inference_time_seconds` |
| E2E | `vllm:e2e_request_latency_seconds` |

每个标签序列分别计算 `Δcount = after.count - before.count` 和
`Δsum_s = after.sum_s - before.sum_s`，检查后再跨序列相加；
均值为 `ΣΔsum_s / ΣΔcount`，分母是该阶段的服务端观察数，绝不使用客户端请求数替代。
没有新观察时保留 count=0、sum=0，mean 为 `null`；有效零时长观察的 mean 可以为 0。
未导出的阶段、标签集合变化、计数下降、创建时间变化、无新增计数却增加 sum、
聚合溢出均有独立不可用原因，不填零。可用的无标签 `process_start_time_seconds`
变化或消失也使差值不可用；两次快照无法排除所有未观测到的重启。
最多保留 4096 个阶段／标签序列；重复、残缺、非有限或非整数计数拒绝。
count 不超过 2⁵³；显式样本时间戳尚不支持，不能将缓存样本冒充本次抓取。

**这是服务级累计观察，不是本次请求的精确阶段分解。** 其他流量可能贡献样本，
exporter 延迟可能纳入 warmup／更早请求，或遗漏刚完成的请求；数量相同也不能证明独占。
这些阶段是服务报告的请求生命周期时间，不能解释为 GPU kernel 时间、阶段资源、能耗，
也不与客户端 TTFT／E2E 拼成同一时间轴；不从两个 sum/count 快照推导分位数。

v2/v3 的可选 `identity.server_metrics` 声明选择、边界和读取预算，参与 identity 哈希与
sweep 条件分组。可选 `server_metrics` 使用独立整数 schema 1，保存 before／after
快照、相对观测起点的开始／结束秒数、原响应 SHA256／字节数、筛选后的序列及重算摘要。
不保留整个 exposition 或 Authorization；SHA256 标识当时读取的响应，不能单凭摘要
独立重建或验证未保留的响应正文。离线报告直接校验保留的序列、差值、时间边界和声明；
缺少本扩展的历史报告继续读取，不回填观测。

HTTP、解析或清理失败会使实验失败：before 失败不启动正式请求，after 失败保留已有
result；负载异常后仍尝试 after，二次错误不覆盖原始异常，sweep 停止后续点。
个别指标缺失或重置只影响服务统计，不改变已完成的客户端请求成功率；CLI 显示状态与原因。
单点 HTML 展示阶段表，sweep 逐点展示有界详情；客户端曲线保持原有口径。

上游依据为 [vLLM metrics 定义](https://github.com/vllm-project/vllm/blob/e336407a3ddb204edf6cfdfdd002cb28b9715438/docs/design/metrics.md)
及 [V1 histogram 实现](https://github.com/vllm-project/vllm/blob/e336407a3ddb204edf6cfdfdd002cb28b9715438/vllm/v1/metrics/loggers.py)。
复用 [Prometheus client_python parser](https://github.com/prometheus/client_python/blob/master/prometheus_client/parser.py)
（安装包许可证为 Apache-2.0 AND BSD-2-Clause）；新增一个纯 Python host 依赖，
`uv.lock` 固定版本。解析只在负载外执行，
不引入 vLLM、GPU 或 Docker 依赖，也不复制上游实现。

## 失败及结果边界

| 情况 | 处理 |
| --- | --- |
| HTTP 非 200／非 SSE 响应 | 失败，保留 HTTP 状态 |
| HTTP 200 中的 `error` 或 SSE error 事件 | 失败，即使此前已有输出 |
| 无 `[DONE]`、截断 UTF-8／SSE、无效 JSON、正文断流 | 协议或传输错误，不计完成成功样本 |
| SSE JSON 含孤立 Unicode 代理 | `invalid_unicode`，请求及实验失败，最终 `load.json` 仍可发布 |
| 超过 run 级字节／事件硬预算 | 对应 run limit 错误，不发送后续请求，实验不成功 |
| 读取无进展 | timeout，保留已收到的输出事件 |
| 用户中断 | cancelled，终止活动连接，保留当前阶段的部分报告；后续清理失败仍报错 |
| 不支持的多选择、tool/function、无效计数 | 明确失败或明确不可用，不静默替换 |
| 无 Token 计数证据／非逐 Token 流 | 相应指标 `null`，记录缺失原因 |

request JSON 中的孤立代理会在联网和创建输出前拒绝；probe 中的同类文本按端点失败处理，
不进入身份产物。合法 surrogate pair 与正常 Unicode 不受此限制。

预热和正式负载的部分证据分别位于 `warmup`／`result`，在外层清理前记入内存报告；阶段异常另存
`phase_errors`，不被后续清理错误覆盖。最终报告写入发生普通异常时保留活动异常并附加写失败诊断，
磁盘文件可能仍是初始 `running` 状态。字段及发布边界见[负载报告](../results/comparisons.md#非流式负载报告)。

单请求运行使用独立的 `stream_load_experiment`／`stream_load_result` schema v2，请求集使用 v3，均保存到新输出目录；
容量报告只读兼容 v1，缺少的 SLO、Token 吞吐和服务身份显示 unavailable，不回填为零或改写历史文件。
原来的 `nonstream_load_*` schema v1 继续读取和执行。
流式结果不写入正式窗口的指标登记表／能耗 CSV，也不混入已有实验的 resume 或补采。
同一共享 GPU 上的并发请求不能直接分别使用整卡能耗作为每请求或每 Token 精确能耗；
未来能耗扩展必须声明归因方法和窗口。

## 测试与上游对照

`tests/execution/operations/test_streaming.py` 用受控时间与字节流核对协议、公式、缺失值；
`test_streaming_load.py` 用真实本地 HTTP 验证两个端点、复用、并发、Poisson、
断流、超时、取消、窗口外一次聚合、运行预算和 Unicode 失败发布；`test_streaming_cli.py` 验证入口、身份与失败产物。
`test_streaming_capacity.py` 独立重算 SLO 与 Token 覆盖，`test_service_identity.py` 验证有界 vLLM 观测；
`tests/analysis/reporting/test_load_report.py` 覆盖 v1/v2 报告、原始记录重建、防篡改与拒绝覆盖。
`test_streaming_request_dataset.py` 与 `test_load_dataset_report.py` 覆盖 JSONL 输入冻结、前置拒绝、
顺序映射、v3 校验、历史兼容及不同请求集的 sweep 分组；浏览器用例检查来源展示和窄屏。
`test_server_metrics.py` 与 `test_server_metrics_report.py` 覆盖服务阶段快照边界、错误发布、
精确模型与标签选择、重置、独立差值期望及离线一致性；headless browser 检查单点与 sweep 展示。

例如 `s=0`，输出事件在 `0.10 / 0.14 / 0.18` 秒到达，
usage 为 5 Token，`[DONE]` 在 `0.23` 秒到达：

- TTFT 为 0.10 s，TPOT 为 0.02 s/token。
- `output_e2e_s=0.18`，`e2e_s=0.23`。
- 输出事件间隔为 0.04、0.04 s；缺少逐 Token 证据时 ITL 为 `null`。

真实 Qwen3-0.6B + vLLM 验收需记录实际版本、checkpoint/tokenizer revision、
prompt、生成参数、thinking、prefix cache、预热和并发。先确认环境可用再运行，
不能把上述离线测试当作 GPU／模型验收。两次工具运行的性能值不要求完全相同；
先逐请求重算公式，再比较同口径分布和运行间差异。

### 复现 Qwen 与 vLLM 对照

P0 已使用 Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`
与 `vllm[bench]==0.11.2` 验证两个端点、连接复用和 Chat 并发，并完成 Completions 的
`vllm bench serve` 对照及独立公式重算。在独立环境固定 vLLM 版本（包含 custom
数据集依赖），预先准备同一 revision 的 checkpoint 与 tokenizer。服务配置示例：

```bash
vllm serve /path/to/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8000 --dtype half --max-model-len 512 \
  --max-num-seqs 4 --max-num-batched-tokens 512 --gpu-memory-utilization 0.55 \
  --enforce-eager --no-enable-prefix-caching --generation-config vllm
```

比较 Completions 时使用相同原始文本，不套用 chat template；固定 `temperature=0`、
生成 `seed=7`、`max_tokens=32`、`ignore_eos=true` 和 `repetition_penalty=1`。
将示例请求转换成 vLLM 的 custom 数据集，再分别运行：

```bash
python - <<'PY'
import json
from pathlib import Path

body = json.loads(Path("examples/streaming/qwen3-completions.json").read_text())
Path("prompts.jsonl").write_text(
    "".join(json.dumps({"prompt": body["prompt"]}) + "\n" for _ in range(4))
)
PY

acprof load --base-url http://127.0.0.1:8000 --protocol openai-completions \
  --request-json examples/streaming/qwen3-completions.json \
  --connections reuse --requests 4 --warmup 1 --output-dir results/acprof-streaming

vllm bench serve --base-url http://127.0.0.1:8000 \
  --backend openai --endpoint /v1/completions --model Qwen/Qwen3-0.6B \
  --tokenizer /path/to/Qwen3-0.6B --dataset-name custom --dataset-path prompts.jsonl \
  --skip-chat-template --disable-shuffle --custom-output-len 32 --num-prompts 4 \
  --max-concurrency 1 --request-rate inf --temperature 0 --ignore-eos \
  --extra-body '{"seed":7}' --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 --save-result --save-detailed \
  --result-dir results/vllm-streaming --result-filename completions.json
```

这是一组小样本协议／公式验收，不作为容量结论或稳定 P99 性能基线。
Chat 另用示例请求验证 role-only 帧不触发 TTFT，并可用
`--scenario concurrent --concurrency 2 --connections reuse` 验证共享 Scheduler 的并发路径。
vLLM `0.11.2` 的 [Chat 流客户端](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/benchmarks/lib/endpoint_request_func.py)
会将 role-only choice 和末尾 usage 纳入其时间边界，
因此此版本不用于 Chat TTFT／E2EL 的数值对齐。

vLLM 详细结果中的 `ttfts`、`itls`、`output_lens` 可独立核对分布。
该版本 Completions 首时间戳与 TTFT 使用相邻两次时钟读取，所以从
`sum(itls)/(N-1)` 重建 TPOT、从 `ttft+sum(itls)` 重建 E2EL 存在微小的时钟差；
不要宣称这些重建值是服务端实测或逐请求原始 latency。
实际验收应保存完整命令、依赖版本、checkpoint 哈希、原始结果和校验报告。

设计参考固定到
[vLLM 5d55ef1590 的 serving metrics](https://github.com/vllm-project/vllm/blob/5d55ef1590c92213e3893dc8e8712f7f2e0bc508/vllm/benchmarks/serve.py)、
[stream client](https://github.com/vllm-project/vllm/blob/5d55ef1590c92213e3893dc8e8712f7f2e0bc508/vllm/benchmarks/lib/endpoint_request_func.py)
及 [WHATWG SSE](https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream)。
该 vLLM 版本的 E2EL 截至最后输出，应与本协议 `output_e2e_s` 比较；
它的 ITL 使用流输出事件间隔，应与本协议 `output_event_interval_s` 汇总比较，
不能直接与严格 `itl_s` 相比。不同安装版本必须检查其实际源码。

[Ray LLMPerf](https://github.com/ray-project/llmperf) 与
[NVIDIA GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf)
仅作为记录／报告组织参考；同名指标可能使用不同分母，未引入这些框架依赖。
