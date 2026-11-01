# OpenAI-compatible 流式测量

[← 返回采集协议](protocol.md)

## 范围与入口

`acprof load` 可对已经启动的 OpenAI-compatible 服务执行独立 HTTP 测量。
`openai-chat` 使用 `/v1/chat/completions`，`openai-completions` 使用
`/v1/completions`；默认的源实验 `/predict` 重放不变。两种协议复用
`measurement.load` 的串行／并发／固定或 Poisson 到达调度和每工作线程连接管理。
本入口不启动或管理模型服务，不声明 Docker／Process 硬件采集能力，不产生正式能耗 CSV。

```bash
acprof load --base-url http://127.0.0.1:8000 --protocol openai-chat \
  --model Qwen/Qwen3-0.6B --prompt "Explain what a token is in one sentence." \
  --max-tokens 64 --temperature 0 --requests 5 --warmup 1 \
  --output-dir results/qwen-streaming

acprof load --base-url http://127.0.0.1:8000 --protocol openai-completions \
  --request-json request.json --scenario concurrent --concurrency 2 \
  --connections reuse --requests 8 --warmup 1 \
  --output-dir results/qwen-completion-streaming
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

协议只接受单个文本生成选择（`n=1`）；tool/function、多选择等尚未支持的输出明确失败。
Streaming Adapter 强制请求 SSE 与 usage。它处理 UTF-8 跨网络读取、BOM、
LF／CRLF／CR、注释心跳、多行 data、JSON 错误、finish reason 和 `[DONE]`。
网络读取块、SSE 事件、模型 Token 是三种不同单位。

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
未被接口观测或使用者明确声明的服务端版本、缓存策略、硬件与运行参数保持未知，
不依据客户端模型名猜测。P1 将继续补充服务端元数据、吞吐量、SLO Goodput 与可视化。

## 失败及结果边界

| 情况 | 处理 |
| --- | --- |
| HTTP 非 200／非 SSE 响应 | 失败，保留 HTTP 状态 |
| HTTP 200 中的 `error` 或 SSE error 事件 | 失败，即使此前已有输出 |
| 无 `[DONE]`、截断 UTF-8／SSE、无效 JSON、正文断流 | 协议或传输错误，不计完成成功样本 |
| 读取无进展 | timeout，保留已收到的输出事件 |
| 用户中断 | cancelled，终止活动连接，保留部分报告 |
| 不支持的多选择、tool/function、无效计数 | 明确失败或明确不可用，不静默替换 |
| 无 Token 计数证据／非逐 Token 流 | 相应指标 `null`，记录缺失原因 |

该协议使用独立的 `stream_load_experiment`／`stream_load_result` schema v1，
保存到新输出目录；原来的 `nonstream_load_*` 协议继续读取和执行。
流式结果不写入正式窗口的指标登记表／能耗 CSV，也不混入已有实验的 resume 或补采。
同一共享 GPU 上的并发请求不能直接分别使用整卡能耗作为每请求或每 Token 精确能耗；
未来能耗扩展必须声明归因方法和窗口。

## 测试与上游对照

`tests/execution/operations/test_streaming.py` 用受控时间与字节流核对协议、公式、缺失值；
`test_streaming_load.py` 用真实本地 HTTP 验证两个端点、复用、并发、Poisson、
断流、超时、取消和统计；`test_streaming_cli.py` 验证入口、身份与失败产物。

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
