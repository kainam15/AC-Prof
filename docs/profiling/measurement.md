# 测量协议与输入契约

[← 返回专题目录](protocol.md)

## 协议不变量

- CPU package、估算 vCPU、GPU device 与 container-attributed 数据分别命名和解释，整机测量不能静默替换容器归因。
- 单请求、workload window、进程生命周期、独立 profiler 使用不同窗口与分母；比较前核对请求数、尺度单位及来源资源。
- 数值 `0`、CSV `nan`、JSON `null` 和字段缺失含义不同；未知历史数据不补成零，推导或代表资源复用不冒充独立实测。
- JSON 保留原生类型；schema 版本由协议与兼容策略决定。只接受当前 schema；可选指标不可用时保持 `nan`，不转换旧字段或推测来源。
- 输入计划、`input_scale_plan_sha256`、workload 素材与模型 revision 必须一致；派生字段复用已有采样，不额外发起请求。
- 静态对象描述与采集过程来源分开保存；补采和修复记录进入 `collection_history.json`，保留原始备份和可回溯信息。

## Profiling mode 与能力证据

`profiling_mode=full` 为默认，保留原有 native Linux、本机 Docker、cgroup v2、RAPL、硬件 instructions
和抓包要求。`basic` 允许 Native Linux 或 WSL2，但仍需本机 Docker 和 cgroup v2，仅要求 application latency、throughput、容器 CPU/memory；
能耗、PMU、packet latency 不请求、不启动、数值保持 `nan`。两种模式不能混作同一画像。
Torch、NCU、Massif、Nsys 仍是显式选择的额外工具，`full` 不意味着自动开启全部工具。

`--dram-energy auto` 在 full 中尝试 DRAM，作为可选能力记录；缺失不改变原有 full 必需项。
`off` 和 basic 的 DRAM 状态为 `not_requested`。只有 `required` 将 DRAM 加入
`requested_measurements`，要求完整 package 覆盖及有效的原始、effective J/request；
权限错误、部分覆盖、采样失败不能标为 verified。`required` 与 basic 组合明确报错。

`capability_report.json` 保存统一的 execution/measurement 对象：`available`（已声明或预检可用）、
`verified`（有当前运行证据）、`unsupported`、`permission_denied`、`not_requested`、`unavailable`、`error`。
每项含 source、detail 和 evidence，运行时验证关联 environment ID。声明不能直接标 verified；
实测缺值不能标零。合法的零值仍是数值，不是能力状态。

`static_meta.json.capability_report` 保存矩阵前的冻结快照，保护恢复时的准备产物身份；
独立 `capability_report.json` 在采集结束后补充实际 CSV/profiler 证据。
它记录主采集收尾时的能力快照；后续 posthoc 补采的状态以对应 profiler plan、CSV error 和
`collection_history.json` 为准，当前不会自动改写该主采集快照。
Capability Report schema v3 将 `requested_measurements_available`（全部要求可采集）与
`requested_measurements_complete`（全部要求已经 verified）分开；预检 available 不再算完成。
`collection_complete` 保留原有“采集行完成且成功”的含义，新增 `collection_finished` 与
`collection_succeeded` 分别表示行已有终态及行执行成功，`row_counts` 区分 succeeded、failed、
not_measured、unfinished。OOM／timeout 是已结束的失败；剪枝未尝试行单列为 not_measured。
这些是已提交审计的行状态，完整计划覆盖仍由 run_state/CSV 唯一键和 audit.coverage 检查。
旧报告不能区分的完成状态为 null，不把 available 或单独的 runtime `status=ok` 冒充验证证据。
读取兼容 schema v1/v2/v3，缺版本按 v1；v1/v2 缺少环境身份时保持 unknown。未知版本及非布尔完成状态明确报错，不隐式转换字符串。
`full_profile_complete` 仅在 native_linux、full 且上述两者满足时成立。工具 permission denied 或输出全为未知数值不能标完整。
主矩阵仍可保留其它成功指标与失败计划；成功测量行缺少当前模式的必需指标时，保存诊断并以失败退出。

模式参与实验恢复身份；旧参数中缺失模式解释为历史默认 full，历史结果不会凭空补出能力证据。
新增 `profiling_mode`、`capability_report` 是 static schema v7 的可选字段，旧 v7 文件继续可读。

### 环境身份与能力支持

static schema v7 新增可选 `platform`、`collection_tier`、`comparability_class`、`environment_class`
和 `platform_runtime`，新采集必须写入。`platform` 包含 environment/native、system/kernel/kernel_version、
machine、WSL generation 与识别证据；`platform_runtime` 保存 `acprof_version`、Git commit、Docker/runtime、GPU driver、
CUDA driver/NVML 版本和探测错误，GPU 硬件身份继续保存在 `gpu_device`。版本不可读时为 null/明确错误，
不合成版本。只在 Preparation 探测，不进入请求窗口。

`platform_runtime.acprof_version` 是无单位的版本字符串，直接取当前执行包的 `acprof.__version__`，
适用于所有安装方式，不依赖 Git 或硬件；已发布版本可对应 `v<version>` Tag。
源码 checkout 额外记录既有 `git_commit`；wheel/standalone 取不到 Git commit 时保持 null 并保留错误原因。
此字段是 static schema v7 的可选来源信息，旧文件缺失表示版本未知，读取时不得用当前安装版本回填。
版本不参与数值聚合，不改变测量窗口、环境比较或 resume 身份。

Capability Report v3 增加相同的环境身份及 `metric_support`，measurement 每项增加独立 `support`。
支持状态为 supported、partial、unsupported、requires_native_validation；它们不替代原来的实测状态。
CSV 新增文本列 `environment_class`（experiment 范围、无数值单位），包括失败占位行。
旧结果缺失、冲突身份读为 unknown，不由当前运行主机反推；CSV 合并、续跑和 Native baseline 拒绝混用。
FULL/PARTIAL 是平台上限，不等于采集成功或 full/basic 模式。完整边界见 [WSL2](../platforms/wsl2.md)。

## Startup probe 与冻结矩阵

startup probe 使用独立容器，只启动服务、等待 `/ready`、检查 Docker State 并记录证据。
它不发送 `/predict`，不启动 client、能耗监控、抓包或 profiler，也不写正式 CSV。
每种 GPU mode 在最低选中 CPU 下，按内存升序探测；仅 `ready=false`、
`State.OOMKilled=true`、`State.Running=false` 且未 restarting 的启动失败计入连续前缀。
第一个非 confirmed startup OOM 就停止该 GPU mode 的后续探测，不能跨过缺口扩大前缀。
错误文本、CUDA OOM、timeout、ready 后或正式采集中的运行期 OOM 均不能增加剪枝依据。
该前缀按现有资源单调性假设推广到其它选中 CPU；所有被剪枝资源配置（包括参考 CPU）
只生成 `result_origin=inferred_not_measured` 的占位行，指标为 `nan`，不冒充 probe 实测性能。

probe 证据使用 `startup_oom_pruning.json` schema v2，逐次原子保存启动结果、Docker State、
错误和时间。中断恢复可复用已记录的尝试；全部 probe 结束后才生成 `matrix_plan.json` schema v1。
禁用剪枝时直接冻结矩阵。正式 case 重新创建容器，不复用 probe 的容器或启动计时。

`matrix_plan.json` 保存完整实验身份、实际 case 顺序、每个 case 的实际 input scale 顺序、
`matrix_order`、`seed`（CLI 的 `--matrix-seed`）、`algorithm_version=sha256-sort-v1` 和 `plan_sha256`。
`seeded` 按 seed 与资源键的 SHA256 排序；input scale 由 seed 与该资源键派生独立 seed 后排序，
不共享可变 RNG 状态，也不受其它 case 数量影响。`declared` 保持参数和输入计划的声明顺序。
hash 对不含 `plan_sha256` 的规范 JSON 计算；探测时间戳不进入计划。
相同身份、probe 前缀、算法和 seed 得到相同计划；输入 payload 本身保持原物化计划不变。
独立随机源的设计参考 [MLPerf LoadGen TestSettings](https://github.com/mlcommons/inference/blob/master/loadgen/test_settings.h)，未引入 LoadGen 依赖。

resume 读取实际冻结顺序并校验身份、内容 hash、case/scale 覆盖及 probe 结论，不重新 shuffle。
`run_state.json` 同时绑定矩阵和 probe 文件 hash，已完成实验恢复时也检查这两份证据。
改变 order、seed、输入计划或剪枝选项须使用新输出目录；已有 case 却没有冻结计划、旧版
startup pruning schema v1、损坏或被改写的计划均明确拒绝恢复，不猜测历史执行顺序。

## Workload Contract

输入计划 schema v2 增加 `scenario: {"type": "serial"}`；正式能耗矩阵保持串行请求。
并发／到达率使用下方独立负载协议，不重写源输入计划的场景或正式 CSV。
资源条件仍来自矩阵 CPU/memory/GPU 字段，输入来自物化计划，场景单独声明。

每次 `/predict` 响应包含 `workload_contract` schema v1，保存 task、batch、scenario、计划尺度、实际尺度，
以及各任务可获得的实际输入／输出形状、计数和生成参数。窗口结束后 CSV 文本列 `workload_contract`
序列化本行已完成请求的摘要：`request_count` 保存总请求数，`variants` 保存各不同 contract 及其 `count`。
相同事实在窗口结束后合并，避免快速模型的重复 JSON 超过 CSV 单字段限制；不同输出数量保留分布，
不把它们简单改写成计划上限。这是工作量计数，不保存请求时间顺序。
比较时保留 variant 计数并比较归一化的联合分布，不能降为不含频率的集合；
分布不同与窗口请求总数不同分别记录，详见[跨独立实验比较](../results/comparisons.md#跨独立实验比较)。
未知事实为 JSON `null` 并保留可用性说明，旧 CSV 缺此扩展列继续可读，不补造历史 workload。

输入计划是 planned，原请求上限是 requested，Handler 观测的张量尺寸、token 数与输出数量是
actual；三者不互相替代。新增任务通过现有 `_workload` 补充事实，保持列与 contract schema 不变。
ONNX 独立验证记录实际 Provider、线程数及制品 SHA256；制品校验在准备／验证阶段完成，
正式服务加载时不增加一次完整权重扫描。协议/任务 sanity 验证与固定参考结果检查分别报告，
通过结构检查不表示模型准确率通过。

文本的 `max_output_tokens` 是请求上限，`actual_output_tokens` 来自真实生成 token ID；
计数排除 causal prompt／已知 decoder-start，包含终止 EOS，排除 EOS 后 padding。
`actual_output_tokens_per_sequence` 和 stop reason 保留单序列事实。`output_token_count` 原有的文本重分词
口径保持不变，不能与生成步骤计数混用。无法取得 token ID 时实际计数为 null，不能用最大长度代替。
音频记录时长、采样率、声道；`processor_input_duration` 是送入处理器的波形时长，
处理器 padding/chunk 后的 `processed_duration` 无法观察时为 null，不能用原始时长替代。
图像记录原始尺寸、可观察的 processor tensor shape；
多模态分别记录 text/image/audio/video，不能用单一尺度代替全部模态。处理器内部尺寸不可观察时明确未知。

完整 raw output 的类型、shape、finite、必需字段、实际尺度和 JSON serialization 校验只在独立验证阶段执行，
不在正式测量窗口重新扫描张量。主请求继续执行既有 Handler 预处理和输出约束，附加轻量工作量摘要。
任务 sanity 不等于完整 accuracy benchmark。
