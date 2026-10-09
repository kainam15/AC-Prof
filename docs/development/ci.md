# CI 与环境回归

[← 返回专题目录](testing.md)

## CI 与环境测试

独立 `lint` job 在 Python 3.10 上只安装开发锁，执行 `pip check` 和完整 pre-commit hooks；
不安装推理依赖，也不需要 Docker/GPU。检查内容及 hook 版本与本地一致。

`.github/workflows/ci.yml` 在 Python 3.10 / 3.12 上安装哈希锁并执行主机回归；
每个版本将完整测试集按排序后的 test ID 轮转分成八片，保留 20 分钟作业超时。
工作流的 `HOST_TEST_SHARDS` 同时传入测试 runner、证据汇总与 coverage 完整性检查；
修改分片数时同步矩阵 index，避免 coverage 开销使单片超时并丢失最终报告。
所有分片都执行完整 discovery，新增测试会自动分配；不使用手写文件白名单。
独立 Python 3.12 `contracts` job 统一执行容器锁、主机锁声明和指标文档同步检查，
不在 16 个 host shard 重复运行。CLI 帮助和 compileall 在每个 Python 版本的第 0 片执行。
每片分别上传 `host.json` 和实时保存的 `host.log`，失败时继续执行其他分片。
七个任务族分别执行 CPU 接口测试，以随机小模型或明确导出的
样例验证真实加载与推理；audio 和 multimodal 在同一作业共享一个 CPU 依赖环境，仍分别执行测试。
网络在容器测试期间关闭。CI Actions 固定为已核验的 commit SHA，作业只授予仓库读取权限。
另有 `nlp-transformers560-cpu` 矩阵项运行新版原生架构与图像 processor 测试，避免主机缺少推理依赖的 skip 掩盖环境回归。
`custom-multimodal-cpu` 矩阵项运行 `test_custom_multimodal_runtime.py`：未知 `auto_map` 架构的
随机小模型经音频、图像、视频输入映射执行，比较共享四阶段与上游 pipeline 结果，并检查
重复推理、真实 Torch profiler 算子及独立 runtime validation；容器禁止网络和跳过。
两个 Transformers 版本线均执行 `test_audio_generation_runtime.py`：保存小型原生模型 snapshot，
再经过共享 Auto 加载、原生音频消息、真实 `generate` 和输出验证；逐参数核对加载权重，防止
组合模型的前缀处理丢失权重。4.57.6 验证 Voxtral/Qwen2 Audio；5.6.0 另验证 Omni 文本子模型。
这些随机权重验证接口，不证明完整 checkpoint 的容量、质量或正式 profiling 指标。

`check_runtime.py` 将所选 `ACPROF_RUNTIME_PROFILE` 与 `ACPROF_MODEL_ADAPTER` 显式传入容器，
让版本核对、adapter、精度和 remote-code 策略都绑定该 profile。共享同一依赖环境的不同任务仍
各用合法 profile：5.6.0 的 NLP 与音频生成分别为 `nlp-transformers560-cpu` 和
`multimodal-transformers560-cpu`，输出目录独立。生成的 NLP custom-code fixture 只在测试
进程注册允许/拒绝 profile，生产信任策略不放宽；basic probe fixture 显式设置 backend、profile 和 adapter，
避免继承主机环境或在导入测试前因缺少配置而失败。
容器测试继续保留版本检查、禁网和 `--require-no-skips`。

`test_cv_runtime.py` 只包含 CPU 可执行的 CV 接口回归；CUDA 专项放在
`test_cv_cuda_runtime.py`，两者复用 `cv_runtime_fixtures.py` 的微型 snapshot 与 dtype 断言。
CUDA 专项须在已安装 CV 依赖且可访问 GPU 的环境中单独执行
`python scripts/run_tests.py --pattern test_cv_cuda_runtime.py --require-no-skips --report <report.json>`。
缺少 CUDA 时跳过不能作为 GPU 验收；CPU job 仍拒绝任何 skip。

本地入口：

```bash
.venv/bin/python scripts/run_tests.py --report internal-testing/host-tests.json
.venv/bin/python scripts/compile_locks.py --check
.venv/bin/python scripts/check_runtime.py --family audio --variant cpu --output-dir internal-testing/audio-runtime
.venv/bin/python scripts/check_runtime.py --profile moss-transformers560 --build-only --output-dir internal-testing/moss-dependencies
.venv/bin/python scripts/check_runtime.py --profile nlp-transformers560-cpu --test-pattern test_transformers5_runtime.py --output-dir internal-testing/transformers5-runtime
.venv/bin/python scripts/check_runtime.py --profile multimodal-transformers560-cpu --test-pattern test_audio_generation_runtime.py --output-dir internal-testing/audio-generation-runtime
.venv/bin/python scripts/render_metric_reference.py --check
```

`run_tests.py` 是 pytest CLI wrapper，保留 `--directory`、重复 `--pattern`、`--report`、
`--require-no-skips` 与分片参数；普通 pytest 也能直接使用后五项参数。
wrapper 显式加载仓库配置；直接 pytest 写入仓库外已有报告时，追加 `-c pyproject.toml`，
避免外部路径影响 pytest 的配置发现和 rootdir 推断。
`tests/conftest.py` 只注册 `acprof.testing.plugin`，evidence 与 sharding 分别位于独立模块。
报告保留 schema v1 的 test ID、outcome、reason、duration、Python/platform/package 版本、
UTC timestamp、discovery counts、完整 suite hash 和 shard 元数据。ID 使用 pytest node ID；
参数化 case 各有 ID，不再合并失败。call 的断言失败为 `failed`，未处理异常及 setup/teardown
失败为 `error`；同一 case 的各阶段合为一条记录并累计耗时，保留失败原因。
xfail/xpass 分别对应 `expected_failure` / `unexpected_success`；后者始终使 evidence 失败。
`--require-no-skips` 同时拒绝 skip 与 xfail。空 pattern、空分片、收集错误和中断不报告成功；
`--collect-only`、`--setup-only` 与 `--setup-plan` 保留 pytest 的诊断退出码，
但 evidence 的 `successful` 始终为 `false`；`collected_ids` 仍可检查选择与分片稳定性。
只有取得 call 与 teardown 结果的 case 才能计入 `passed`；仅完成 setup 或中断清理的 case
不填通过结果。已发生的 skip、`xfail(run=False)` 与 setup 失败仍保留原因；中断时保留
已完成测试的部分证据，并由选中数与记录数的差异显示尚未完成的范围。
筛选先于分片：对完整筛选后 node ID 排序，用 `index::count` 分配，SHA256 包含整个筛选集。
新增测试或参数改变 suite hash；同一 revision、参数与依赖环境的重复收集必须一致。

schema v1 报告新增 `provenance`：内容指纹分别覆盖显式项目源码/配置/资源/文档、测试及
fixture、依赖锁。源码树外的测试目录也按相对路径和实际文件内容计入测试指纹，不依赖 Git
或绝对 checkout 路径。仅在请求 `--report` 时采集，并在执行前后复核；输入改变或读取失败时
保留测试记录和 `provenance_error`，但不报告成功。生成目录、缓存、`.env.local` 不在指纹范围内。
汇总同一 Python 版本时要求指纹及实际 `packages` 完全一致；不同 Python 版本允许依赖差异，
但源码、测试与锁指纹必须一致。缺少身份的旧报告可以阅读，不能用于当前完整套件的成功验收。

容器接口验证按 runtime platform 的 Python、Linux wheel tags 和依赖 markers，
从 `runtime-test.lock` 生成保留原始 hashes 的 `runtime-test-target.lock`，下载对应 wheelhouse。
这避免 Python 3.12 主机漏掉 Python 3.10 容器所需的 backport；普通主机安装仍遵守版本条件。
依赖条件沿用 [pytest-asyncio 的声明](https://github.com/pytest-dev/pytest-asyncio/blob/v1.2.0/pyproject.toml)，
目标筛选复用项目锁工具，避免 [pip 按主机解释 markers](https://github.com/pypa/pip/issues/6117) 的跨版本下载问题。
下载后在 `--network none` 容器内
创建 `/tmp/acprof-tests` 临时环境，复用镜像的推理依赖；不向推理镜像添加 pytest，也不改变
环境身份或硬件采集入口。硬件 workflow 仍调用真实 `check_hardware.py`，pytest 通过不能替代它。
Runtime policy 的最小固定回归在 `test_runtime_preflight.py`：覆盖 GLM-OCR task registry、
SAM/SAM2 dtype、RMBG/skimage、manga-ocr/fugashi、缺少结构化模型 contract，均不下载权重。
`test_runtime_validation.py` 验证预算耗尽与阶段存证；`test_runtime_evidence.py` 验证
CSV/TUI/audit/report 统一原因、质量与能力独立、selected artifact 预算；`test_loading_quality.py`
验证 loading info 返回约定及失败后的恢复。对应真实 smoke 仍须另外记录 checkpoint SHA、
镜像 ID、Transformers 版本、CPU/GPU、实际 dtype 和输出验证；mock 与随机权重不替代该证据。

`test_metric_reference.py` 在关闭 UTF-8 mode、启用 `EncodingWarning` 错误的独立进程中
验证指标文档生成与检查：生成固定使用 UTF-8（无 BOM）和 LF；检查接受 UTF-8 的 LF/CRLF
工作区文件，对缺失、GBK 编码或内容过期返回非零并提示重新生成，不改写文档。
本地可用 `--shard-index 0 --shard-count 8` 重现一个 CI 分片；省略参数执行完整测试集。
pytest 的 session autouse fixture 为测量锁注入临时目录，覆盖 setup/call/teardown；
CLI、IDE 和 wrapper 均受保护。生产入口仍固定使用 `/tmp` 的同用户锁，`TMPDIR` 不能绕过。
本地汇总可执行 `python scripts/aggregate_test_reports.py <下载目录> --shard-count 8 --report <新报告.json>`；
单版本验证用 `--python-versions 3.12`。缺分片、失败、计数／摘要不符或测试归属错误均使汇总失败。
报告的 `shard` 记录编号、总片数、完整发现数、选中数和排序后 test ID 列表的 SHA256。
`host-summary` job 自动下载两个 Python 版本的全部分片到各自目录；
`scripts/aggregate_test_reports.py` 确认同一 Python 版本分片齐全且测试集摘要相同，所有 test ID 无重复，
总执行数等于完整发现数；单片通过不代表主机回归完成。

### Host coverage baseline

`requirements/dev.in` 与带 hash 的开发锁固定 `coverage[toml]==7.15.2`，支持 Python 3.10+。
CI 仅在 Python 3.12 的八个 host shard 使用 branch coverage；Python 3.10 使用相同 pytest wrapper。
每片单独上传 `.coverage.*`，显式开启 hidden files；`host-summary` 先验收测试证据，
再确认八个 coverage 目录齐全，执行 `coverage combine --keep`、`xml`、`json`、`html`，
发布 `coverage-baseline-3.12` artifact。缺分片不能生成完整 baseline。

第一阶段只记录覆盖情况，不设全仓百分比 gate，也不排除错误与清理路径来提高数字。
`pyproject.toml` 固定 `source=["acprof"]`、branch 和 relative paths；未运行的模块仍进入报告。
默认本地产物位于已忽略的 `internal-testing/coverage/`，CI 用 `COVERAGE_FILE` 指向独立 evidence 目录。

```bash
# 使用装有 host/dev lock 的 Python 3.12 环境，每个 index 执行一次。
python -m coverage erase
for shard in 0 1 2 3 4 5 6 7; do
  python -m coverage run --branch --parallel-mode scripts/run_tests.py \
    --shard-index "$shard" --shard-count 8 \
    --report "internal-testing/coverage/host-3.12-$shard/host.json" || exit 1
done
python scripts/aggregate_test_reports.py internal-testing/coverage \
  --python-versions 3.12 --shard-count 8 --report internal-testing/coverage/host-summary.json
python -m coverage combine --keep
python -m coverage xml
python -m coverage json
python -m coverage html
```

优先查看 schema/merge、matrix planning、run recovery、energy math、error/cleanup 的未覆盖分支。
该 baseline 只度量 host runner 进程；未启用对子进程的自动注入，不能代表子进程、Docker、
GPU、RAPL 或 perf 的实际路径。测试中的 mock 覆盖也不等于 Native validation。
设计复用 [Coverage.py 官方合并流程](https://github.com/coveragepy/coveragepy/blob/main/coverage/data.py)
（Apache-2.0）与统一 pytest/evidence 插件；锁定版本、仅安装开发依赖，不进入采集环境。
hidden files 行为见 [upload-artifact #602](https://github.com/actions/upload-artifact/issues/602)。

### 恢复与运行环境回归

空测试集必定失败；容器作业带 `--require-no-skips`，跳过或 expected failure 都不算环境验证通过。
普通主机测试允许缺少推理依赖时跳过，报告明确列出范围。`check_runtime.py` 的目录必须为空；
`runtime.json` 另记录逻辑 profile、环境 ID、平台/环境 image ID、完整运行清单和退出结果，不覆盖旧验证。
`--build-only` 仅证明依赖构建和清单核验；`--family` / `--profile` 共用主构建的环境准备入口。

可靠性回归包括 `test_run_identity.py`（跨 TMPDIR 的进程互斥、声明/资源变化拒绝续跑）、
`test_monitor_cleanup.py`（实际 client 循环的清理失败与请求证据）、`test_container_ownership.py`
（独立名称、失败/取消时按完整 ID 回收）和 `test_release_gate.py`（同提交的发布前置依赖）。
`test_tui_process_lifecycle.py` 补充回调异常、忽略信号的真实子进程、锁释放及清理状态；
`test_progress_events.py` 检查控制事件不受日志措辞影响，`test_hardware_conditions.py` 与
`test_result_comparison.py` 检查 affinity、比较目的和未知证据，`test_run_recovery.py` 检查 CPU 集合续跑身份。
服务上下文及 host/TUI 变化不重建服务的约束由镜像构建测试覆盖。mock 测试不代替实际 Docker 采集。
普通接口测试固定使用 CPU，即使选择 CUDA wheel；GPU 和自定义 MOSS adapter 由完整服务镜像的
独立 `runtime_validation` 验证。依赖迁移需逐一构建全部唯一环境，再分别验证共享环境的各 profile。

更新锁前后比较原完整包版本，验证目标 wheel 的 ABI/平台和全部制品 SHA256。默认锁生成保留原
版本，显式 `--upgrade` 才更新环境包；uv 固定 0.12.13，生成 wheel 锁的脚本使用 Python 3.11+，
只读检查兼容 Python 3.10+。系统锁可指定同一 snapshot 再生成并比较，过程只修改一次性容器和
指定输出锁。具体命令见[当前配置](../models/environment.md#当前配置)。

源代码、模型权重和依赖层保持分离，CPU 容器测试不下载 Hub 模型，不代替真实 GPU/PMU/抓包实验。

通用接口回归见 `test_generic_model_interfaces.py`：任意 checkpoint 名称的路由、固定 revision
元数据读取、任务／架构到版本线的选择、制品拒绝、prompt 预算及 workload 参数重放。
`test_cv_runtime.py` 在离线容器比较两类原生 timm 快照与官方模型输出；`test_timeseries_runtime.py`
比较 Chronos-Bolt／Chronos-2 的多序列预测；`test_nlp_runtime.py` 比较完整句向量模块图、默认 prompt
和归一化数值。`test_transformers5_runtime.py` 用旧环境没有的 Qwen3.5 原生小配置验证共享 NLP 入口，
并验证新版图像 processor 的输入形状、检测框及依赖 OpenCV 的多边形输出。该测试需在
`nlp-transformers560-cpu` 共享环境显式运行；随机小配置不是热门 checkpoint 的采集或准确率证据。

候选发现、冲突处理、本地模型声明、镜像／恢复身份和输入宽度回归见
`test_model_discovery.py`；`test_validation_stages.py` 检查成功步骤、失败位置及原异常保留。
`test_custom_pipeline_runtime.py` 在断网 NLP 容器创建微型随机 BERT snapshot 与自定义 pipeline，
比较原生接口和自定义接口的 label/score，并执行真实独立验证、失败阶段检查及自定义 `auto_map`
架构加载。它已加入默认 NLP 容器测试集，由现有 CI 的 NLP job 执行；也可单独运行：

```bash
.venv/bin/python scripts/check_runtime.py --family nlp --variant cpu \
  --test-pattern test_custom_pipeline_runtime.py \
  --output-dir internal-testing/custom-pipeline-runtime
```

该证据覆盖锁定环境中的标准文本分类桥接，不证明任意自定义模型、其它任务或 GPU 兼容。
多模态共享接口的离线测试可运行：

```bash
.venv/bin/python scripts/check_runtime.py --profile custom-multimodal-cpu \
  --test-pattern test_custom_multimodal_runtime.py \
  --output-dir internal-testing/custom-multimodal-runtime
```

主机协议／环境选择回归在 `test_custom_multimodal.py`；依赖 commit、文件哈希与镜像复用检查在
`test_model_download.py`、`test_image_reuse.py`；TUI 输入、命令、持久化和语言切换在
`test_tui_model_spec.py`。真实 Ultravox checkpoint 还需对应基础仓库访问权限和设备容量。

自动契约的主机回归使用 `test_model_contract.py` 和
`tests/fixtures/custom_pipeline_audio_like/`：覆盖 SHA 绑定、来源状态、输入映射、确定性参数、
动态表达式／冲突拒绝、依赖候选、缓存身份和报告导出；恶意源码检查在隔离进程中确认没有执行
模型顶层代码，也没有导入 Torch／Transformers。共享 fixture 的 Hub mock 按仓库 ID 区分主模型
与依赖；依赖测试通过 `dependency_lookup` 注入响应并检查调用，避免内外两层 patch 同一个
`HfApi.model_info` 时覆盖依赖 SHA 和文件列表。生成契约与通用 handler 的真实衔接使用随机小权重：

```bash
.venv/bin/python scripts/run_tests.py --pattern test_model_contract.py \
  --report internal-testing/model-contract-tests.json
.venv/bin/python scripts/check_runtime.py --profile custom-multimodal-cpu \
  --test-pattern test_model_contract_runtime.py \
  --output-dir internal-testing/model-contract-runtime
```

第二条命令核验锁定环境并在断网 CPU 容器执行加载、预处理、推理及输出验证，拒绝跳过。
同一容器测试还覆盖 Interface Probe 仅导入／签名、不加载权重，以及经显式审阅生成的嵌套 `turns` 模板。
M4～M6 的主机回归使用 `test_model_dependencies.py`、`test_model_review.py`、`test_model_transforms.py`、
`test_model_probe.py`、`test_model_inspection.py`：覆盖依赖角色／SHA／过滤、条件和动态路径、逐字段决策、
DSL 深度和引用限制、只读断网命令、证据完整性、CLI 导出及失败状态。`test_interface_probe.py` 验证
不进入模型准备或 Model Store、缓存零联网、缺图拒绝、文件边界、SHA、递归源码与取消清理。
TUI 的一键开始、只读字段、resolver 重新确认、同窗错误／重试和中英文三种终端尺寸由
`test_tui_model_resolution.py` 与 `test_tui_collection_workflow.py` 验证；后者用真实子进程检查取消后
临时目录、锁和进程释放。`test_collection_workflow.py` 阻断 postprocess／validate_output 失败及非成功报告；
`test_runtime_validation.py` 检查 CPU/GPU、最小输入和完整阶段证据。
真实断网容器检查之外，外部依赖验收应保留 Hub SHA、实际选择文件和缓存内容，确认未下载无关权重；
真实只读 Probe 应使用独立测试镜像和目录，不能只用 mock Docker 命令代替。
`test_dependency_flow.py` 覆盖 active/inactive/unknown、参数绑定、main-model 转发、primary/fallback、
角色分离及动态条件／kwargs／装饰器／递归边界。`test_ultravox_dependency_flow.py` 使用保留许可与 SHA256
的固定上游源码文本，验证零未决项、与示例声明的角色语义一致、未知 Transformers 版本保持 review，
以及换成任意主模型 ID 仍得到相同规划。原有
`test_conditional_weight_load_does_not_download_a_potential_base_model` 必须继续通过；不能简单取消条件拦截。
这些测试不导入或执行 snapshot Python，不访问真实 Hub、不下载权重。

真实 Ultravox 的静态 draft 验证不下载权重、不证明所有动态依赖完备或大模型推理成功；GPU／profiler
需各自取得运行证据，不能从这项 CPU fixture 验证外推。

生产模型声明与服务镜像的完整链路可使用 [Iris 示例](../models/pipelines.md#本地模型声明与自定义-pipeline)，
按 `resolve → interface validation → prepare runtime → runtime validation → matrix measurement` 分层验收；
接口检查、Smoke、预热、正式行和 profiler 结果分别计数，Smoke 不写正式 CSV。

### 无 Torch 运行时验收

```bash
.venv/bin/python scripts/check_runtime.py --profile onnxruntime-cpu \
  --basic-e2e --output-dir internal-testing/onnx-runtime
```

ONNX 的三个 profile 按 runtime 选择专用测试集，不按 family 误选 Torch 用例。
`.github/workflows/ci.yml` 的独立 ONNX CPU job 在 GitHub-hosted runner 必跑上述命令；
它检查 Torch、Transformers **未安装**，并要求 ORT/ONNX/Pillow/Tokenizers 等实际可导入。
任何指定测试模式为空、skip、expected failure、依赖缺失、超时、输出／必需字段错误或清理失败
都会失败。普通宿主测试仍可明确 skip 缺失的可选推理依赖，不代表容器通过。
`--test-pattern` 可重复覆盖默认测试，报告记录每个 pattern 的发现数；专用 CI 不覆盖默认集。

离线 fixture 固定 IR10/opset17，真实执行表格、图像、多输入文本，覆盖数值参考、动态维度、
样本顺序、dtype/shape 与固定形状拒绝；不会下载 Hub 模型。basic 回归复用生产 client、
ResourceUsageMonitor、CSV 合并及 audit，验证应用延迟、原有 batch/latency 吞吐口径、真实
cgroup CPU/内存、actual workload、输出验证和能力状态。合成图只证明接口与执行链路。
验收脚本将能力报告中的实际主机环境身份同步写入 `static_meta.json` 和 `run_state.json`，
与 client 写入 CSV 的身份交叉核对；不根据 CSV 猜测主机，也不放宽环境不一致的审计错误。
`test_onnx_basic_audit.py` 在 Docker/HTTP 边界模拟下覆盖 Native Linux、WSL2 的身份保存和不一致拒绝，
保留真实 CSV 合并及审计链路；此测试不代替上述容器端到端验证。
`--basic-e2e` 依次执行表格、图像、文本三种场景，每种包含一次 warmup 和两次正式请求；
任一场景失败即返回失败。图像检查原始与处理后尺寸，文本检查具名整数输入及实际 token 数。
真实 ORT、WordPiece tokenizer、HTTP probe 与既有规划器的联合回归还覆盖超限拒绝和自动尺度规划。
真实 HTTP 回归还验证异步等待、后台失败、超时及请求内不执行输出验证。
服务只发布 loopback 端口，不使用 privileged、Docker socket 挂载或硬件 runner；完成或失败后
清理本次容器。已有 `.github/workflows/hardware.yml` 仍仅手动触发。

两个预训练小模型可另行验证；第一步联网准备，第二步在同一无 Torch 环境断网执行。
以下目录需尚未存在或为空，重复检查请换新目录：

```bash
.venv/bin/python examples/onnxruntime/real_models.py prepare mnist \
  --directory internal-testing/pretrained-mnist
.venv/bin/python examples/onnxruntime/real_models.py prepare bert-tiny \
  --directory internal-testing/pretrained-bert-tiny
runtime_image=$(.venv/bin/python -c 'import json; print(json.load(open("internal-testing/onnx-runtime/runtime.json"))["image_id"])')
docker run --rm --network none --cpus 2 --memory 2g \
  -e ACPROF_RUNTIME_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace:ro" -w /workspace "$runtime_image" \
  python examples/onnxruntime/real_models.py validate mnist \
  --directory internal-testing/pretrained-mnist
docker run --rm --network none --cpus 2 --memory 2g \
  -e ACPROF_RUNTIME_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace:ro" -w /workspace "$runtime_image" \
  python examples/onnxruntime/real_models.py validate bert-tiny \
  --directory internal-testing/pretrained-bert-tiny
```

脚本保留来源、许可证说明、revision 和制品哈希，参考检查使用独立的 ONNX ReferenceEvaluator。
它验证一个样例的数值一致性，不执行训练数据准确率评测，也不输出正式性能结果。
真实 full 矩阵仍需相应硬件和权限；basic 成功不能替代 full 指标验收。
