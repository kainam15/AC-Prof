# 命令行、输入清单与界面设置

查 CLI 参数、workload 清单或 TUI 持久化契约时查阅。交互操作见 [TUI 用户指南](TUI.md)，运行示例见[安装与运行](Getting_Started.md)，实现依据为当前入口的 `--help`。文中的命令从仓库根目录执行。

[文档导航](README.md)

## 主机环境与 Hugging Face 认证

CLI 启动时读取当前工作目录的 `.env` 和 `.env.local`；同名值的优先级为
**进程环境 > `.env.local` > `.env`**。文件中的值不执行 shell 命令或变量展开。
凭据保存方式见[认证配置](Getting_Started.md#hugging-face-认证)。

读取完成后，Hugging Face 初始化按去除首尾空白后的非空值选择配置：

- 默认 `HF_DOWNLOAD_MODE=auto`。缓存命中先离线复用；否则 Hub 入口依次取主地址（`HF_ENDPOINT`、`HF_HUB_ENDPOINT`，默认 `https://hf-mirror.com`）、`HF_FALLBACK_ENDPOINTS`、`https://huggingface.co`。metadata 与权重下载都可回退，revision 保持固定。
- `--download-mode` 仅供高级/兼容用途：`official` 只选官方入口，`mirror-preferred` 与 `auto` 相同；`mirror-only` 不主动选择备用入口，但仍允许合法 Hub/CDN 重定向。旧 TUI 设置中的三种模式迁移为 `auto`。`ACPROF_ALLOW_PROXY_FALLBACK`、`ACPROF_DIRECT_HOSTS` 已无作用。
- HTTP 路径允许 HF 控制的可信存储域名，包含 Xet bridge；原生 Xet/hf_transfer 加速器仍禁用，以保证每跳请求可审计。AC-Prof 不负责配置 VPN，不改写标准代理变量或默认路由；公网出口由系统或上游网络负责，不能从 `direct-socket` 推断没有 VPN。
- 令牌依次取 `HF_TOKEN`、`HUGGING_FACE_HUB_TOKEN`，都为空时调用已有
  `huggingface_hub.utils.get_token()`；找到令牌后回填缺失或仅含空白的令牌变量。
  两个变量都有非空值时保留各自值，解析结果以 `HF_TOKEN` 为准；没有令牌或本地读取失败时返回匿名状态。

认证优先级参考官方维护的
[Hugging Face Hub 实现](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_auth.py)
（[Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE)）。
项目沿用已安装的公开接口，只在自身初始化层处理空白值和变量回填，不复制上游内部实现、
不新增依赖；该初始化发生在主机准备阶段，不进入正式测量窗口。
令牌仅用于主机检测与 Model Store 下载，不传入 Docker 构建或正式推理容器；运行阶段只读挂载固定 snapshot 并离线加载。
配置自定义 endpoint 也决定 Hugging Face 请求及认证令牌的接收方，应只选择信任的服务。
Hub endpoint 是传输入口，不作为模型身份，也不写入离线容器的网络配置。
模型身份由 source、模型 ID、固定 revision 和文件哈希确定；endpoint 与脱敏后的 redirect chain 单独保存在下载 provenance 中。

AC-Prof 仅使用启动进程继承的标准代理环境变量（包括小写形式与 `NO_PROXY`），不在 TUI 提供代理设置。
旧 `.env`/`.env.local` 中的代理字段读取后忽略，原文件中的值保留。需要显式应用层代理时，在 shell 或系统配置后启动，例如：

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1,::1
acprof tui
```

端口仅为示例，请使用自己配置的代理地址。透明代理、系统 VPN、路由与公网出口由用户的系统网络管理。
显式使用另一来源：`acprof run --model-source modelscope --model <ModelScope-ID>`；可通过系统环境或本地配置提供 `MODELSCOPE_API_TOKEN`。
ModelScope branch/tag 通过轻量 `git ls-remote` 固定为仓库 commit（需安装 Git）；也可提供完整 commit SHA。

## TUI 本地设置

`F2` → **应用设置** → **管理连接与权限**管理 Token、下载预算、Model Store 与企业微信 Webhook。Hub 入口高级配置仅通过 CLI/环境变量提供。
连接配置保存到工作目录的 `.env.local`，原文件备份为 `.env.local.bak`，两个文件权限均为 `0600`。
保留其他键和注释；文件在表单打开后被外部修改时拒绝覆盖，符号链接也不会被写入。
显式保存同步更新当前 TUI 进程及后续子进程的受支持配置，并同步认证别名；已有代理环境不修改。
重启后仍按上述进程环境优先级加载。清空 Webhook 会写入空值，屏蔽旧文件中的同名值。
关闭窗口不保存；凭据不写入 `tui.json`、命令预览或日志。保存和权限检查不发送通知。
采集权限的系统授权单独操作，见[最小权限安装](Getting_Started.md#最小权限安装)。

`acprof tui --model <ID> --preset smoke --output-dir <目录>` 可覆盖本次初始表单。
显式 preset 优先于已保存的实验默认参数，显式输出目录再覆盖 preset 的目录；未传入的 model
沿用已保存模型。启动参数不直接写入设置文件，保存时机仍遵循下方约定。
smoke 为 basic CPU 单次请求配置；`main` 和 `default` 保持 full。

`acprof/tui/settings.py` 管理项目隔离的 `tui.json`，当前版本为 v4；
路径与操作方式见 [TUI 设置文件](TUI.md#设置文件)。
`ui.language` 是字符串，仅接受 `zh`（简体中文，默认）和 `en`（English），不使用系统 locale 自动推断。
只读取 version 4 设置；缺少版本、v1/v2/v3 文件或仍包含已删除的 `allow_cgroup_v1` 字段时直接报错，原文件保持不变。归档旧设置后可重新配置。
未知语言值或错误类型遵循现有校验规则：提示、使用默认设置，并保留原文件，直到用户主动保存。

切换语言仅更新当次界面，点击“保存设置”后持久化；“恢复界面默认”将当次语言恢复为中文。
自动记住模型 ID、结果路径或显式记住实验配置时，不会顺带保存尚未保存的界面偏好。
语言只影响 TUI 文案，原始子进程日志、命令参数、结果文件和进度解析状态值保持原有语义。
切换时复用已挂载控件和已读取摘要，不重新读取结果 CSV，也不启动定时刷新；任务运行期间语言控件随其他偏好锁定。

表头边界拖动产生的列宽只保留在当前 TUI 会话中，不写入设置文件，也不随“保存设置”持久化。
排序、筛选、刷新、视图与语言切换和终端缩放保留手动列宽；重启后恢复各表格默认宽度。任务运行期间暂停列宽拖动。

v4 新增顶层字符串 `last_result_dir` 和 `last_result_csv`，分别保存最近使用的结果目录和 CSV 路径。
TUI 中，结果目录位于“补采工具”页，结果 CSV 与摘要位于“绘图工具”页；切换页面保留输入草稿和工具勾选。
两者默认均为 `""`；当前版本文件缺少可选字段时保持空值，不从模型草稿推测历史输出目录。
TUI 保存实际使用的绝对路径，相对输入以启动时的工作目录为基准，支持 `~`、空格和中文。
确认采集时保存 `last_model`；输出目标保留在本次运行配置中，不提前覆盖结果路径。
采集结束后，仅当既有 `run_state` 的新增 attempt 与子进程匹配，且 CSV 为本次新产物时，才更新结果路径。
预检失败、没有新 CSV 或仅复用已完成实验时保留历史选择；路径记忆不保证文件此后仍然存在。
读取摘要成功或启动绘图只更新 CSV 字段，启动补采（含 dry-run）只更新目录字段；取消确认和无效输入不更新。
恢复路径不扫描目录、不读取 CSV、不检查文件存在性；实际读取摘要、绘图或补采时再验证。
自动写入复用现有原子替换和错误隔离，保留已保存的 UI 偏好、实验默认参数及未涉及的历史字段。

### TUI 终端颜色

TUI 默认使用 `--color-system truecolor`，直接输出主题中的 RGB 颜色。默认主题为“石墨灰 · 深色”
（`acprof-graphite`）：背景 `#202126`、输入框表面 `#292b32`、面板 `#383b45`、强调色 `#66b8c4`、文字 `#eceef3`。
首次启动、设置中缺少主题或“恢复界面默认”时使用此主题；已保存的有效主题选择优先于默认值。
全部主题均使用明确颜色，控件不依赖终端可重定义的 ANSI 基础色；原有主题选择及保存格式不变。
主题控制背景与层次，操作颜色保持固定语义：青色为选择和主要操作、白色为普通操作、黄色为可恢复的风险操作、
红色为删除或终止、灰色为禁用、绿色为成功或 Ready。浅色主题使用深色普通文字和较深的同色系颜色；
悬停、聚焦及确认弹窗沿用操作原有的颜色。页面布局与操作位置见 [TUI 说明](TUI.md#启动和页面)。

Windows Terminal、VS Code 集成终端等支持真彩色的客户端，通过 SSH 运行时即使缺少
`COLORTERM`，默认模式也保留 RGB 输出。颜色模式仅作用于 TUI 渲染器，不修改 shell 配置、
环境变量或采集子进程的环境，也不启动终端探测或刷新定时器。

确实只支持 256 色的终端可运行 `acprof tui --color-system 256`，颜色按扩展调色板近似，
不能精确还原 RGB。`--color-system auto` 恢复 Textual 的环境检测：通常由 `TERM` 和
`COLORTERM` 决定色深；显式设置的 `TEXTUAL_COLOR_SYSTEM` 也只在此模式下沿用。
`NO_COLOR` 在所有模式下仍由 Textual 转为单色显示。颜色模式只影响本次启动，不写入 `tui.json`。

## 企业微信通知

先在企业微信群中添加群机器人，把完整 Webhook 只保存在项目根目录的
`.env.local`（该文件已被 Git 忽略）：

```env
ACPROF_WECOM_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
```

建议限制本地配置文件权限：

```bash
chmod 600 .env.local
```

配置 Webhook 后，`acprof run` 默认启用企业微信通知，无需额外参数：

```bash
acprof run --model google-bert/bert-base-uncased
```

`--notify` 默认为 `auto`（检测到 Webhook 自动启用）；传入 `--notify wecom` 显式启用，传入 `--notify none` 临时关闭：

```bash
acprof run --model google-bert/bert-base-uncased --notify none
```

通知覆盖实验开始、已启用 profiler 的各工具阶段完成、每个资源 case 完成和最终总结。
CPU Torch、GPU Torch、NCU、Massif、Nsys 各自汇总实际采样项、失败数、阶段耗时和累计耗时；
vendor 模式的 CPU Advisor 同样适用。阶段状态区分成功、部分失败、失败和无结果。
关闭或不适用的工具不发阶段通知，代表资源复用不重复计数。最终总结区分成功、部分成功、
无结果、失败和用户取消。TUI 启动的 `acprof run` 使用同一设置，独立 `acprof profile` 不发送这些通知。

通知在测量窗口外发送：profiler 阶段返回后、下一个阶段前，或 case 的容器、监控器与抓包
全部停止后。input scale、warmup 和 repeat 窗口内部不发送网络通知。每次发送超时 5 秒，
最多尝试两次；失败只产生警告，不改变测量结果或原退出码。Webhook 只保存在本地环境配置中。

## CLI 参数

源码开发与安装环境的唯一公开入口为 `acprof <command>`；公共子命令有 `run`、`tui`、`probe`、`plot`、
`doctor`、`profile`、`audit`、`stats`、`inspect`、`auto`、`coverage`、`compare`、`load`、`model-store`、`report`。
`acprof --version` 查看版本，`acprof <command> --help` 查看对应帮助。

### `acprof doctor`

`doctor` 只检查环境，不下载模型、启动容器、安装工具或修改权限。
检查覆盖 Native Linux / WSL2 x86_64、cgroup v2、本机 Docker、Buildx、安装资源与输出目录。
输出 Environment、Collection tier、Native benchmark 和按指标的支持策略；WSL2 使用 `--profiling-mode basic`。
`full` 另外检查 RAPL、perf instructions、抓包工具和网卡；`--gpus on` 检查 NVIDIA driver
和 Docker NVIDIA runtime。模型下载、容器 GPU、任务输出与正式测量仍由实际运行验证。

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--profiling-mode basic/full` | `full` | 与主流程保持相同的必需能力范围 |
| `--gpus off/on` | `off` | 是否检查 NVIDIA 主机和 runtime |
| `--sniff-iface` | `docker0` | full 模式抓包网卡 |
| `--output-dir` | 当前工作目录 | 检查输出路径可写性与所在磁盘余量 |
| `--json` | 关闭 | 输出包含 `ready`、`checks`、`scope` 的 JSON |

退出码为 `0`（必要检查通过）、`1`（必要条件缺失）、`2`（参数错误）。
检查状态是 `available`、`unavailable`、`not_requested` 或 `warning`；
`available` 不等于真实 workload 的 `verified`。低磁盘余量为提示，不自动删除镜像。
`doctor` 不检测所有模型的联网和容量需求，`run` 仍在正式采集前执行权威 preflight。

以下参数表对应 `acprof run`。示例命令见[运行指南](Getting_Started.md#运行正式实验)，
默认值与实际选项以当前入口的 `--help` 和 [acprof/config.py](../acprof/config.py) 为准。

[acprof run](#acprof-run) · [acprof probe](#acprof-probe) · [acprof profile](#acprof-profile) · [其他入口](#其他入口) · [输入规模与音频清单](#输入规模与音频清单)

### `acprof inspect`

`acprof inspect MODEL` 只进行静态解析；`--explain` 显示固定 revision、字段来源和未决项。
`--output-dir DIR` 导出 `model_resolution.json`。静态缺口退出 2，保留 draft；模型访问或网络错误退出 1，
分类见[共享接口解析](Runtime_Compatibility.md#共享接口解析)。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model-spec`、`--task`、`--backend` | 自动解析 | 声明或选择覆盖，仍检查冲突 |
| `--expected-revision` | 空 | 要求模型 SHA 与已审阅 SHA 相同 |
| `--revision` | Hub 默认分支 | 解析 branch、tag 或完整 SHA，文件固定到解析后的 SHA |
| `--probe-interface` | 关闭 | 使用纯源码 bundle，在依赖镜像内检查 import 和 method signature |
| `--cpus`、`--mems` | `2`、`4` | 接口检查的 CPU 核数和 GiB 内存上限 |
| `--timeout-seconds` | `300` | 单次接口检查容器超时，不包括依赖镜像准备 |

```bash
acprof inspect MODEL --explain
acprof inspect MODEL --probe-interface --output-dir results/inspection/model-interface
```

接口检查不调用 `prepare_image()`、`prepare_model()` 或 Model Store，不下载模型权重、不执行推理。
首次可能准备依赖基础镜像。未指定输出目录时使用独立 `results/inspection/` 子目录，已有接口报告的
目录不能复用；写入 `interface_validation.json` 和 `logs/interface_validation.log`，不生成正式 CSV。
源码图缺失直接失败，禁止整仓下载回退。容器使用断网、只读文件系统和只读源码挂载。
旧 `--probe basic/full`、`--gpus`、`--skip-build` 不再用于 inspect；完整运行验证由采集流程自动执行，
批量独立验证可用 `coverage run --validate-runtime`。接口与运行证据边界见[运行兼容](Runtime_Compatibility.md#自动生成模型契约m1m6)。

### `acprof auto`

`acprof auto MODEL` 接受精确 Hub ID 或 Hub 本身支持的别名，检查主模型及声明依赖的访问权限，
导出静态裁决，检查主机，然后复用 `run` 的镜像准备、输入规划、独立 runtime validation、
正式矩阵和报告。模糊名称不按下载量替换，访问失败不改选其他模型；语义冲突保留解释和 draft 后退出。

除模型改为位置参数外，其余资源、输入、窗口和 profiler 参数与 `run` 相同，默认矩阵也相同。
首次验证建议显式限制矩阵：

```bash
acprof auto google-bert/bert-base-uncased --profiling-mode basic \
  --cpus 1 --mems 4 --gpus off --input-scales 32 --warmup 0 --repeat 1 \
  --repeat-in-window 1 --output-dir results/auto-first
```

默认仍为 `--profiling-mode full`。只有显式选择 `--profiling-mode auto`，才允许因
RAPL/perf/packet 不可用而选择 basic；Docker、Linux/cgroup、安装资源或所选 GPU 不可用时仍停止。
`--dram-energy required` 等显式要求仍由正式入口检查，不能通过 auto 绕过。
`profiling-mode basic/full` 只决定性能指标范围；两种模式都必须通过完整运行验证。

`auto_report.json` 位于 `OUTPUT/MODEL--NAME/`，保存请求模式、实际模式、来源身份、预检和最终状态。
只有主采集与所需指标均成功才退出 0；冲突、缺条件或部分失败退出 2。已有实验不会覆盖；
`--resume` 需要原资源/输入/模式选项，并复用记录的模型 SHA、镜像和计划，不重新解析浮动分支。
恢复时显式指定 `--revision` 必须与保存的完整 SHA 一致。换模式、依赖或输入计划需使用新目录。
当前不做环境修复、pad token 替换、缩小尺度或预算重规划。

### `acprof coverage`

固定回归集与滚动样本使用同一个 schema v1 manifest。每个条目需有 `model_id`、完整
`revision` 和非负 `weight`，顶层需说明 `sampling`、`weight_basis`。样本、权重和 SHA 一经
冻结不在运行中刷新；重新 snapshot 才产生新一轮滚动样本。

```bash
acprof coverage run examples/coverage/regression.json \
  --output-dir internal-testing/coverage-fixed
acprof coverage snapshot --stratum fill-mask:transformers \
  --stratum image-classification:transformers --limit 5 \
  --output internal-testing/coverage-sample.json
acprof coverage run internal-testing/coverage-sample.json \
  --output-dir internal-testing/coverage-static
# 显式运行容器验证；可能构建镜像和下载权重
acprof coverage run internal-testing/coverage-sample.json --validate-runtime \
  --cpus 2 --mems 4 --gpus off --timeout-seconds 300 \
  --output-dir internal-testing/coverage-runtime
# 在下载前应用 conservative 预算；大小来自选中的制品而非整个仓库
acprof coverage run internal-testing/coverage-sample.json --validate-runtime \
  --max-parameters 1000000000 --max-download-bytes 4294967296 \
  --output-dir internal-testing/coverage-budgeted
# 中断后使用原参数继续；已经完成的模型不重复执行
acprof coverage run internal-testing/coverage-sample.json --validate-runtime \
  --cpus 2 --mems 4 --gpus off --timeout-seconds 300 --resume \
  --output-dir internal-testing/coverage-runtime
# 只重试超时失败项；新 attempt 保留提高预算前的证据
acprof coverage run internal-testing/coverage-sample.json --validate-runtime \
  --timeout-seconds 600 --resume --retry-reason request_timeout \
  --output-dir internal-testing/coverage-runtime
# 只读取已有结果，输出统一的 CSV、JSON 和 Markdown 报告
acprof coverage report results/model-a results/model-b \
  --output-dir internal-testing/coverage-recorded
```

snapshot 按 `TASK:LIBRARY` 各取下载量前 N 个，属于所选样本统计，不代表全 Hub 或随机长尾。
run 默认只做静态检查；`--validate-runtime` 的验证时间限制不包含构建和下载。`coverage.json` 的分母始终是
冻结样本总权重，分别报告解析、适配、运行、拒绝、权限与资源限制；静态检查不检查权重读取权限，
运行成功率、权限拒绝率与资源限制率均为 null。
独立审阅的 `semantic_reference: {"task": "...", "source": "..."}` 才用于语义正确率；
snapshot 不把 Hub 标签自动当成正确答案。零总权重和没有审阅样本的比率为 null。
报告生成成功退出 0 不表示所有模型成功；逐模型失败保留在 rows 中，不生成正式性能 CSV。

run/report 同时输出 `coverage.json`、`models.csv` 和 `REPORT.md`。失败列保留稳定的
`reason_code` 及 evidence，质量警告单独保留在 `quality_checks`；TUI 的 `/report <coverage.json>`
可读取两种报告。report 不执行模型，不修改源结果，也不从旧日志猜测缺少的原因。
Runtime Validation 成功只表示独立推理验证通过；已有采集是否完成由记录的 `full_profile_complete` 决定。

`--max-parameters` 和 `--max-download-bytes` 默认不设置，传入时须为正整数。
超预算或无法确定所需大小时保存 `resource_limit` 与 `unverified`，不下载权重、不宣称实测 OOM。
selected artifact size 包括显式模型依赖；参数量不是峰值 RAM/VRAM 预测。
`--timeout-seconds` 默认保持 300；例如复核 60 秒耗尽的条目，可用同一固定 manifest，显式传入
`--timeout-seconds 600 --resume --retry-reason request_timeout`。每个选中模型在本次调用中只执行
一次 probe，不自动无限重试。
timeout 展示为 `inconclusive`，与 `inference_failed` 分开；详细定义见
[质量与失败产物](Profiling_Protocol.md#质量与失败产物)。

新的 `coverage.json` 使用 schema v2；manifest 仍为 schema v1。`--resume` 核对原始
`sample.json` 与传入 manifest 的完整摘要、各模型 revision、probe 模式、设备（GPU 使用物理 UUID）、
资源、预算、有效下载策略与 Hub endpoints、运行环境及主机/源码身份。
有效下载条件同时记录到 attempt 配置。普通续跑只能沿用原参数，继续没有完成记录的模型，
不会自动重试已记录的失败；中断的重试则沿用该次 attempt 的条件和剩余选择。
历史 schema v1 缺少完整预算与 attempt 证据，保持可读，恢复时明确要求新目录，不猜测旧条件。

`--retry-failed` 重试所有未验证成功的已记录模型；`--retry-stage STAGE` 和
`--retry-reason REASON` 可分别重复指定，同类选择取并集、阶段与原因之间取交集。
这些选项都要求 `--resume`，没有匹配项时明确报错；已经验证成功的模型始终跳过。
显式重试允许调整资源、设备、时间和下载预算等配置，但不允许替换冻结样本或切换静态/full 模式。
配置变化逐项保存在 `configuration_changes`，不同条件的结果同时存在时标记
`mixed_configurations=true`；汇总只描述各模型最后一次观测，不代表同一配置下的统一验证。

每次执行写入独立的 `attempts/attempt-NNNNNN/attempt.json` 和逐模型目录。
完成记录先原子落盘，再更新可重建的 `coverage.json`、CSV 和 Markdown；进程被强杀后仍可从
attempt 恢复已完成模型，进行中的模型使用新目录重新验证。旧 attempt 的预算、失败和部分产物不覆盖，
顶层 `resources`/`budgets` 保留第一次条件，每行 `attempt_id` 指向实际验证配置。

清理状态为 `incomplete` 时，先保存当前 model/attempt、原始运行失败和清理证据，再中止批次；
后续必须显式重试并包含所有清理未完成的模型（可用 `--retry-stage cleanup` 选择），
优先执行既有 owner 标签和不可变容器 ID 的清理核验，再继续其他模型。
清理问题独立于最新解析或推理结论保留。重试先核对原主机、原 owner UID 与原 Docker endpoint/context，再取得测量锁，
确认原容器已不存在后才重新解析模型；失败、超时或无法确认时不执行后续模型。
`cleanup_recovery` 保存核验结果，原始 attempt 不改写；旧重试记录缺少明确清理证明时继续阻断。
冻结条件还包括 `ACPROF_NLP_TORCH_INDEX_URL`、`ACPROF_NLP_TORCH_SPEC`、`ACPROF_HOST_CUDA_VERSION`；
修改这些运行依赖选型参数须显式重试，attempt 会记录前后差异，普通续跑不能静默混用锁定环境。
同一报告目录通过现有目录锁排除并发写入，运行验证继续使用原有测量锁。

恢复语义参考 [Ray Tune 的 `Tuner.restore`](https://github.com/ray-project/ray/blob/f8a314bf077c9772fee2a8a1073368ca2ef9360b/python/ray/tune/tuner.py)
对已完成、未完成和失败任务的区分（Apache-2.0，维护中的实现）。AC-Prof 沿用已有 JSON 原子写入和目录锁，
只借鉴选择与证据保留原则，不引入 Ray、训练 checkpoint 或 pickle 依赖；记录发生在独立验证之外，不进入正式测量窗口。

### `acprof run`

#### 模型与资源矩阵

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | required | Hugging Face model ID，例如 `google-bert/bert-base-uncased`。 |
| `--revision` | Hub 默认分支 | 指定模型分支、tag 或完整 commit SHA；运行时固定解析后的 SHA。 |
| `--task` | auto | 覆盖 `pipeline_tag`，例如 `fill-mask`、`text-generation`。 |
| `--task-family` | auto | 覆盖任务族：`nlp`、`cv`、`audio`、`timeseries`、`diffusion`、`multimodal`、`structured`。 |
| `--backend` | auto | 覆盖声明清单中的 runtime backend，例如 `transformers_pipeline`、`chronos`、`diffusers`、`onnxruntime`。 |
| `--model-spec` | 无 | 本地 `acprof_model.json` 格式的模型接口声明，优先于仓库声明，固化到服务镜像并参与恢复身份。用于缺少任务元数据、制品选择、custom pipeline 输入映射与固定离线依赖；TUI 对应“高级参数 → 识别覆盖 → 模型接口声明”，见[模型声明](Runtime_Compatibility.md#本地模型声明与自定义-pipeline)。 |
| `--profiling-mode` | `full` | `full` 保留 Native Linux 的 RAPL、perf 和 packet latency 必需条件；`basic` 要求 application latency、吞吐、容器 CPU/内存，跳过能耗、PMU、抓包，允许 WSL2 PARTIAL。两者均要求本机 Docker 和 cgroup v2；不自动降级，见 [WSL2](platforms/wsl2.md)。 |
| `--cpus` | `1,2,4,8` | CPU core 限制列表。 |
| `--cpuset-cpus` | 空 | 可选固定 CPU ID/范围，如 `0-3,8`，应用于正式采集和 startup probe；留空保留原有配额调度。规范化集合参与恢复身份，采样前核验实际 affinity。 |
| `--mems` | `2,4,8,16` | Memory cap GB 列表。 |
| `--gpus` | `off,on` | GPU mode 列表。`on` 只向容器暴露选定的物理 GPU。 |
| `--gpu-device` | 环境变量或 `0` | 单个主机 GPU index 或完整 UUID，优先级为此参数、`ACPROF_GPU_DEVICE`、`DEVICE_INDEX`、`0`。运行前解析并固定 UUID；不接受 `all`、设备列表或 MIG。`acprof probe` 使用同样的环境变量，post-hoc GPU 补采使用原实验记录的 UUID。 |
| `--prune-startup-oom` / `--no-prune-startup-oom` | enabled | 正式矩阵前，用最低选中 CPU、内存升序执行独立 startup probe，只启动并等待 `/ready`，不产生性能结果。仅 Docker 确认启动 OOM 的连续低内存前缀用于剪枝；遇到 ready、timeout、CUDA OOM 或普通错误即停止扩展。所有被剪枝 case 标为 `inferred_not_measured`。禁用后逐格正式尝试。 |
| `--matrix-order` | `seeded` | `seeded` 用版本化的确定性 hash 排序资源 case，并用每个 case 的独立派生 seed 排列 input scale；`declared` 保持资源参数与物化输入尺度的顺序。probe 完成后将实际顺序冻结到 `metadata/matrix_plan.json`。 |
| `--matrix-seed` | `0` | 整数 seed；相同实验身份、probe 结论、算法版本和 seed 生成相同计划。resume 校验并复用已冻结顺序，不重新排序；不能在原目录改变 seed 或 order。 |
| `--dram-energy` | `auto` | `full` 中独立采集可用的 DRAM RAPL 域；缺失或不可读时保持 `nan`，不因此使 full 失败。`off` 不采集，`required` 要求所有选中 package 的 DRAM 都可用且取得有效测量；不能与 `basic` 同用。DRAM 不加入 container-attributed energy。 |

#### 请求窗口与采样

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--batch-size` | `1` | 每个 request 的 batch size。 |
| `--warmup` | `2` | 每个资源配置、每个 input scale 的 warmup 行数。 |
| `--repeat` | `5` | 每个资源配置、每个 input scale 的正式测量行数。 |
| `--repeat-in-window` | `0` | 每一行内部连续发送的 `/predict` request 数量。`0` 表示 auto 模式：每行至少发送 1 个请求，并持续到累计 `latency_app_s` 达到 `--repeat-window-seconds`。 |
| `--repeat-window-seconds` | `10.0` | `--repeat-in-window 0` 时的目标 workload window 秒数。auto 模式不再额外跑一个 10 秒校准窗口。 |
| `--request-timeout-seconds` | `300.0` | 正式矩阵中每个 `/predict` 请求的连接等待和读取无进展上限（分别应用），必须是大于 0 的有限值；它适用于 warmup、auto-window warmup 和正式请求，不是请求的严格总截止时间，也不限制整行、整个 case 或整条命令的总运行时间；持续有数据到达可使总耗时超过此值，不自动重试。超时后保留已完成行，并将触发请求及后续未测计划行分别写成可诊断的 error 占位。 |
| `--sample-hz` | `20.0` | GPU power sampling rate，单位 Hz；CPU workload 和 matched control window 期间也用它控制 RAPL、container cgroup、CPU frequency 和 GPU/resource usage 的采样间隔，以估计 average/peak power、vCPU share、CPU utilization 和 CPU cycles。perf MIPS 使用独立的 `perf stat` 窗口，不受该采样率影响。 |
| `--idle-seconds` | `20.0` | 每个 workload window 前 matched control window 的目标时长。CPU、GPU、resource usage 以及启用时的 perf MIPS monitor 会按与 workload 相同的 `start()` / `stop()` 生命周期同时运行，但 control window 内不发送 `/predict` 请求。CPU baseline 为整段 RAPL 能耗 / 实际 duration；GPU baseline 为 NVML samples 的时间加权平均功率。case 结束后会复查该 case CSV 中所有有效 CPU/GPU baseline 的相对极差，达到或超过 5% 会输出 warning，实验继续运行。 |
| `--idle-cooldown-seconds` | `5.0` | 每个 workload window 采集 idle baseline 前的统一冷却等待时间。CPU-only 和 GPU+CPU case 都使用同一个值，避免上一轮推理刚结束后的短时热状态、Docker/server 收尾或 GPU clock/power 瞬态直接进入 idle baseline。 |
| `--idle-debug` | false | 开启 baseline 调试输出。主 CSV 会填充 GPU 的 `gpu_idle_measured_at` / `gpu_idle_rel_range_so_far` 和 CPU 的 `cpu_idle_measured_at` / `cpu_idle_rel_range_so_far`，并写出 `debug/idle/<case-id>.jsonl`。诊断文件记录 matched control window 的 GPU NVML trace、CPU RAPL 子窗口、host/container CPU delta，以及 control 结束后的 `nvidia-smi`、loadavg、top CPU processes、Docker 容器和 `docker stats` 快照。为避免诊断本身污染 baseline，逐进程 `/proc` 快照移到 control window 外，不再归入 RAPL control 能量。 |

#### 输入

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input-scales` | auto | 手动覆盖 input scale 列表；未提供时通常自动规划 6 档，自定义音频清单按声明档数。 |
| `--workload-spec` | task default | NLP、cv、audio、multimodal、diffusion 或 structured 的 workload 清单 JSON。NLP 可声明 embedding／生成参数；读取音频的任务默认复用内置 LibriSpeech 语音；结构化清单声明输入宽度、尺度与种子。各任务须使用对应的清单格式。 |

#### 计算分析器

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--compute-profile-tool` | `none` | 默认跳过全部 compute probe；`both` 独立采集 `torch_profiler_eager` 逻辑 FLOP，并在 `gpu_mode=on` 时采集 NCU GPU 实际执行 FLOP。`torch`、`ncu` 用于单工具诊断，`vendor` 用于 CPU Advisor 与 GPU NCU；`auto` 不再接受。 |
| `--advisor-root` | auto | Host Intel Advisor install root or executable；显式值优先于自动检测。 |
| `--ncu-root` | auto | Host Nsight Compute install root or `ncu` executable；显式值优先于自动检测。 |
| `--advisor-repeat` | `20` | `vendor` CPU Advisor probe 的推理重复次数；最终 FLOP 会除回单 request。 |
| `--torch-profiler-repeat` | `1` | `torch_profiler_eager` probe 的推理重复次数；CPU/GPU 结果分别除回单 request。 |
| `--ncu-repeat` | `1` | NCU GPU probe 的推理重复次数；FLOP、kernel 数和 kernel 时间最终都除回单 request。 |
| `--compute-profile-cpus` | host logical CPUs | 临时 compute profiler container 的 CPU core cap。 |
| `--compute-profile-mem` | 75% host memory | 临时 compute profiler container 的 memory cap，单位 GB。 |
| `--keep-compute-profiles` | true | 保留 raw profiler artifacts；这是默认行为。artifact 位于模型结果目录的 `raw/compute_profiles/`，路径不写入结果行。 |
| `--discard-compute-profiles` | false | 汇总完成后删除 raw profiler artifacts。 |

#### 执行分析器

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--execution-profile-tool` | `none` | 显式启用高开销 execution profiler：`massif` 用于 CPU-only、`nsys` 用于 GPU，`both` 同时选择两者；默认 `none` 不运行。 |
| `--massif-sampling` | `per-scale` | `per-scale` 使用一个代表 CPU/内存逐 input scale 采集并复用；`full` 采完整 CPU × memory 矩阵。 |
| `--massif-reference-cpu` / `--massif-reference-mem` | 最大选中值 | Massif `per-scale` 的代表 CPU 与内存；必须存在于本次 `--cpus` / `--mems` 中。 |
| `--massif-repeat` | `1` | 每个 Massif probe 内执行的 inference 次数。Massif peak 仍是包含加载和预热的 process-lifetime peak，不按此值归一化。 |
| `--nsys-sampling` | `per-cpu-scale` | `per-cpu-scale` 保留全部 CPU、只用一个代表内存；`per-scale` 只用一个代表 CPU/内存；`full` 采完整矩阵。 |
| `--nsys-reference-cpu` / `--nsys-reference-mem` | 最大选中值 | Nsys 缩减采样的代表资源；`per-cpu-scale` 只使用代表内存，`per-scale` 同时使用两者。 |
| `--nsys-repeat` | `1` | 每个 Nsight Systems `acprof_compute` NVTX range 内的 inference 次数；time、count 和 bytes 汇总会除回单 request。 |
| `--nsys-root` | auto | Host Nsight Systems install root 或 `nsys` executable；显式值优先于自动检测。 |
| `--keep-execution-profiles` | true | 保留 `raw/execution_profiles/` 下的 raw Massif `.out` 与 Nsight Systems `.nsys-rep`；这是默认行为。stats 导出的 `.sqlite` 缓存会自动删除。 |
| `--discard-execution-profiles` | false | 汇总成功后删除 raw execution-profiler artifacts，保留 plan、CSV 数值与错误诊断。 |

#### 输出与运行环境

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sniff-iface` | `docker0` | 本机 Docker 默认 bridge 对应的 `tcpdump` 抓包网卡。只有 daemon 改过 bridge 名时才覆盖。 |
| `--output-dir` | `results` | 输出根目录。最终还会追加 model name 子目录。 |
| `--resume` | false | 使用原参数和目录恢复实验；逐项报告身份差异，保留完成 case，备份后重测中断 case；准备未完成时先归档准备证据再重试。已完成实验不重测。TUI 的“恢复 / 重试”也可自动分配新实验目录。 |
| `--skip-build` | false | 核验构建指纹和环境清单后复用镜像；不存在时自动构建，不匹配时退出。 |
| `--model-source` | `huggingface` | `huggingface`、`modelscope`；不同来源独立记录身份，不静默替换。 |
| `--download-mode` | `auto` | 高级/兼容参数：`auto`、`mirror-only`、`mirror-preferred`、`official`；显式参数优先于 `HF_DOWNLOAD_MODE`，TUI 固定使用 `auto`。 |
| `--max-download` | 不设上限 | 下载前核验全部批量 payload 预算，例如 `5GB`、`5GiB`；`0` 只允许缓存命中。任何来源大小未知或总量超限时，在 pull/build/权重下载前停止。环境变量为 `ACPROF_MAX_DOWNLOAD`。 |
| `--model-store` | `~/.cache/acprof/model-store` | 单份主机模型目录，对应 `ACPROF_MODEL_STORE`；运行容器只读挂载。 |
| `--model-store-max` | 不设上限 | Model Store 容量上限，对应 `ACPROF_MODEL_STORE_MAX`；超限须先显式清理。 |
| `--model-download-policy` | `auto` | `auto` 按已覆盖的加载器规则筛选文件，未知结构保留完整快照并记录原因；`full` 下载固定 commit 的完整仓库。策略进入镜像指纹，不能相互误复用。采集和探测入口均支持。 |
| `--notify` | `auto` | `auto` 在配置 Webhook 后启用企业微信；`none` 关闭，`wecom` 显式选择企业微信。配置见[企业微信通知](#企业微信通知)。 |
| `--help` | — | 显示此入口的全部公开参数后退出。 |

`--allow-cgroup-v1`、`--no-compute-profile` 和 `--compute-profile-tool auto` 已删除，使用它们会在参数解析时退出。关闭计算分析使用 `--compute-profile-tool none`。

结果目录存在异常中断留下的 `result_case_*.csv` 时，`acprof run` 会先读取同目录 `static_meta.json/cgroup_version`。只有版本与当前 host 一致才允许续写；版本不同、缺失或元数据不可读时会退出，避免把 v1/v2 窗口合并到同一结果文件。

### 输入规模与音频清单

`--input-scale-policy auto` 是普通 run 的默认范围规划；`minimal` 在任务解析后从对应 workload 的默认尺度
（或同一任务族既有范围）选取最小单一尺度，仍复用原有 tokenizer、音频采样率和模型上限校验。
TUI Smoke 使用 `minimal`；显式 `--input-scales` 优先且严格校验，不被该策略改小。
未显式使用 `minimal` 的历史恢复参数保持原记录语义；更改规划策略或输入尺度属于新实验。

`input_scale` 是每个任务族的主输入尺度，语义由 `static_meta.json` 的 `input_scale_type` 决定：

| task family | `input_scale_type` | 含义 |
| --- | --- | --- |
| `nlp` 文本任务 | `seq_length` | 输入 token length；问答取 context，检索／排序取候选文本 token 数的最大值，固定 query。 |
| `nlp` 表格问答 | `table_rows` | 每张表的行数；query 和列结构固定。 |
| `cv` | `resolution_scale` | 图像／视频帧基础边长 224 像素的缩放倍率；视频帧数固定。 |
| `audio` 读取音频 | `duration_s` | 输入音频时长，单位秒。 |
| `audio` 文本到语音／音频 | `seq_length` | tokenizer 实测输入 token 数；不是生成音频的秒数。 |
| `timeseries` | `context_length` | 时间序列 context length。 |
| `diffusion` | `resolution_px` 或 `denoising_steps` | 图像／视频生成是方形输出边长（像素），帧数固定；无条件图像与 Shap-E 3D 为去噪步数，分辨率／网格解码设置固定。以输入计划为准。 |
| `multimodal` 图像理解／问答／文档检索 | `resolution_px` | 输入图像边长，单位像素；模型 processor 可能重新缩放、切块或固定尺寸。 |
| `multimodal` 音频理解 | `duration_s` | 输入 WAV 的实际样本数 / 采样率，单位秒。 |
| `multimodal` 视频理解 | `frame_count` | 输入 PNG 帧数；FPS 和帧分辨率固定并写入计划。 |
| `multimodal` Any-to-Any | 由 `scale_modality` 决定 | 单一输入模态随尺度变化，其余固定；默认改变音频秒数。 |
| `structured` 表格 | `table_rows` | 每个 batch 项的表格行数，特征宽度固定。 |
| `structured` 策略 | `observation_count` | 每个 batch 项的独立向量观测数，观测宽度固定；不是交互时间步。 |
| `structured` 图 | `node_count` | 每张图的节点数，特征宽度固定，默认双向环有 2 × 节点数条边。 |

未提供 `--input-scales` 时，当前内置 workload 配置通常会为一次 profiling run 规划 6 档 input scale；自定义音频清单则使用清单中声明的档数：

- `nlp` 文本任务会启动容器读取 tokenizer / handler 的可用最大输入长度，最后一档尽量贴近有效上限。Decoder-only 生成额外预留 `max_new_tokens`；encoder-decoder 不从 encoder 输入预算扣除 decoder 输出长度。表格问答按 `1,2,4,8,16,32` 行规划，不进入 token 二分搜索；超出模型容量时明确失败。
- 读取音频的任务从 workload 清单读取默认尺度；内置英文语音清单为 `1,2,5,10,20,30` 秒。文本到语音／音频进入同一 token 规划器；tokenizer 没有有限上限时采用显式 512-token 采集上限，该值不是模型最大容量，Bark 使用自身 semantic 输入限制。`cv` 使用 generator 最大尺度；`timeseries` 读取已加载 Chronos 的 context limit，并与 workload 上限取较小值，拒绝静默截断。
- `structured` 表格／策略默认 `1,8,32,128` 行／观测，图默认 `8,32,128,512` 节点；可用清单或 CLI 覆盖，当前生成器限制非图不超过 4096、图不超过 2048。该限制是 workload 的输入大小限制，不是模型容量。
- `diffusion` 图像／视频任务默认使用 `128,192,256,320,384,512` 像素输出边长；提示词、随机种子、guidance scale 和去噪步数在各尺度间保持不变。`unconditional-image-generation`、`text-to-3d`、`image-to-3d` 改为扫描 `1,2,4,8,16,20` 个去噪步；Shap-E 使用真实 mesh 解码，渲染图片尺寸不作为网格计算规模。
- `multimodal` 从任务 workload 读取默认尺度：图像边长 `224,336,448`；音频 `1,2,5,10` 秒；视频 `2,4,8` 帧。清单 `input_scales` 可覆盖默认值，CLI `--input-scales` 优先。`static_meta.input_scale_type` 在计划完成后取实际 workload 的单位，不能将所有多模态任务统一解释为 token 数。
- 同一次 run 的所有资源配置共用同一组 scale。
- 所有任务族都会把已确定尺度的 payload 写入唯一的 `input_scale_plan.json`；主采集与 compute profiler 共同读取该文件，保证实际执行 payload、FLOP profiling 和 CSV 中记录的 `input_scale` 一致。
- 手动传入 `--input-scales` 时以手动值为准；workload 会在 sweep 前验证合法性（图像／视频生成分辨率至少为 64 且必须是 8 的倍数；去噪步数是正整数；CV 倍率为有限正数）。

#### NLP workload 参数

NLP 清单使用 `schema_version=1`、与检测结果一致的 `task` 和 `params`，不接受额外字段。
例如 `feature-extraction` 的句向量归一化：

```json
{"schema_version": 1, "task": "feature-extraction", "params": {"normalize_embeddings": true}}
```

`feature-extraction`／`sentence-similarity` 可指定字符串 `prompt` 或 `prompt_name`（二选一）以及布尔
`normalize_embeddings`；命名 prompt 必须由模型声明。特征提取的这些参数要求
`sentence_transformers` backend，普通 Transformers token 特征不能忽略参数继续运行。
未指定时保留模型默认 prompt 和模块图的归一化行为。`normalize_embeddings=false` 不会移除仓库已有的
Normalize 模块，遵循原生 `encode` 语义。尺度表示正文 token 数；容量预算额外扣除
prompt 和 special tokens，预处理还验证拼接后的实际长度，防止编码器隐式截断。

生成任务接受正整数 `max_new_tokens`，例如 `{"schema_version":1,"task":"text-generation","params":{"max_new_tokens":16}}`。
清单通过 `--workload-spec /path/to/workload.json` 传入；参数、清单 SHA256 及每档实际 payload
写入输入计划，主采集与 profiler 重放同一参数。本接口不加载用户执行代码，也未增加任意 chat 模板或非对称检索任务。

#### 真实音频 workload

`automatic-speech-recognition` 默认使用 `assets/audio/librispeech-clean-test-en-30s/source.json`。该清单引用 LibriSpeech `clean/test` 中同一说话人、同一章节的三条连续语音，按固定顺序拼接后截取前 30 秒；素材是单声道 16 kHz PCM16 WAV，许可证为 CC BY 4.0。每一档输入都从同一个 30 秒基准音频取前缀，不做逐档归一化、补全或循环。

音频分类、Encodec／DAC 音频重建和 Silero VAD 默认复用相同语音前缀。codec 按模型需要在预处理阶段重采样，输入规模仍按源音频时长记录。生成／重建响应的 `audio_num_samples`、`audio_sample_rate`、`audio_duration_s` 描述输出波形，不能写入文字 token 计数；VAD 的 `segments` 是秒为单位的连续阈值帧区间，阈值与分帧策略随响应记录，不是识别文本。Silero 每个请求重置状态，profiler 重复调用也不会延续上一请求的隐藏状态。

新增任务使用当前 CSV 字段、静态 schema v7 与输入计划 schema v2；旧 schema 会被拒绝。`input_units_per_request = effective_input_scale × batch_size`：表格／策略是总行数／观测数，图是总节点数。结构化 `input_num_samples` 对表格／策略记总行数／观测数，对图记图数量，另外在计划记录总节点数与边数。NLP 检索／排序的单位仍为候选文本尺度乘 batch，不再乘候选数量；固定 query、候选数及重复编码成本属于该请求，比较实验时必须保持一致。零样本 NLI 的候选标签推理成本同样包含在请求中。

结构化输入为固定种子的合成矩阵／环图，保存特征宽度、种子、结构、清单 SHA256 及实际 payload；各资源组合和 profiler 使用同一计划。模型缺少 SafeTensors 元数据时，参数量／权重字节数保持 `null`，不能以输入大小代替。skops 与 Silero 的 GPU 配置失败不会生成虚假的 GPU 指标；TorchScript 不支持加载时更换 attention implementation，eager FLOP 采集明确失败并保留工具状态，不能把缺失 FLOP 当作 0。

音频请求采用 JSON 内的 Base64 WAV；handler 仍能读取历史 `audio_samples` 浮点数组。短音频模式会读取模型 feature extractor 的约束并拒绝超过 receptive field 的尺度。对于 Whisper，30 秒是音频 receptive field；当前 feature extractor 会把接受的短音频补齐为固定的 480,000 samples / 3,000 frames，`/scale_meta` 会显式记录这一点。`max_target_positions=448` 是解码器输出 token 上限，不是音频输入上限，因此框架不会把 latency 必须随 `duration_s` 单调增加作为正确性条件。

自定义素材时可复制内置 `source.json`，设置新的 `workload_id`，再修改相对素材路径、SHA256、provenance 和 inference 字段。自定义 provenance 可以描述单条录音或既有重采样/增益流程；运行时仍会严格验证派生 WAV 本身是单声道、16 kHz、PCM16 且哈希匹配。然后传入：

```bash
acprof run --model openai/whisper-large-v3 \
  --workload-spec /path/to/source.json
```

当前音频 request 只实现 `batch_size=1` 和 `short_form`。清单会拒绝非空的 `chunk_length_s` / `stride_length_s`；长音频 sequential/chunked 应使用独立 workload，不能通过把本清单尺度直接扩展到 30 秒以上来混测。

视觉、多模态与图像条件生成也接受 `--workload-spec`，清单和支持边界见 [README 视觉任务](Runtime_Compatibility.md#视觉任务) 与 [多模态任务](Runtime_Compatibility.md#多模态任务)。输入计划沿用 schema v2，新增信息写在扩展的 `workload`、`input_metadata` 和 `payload` object 内；CSV 未增加列，当前 schema 的可选字段允许缺失。`workload` 保存素材路径／SHA256、清单 SHA256、提示词、参数、尺度单位和固定条件；实际序列化 payload 及计划 SHA256 是重放依据。

CV 每请求一个图片／视频样本，`input_num_samples=1`；视频帧数及每帧 SHA256 单独记录在 `input_metadata`。零样本标签、VitPose 的人物框和参数也随 payload 重放，不增加隐藏的人物检测请求。CV 响应按任务区分 classification、detection、caption、depth、segmentation、masks、features、keypoints；大张量／掩码／深度图只返回摘要。未产生文本的任务不填写输出 token 数，相关 CSV 指标保持 `NaN`。

无条件图像与 3D 的 `input_units_per_request` 单位为去噪步；其 `task_param.num_inference_steps` 随输入尺度变化。Shap-E 响应 `output_type="mesh"`，输出长度表示网格数量，顶点和三角面数量另行记录，不冒充文本长度、像素数或 token 数。该任务的 profiler 覆盖去噪及真实网格解码；图像条件预处理仍属于请求应用延迟。此扩展不改变历史图像生成的分辨率单位，读取时应依据每次实验的 `input_scale_type`，不能只按 diffusion 任务族判断单位。

多模态 `input_num_samples=1` 表示一个请求样本；其中可含图像、音频、视频或 query＋文档。音频 PCM 样本数与视频帧数分别保存在 `input_metadata.audio_num_samples` / `video_num_frames`。`input_units_per_request` 继续等于有效 `input_scale × batch_size`，因此其单位取决于上表，不可横跨不同尺度类型直接比较每单位延迟。

文字生成／问答的 `output_length_avg` 是返回文字的字符数窗口均值；`output_token_count_avg` 是对输出文字重新分词的 token 数，不代表所有解码步或 Omni 音频 token。Omni 音频摘要另含 24000 Hz 采样率、PCM 样本数和秒数，不混入文字长度。图像生成 `output_length` 为图像数，视频生成为总帧数；检索仅返回 query-by-document 分数矩阵，不产生文字长度／token 字段，其对应 CSV 值保持 `NaN`。

主请求延迟包含输入预处理、模型推理和结果摘要。后置 profiler 测量 handler 的 `predict()`：多模态理解、视频分类和关键点的直接模型路径不包含在 `preprocess()` 中运行的 processor；使用 Transformers pipeline 的 CV 路径仍包含 pipeline 内部预处理和后处理。Base64 解码与 handler 的摘要处理均在 profiler 的预测段之外；Diffusers 的完整生成／媒体解码、检索的两个编码器 forward 与 MaxSim 则在预测段内。Omni GPU 采用 thinker/talker FP16 和 Token2Wav FP32；其 eager FLOP 请求明确不支持，NCU/Nsys 仍可测完整推理。Shap-E 及 Diffusers Transformer 视频架构在没有可验证 eager 替换时同样报告工具错误，不伪填 FLOP。生成媒体不会作为图片／音频／视频／网格响应体返回，因此网络指标描述当前摘要服务协议。

### `acprof probe`

`--revision` 接受 branch、tag 或完整 commit SHA；TUI 的模型候选、采集前自动解析和最大输入探测共用当前填写的 revision。

复用 `--model`、`--task`、`--task-family`、`--backend`、`--model-spec`、`--batch-size`、`--workload-spec`、
`--output-dir` 和 `--skip-build` 的参数及默认值。
资源列表与超时的用途如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--cpus` | `1,2,4,8` | 只选择列表中的最小 CPU。 |
| `--mems` | `2,4,8,16` | 从低到高实测的候选内存上限。 |
| `--gpus` | `off,on` | 包含 `off` 时优先 CPU-only，否则使用 `on`。 |
| `--input-scales` | 自动规划 | 只探测已确定尺度中的最大值。 |
| `--timeout-seconds` | 不设超时 | 单次探测请求的等待上限；显式值必须有限且大于 0。 |

`acprof probe` 不接收 `acprof run` 的 `--request-timeout-seconds`、warmup/repeat、能耗采样或 profiler 参数。
详细用法见[先探测最大输入](Getting_Started.md#先探测最大输入)。

### `acprof profile`

位置参数 `result_dir` 是已完成的模型结果目录。操作和恢复规则见
[补采说明](Profilers.md#补采已有结果)。

TUI 使用四项复选框选择补采工具（初始勾选 `torch`、`ncu`），将勾选结果传给
`--tools`；未勾选任何工具时不启动补采。工具适用范围、采样策略和指标口径与 CLI 相同。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tools` | `torch,ncu,nsys,massif` | 选择需要补齐的工具，逗号分隔。 |
| `--dry-run` | 关闭 | 验证已有文件并展示计划，不启动分析器或改写结果。 |
| `--force-reprofile` | 关闭 | 强制重采并替换所选工具已经成功的值。 |
| `--massif-sampling` | `per-scale` | 可选 `per-scale`、`full`。 |
| `--nsys-sampling` | `per-cpu-scale` | 可选 `per-cpu-scale`、`per-scale`、`full`。 |
| `--massif-reference-cpu` / `--massif-reference-mem` | 结果矩阵最大值 | 在已有 CPU-only 资源配置中选择代表值。 |
| `--nsys-reference-cpu` / `--nsys-reference-mem` | 结果矩阵最大值 | 在已有 GPU 资源配置中选择代表值；`per-cpu-scale` 只用代表内存。 |
| `--torch-profiler-repeat` / `--torch-repeat` | `1` | 同一参数的两个名称；控制 Torch probe 内推理次数。 |
| `--ncu-repeat` / `--nsys-repeat` / `--massif-repeat` | `1` | 对应工具的 probe 内推理次数，归一化口径同 `acprof run`。 |
| `--ncu-root` / `--nsys-root` | 自动检测 | Host 工具安装目录或可执行文件。 |
| `--compute-profile-cpus` / `--compute-profile-mem` | host 逻辑 CPU / 75% host memory | 临时 compute profiler 的 CPU/内存上限，内存单位 GB。 |

### 其他入口

`acprof report <实验目录或 CSV> [更多输入 ...]` 生成 Comparison Matrix、Pareto 与 Scaling 的离线 HTML。
`--output <新文件.html>` 指定输出，默认首个实验目录下 `report.html`，拒绝覆盖已有文件；
`--baseline <config_id>` 预选配置，单配置 run 也可直接使用 run ID。无需 GUI 或 Web 服务，
不递归扫描输入目录中的备份，源 CSV 保持不变。详细语义见[交互式配置比较报告](Metrics.md#交互式配置比较报告)。

`acprof plot` 接收结果 CSV 路径，`acprof tui` 可用 `--model` 预填模型、用 `--preset` 选择预设。
`acprof audit <目录或 CSV>` 只读校验结果；`--json` 输出报告，`--require-complete --require-ok`
用于验收新实验。`acprof stats <目录或 CSV>` 按测量窗口计算置信区间，支持重复 `--metric`、
`--confidence`、`--resamples`、`--seed`、`--block-size`；定义见[结果分析](Metrics.md)。
省略输出选项时向 stdout 输出报告 JSON。`--output FILE` 保存到指定新文件，禁止覆盖；
`--output-dir DIR` 在指定目录中比较完整 JSON 内容，相同则复用已有文件，否则以本地日期时间
`window-statistics-YYYYMMDD-HHMMSS-ffffff.json` 保存。两个输出选项互斥。
目录模式向 stdout 输出一行 `ACPROF_STATS {"report_path": "绝对路径", "reused": false}`；复用时 `reused` 为 `true`。
TUI“统计报告”页的“计算统计”使用目录模式和默认统计参数，报告位于 v2 结果目录的 `plots/analysis/`（旧目录为 `analysis/`），复用时提示已有报告并显示其内容。
`/stats [csv/dir]` 与按钮等价；`/report [json]` 或“查看报告”读取已有窗口统计、监测开销或 CLI/TUI 对照报告。
这些操作需要 TUI 空闲；开销实验仍通过独立脚本显式运行。报告展示与路径带入方式见 [TUI 说明](TUI.md#统计报告)。
`/images` 打开“镜像管理”页并自动读取数据，空闲时每轮读取完成后 5 秒更新；离开页面或运行任务时暂停。
默认视图为镜像树，可切换到镜像列表或层共享。
树中 `←/→` 折叠/展开、空格勾选；列表表头点击排序，再次点击反向。筛选支持模型、逻辑环境名、标签、ID 和平台。
镜像列表优先显示逻辑名称和完整/新增大小，右侧保留 `Repository` 和 `Tag`；无标签镜像显示短 image ID，`Tag` 为 `—`。
“选择同模型”勾选模型镜像；“删除所选”确认全部标签及按集合去重的释放估算。层共享视图只浏览层，仍可清除或删除先前勾选的镜像。
视图、筛选、排序、折叠、勾选与手动调整的详情高度只在本次会话保留，不写入设置文件；自动刷新保留仍有效的勾选和浏览位置。
详情高度通过列表下方的分隔条调整；窗口缩小时限制显示高度，放大后恢复手动值，`Home` 恢复默认高度。
镜像消失、标签变化或新增容器引用时取消对应勾选，Docker 环境变化时清空勾选。
关系证据和空间口径见[镜像管理与清理](Runtime_Compatibility.md#镜像管理与清理)。
各入口的完整帮助可直接运行：

```bash
acprof run --help
acprof probe --help
acprof profile --help
acprof plot --help
acprof audit --help
acprof stats --help
acprof tui --help
```

### 独立比较与负载

`acprof compare --left <实验> --right <实验>` 支持重复指定两侧独立实验，输出差值、比值和跨实验区间；
参数与统计假设见[跨独立实验比较](Metrics.md#跨独立实验比较)。
`--purpose` 支持 `same-hardware`、`cross-hardware`、`resource-scaling`；组内重复始终按相同硬件核验。
`audit --compare` 和 `report` 使用同名取值的 `--comparison-purpose`；HTML 可在生成后切换用途。
资源扩容只放开 CPU/内存配额，双方资源坐标和仍需匹配的条件见[比较规则](Metrics.md#跨独立实验比较)。
TUI“统计报告 → 独立实验比较”复用同一入口，可选择左右组、基线和比较用途，或直接读取 CLI 输出的 JSON。
`acprof load <源实验> --gpu off --scenario concurrent --concurrency 4 --output-dir <新目录>`
执行独立 HTTP 负载；到达率使用 `--scenario arrival-rate --rate 10 --arrival poisson`。
协议、连接复用前提和失败口径见[独立非流式负载](Profiling_Protocol.md#独立非流式负载)。
