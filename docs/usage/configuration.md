# 主机配置与通知

[← 返回专题目录](cli.md)

## 主机环境与 Hugging Face 认证

CLI 启动时读取当前工作目录的 `.env` 和 `.env.local`；同名值的优先级为
**进程环境 > `.env.local` > `.env`**。文件中的值不执行 shell 命令或变量展开。
凭据保存方式见[认证配置](../getting_started/installation.md#hugging-face-认证)。

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
采集权限的系统授权单独操作，见[最小权限安装](../getting_started/installation.md#最小权限安装)。

`acprof tui --model <ID> --preset smoke --output-dir <目录>` 可覆盖本次初始表单。
显式 preset 优先于已保存的实验默认参数，显式输出目录再覆盖 preset 的目录；未传入的 model
沿用已保存模型。启动参数不直接写入设置文件，保存时机仍遵循下方约定。
smoke 为 basic CPU 单次请求配置；`main` 和 `default` 保持 full。

`acprof/tui/settings.py` 管理项目隔离的 `tui.json`，当前版本为 v4；
路径与操作方式见 [TUI 设置文件](tui-experiments.md#设置文件)。
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
悬停、聚焦及确认弹窗沿用操作原有的颜色。页面布局与操作位置见 [TUI 说明](tui-pages.md#启动和页面)。

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
