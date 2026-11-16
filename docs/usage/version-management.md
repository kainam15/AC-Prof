# 版本管理与恢复

[TUI 指南](tui.md) · [安装](../getting_started/installation.md) · [发行版本策略](../development/distribution.md#版本策略与失败恢复)

在 **应用设置 → 版本管理** 中查看当前版本、Alpha / Beta / RC / Stable 类型、生效渠道、
最新正式版、最新预览版和最近检查时间。渠道选择是待确认的意图；安装成功并验证后才生效，
无需点击页面底部的“保存设置”。

Stable 从兼容当前 Python 的正式版中选择最高版本。Preview 从正式版和 Alpha / Beta / RC
中选择最高版本，因此可以从 Alpha 升级到 Beta、RC，再升级到正式版。
候选使用 PEP 440 比较，排除 Dev、本地版本、没有可用文件、全部撤回或 Python 不兼容的发布。
最新预览版单独显示，可能低于最新正式版；此时 Preview 渠道仍选择更高的正式版。
从较高的预览版返回 Stable 会明确要求确认降级，不会在检查更新时自动降级。

## 更新与重启

1. 点击“检查更新”，选择渠道，再点击“更新 / 切换版本”。
2. 确认源版本、目标版本、PyPI 来源、安装路径和预览版风险；Esc 可取消。
3. 等待备份、依赖解析、下载、安装和验证。阶段来自实际执行与安装器输出，未知总进度不显示百分比。
4. 验证成功后点击“重新启动 TUI”。重启前禁止开始新任务；当前运行版本与已验证安装版本分开显示。
   无交互终端时提供绝对路径的重启命令。重启保留原 CLI 参数与工作目录，不自动保存表单草稿。

页面关闭或切换不会取消安装。独立更新进程使用工具环境外的 Python 与预先复制的标准库脚本，
不依赖正在被替换的 AC-Prof。TUI 意外退出时安装继续，重新进入后可读取保留的阶段和日志。
安装器和更新进程继承现有测量锁；采集、预检、补采、读结果和其他互斥操作运行时不能开始安装。
检查网络也使用同一测量锁，HTTP 总期限为 3 秒，响应最多 1 MiB；断网只影响检查状态。

自动更新核对 `sys.prefix`、包位置、`uv-receipt.toml`、`uv tool dir`、解释器与命令 symlink，
拒绝指向另一安装的环境覆盖、额外 requirements / extras、direct URL、开发安装和不明确的入口。
安装使用参数数组，不执行 Shell、sudo 或权限提升；只替换当前工具环境，不改实验目录、模型缓存或 TUI 设置。
备份与安装必须在同一文件系统，且必须有备份和替换空间。依赖仅使用 Wheel，不能安装的依赖会明确失败。

安装要求保持 `acprof` 无固定版本约束，并保存 `allow` / `disallow` 策略，后续普通
`uv tool upgrade acprof` 不会被本次目标版本固定。安装后核对 distribution 版本、包版本、路径、
命令入口与 receipt；若确认之后发布了新版本或依赖解析选择了其他版本，会验证失败并恢复，重新检查即可。

## 日志与故障恢复

“安装日志”可查看末尾 64 KiB 并复制完整日志路径。更新状态独立存放在
`${XDG_STATE_HOME:-~/.local/state}/acprof/updates/<安装路径摘要>/`，包含：

- `check.json`：最近成功检查时间；`channel.json`：验证成功后的渠道。
- `status.json`：当前或最近一次事务状态；`active.lock`：操作占用标记，锁随进程退出释放。
- `jobs/<事务 ID>/`：`plan.json`、独立 `worker.py`、`worker.log`、`install.log` 和备份。

错误不会以成功状态掩盖。权限不足时检查所示目录的所有者与写权限；磁盘不足时释放工具环境备份所需空间；
网络或超时错误检查 PyPI、代理和证书；依赖／环境不匹配时核对 Python、安装路径和日志，再重新检查更新。
不要为修复更新而删除实验结果、模型缓存或 `tui.json`。

安装前完整复制原工具环境及 receipt，保留命令 symlink。安装或验证失败后先终止并回收安装器，
再将失败环境留在 `failed-environment/`，原环境恢复到原绝对路径并重新验证。原渠道保持不变。
成功后仍保留备份；确认新版本可用后，由用户按事务目录清理，不自动删除历史证据。

若系统断电或更新进程被强制杀死，可能只留下 `running` 状态；TUI 将明确提示中断，禁止继续采集。
先确认更新器和安装器已经退出，保留整个事务目录。按 `plan.json` 中的绝对路径处理：

1. `backup-complete.json` 存在时，`backup/` 才是完整备份；不存在时不要用半份备份覆盖安装。
2. 若 `backup/` 仍在，将当前 `prefix` 改名保留，再将 `backup/` 移回原 `prefix`。
   不将备份当作另一个路径下的正常安装使用，因为命令 shebang 包含原路径。
3. 按 `entrypoint.json` 保存的 symlink 目标恢复 `entrypoint`，执行该绝对路径的 `--version`。
4. 确认恢复正确后保留故障 `status.json` 的副本，并移走该陈旧状态文件，再重新打开 TUI。
   不移动 `channel.json` 或现有用户设置。若恢复失败，保留原始错误与两份环境，请管理员处理。

## 其他安装方式

| 安装方式 | 当前行为与安全路径 |
| --- | --- |
| `uv tool` | Linux 标准独立工具环境支持一键更新、双向切换、验证与回退；自定义 requirements / 入口拒绝自动操作。 |
| `pip` / `uv pip` | 识别当前虚拟环境，复制绑定该解释器的命令；自动更新仍需依赖冲突审阅、环境快照和恢复方案，不修改共享环境。 |
| `pipx` | 识别工具环境，给出绑定 `PIPX_HOME` 的手动命令。自动切换尚未启用：还需保护 pipx shared libraries、注入包与应用入口，不能只复用 uv 的目录替换。 |
| Standalone | 提供 Release 入口；需下载对应平台文件、校验 SHA256SUMS、保留旧文件、原子替换并验证，失败恢复旧文件；未启用自动替换。 |
| 源码 / 系统 Python / 未知环境 | 显示原因与手动指引；源码不会被安装器覆盖，系统环境不会自动提权。 |

实现参考 [uv 工具安装与升级](https://github.com/astral-sh/uv/blob/main/docs/concepts/tools.md)、
[uv install 源码](https://github.com/astral-sh/uv/blob/main/crates/uv-tool-commands/src/install.rs) 和
[pipx 环境事务](https://github.com/pypa/pipx/blob/main/src/pipx/commands/install.py)。
TUI 借鉴 Carbon 的 [Loading](https://www.carbondesignsystem.com/building-blocks/core/components/loading/guidelines)
与 [Modal](https://www.carbondesignsystem.com/building-blocks/core/components/modal/guidelines) 原则，复用现有 Textual 主题和确认对话框。
