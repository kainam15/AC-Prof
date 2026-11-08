# PNG Golden Snapshot 视觉回归（VHS）

保留 [SVG/Textual 回归](devtools.md#辅助开发工具)作为稳定的控件、行为与结构基线；
另用 [charmbracelet/vhs](https://github.com/charmbracelet/vhs) 驱动 ttyd + xterm.js + 无头 Chrome，
通过 VHS 的 `Output frames/` PNG 帧序列取稳定的最终文本帧，检查中文字符、连续边框、截断、弹窗与实际终端模拟器的布局。这样不会依赖 VHS v0.12.1 偶发静默漏文件的单独 `Screenshot` 命令。
这是**开发测试工具**，不是运行时依赖，不在正式实验的测量窗口调用。

## 前置环境

需要在执行截图的 Linux 开发机或临时 CI runner 安装：
- VHS（固定 v0.12.1，支持 `Set Columns` / `Set Rows` 与 `Screenshot`；v0.12.0 存在 [退出码为零却不生成文件的上游缺陷](https://github.com/charmbracelet/vhs/issues/787)）
- `ttyd`、`ffmpeg`、Chrome/Chromium（支持 headless）、`fontconfig`
- 字体 `Noto Sans Mono CJK SC`（Ubuntu: `fonts-noto-cjk`）
- Golden 基线采用 Ubuntu 24.04 / VHS v0.12.1 / Chrome 147.0.7727.116；CI 从 Chrome for Testing 下载固定版本，并验证字体文件 SHA-256。跨系统运行可能产生噪声，请在与 CI 一致的环境制作基线。
- Ubuntu 24.04 的 AppArmor 可能阻止临时目录中的 Chrome for Testing 创建用户命名空间。CI 仅在对应限制启用时，为该次 runner 中**固定路径的 Chrome 可执行文件**加载 `userns` AppArmor profile；不关闭全局限制、不使用 `--no-sandbox`。参考 [Chromium 官方说明](https://chromium.googlesource.com/chromium/src/+/main/docs/security/apparmor-userns-restrictions.md)。

Ubuntu 示例（VHS 本体可从 [官方 Releases](https://github.com/charmbracelet/vhs/releases) 下载、核验校验和后安装）：

```bash
sudo apt-get update
sudo apt-get install -y ttyd ffmpeg fonts-noto-cjk
fc-match 'Noto Sans Mono CJK SC'
# 可独立运行浏览器/VHS 冒烟测试，不依赖 AC-Prof 模型或 TUI
python3 scripts/smoke_tui_png_environment.py
uv sync --locked --no-default-groups --group test
```

无需 X11、Wayland、远程桌面、Windows、dor-jump、Docker、GPU 或真实模型；
Chrome 仅用于无头启动预检与 VHS 截图。系统安装的工具不会加入 `pyproject.toml` 正式依赖或任何推理镜像。

## 运行与结果

```bash
uv run python scripts/capture_tui_png.py --scenario form
uv run python scripts/capture_tui_png.py --scenario error-modal
uv run python scripts/capture_tui_png.py --scenario near-match
uv run python scripts/capture_tui_png.py --all --output internal-testing/shared/tui-png

# 与 Git 中已审阅的 PNG 基线比较，失败时生成 diff 与 report.json
uv run python scripts/compare_tui_png.py
# 基于 report.json 和 PNG 证据生成独立 HTML（比较失败后也可运行）
uv run python scripts/render_tui_png_report.py

# 显式启用真实 VHS 截图 + Golden PNG 比较；普通 pytest 不要求 VHS
ACPROF_PNG_VISUAL=1 uv run pytest tests/visual_png -v
```

默认结果为 `internal-testing/shared/tui-png/` 下的 12 张 PNG；其中 `form-small`、`statistics`、`statistics-en`、
`docker-frontend-small` 和 `docker-frontend-en-small` 使用 **80×24**，其他场景使用 **120×36**，目录已由 Git 忽略。
新增的 `docker-frontend`、`docker-frontend-small`、`docker-frontend-en`、`docker-frontend-en-small`
模拟 Dockerfile 前端解析静默 58 秒，检查未知下载量、等待时长、来源、元数据说明和停滞提示；不会启动 Docker。
所有场景使用固定字体、16 px 字体大小；`statistics-en`、`docker-frontend-en` 和 `docker-frontend-en-small` 使用英文，其余使用中文，
Graphite 主题、UTC、固定测试数据与隔离的临时 HOME/缓存。
测试夹具还将环境身份固定模拟为 `Native Linux / FULL`，避免 CI 虚拟机的实际身份引入 PNG 差异。
该模拟仅在测试夹具进程中生效，不修改正式运行时的环境检测。
截图入口在构造 App 前临时移除 `NO_COLOR`，退出后恢复原值，避免本地灰度设置进入 Golden。
这与现有 SVG fixture 一致；[Textual 8.2.8](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/app.py)
只要发现该变量就启用灰度过滤，设置为空字符串也不能关闭。沿用现有 MIT 依赖和标准库，
不复制上游代码或引入运行时依赖。
SVG 与 PNG Golden Fixture 共用 `tests/support/tui_fixtures.py` 的 `SNAPSHOT_TITLE`，
渲染前固定为已审阅的 `AC-Prof 0.3.0`，使发布版本变化不触发整套基线更新。
该值仅用于视觉测试输入；正式 TUI 仍展示 `acprof.__version__`，由双语启动回归核对可见页头。
沿用 Textual 的 `title` / `TITLE` 公开接口，不替换截图文本、不遮罩标题区域，也不放宽比较阈值；
实际布局差异仍按 [Textual 上游的快照审阅流程](https://github.com/Textualize/pytest-textual-snapshot)检查。
`--all` 逐场景启动，避免一个场景遗留 UI 状态或本地设置影响另一个。

测试复用真实 `AcprofTui` 的页面与弹窗组件，但由
`tests/visual_png/fixture_app.py` 注入测试设置、模拟配置和错误；
禁用真实预检、采集子进程、历史结果读取与镜像刷新，设置离线模式；near-match 场景仅注入一份不访问宿主机的模拟环境检查成功状态。
不会解析下载模型、联系外部服务、采样真实硬件或写入正式实验结果。
VHS 只操作该测试应用，没有向 AC-Prof CLI 添加截图选项。

## 全流程走查与覆盖矩阵

现有 SVG、PNG 与交互测试通过 [流程场景索引](../../tests/visual/flow_catalog.json) 汇总，不新增平行截图框架：

- `tests/visual/flow_catalog.json` 定义六个主页面（含“补采实验”）及按采集阶段排列的检查点，标注触发条件、预期状态、可选/异常路径，并引用现有 PNG、SVG 与 Pilot 测试文件；索引由 `scripts/tui_flow_catalog.py` 校验。页面顺序与可见标签通过 headless TUI 回归核对；图标入口用可选的 `tab_label` 记录实际标签，`title` 保留可读页面名称。图库数量按 Git 追踪的 SVG 清单验证。新增并纳入 Git 的 SVG 自动出现在图库中，未关联场景会明确提示；未跟踪的本地截图不会公开。
- `scripts/render_tui_png_report.py` 仍以 `report.json` 作为**唯一 PNG 比较结论**来源。在原有 HTML 中加入「流程走查」「视觉对比」「覆盖矩阵」三个视图；视觉对比同时展示现有 PNG 差异和只读 SVG 基准图库。没有 `report.json` 时必须显示「未比较」，不能推断通过。SVG 通过情况仍由独立的 SVG CI 决定。
- **历史 SVG 的网页预览**：`uv run --locked --no-default-groups --group test python scripts/tui_svg_preview_png.py` 只读取 Git 追踪的 `tests/visual/__snapshots__/*.svg`，在隔离的无头 Chrome 中去掉 Rich 的外部 Fira Code 字体引用，固定使用 `Noto Sans Mono CJK SC`，调整字符宽度渲染，将 PNG 写入 `internal-testing/shared/tui-png/svg-preview/` 并记录源文件和预览的 SHA-256。HTML 优先展示这些 PNG，可展开原始 SVG。预览是**旧 SVG 的可读渲染版本**，不等于现有 VHS 真实终端 PNG，也不属于 Golden，不改变 SVG/PNG 比较规则、覆盖数量或结论。旧 SVG 的原始定位和字符内容不会被修改，极端历史字距异常未必能完全修复。未运行生成器时 HTML 保留原 SVG 作为回退；无有效 manifest 或源哈希不一致时不接受旧预览。
- 使用 `uv run --locked --no-default-groups --group test python scripts/tui_svg_preview_png.py --smoke` 仅验证一张已追踪的中文 SVG 能用当前无头 Chrome 生成有效 PNG，不触碰 Golden 或完整预览目录；CI 在批量生成前独立执行这一验证。截图进程使用独立进程组，取得有效图片后会清理浏览器及子进程，避免 Chrome 延迟退出造成假超时。若生成失败，限定的 HTML、stderr/stdout 及可能存在的 PNG 会写入 `internal-testing/shared/tui-png/svg-preview-failures/scene-*/` 并上传到 CI Artifact，不公开到 GitHub Pages；未成功完成批量生成时不写入 `manifest.json`，报告保持原 SVG 回退。
- 流程检查点的测试文件只表示**可追溯的交互测试入口**，不代表这些测试已在本次运行通过；缺少固定截图的节点会列为视觉覆盖缺口。只有高价值稳定场景才需要增加长期 Golden。
- 所有嵌入网页的截图仅来自受版本控制的 `tests/visual*/__snapshots__/` 或现有隔离 PNG 夹具输出；报告不读取真实采集日志、实验结果、用户配置或密钥。GitHub Pages 是公开网站，禁止将真实运行证据合并到自动发布的报告中。

直接执行 `python3 scripts/render_tui_png_report.py` 即可重新生成带流程视图的离线 HTML；既有 PNG 截图、差异计算、Artifact 与 Pages 发布命令不变。需要真实采集验收时继续使用独立的硬件验证入口，并在受控位置保存证据，不能把它当作 Golden 图片来源。
本地首次使用新的 SVG CJK 预览，先执行上面的 `scripts/tui_svg_preview_png.py`，再运行 `scripts/render_tui_png_report.py`；CI 会在生成 HTML 前自动执行预览脚本。

## 连续 Pilot 跨阶段走查（无真实采集）

`uv run --locked --no-default-groups --group test python scripts/capture_tui_journey.py`
会在独立的临时配置目录中运行 **同一个 Textual 应用实例**，模拟实际鼠标点击、恢复检查、下载确认、下载进度、正式窗口、完成核验、结果读取与页面切换。测试夹具从不启动真实模型、Docker、下载、计量或 profiler；成功状态通过明确的合成 `RunResult` 注入，不代表生产采集结果已验证。

- 截图放在 `internal-testing/shared/tui-png/journey/`，包含 `zh-wide`（120×36）、`zh-small`（80×24）、`en-small`（80×24）三组可复现的 SVG、ANSI 终端状态与一个 `manifest.json`。每个步骤记录动作、Tab、屏幕、焦点、按钮与布局位置。它们是**每次执行生成的诊断证据**，不是 Golden SVG，不添加到 `tests/visual/__snapshots__/`，也不进入 PNG 像素比较。
- 使用 `--png-preview` 时，复用现有 VHS/xterm.js 与 `Noto Sans Mono CJK SC` 从同一实例的 ANSI 状态生成清晰的 PNG 预览（不启动真实采集）；HTML 优先显示 PNG，SVG 留作诊断。Rich 原生 SVG 的字体默认是 Fira Code，且其 CJK `textLength` 在浏览器的 `<img>` 中可能挤压字符，单纯替换字体无法保证排版。生成的诊断 SVG 也会改为引用本机 Noto 字体，不再请求 Fira Code CDN，但不以此代替 PNG 验收。对走查过程中动态变为 80×24 的检查点，使用独立终端尺寸生成 PNG，避免错用宽屏图。缺少 VHS 时仍能运行不带该选项的原有 Pilot 测试；正式运行和普通 pytest 不需要 VHS。
- 现有 HTML 报告的「流程走查」会把连续 Pilot 截图按顺序展示，并在「覆盖矩阵」标明每个检查点的新截图数量；若未执行走查，报告必须标注证据不可用。合成截图不得与真实采集的通过证据混同。
- `tests/tui/workflows/test_tui_visual_journey.py` 提供交互回归，核验 80×24 下的实际按钮点击、任务执行时配置锁定、窗口结束后停留监控页，以及点击结果页后的成功 CSV 读取。测试同时说明：**新目录直接启动监控；已有实验目录才显示恢复检查弹窗**。
- TUI PNG Visual 工作流会在生成 HTML 之前以 `--png-preview` 自动采集这些纯合成截图和 Noto PNG 预览；可公开的报告仅嵌入经过固定场景身份校验的证据，不接受任意路径、当前项目日志或真实实验目录。任何真实模型及硬件验收仍使用原有独立入口。

## 长期维护的 Golden PNG

仓库将人工审阅的基线保存在 `tests/visual_png/__snapshots__/*.png`：

- `form`：中文普通表单（120×36）
- `form-small`：小终端、截断与滚动（80×24）
- `error-modal`：左上角环境阻断悬浮面板（沿用原场景标识），含状态圆点、悬停详情及重检操作
- `near-match`：BAAI/bge-m3 自定义配置
- `measuring`：模拟测量阶段与日志布局
- `task-selection`：多任务选择弹窗
- `statistics`、`statistics-en`：中文／英文统计页（80×24）
- `docker-frontend`、`docker-frontend-en`：中文／英文 Dockerfile 前端解析与静默提示（120×36）
- `docker-frontend-small`、`docker-frontend-en-small`：相同解析状态在窄屏中的换行与滚动（80×24）

比较器使用 Pillow 对每个 RGB 像素取最大通道差，差异小于 `32/255` 的像素视为渲染噪声，每张图**最多允许 120 个明显不同的像素**；尺寸不一致、缺失、无效图片或超过阈值均判失败。比较器只读基线，绝不会自动创建/修改它。失败时在 `internal-testing/shared/tui-png/diff/` 生成红色差异覆盖图和 `diff/expected/` 中的原基线副本，同时生成机器可读的 `report.json`。独立的 `scripts/render_tui_png_report.py` 只读取已有比较结论及 PNG 证据，输出忽略目录下的 `internal-testing/shared/tui-png/report.html`；HTML 内嵌旧图、新图与差异图，单文件可离线查看，支持按通过/失败筛选，不重新比较像素，也不更新基线。没有 `report.json` 时明确标注“比较未执行”，不误报通过；损坏的报告会导致生成命令报错。

有意调整界面时，用相同的 Ubuntu/Chrome/VHS/字体环境生成候选截图：

```bash
uv run python scripts/capture_tui_png.py --all --output internal-testing/shared/tui-png/candidates
# 仔细人工审查全部 12 张 PNG，不应包含中文重叠/边框缺口
cp internal-testing/shared/tui-png/candidates/*.png tests/visual_png/__snapshots__/
ACPROF_PNG_VISUAL=1 uv run pytest tests/visual_png -v
```

### 在 HTML 报告审核 Golden 变更

主分支 `TUI PNG Visual` 的 [GitHub Pages 报告](https://kainam15.github.io/AC-Prof/) 在有有效的 `visual-regression` / `size-mismatch` 结果时，会在 **视觉对比** 页显示 **Golden 基准审核**。使用方法：

1. 打开 **视觉对比**，逐项查看左侧 Golden、当前 PNG 与差异图，勾选确认接受的场景；也可全选或清空。
2. 点击 **通过 GitHub 提交更新请求**。静态页面会打开预填信息的 GitHub Issue 创建页（需要登录）；审阅标题、场景和 CI run 后，手动点击 **Submit new issue**。
3. 只有由仓库所有者创建的有效 Issue 才会启动 [TUI PNG Golden Review](../../.github/workflows/tui-png-golden-review.yml)。工作流校验来源属于 main 的 `tui-png.yml`、run 已完成、head SHA 与当前 main 一致、场景处于已完成的视觉回归状态，然后从指定 run 的 Artifact 提取 **仅被勾选的 PNG**，在独立分支创建 PR。
4. 在 PR 中继续审阅并验证变更，确认后合并。未选中的 Golden PNG 不会变化；HTML 或 Issue 本身都不能直接修改 main。

**安全边界**：网页和 Issue 均公开，但页面不存储任何 GitHub Token；只有创建 Issue 的仓库所有者具备更新权限。Issue 触发的工作流不执行 Issue 中的命令，只解析限定字段并校验 CI provenance。**若 main 已经前进，旧报告会被拒绝**，需要在最新 main 手动重新运行 `TUI PNG Visual` 并刷新网页。缺失截图或运行环境失败不能被审核通过。若 Artifact 已过期，也必须重新运行。GitHub Actions 使用内置 `GITHUB_TOKEN` 创建 PR：仓库需在 **Settings → Actions → General → Workflow permissions** 允许 Actions 创建 Pull Request；若 bot 创建的 PR 被标为需要批准工作流，应由有权限的维护者批准后再合并。此功能不会自动合并 PR，也不自动更新 Golden。

无需线上审核时仍可沿用上面的**本地候选截图 + 人工提交 PNG**流程；离线 HTML 没有 GitHub 主分支审核上下文，保持纯查看报告。

必须在 PR 审阅中明确确认 Golden PNG 变化；**CI 绝不在测试失败时更新基线**。基线是经过审阅的变化检测依据，不代表程序能够从零判断中文排版是否美观。Textual 几何断言与 SVG 快照继续发挥互补作用。

## 首页统一验收状态（同一次 CI）

`TUI PNG Visual` 工作流会**在同一提交、同一次工作流运行中**分别执行与主 CI 相同的
`tests/visual` SVG 快照测试、现有 VHS PNG Golden 回归，以及连续 Pilot 测试与截图。
SVG 在这个工作流重新运行，是为了避免跨工作流状态竞争；主 CI 的 SVG 检查仍独立存在，
**本页的 SVG 通过不宣称另一条主 CI 已通过**。

完成（或失败、跳过）后，`scripts/tui_ci_status.py` 使用三个 `steps.<id>.outcome` 和各自的真实产物
核对状态，再写入 `internal-testing/shared/tui-png/ci-status.json`：

- **SVG**：工作流的 SVG pytest 步骤成功，JUnit XML 有实际执行的通过用例且没有失败/错误；
- **PNG**：VHS/pytest 步骤成功，`report.json` 与本次全部 Golden 场景一一对应、每张状态为 `passed`，并有实际 PNG 文件；
- **Pilot**：Pilot 交互 pytest 与连续截图步骤均成功，并且合成走查 manifest 与截图通过安全校验。

首页在「流程走查」等三个视图上方显示三项状态、证据数量、完整 Commit SHA 的前缀、生成时间及对应
GitHub Actions Run 链接。**任何一步失败、跳过、缺失产物、数据损坏或发现不同提交 / 不同 CI Run 时均不能显示通过**。
在本地直接生成 HTML 而没有本次 CI 元数据时，三项一律显示「未运行 / 无法核验」，不会凭已有 Golden 截图猜测成功。
这份 JSON 与 SVG JUnit 也保存在该次 Actions Artifact，可用于人工诊断；不改写长期 Golden，也不包含真实模型采集数据。

当前 Pages 行为**保持不变**：main 的工作流即便出现视觉回归，仍可发布红色诊断报告；如无法生成 HTML 则
保留上一次已部署的网页。PR 仅产生 Artifact，不发布。若要实现「只有全部通过才覆盖正式首页」，需要另行
调整发布策略，不能把它当成本次状态合并已经实现。

## CI 与验证边界

独立的 [TUI PNG Visual](../../.github/workflows/tui-png.yml) 工作流在 **push main / Pull Request 修改 TUI、任务呈现、视觉夹具、锁文件或测试配置时自动运行**，也可通过 `workflow_dispatch` 手动触发。安装截图工具后，先用 `scripts/smoke_tui_png_environment.py` 验证 Chrome DevTools 调试端点就绪（并安全结束其进程组），再检查 VHS 能否产生有效 PNG；若环境失败，直接输出浏览器 stderr，避免误报为 TUI 快照差异。冒烟通过才运行 PNG Golden 比对，并在 PNG 测试步骤后（无论通过还是失败）用 Python 标准库生成 HTML，并通过 GitHub Actions Artifact 上传 `report.html`、actual / expected / diff / `report.json`。截图环境提前失败而没有 `report.json` 时，HTML 只说明无法取得比较结论，具体原因查看 Actions 日志。报告是临时生成物，不提交到 Git 仓库。主分支 main 的 PNG 工作流在生成 HTML 后，将单文件报告作为网页入口 index.html 发布到 [GitHub Pages](https://kainam15.github.io/AC-Prof/)；无须下载 Artifact 即可浏览最新一次**成功部署**的报告。PNG Golden 比较失败时仍尝试发布失败诊断，但无法生成 HTML 时保留此前网站版本；手动从 main 触发也会发布，PR 仅上传 Artifact、不部署。首次使用需在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**，并建议将 github-pages 环境限制为仅 main 可部署。网页公开可读，不能把敏感运行数据加入测试夹具。GitHub Pages 为最新报告展示入口，旧运行证据继续从各次 Actions Artifact 查看；仍不影响原有 CI 的 SVG 快照。正式运行与完整测试环境无需安装 VHS 或桌面。

历史 Rich SVG 的 CJK PNG 预览在安装 test 依赖后、长时间 VHS/Pilot 采集前生成：先对已提交的中文 SVG 运行一次真实 Chrome 截图冒烟，再批量渲染。该离线浏览器进程禁用后台网络更新与同步，仍保留 sandbox；无有效 PNG 时保存受限的截图/日志诊断并使 CI 失败，不跳过校验、不自动更新 Golden。

VHS 的 xterm.js + Chrome 反映的是一个确定的无头终端模拟环境，
不是 VS Code Windows 终端、SSH/tmux 或用户本地字体渲染的完全复刻；
涉及终端原生渲染差异时仍需在实际客户端截图验收。
参考上游 [VHS 官方命令文档](https://github.com/charmbracelet/vhs#vhs-command-reference)
与 [Textual 测试指导](https://textual.textualize.io/guide/testing/)。
