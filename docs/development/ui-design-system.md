# AC-Prof UI/UX 设计规范（IBM Carbon 适配）

> **项目约定**：AC-Prof 将 [IBM Carbon Design System](https://carbondesignsystem.com/) 作为 **首要 UI/UX 设计参考**，而非宣称产品实现、组件库或无障碍认证与 Carbon 完全一致。本文是项目内的**权威适配规范**，适用于 Textual / Rich TUI 和离线 HTML 实验报告。

## 适用范围与优先级

- 新增或调整 UI 时，优先参考 Carbon 的状态语义、信息层级、视觉反馈、通知、颜色 Token 与无障碍原则。已有界面遵循**渐进式统一**，不为追求像素级一致而全局重构。
- UI 决策遵守以下优先级：**测量安全和真实业务状态 → 现有用户操作契约与终端可用性 → 本规范 → Carbon 上游设计指南**。上游 Web 组件的像素尺寸、字体、SVG 图标、动效和 React 实现不直接移植到 TUI。
- Textual / Rich 是 TUI 实现层；继续复用现有控件、`acprof/tui/tui.tcss`、`acprof/tui/ui/themes.py` 的语义化主题颜色及 i18n。不得仅为套用 Carbon 额外引入 Web 前端框架、字体或运行时依赖。
- HTML 报告可更直接借鉴 Carbon 的 Web 页面层级与组件模式，但必须保持当前离线、自包含、报告数据和交互契约，不依赖联网加载 Carbon 资源。
- Carbon 是外部参考来源，不代表 IBM 对 AC-Prof 的认可、合作或认证。本项目只借鉴规则并注明来源，不直接复制外部资产或文档。

## 状态词汇与统一表现

用户应从 **文字标签 + 符号/形状 + 颜色** 理解状态，**禁止仅靠颜色**。具体业务状态的枚举和采集门禁以现有程序为准，视觉样式不得反向改变状态判断。

| 语义 | AC-Prof 典型场景 | 建议文字与符号 | 主题角色 | 行为 |
| --- | --- | --- | --- | --- |
| 正常 / 完成 | 环境就绪、已成功保存 | `✓ 就绪`、`✓ 完成`；缺字形时使用 `OK` | `$success` | 只在确实完成或通过检查时使用 |
| 警告 / 受限 | 部分采集能力不可用 | `! 受限` | `$warning` | 解释限制和可执行的补救措施；不得冒充完全就绪 |
| 错误 / 阻断 | 前置条件不满足、任务失败 | `× 阻断`、`× 失败`；缺字形时用 `X` | `$error` | 给出原因和恢复入口；继续遵守实际启动门禁 |
| 信息 / 进行中 | 预检、下载、分析中 | `… 检查中`、`… 处理中` | `$accent` 或现有信息角色 | 展示真实阶段或进度，未知比例不得伪造百分比 |
| 中性 / 待确认 | 未检查、已过期、未知 | `- 未检查`、`- 已过期` | `$text-muted` | 提供重新检查或进一步查看入口，不推断成功 |
| 不适用 | 功能在当前 Runtime 不支持 | `- 不适用` | `$text-muted` | 与失败、未采集、零值严格区分 |

环境检查至少区分「检查中、就绪、受限、阻断、已过期、检查失败」；**已过期**不等于**检查失败**，**检查失败**不等于**已确认阻断**。聚合状态遵循影响优先：真实阻断优先于受限，受限优先于就绪；检查中/未知须明确显示，不能掩盖为就绪。视觉颜色不决定能否开始采集。

**状态图标须维护少量、固定、易识别的项目词汇**。避免用彩色 emoji 作为唯一图标；先验证 Unicode 在目标终端的字宽、字形和对齐，必要时使用等宽 ASCII 文字回退。禁止在一处把相同符号解释为成功，另一处解释为错误。具体状态显示文案使用现有中英文翻译目录。

## 视觉系统

- **语义 Token 优先**：背景、表面、正文、次要文字、主操作、成功、警告、错误使用现有 Textual Theme/TCSS Token；不要在单个 Widget 中散落硬编码 RGB。现有主题不强制替换为 IBM 蓝或 Carbon 的十六进制色值。
- **文字优先于装饰**：界面呈现任务、当前环境、运行阶段、关键配置及下一步操作。次要诊断按需展开；避免把无须用户行动的每条信息渲染成高注意力 Badge。
- **一致的层级**：页面标题、分组标题、摘要、说明和控件保持可预测的位置与对齐；按钮的主次、危险性、禁用和焦点状态清晰可辨。布局按终端 **cell** 与响应式宽度适配，不能直接移植 Carbon 的 `px` 间距或字体规格。
- **克制地使用颜色**：绿色只表达明确成功，黄色表示需关注，红色表示失败/阻断/危险操作，蓝色或强调色表达信息与交互焦点。状态与交互焦点不能仅通过不同颜色区分。
- **可读性目标**：参考 [WCAG 2.2](https://www.w3.org/TR/WCAG22/) 的颜色使用、文本对比和非文本对比原则；标准文字对比建议达到 4.5:1，必要的控件边界/状态提示建议达到 3:1。终端实际颜色可能被用户主题或 256 色模式重映射，不能把设计目标写成已经实测符合或通过 WCAG 认证。

## 交互与反馈

| 组件/流程 | 统一规则 |
| --- | --- |
| 按钮与危险操作 | 主操作突出、次操作收敛；取消/返回/终止语义固定；真正不可执行时禁用并说明原因，不能用颜色或隐藏控件替代解释 |
| 通知与错误 | 普通成功或信息反馈优先就近展示、非阻断；需要用户处理的警告/失败提供明确原因与下一步；只有涉及安全确认或无法继续的操作才使用阻断式对话框 |
| 进度与长任务 | 标明当前阶段、已知进度、更新时间及等待原因；异步任务保持界面可操作；禁止显示虚假完成、虚假成功或无依据的百分比 |
| 悬停与弹出层 | 悬停可以揭示详情，但**不得成为获取关键状态或执行关键操作的唯一方式**；键盘焦点、点击/Enter 和关闭/返回路径必须可用 |
| 表格与历史结果 | 状态标记与文字左对齐；同列状态语义一致，保留稳定列与数据单位；区分 `0`、空值、未采集、不适用、失败和未知 |
| 键盘与鼠标 | `Tab` / `Shift+Tab`、方向键、`Enter`、`Esc` 的效果应可发现；焦点、选中、悬停、禁用是不同状态；终端/SSH 宿主保留的快捷键不可擅自覆盖 |

## TUI / HTML 特殊约束

- **终端**：检查中文双宽字符、字体覆盖、80×24 / 120×30 / 150×45、深浅主题、颜色降级、SSH 与不同客户端；无图形字形时状态仍应由文字理解。不要为图标美观破坏对齐或窄屏操作路径。
- **正式测量窗口**：不得因 UI 改造新增采集期刷新、轮询、动画、统计、文件扫描或下载；具体不变量以 [TUI 架构](tui-architecture.md) 和 [测量协议](../profiling/measurement.md) 为准。
- **HTML 报告**：优先统一信息组织、筛选状态、颜色含义、通知和可读性；保留既有独立 HTML 生成、离线打开与结果来源校验，不能为了视觉一致性改变统计计算、CSV 或字段语义。
- **实现纪律**：优先复用项目现有 Textual/Rich 控件、主题和测试夹具；确有偏离 Carbon 的终端适配时，在相关代码审查或设计说明中说明原因，避免增添平行设计系统。

## 导航图案预览

在源码目录运行以下命令，按侧栏六类导航列出筛选后的适用候选：

```bash
uv run --locked python scripts/preview_nav_icons.py --list
uv run --locked python scripts/preview_nav_icons.py --list --nerd-font
uv run --locked python scripts/preview_nav_icons.py --page settings
```

普通清单共 119 项（不同页面可以使用相同图案），包含 Unicode、Emoji、ASCII 和无图案；
`--nerd-font` 额外显示 24 项 Codicons。每组当前图案排在第一项，并显示名称、码位、
类别与实际标题。`--page` 接受 `run`、`monitor`、`results`、`backfill`、`images`、`settings`；
`--page` 或 `--nerd-font` 单独使用也会进入列表模式。80 列及以上使用表格，窄终端使用逐项换行。
清单按页面去重，编号与交互预览独立。

不加参数仍进入“实验配置”的交互预览，保留原有 36 个选项和编号，
同时查看当前项与候选的展开、折叠效果。
方向键或点击切换，`L` 切换中英文标题，`T` 切换现有主题，`R` 回到当前图案；
点击“选定并退出”或按 `Ctrl+S` 仅将选择输出到终端，`Esc` 取消。

工具参考 Textual 8.2.8 的 [OptionList 示例](https://github.com/Textualize/textual/blob/v8.2.8/docs/examples/widgets/option_list_options.py)，
复用现有 MIT 依赖、主题与导航符号，无新增依赖、设置写入或正式采集逻辑。
列表复用 [Rich Table](https://github.com/Textualize/rich/blob/main/rich/table.py)（MIT），
Nerd Font 名称和码位核对上游 [glyphnames.json](https://github.com/ryanoasis/nerd-fonts/blob/master/glyphnames.json)。
这两个项目均有持续维护；只借鉴列表展示和码位命名，不复制渲染实现或引入新依赖。
所选 [Codicons](https://github.com/ryanoasis/nerd-fonts/blob/master/src/glyphs/README.md) 字体资产采用 CC BY 4.0，
本工具未引入字体资产，也不下载或安装字体。Nerd Font 需要显示终端配置相应字体（SSH 时为客户端）；
Unicode、Emoji 与文本样式变体的支持、字宽仍需以实际终端显示为准，离线输出不能确认客户端字形。

## 设计审查与验收

对新增页面、视觉重排和交互变化，记录「**主要动作 → 必须可见的信息 → 状态/失败反馈 → 鼠标与键盘路径 → 所需终端验收**」。至少核查：

1. 语义是否与现有采集、下载、结果和错误状态一致；状态能否在无颜色情况下被理解。
2. 颜色是否复用主题 Token；是否误把警告渲染为成功、把未知渲染为零值；深浅主题与颜色降级有无可读性问题。
3. 长文案、中英文、窄窗和多终端下主要按钮及焦点能否到达；悬停入口是否有等价键盘操作。
4. 是否复用现有组件，避免引入第二套状态映射、重复轮询或正式测量期开销。
5. 按改动范围选择 [设计审查](../../.agents/skills/acprof-textual-regression/references/design-review.md)、[终端验证](terminal-validation.md) 和 [PNG 视觉回归](png-visual-validation.md)；快照与 headless 结果不能代替真实终端验收。

## 外部设计依据

- [IBM Carbon — About Carbon](https://carbondesignsystem.com/getting-started/about-carbon)：设计系统的范围与定位。
- [IBM Carbon — Status indicators](https://carbondesignsystem.com/building-blocks/core/patterns/status-indicators)：图标、形状、文本、严重程度与状态聚合。
- [IBM Carbon — Notifications](https://carbondesignsystem.com/building-blocks/core/patterns/notifications)：信息、成功、警告、错误及通知打扰程度。
- [IBM Carbon — Color](https://carbondesignsystem.com/building-blocks/foundations/color/overview)：语义 Token 与主题。
- [W3C WCAG 2.2](https://www.w3.org/TR/WCAG22/)：可访问性基准（特别是 1.4.1、1.4.3、1.4.11）。

本规范于 2026-11-07 对照官方文档制定。Carbon 上游规范更新时，应评估与 AC-Prof 终端约束的兼容性后再调整；**不自动继承上游视觉变更**。
