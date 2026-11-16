---
name: acprof-html-report-regression
description: 修改或验证 AC-Prof 离线 HTML 报告、浏览器导入、筛选、图表交互和导出；Textual TUI 或真实模型采集不触发。
metadata:
  reference: anthropics/skills
  revision: 1780c5e5a552fd4d1c9d6a7b0cab26b0c2bef064
---

# AC-Prof HTML 报告回归

先按[开发质量检查](../../../docs/development/quality.md)定位浏览器环境与运行命令，并读取受影响的[报告用例](../../../tests/analysis/reporting/test_report_browser.py)或[sweep 用例](../../../tests/analysis/reporting/test_load_sweep_browser.py)。沿用现有合成数据、Fixture 和 Playwright，不新增第二套浏览器测试基础设施。

1. 明确报告类型、输入身份、当前筛选和预期交互。区分 Python 生成、浏览器重算、展示和导出各自的责任。
2. 优先用 Fixture 或已有 HTML 复现。先检查渲染后的 DOM、控制台和可见状态，再选择稳定的 role、label 或项目现有 selector。
3. 页面保持 `offline=True`，禁止 HTTP(S) 请求和外部字体/CDN 依赖。先确认锁定环境与 Chromium 是否已准备；环境准备的网络访问不属于页面离线证据。
4. 等待具体元素、报告 ready 条件或目标状态，不能仅用 `networkidle` 或固定 timeout 证明逻辑完成。按实际行为使用 Playwright 断言及现有等待工具。
5. 仅覆盖本次改动涉及的导入、筛选、排序、跨图选择、比较隔离、窄屏、键盘和导出场景。涉及结果时对照 Python 的字段、单位、unknown 和来源语义，不只看截图。
6. 保存所需 HTML、截图、DOM、console/page errors、trace 和 evidence JSON；核对 `ACPROF_BROWSER_TESTS=1` 与 skip，未启用浏览器用例不能报告浏览器验证通过。

布局设计需要补充信息时，可选用全局 `ui-ux-pro-max` 的 chart、ux、color 查询，仍保留单文件离线约束；不因设计建议切换前端框架。终端 UI 使用[Textual 回归](../acprof-textual-regression/SKILL.md)。

此流程遵守[验证范围](../../../docs/development/validation-scope.md)，避开正式测量窗口。Headless Chromium 通过不等于 Windows/WSL/macOS 用户浏览器验收，不启动真实模型来制造报告数据。

参考 [Anthropic webapp-testing](https://github.com/anthropics/skills/blob/1780c5e5a552fd4d1c9d6a7b0cab26b0c2bef064/skills/webapp-testing/SKILL.md) 的先观察再操作模式；未复制脚本，使用项目现有 runner，并以具体状态等待替换上游的通用 `networkidle` 建议。
