# 测试与验证

唯一测试入口是 `python -m pytest`；CI 的 `scripts/run_tests.py` 仅转发 pytest 参数并启用同一插件。

本次迁移的基线、集合映射与验证限制见 [2026-10-03 pytest 迁移验收记录](../reviews/2026-10-03-pytest-migration.md)。

选择本次改动能改变的行为和失败路径。下列命令均从仓库根目录执行，使用已有 `.venv`。
测试数量、设备余量和镜像可用性由本次执行确认，不把历史通过记录作为当前验证结果。

## 专题索引

- [验证策略与 Python 修改工作流](validation-scope.md)：验证范围、工作流与文件规模。
- [开发质量检查](quality.md)：开发质量、提交信息和 Ruff。
- [类型与模块边界检查](static-checks.md)：mypy、private API、导入方向和控制器回归。
- [开发辅助工具与 MCP](devtools.md)：本地开发辅助工具、PyCharm MCP 边界。
- [自动化测试入口](test-commands.md)：pytest 运行命令与测试筛选。
- [CI 与环境回归](ci.md)：CI、coverage、恢复和无 Torch 环境回归。
- [终端与采集环境验证](terminal-validation.md)：TUI 终端证据和真实测量隔离。
- [测试依据与文档检查](documentation-checks.md)：方法参考与文档、Skill 校验。
