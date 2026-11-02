- [进度表](roadmap.md)
- [开发目标](GOAL.md)
- [文档导航](README.md)

# 文档维护约束

- 新增文件时，需要整理到对应的主题文件夹（缺少对应主题的文件夹，则新增主题文件夹）；
- 新增、移动或删除文档后需要修复入站链接，使导航指向对应专题；
- 测试与维护报告应该放到 `internal-testing/` ；
- 长期技术事实按专题维护：测量协议见 `profiling/`，指标与数据见 `results/`，开发环境、Git 和验证策略见 `development/`；避免在根目录 `AGENTS.md` 重复操作细则；
- 临时计划、开发日志及失败证据遵循 `internal-testing/AGENTS.md`，不要提升为常驻 Agent 规则；
- `README.md` 与 `docs/i18n/README_zh-CN.md` 中对应的内容必须保持同步，包括命令、链接、功能说明和排版。
