# 本地验证产物约束

- 本目录存放临时验证脚本、测试日志和终端截图，按任务隔离输出；不将这里的结果标作正式 profiling 数据。
- 保存证据时注明验证命令、终端尺寸和运行范围；历史截图或日志不能证明当前实现已通过。
- 可复用验证进入 `tests/`，长期结论进入对应 `docs/` 专题；临时计划不提升为 Agent 常驻指令。
- 新运行目录统一通过 `scripts/internal_testing.py new` 创建并写入 manifest；测试结束后标记实际状态。
- 顶层只允许 `AGENTS.md`、`runs/`、`evidence/`、`shared/`、`cache/`：固定路径输出放 `shared/`，可重建缓存放 `cache/`，历史证据归入 `evidence/legacy/`，不得新建其他顶层条目。
- 旧证据、失败或中断数据、正式 `results/` 以及含科研原始证据的任务不得因调整布局而删除；清理默认只预览。
- 使用 `python scripts/internal_testing.py inventory --check` 检查顶层布局；CI 同样执行检查。

目录命名、生命周期和清理边界见[本地测试产物规范](../docs/development/internal-testing-layout.md)。
