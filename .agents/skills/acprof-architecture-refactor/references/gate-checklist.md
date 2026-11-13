# 架构治理门禁与检查清单

命令均在当前任务 worktree 根目录通过独立 `.venv` 执行；仅做本地静态与离线回归，不启动真实模型采集或长矩阵。

## 1. 模块规模与复杂度线索

```bash
uv run python scripts/check_module_sizes.py --top 0
uv run ruff check --select C901,BLE001 acprof/<target_module>.py
```

- 超过 800 行的非 `data-review` 模块优先按职责、长函数分支与可测试性评估拆分，不把 500 行软目标当成硬性机械拆分线。
- 新纳入复杂度或异常收窄约束的模块，同步评估接入 [.pre-commit-config.yaml](../../../../.pre-commit-config.yaml) 的 `core-complexity`、`shared-boundaries` 或对应局部 hook，并确认 [test_architecture.py](../../../../tests/quality/test_architecture.py) 通过。

## 2. 导入方向与跨模块 Private API

```bash
uv run python scripts/check_import_boundaries.py
uv run pytest -q tests/quality/test_import_boundaries.py
uv run python scripts/check_private_api.py
uv run pytest -q tests/quality/test_private_api_guard.py tests/quality/test_architecture.py
```

- 依赖契约唯一来源为 [pyproject.toml](../../../../pyproject.toml) 的 `[tool.importlinter]`；优先修复反向或传递耦合，不添加全局 `ignore_imports`。
- 消除跨模块私有符号调用后，使用 `uv run python scripts/check_private_api.py --prune` 清理失效 baseline 条目，不引入未登记的新依赖。

## 3. 渐进类型收紧

```bash
uv run mypy
```

- 保持全包 `acprof` 检查覆盖；抽离出的纯类型协议或已完整注解模块可评估加入 [pyproject.toml](../../../../pyproject.toml) 的 `[[tool.mypy.overrides]]` 严格列表。
- 对 `TypedDict`、`TypeGuard` 与 JSON 边界先执行真实字段和类型校验，不以 `ignore_errors`、`cast` 或 `Any` 绕过检查。

## 4. 关键分支覆盖与文档同步

- 涉及 [check_critical_coverage.py](../../../../scripts/check_critical_coverage.py) 中 `BRANCH_FLOORS` 的模块搬迁或拆分时，核对新旧模块的分支覆盖与文件路径清单，确保关键路径覆盖不回落。
- 涉及目录职责、调用边界或命令行为变化时，更新对应专题文档与 [roadmap.md](../../../../docs/roadmap.md)，并执行：

```bash
uv run python scripts/check_docs.py
git diff --check
```
