# 测试目录

测试按**被验证的功能域**组织，而不是按 pytest marker 或硬件可用性分文件夹。
pytest 从 `tests/` 递归收集；保持 `test_*.py` 文件名与现有 markers，
不依赖原来的平铺路径。

```text
tests/
├── analysis/        # 统计、绘图、比较、报告
├── cli/             # 安装、配置与命令行
├── execution/       # 流程编排、计划、进程与验证
├── measurement/     # 硬件、采样窗口与 profiling
├── models/          # 模型发现、契约、缓存与下载
├── network/         # 数据包、网络与下载预检
├── quality/         # 架构、CI、覆盖率与发布门禁
├── results/         # 结果存储、分层发布与恢复
├── runtime/         # 容器、镜像、依赖与平台环境
├── tui/             # 目录选择、日志、结果、控件与工作流
├── workloads/       # audio、multimodal、nlp、onnx、structured、vision
├── support/         # 可复用测试构造器与辅助模块
├── fixtures/        # 静态 JSON、二进制样本等测试数据
├── visual/          # Textual SVG 快照、对应测试及局部 conftest
├── manual/          # 非 pytest 自动收集的手动 smoke 脚本
└── conftest.py      # 全局 pytest 插件注册
```

`tests/__init__.py` 及领域目录的 `__init__.py` 使测试模块能够以
`tests.<domain>.<subdomain>.test_*` 方式被明确导入。
共享构造器统一从 `tests.support` 导入；只有目录特有的 pytest fixture
才放在对应目录的 `conftest.py`。不要为旧的平铺模块路径添加导入兼容层。

```bash
uv run pytest --collect-only -q
uv run pytest tests/models -q
uv run pytest tests/tui/widgets/test_tui_input.py -q
uv run pytest tests/visual -q
```

运行策略、标记及 CI 分片规则见 [开发测试指南](../docs/development/test-commands.md)。
