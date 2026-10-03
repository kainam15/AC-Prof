# CI 失败模式与复现入口

命令从仓库根目录执行。下列参数是入口示例，测试文件、解释器和分片编号按失败 job 选择；不固定复用历史测试数量。

```bash
.venv/bin/python scripts/run_tests.py --pattern 'test_metric_registry.py' --report internal-testing/ci-triage/targeted.json
.venv/bin/python scripts/render_metric_reference.py --check
git diff --check
```

为本次任务选择独立报告目录，创建目录后再写报告。`--require-no-skips` 仅用于该环境本应执行全部用例的集合；真正缺少的运行条件须明确记录。分片与汇总命令以[当前测试指南](../../../../docs/Testing.md#ci-与环境测试)及脚本 `--help` 为准。

| 失败证据 | 核验方法 | 修复约束 |
| --- | --- | --- |
| Hub 依赖返回主模型的 SHA/文件 | 查实际 lookup 位置和嵌套 patch，复现单个依赖用例 | mock 按 repo ID 路由，保留 revision/文件过滤断言 |
| RAPL fixture 被识别为 unavailable | 对照读取器要求的 sysfs 内容与 fixture | 补齐接口事实，保留循环链接/域过滤检查 |
| 新指标导致历史字段或文档检查失败 | 对照字段声明、顺序契约及生成器输出 | 更新新增字段约定和生成文档，不删除历史顺序保护 |
| SVG 基线不匹配 | 查看实际差异与该提交的用户可见变化 | 审阅后只更新相关基线；再执行不带更新选项的比较 |
| TUI 测试偶发截图旧页面 | 检查排队事件、布局 region 和生命周期 | 等待可观察就绪条件，不持续加长固定 sleep |
| 本地测试争用测量锁 | 核对 runner 是否隔离测试锁、实际 test ID 与临时目录 | 复用项目 runner，不绕过生产锁；pytest 的 autouse fixture 对所有入口隔离锁 |
| job 通过但 artifact 缺失或只跑部分用例 | 核对报告中的 SHA、版本、分片和每项测试记录 | 缺片、重复归属、零测试或摘要不符不算完整成功 |

这些是历史故障类别，当前失败仍需日志和复现支持。只看名字相似不能归因。

## 远端证据

优先复用可访问 run/jobs/logs/artifacts 的 GitHub 工具；原始 run 正在执行时只描述当前状态。注意 job API 的分页和 retry attempt，避免混合不同尝试的日志。

已有 `gh` 时可用：

```bash
gh run view RUN_ID --json headSha,status,conclusion,jobs
gh run view RUN_ID --log-failed
```

无法取得日志时保留 run URL 和缺口，不按截图中的红色标记推断根因。报告生成器退出 0、单个 job 成功或本地复现通过，均不等于整次 Actions 通过。
