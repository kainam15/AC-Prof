# internal-testing 目录与生命周期规范

[← 返回测试专题](testing.md)

`internal-testing/` 是 **被 Git 忽略的开发验证产物目录**，不属于 AC-Prof 正式采集的 `results/`。
正式实验的 CSV、PCAP、能耗窗口、断点恢复和 provenance 均由正式结果协议管理，不能随测试日志一起回收。

## 目录职责

```text
internal-testing/
├── AGENTS.md                       # 唯一被 Git 跟踪的本地目录约束
├── runs/
│   └── YYYY-MM/
│       └── YYYYMMDDTHHMMSSZ--task-name--a1b2c3d4/
│           ├── manifest.json
│           ├── logs/
│           ├── reports/
│           ├── screenshots/
│           └── artifacts/
├── evidence/                       # 人工确认的长期验证证据；不自动删除
├── shared/                         # 将来确需固定路径的工具输出；不自动删除
└── cache/                          # 可重建环境/下载物；目前不自动删除
```

| 类别 | 保存什么 | 清理边界 |
| --- | --- | --- |
| `runs/` | 一次独立执行的日志、报告、截图、临时产物 | 仅受管 manifest + 合格完成状态可成为候选 |
| `evidence/` | 人工保留的验收、故障复现或研究判据 | 绝不由 `prune` 自动删除 |
| `shared/` | 确定需要稳定地址的开发工具输出 | 由具体工具维护；不会自动清理 |
| `cache/` | 可重建的虚拟环境、构建物与安全数据库 | 需要另行证明可重建和无正在使用者 |
| `evidence/legacy/` | 早期单次验证、失败日志、旧归档、模型测试和历史数据 | 保留原始证据与迁移映射，绝不自动删除 |

**顶层严格只有上述四个目录及 `AGENTS.md`**，不得为历史任务创建同级目录或兼容软链接。固定输出（例如 `shared/tui-png/`、`shared/distribution/`、`shared/ci-triage/`、`shared/coverage/`）放入 `shared/`；虚拟环境和可重建下载物放入 `cache/`。原来的根目录输出已归档至 `evidence/legacy/`；涉及引用的调用方、CI 和文档必须同时改成新路径。

## 统一命名与运行身份

新任务由 `scripts/internal_testing.py` 分配 `runs/YYYY-MM/<UTC时间>--<任务>--<8位随机标识>/`。
示例：`runs/2026-10/20261030T110425Z--local-tests-fast--a1b2c3d4/`。

- UTC 时间使用 `YYYYMMDDTHHMMSSZ`，月份只用于快速查找。
- 任务名限定小写 `kebab-case`、不超过 48 字符，不能包含路径分隔符。
- 随机后缀是 **运行唯一 ID**，不是 Git commit；允许同一秒并行分配不同目录，不覆盖已有内容。
- Git commit 与 `git_dirty`（标识运行时源码是否含未提交改动）、开始与结束时间、运行状态、保留类别记录在 `manifest.json`，不依赖目录名表示状态。
- 目录中的 `logs/`、`reports/`、`screenshots/`、`artifacts/` 用于分层保存一次执行的内容。
- manifest 仅记录安全且经过确认的命令；禁止自动写入原始环境、Token、密码或含凭据的完整命令行。

状态转换：`running → success / failed / interrupted`；失败或中断解决后可
`failed / interrupted → resolved`。`--keep` 与根目录下的 `.keep` 标记可保护单次运行。
只有已结束的 `success` 或 `resolved` 且 `retention=temporary` 可以进入清理候选。

## 创建、使用与查看

从 Git 工作树根目录运行：

```bash
# 首次建立四个管理目录（现有文件不会移动）
python scripts/internal_testing.py init

# 新建任务；输出完整任务目录路径
RUN=$(python scripts/internal_testing.py new --task tui-layout)
printf '%s\n' 'validation detail' > "$RUN/logs/check.log"

# 执行验证后，根据真实结果标记状态；失败时务必保留原始证据
python scripts/internal_testing.py finish "$RUN" --status success
# 失败：--status failed；中断：--status interrupted；修复后：--status resolved
# 需要长期保留：new --keep，或 finish --keep

# 默认只读、只列结构；--json 供脚本读取
python scripts/internal_testing.py inventory
python scripts/internal_testing.py inventory --json
# 严格检查顶层是否仅有规定的五项；CI 同样执行
python scripts/internal_testing.py inventory --check

# 只预览候选，完全不删除；明确调用才允许删除
python scripts/internal_testing.py prune --dry-run
python scripts/internal_testing.py prune --apply --confirm DELETE-ELIGIBLE-RUNS
```

`scripts/run_local_tests.py` 默认使用这套目录与 manifest，成功、失败、中断均会更新状态。
显式调用 `scripts/run_local_tests.py --output` 只能在 `internal-testing/shared/` 下建立新的固定输出目录；默认不传该参数时创建受管 `runs/`。独立工具的新运行必须先使用 `new`，结束后调用 `finish`；不能把已有目录冒充新运行，也不能向顶层直接写文件。

## 回收策略：默认不删除

清理候选必须 **同时** 满足：目录在受管的 `runs/YYYY-MM/` 路径、非符号链接、
名称和 manifest 身份完全一致、存在有效完成时间、保留类别为 `temporary`、
没有 `.keep`、没有需要保护的科研数据类型。

| 状态 | 最早候选时间 |
| --- | --- |
| `success` | 结束 14 天后 |
| `resolved` | 标记解决 30 天后 |
| `running` | 永不自动清理 |
| `failed` / `interrupted` | 未标记解决前永不自动清理 |
| `retention=keep` | 永不自动清理 |

即使是成功运行，只要出现 `raw/`、`.acprof/`、恢复检查点或 `.csv`、`.pcap`、
`.jsonl`、`.parquet`、`.npy` 等数据，工具也会保守跳过。它不会解析文件是否真能复现；
人工确认更为优先。清理不读取或删除 `results/`、`evidence/`、`shared/`、`cache/`、
旧顶层目录、活动任务以及未完成或损坏的 manifest。

`prune` 的默认行为与 `--dry-run` 相同。执行 `--apply` 需要显式确认短语；
不应在 GitHub Actions、cron、定时任务或正式测量期间默认调用实际删除。
目录被 Git 忽略只意味着不提交，不代表获得删除许可。

## 老目录迁移的前提

历史目录迁移记录位于 `internal-testing/evidence/maintenance-20261031/`，保留每个原路径、目标路径、引用及校验记录。迁移固定路径时，必须同步修正调用方、CI artifact 路径、文档与测试并验证；历史证据一律进入 `evidence/`，不能伪造成受管 `runs/`。不得仅凭时间、`archived` 名称或 Git 忽略状态删除。

## 设计参考

- [pytest 的临时目录和保留策略](https://github.com/pytest-dev/pytest/blob/main/doc/en/how-to/tmp_path.rst)：
  借鉴每次运行隔离、明确保留窗口；避免 `--basetemp` 直接清空共享目录的风险。
- [ASV benchmark 结果与 Git commit 身份](https://github.com/airspeed-velocity/asv/blob/main/docs/source/using.rst)：
  借鉴结果与运行上下文关联，但不把短 hash 当作完整实验身份。
- 标准库 `tempfile` + `os.replace` 用于 manifest 原子写入；没有引入新依赖或测量期逻辑。
