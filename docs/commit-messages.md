# Git Commit Message 规范

AC-Prof 使用基于 [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 的 Git commit message 规范。

目标是让提交历史同时满足：

- 人类能够快速理解变更；
- Agent 能够可靠判断提交意图；
- GitHub history 和 blame 保持清晰；
- 后续可以用于生成 changelog、release notes 和版本信息；
- 一个 commit 对应一个清晰、可审查的逻辑变更。

## 格式

基本格式：

```text
<type>(<scope>): <summary>

[optional body]

[optional footer]
```

`scope`、body 和 footer 都是可选的。

Breaking change 可以写成：

```text
<type>(<scope>)!: <summary>
```

例如：

```text
feat(provenance): record installed package version in results

- Add the AC-Prof package version to preparation-time platform metadata.
- Preserve unknown versions when reading historical results.
- Cover missing Git and hardware without changing measurement semantics.
```

简单提交不需要为了格式强行添加正文：

```text
docs: fix installation example
```

## Type

优先使用以下类型。

| Type | 用途 |
| --- | --- |
| `feat` | 新增用户可见功能或能力 |
| `fix` | 修复错误行为或 bug |
| `refactor` | 重构实现，不应改变预期行为 |
| `perf` | 性能或资源开销优化 |
| `test` | 新增、迁移或修改测试 |
| `docs` | 仅文档变化 |
| `build` | 打包、构建系统、依赖构建逻辑 |
| `ci` | CI / GitHub Actions |
| `style` | 仅格式、排版，不改变行为 |
| `chore` | 无更合适类型的维护性工作 |
| `revert` | 回滚已有提交 |

优先选择能够准确表达主要意图的具体 type。

不要因为不知道该选什么就默认使用 `chore`。

## Scope

Scope 表示主要受影响的子系统，而不是文件名。

AC-Prof 常用 scope 包括：

```text
cli
tui
probe
model
runtime
workload
collector
packet
energy
provenance
platform
schema
analysis
plotting
build
ci
docs
```

例如：

```text
fix(probe): avoid downloading model weights during inspection
feat(tui): add automatic environment validation
refactor(runtime): separate dependency resolution from execution
perf(collector): reduce idle sampling overhead
```

Scope 不是封闭枚举。

如果某次修改跨越整个仓库，或者没有一个 scope 能准确描述，可以省略：

```text
test: migrate the test suite to pytest
docs: reorganize development documentation
```

不要为了满足格式使用过宽或没有信息量的 scope，例如 `core`、`misc`，除非它们在项目中确实代表一个明确子系统。

## Summary

Summary 是整个 commit 最重要的一行。

要求：

- 使用英文；
- 使用祈使语气 / 动词原形；
- `<type>` 和 `<scope>` 使用小写；
- summary 首词通常使用小写；
- 不以句号结尾；
- 描述“这个 commit 做了什么”，而不是“我做了什么”；
- 保持简洁，建议不超过约 72 个字符；
- 避免实现细节堆积在标题中。

推荐：

```text
fix(model): preserve unknown revisions in historical results
feat(provenance): record installed package version in results
refactor(cli): remove legacy root-level launchers
test(runtime): cover missing dependency metadata
docs: document the model inspection lifecycle
```

避免：

```text
fix bug
update files
misc changes
some fixes
fix stuff
updated probe
WIP
```

也不要使用过去式：

```text
feat(tui): added disk usage dialog
```

应写为：

```text
feat(tui): add disk usage dialog
```

## Body

Body 可选。

只有当标题不足以解释变更时才写正文。

适合增加 body 的情况包括：

- 一个 commit 包含多个彼此相关的行为变化；
- 需要解释为什么这样设计；
- 存在不明显的兼容性处理；
- 修复较隐蔽的失败路径；
- 改变模块边界；
- 需要说明哪些语义明确保持不变；
- reviewer 仅看标题无法理解风险。

正文与标题之间留一个空行。

可以使用自然段：

```text
refactor(probe): separate static inspection from runtime validation

Static inspection must remain usable before a workload is selected and
must not require downloading large model weights.

Runtime-dependent validation is deferred until the execution contract
is available.
```

也可以使用 bullet：

```text
feat(provenance): record installed package version in results

- Add the AC-Prof package version to preparation-time platform metadata.
- Preserve unknown versions when reading historical results.
- Cover missing Git and hardware without changing measurement semantics.
```

两种形式都允许。

优先说明：

1. 为什么需要这次改动；
2. 哪些重要行为发生变化；
3. 哪些容易被误解的行为明确没有变化。

不要把 body 写成单纯的文件清单：

```text
- Changed acprof/foo.py.
- Changed tests/test_foo.py.
- Updated docs.
```

Git diff 已经能够表达这些信息。

也不要为了显得“完整”而机械凑三条 bullet。

## Footer

只有存在额外元数据时才使用 footer。

例如关联 issue：

```text
fix(runtime): reject unresolved execution profiles

Refs: #123
```

或者：

```text
fix(runtime): reject unresolved execution profiles

Closes #123
```

Footer 与正文之间留一个空行。

## Breaking Changes

如果一个 commit 有意改变已经建立的外部行为，可以使用 `!`：

```text
refactor(cli)!: remove legacy root-level launchers
```

需要进一步解释时：

```text
refactor(cli)!: remove legacy root-level launchers

Use the installed `acprof` command as the single public entry point.

BREAKING CHANGE: direct execution of the legacy root-level launchers is no longer supported.
```

在 AC-Prof 当前开发阶段，不要因为普通内部重构就使用 `!`。

优先在以下变化中考虑 breaking change：

- 已建立的公开 CLI 行为；
- 持久化配置格式；
- result schema 或 artifact protocol；
- 已发布给用户依赖的接口；
- 明确承诺长期稳定的测量语义。

单纯移动内部函数、重命名私有模块或删除未发布的兼容 shim，通常不需要标记 breaking change。

## 一个 Commit 应包含什么

一个 commit 应表达一个完整的逻辑变化，而不是固定数量的文件。

以下内容通常可以放在同一个 commit：

```text
implementation
+ directly related tests
+ directly related documentation
```

它们共同组成一个功能，不需要机械拆成多个 commit。

应拆分的情况：

```text
runtime bug fix
+
unrelated TUI cleanup
+
README wording cleanup
```

这些属于不同逻辑主题，应分别提交。

如果一个 diff 很自然地对应两个不同的 type 或两个完全不同的 scope，通常也是应该拆分 commit 的信号。

## Tests 与 Commit

提交前应验证与该逻辑变化直接相关的行为。

不要求为了每个小 commit 都执行完整测试矩阵。

例如：

```text
fix(tui): correct validation status rendering
```

通常只需要相关 TUI / unit tests。

而修改：

```text
refactor(schema): change result manifest resolution
```

则应覆盖相关 schema、resume、historical result 和 integration 行为。

昂贵的 Native Linux、GPU、Docker 或正式 profiling 验证按照项目测试与测量规范执行，不能因为 commit message 规范而自动触发。

## Agent 创建 Commit 时

创建 commit 前：

1. 检查实际 staged diff，而不是仅根据任务描述猜 commit message。
2. 确认 staged changes 属于同一个逻辑主题。
3. 将无关改动留给其他 commit。
4. 根据实际行为选择 type 和 scope。
5. 只有在标题不足以解释修改时才添加 body。
6. 不在 commit message 中声称尚未验证或实际没有发生的行为。
7. Commit message 应描述最终 diff，而不是开发过程中尝试过的方案。

例如，开发过程中可能先增加兼容层、随后又删除。最终 diff 中没有兼容层时，就不应把“add compatibility layer”写进 commit message。

## 示例

### 简单修复

```text
fix(tui): keep model validation status visible
```

### 带正文的新功能

```text
feat(provenance): record installed package version in results

- Add the AC-Prof package version to preparation-time platform metadata.
- Preserve unknown versions when reading historical results.
- Cover missing Git and hardware without changing measurement semantics.
```

### 架构重构

```text
refactor(probe): separate static inspection from runtime validation

Static inspection should resolve metadata and source requirements without
requiring model weights or an executable workload.

Defer runtime-dependent checks until the execution contract is available.
```

### 性能优化

```text
perf(collector): reduce sampling overhead outside measurement windows
```

### 测试

```text
test(runtime): cover unresolved adapter dependencies
```

### 文档

```text
docs: add commit message guidelines
```

### 构建

```text
build: move the Hatch hook into the packaging directory
```

### Breaking change

```text
refactor(cli)!: remove legacy root-level launchers

Use `acprof <command>` as the single supported command-line entry point.

BREAKING CHANGE: direct execution of the removed wrapper scripts is no longer supported.
```

## Quick Reference

最常见的形式：

```text
<type>(<scope>): <imperative summary>
```

复杂提交：

```text
<type>(<scope>): <imperative summary>

- <important behavioral change>
- <important behavioral change>
- <important invariant or compatibility detail>
```

Breaking change：

```text
<type>(<scope>)!: <imperative summary>

<optional explanation>

BREAKING CHANGE: <description>
```

正文、scope 和 footer 都是按需使用，不要为了满足模板添加无信息内容。
