# AC-Prof 文档导航

文档按主题目录维护。每个概念、命令或协议只在一个权威位置完整定义；其他页面只放导航和链接。

长期建设与开发目标看[ Roadmap](roadmap.md)。

## 文档导航

| 分类                               |                                                                                                 权威文档                                                                                                 |
| :--------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **roadmap.md** · 路线图     |                                                                                     [长期建设目标与里程碑](roadmap.md)                                                                                     |
| **getting_started/** · 入门 |                                                                             [安装与主机准备](getting_started/installation.md)                                                                             |
| **usage/** · 使用           |                                          [实验指南](usage/experiments.md) · [TUI](usage/tui.md) · [CLI 参考](usage/cli.md) · [运行排障](usage/troubleshooting.md)                                          |
| **models/** · 模型          |                                                                                    [运行兼容与适配](models/runtime.md)                                                                                    |
| **profiling/** · 采集       |                                                  [采集协议](profiling/protocol.md) · [能耗测量](profiling/energy.md) · [Profiler](profiling/profilers.md)                                                  |
| **results/** · 结果         |                                                            [指标与分析](results/metrics.md) · [自动生成字段登记表](results/metric_reference.md)                                                            |
| **development/** · 开发     | [架构](development/architecture.md) · [测试](development/testing.md) · [测试产物目录](development/internal-testing-layout.md) · [发行包](development/distribution.md) · [历史来源](development/history.md) |
| **platforms/** · 平台       |                                                             [Linux Runtime 与云环境](platforms/linux-runtimes.md) · [WSL2](platforms/wsl2.md)                                                             |
| **i18n/** · 翻译            |                                                                                  [简体中文 README](i18n/README_zh-CN.md)                                                                                  |

## 文档、规则与流程的分工

[安装包、standalone 与发布](development/distribution.md)维护 wheel 资源、工作目录、PyPI Trusted Publishing、GitHub Release 和 GHCR 发布约定。

## Skill 与 Agent 文档编写

新建或修改 Skill、`AGENTS.md` 时，参考[OpenAI：重新思考 GPT-6 Astra 的技能与提示词](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。
