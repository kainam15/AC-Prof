---
name: acprof-hub-research
description: 为 AC-Prof 调研 Hugging Face 或 ModelScope 模型仓库、任务、元数据、制品布局和固定 revision；保持静态证据，不启动权重下载、远程模型代码或真实推理。
metadata:
  reference: huggingface/skills
  revision: 25910b3d0ba31a2ba102f901cdc9e008c278a69c
---

# AC-Prof Hub 静态调研

先确认平台和完整 model ID，核对[模型文件选择](../../../docs/models/adaptation.md#模型文件选择规则)、[Model Store](../../../docs/models/model-store.md)与[扩展声明](../../../docs/models/extensions.md)。昵称或不完整 ID 无法唯一识别时保留歧义，不猜测模型。

1. 优先读取已有 Model Store 元数据与项目静态 inspect 能力。需要远端资料时使用已连接的 Hub 工具、官方 API 或固定 revision 的小型元数据文件；遵守本次任务的离线约束。
2. 记录来源平台、仓库 ID、revision、文件路径和证据指针。区分作者 model card、config 声明、实际文件布局与推断；不只凭任务标签宣布兼容。
3. 沿用项目 Hub SDK、认证和代理策略。使用 `hf` 前先确认本机版本与相关子命令帮助；上游生成的 CLI 手册可能高于项目依赖版本，不为适配示例自动升级 Hub SDK、安装 CLI 或改写锁。
4. 只获取完成判断所需的配置、索引、文件清单和文档，不批量拉取权重，不导入远程代码或触发 checkpoint 加载。查询与记录不暴露 token。
5. 对齐 task、architecture/model type、pipeline/components、制品、adapter、运行时依赖和所需输入。未知字段明确保留 unknown，证据冲突单独列出。
6. 输出已知事实、来源、未确认项和下一步。静态 candidate、接口验证、真实推理和正式采集分别表述，不推导未经实测的通过率或性能。

需要实现适配时转入[模型适配](../acprof-model-adaptation/SKILL.md)；模型集审计使用[兼容性审计](../acprof-compatibility-audit/SKILL.md)。它们沿用已有用户授权，无需逐步重新确认；真实实验仍按[验证边界](../../../docs/development/validation-scope.md)执行。

参考 [Hugging Face skills](https://github.com/huggingface/skills/tree/25910b3d0ba31a2ba102f901cdc9e008c278a69c) 的 Hub 发现与 CLI 自描述思路。未复制按新版 CLI 生成的命令表，也未安装训练、发布或云任务流程；ModelScope 仍以对应平台原始证据为准。
