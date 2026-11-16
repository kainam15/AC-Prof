---
name: acprof-github-actions-workflow
description: 新增、修改或审查 AC-Prof GitHub Actions 工作流、矩阵、权限和发布门禁；已有失败 run 的日志定位优先使用 acprof-ci-triage。
metadata:
  reference: wshobson/agents
  revision: 9c61ab243def98f72da12abd41cfcf5bcd91af9d
---

# AC-Prof GitHub Actions 维护

先读[CI 与发布验收](../../../docs/development/ci.md)及受影响的[工作流](../../../.github/workflows)，核对事件、分支/标签、job 依赖、环境、权限与实际运行命令。排查某次失败先转入[CI triage](../acprof-ci-triage/SKILL.md)。

1. 复用现有 workflow、composite action、uv 锁、依赖组和 evidence runner；模板仅提供结构参考，不替换项目 Python/uv 版本或安装方式。
2. 按实际任务维护 host、浏览器、Native 和硬件作业边界；检查矩阵、分片完整性、required gate 与 skip 条件，不能通过跳过失败路径获得绿灯。
3. 核对事件信任边界和最小权限，按既有约定固定第三方 Actions。特别检查 fork/PR 代码是否能进入 secrets、写权限或发布环境。
4. cache key 反映平台、解释器与锁；缓存命中不证明产物有效。日志和 evidence 在失败时仍须上传，不把工作目录旧文件误作本次结果。
5. 发布限制沿用项目对 main、稳定标签和手动事件的规则。修改 YAML 不等于配置了 GitHub 平台分支、Environment 或仓库保护。
6. 运行相称的 YAML、脚本、文档和定向检查；本地通过、工作流语法通过、同提交远端 CI 与硬件验收分别报告。推送、远端 rerun、发布和真实硬件作业依据用户已有授权，安装 skill 本身不执行这些动作。

参考 [github-actions-templates](https://github.com/wshobson/agents/blob/9c61ab243def98f72da12abd41cfcf5bcd91af9d/plugins/cicd-automation/skills/github-actions-templates/SKILL.md) 的矩阵、缓存、权限与 artifact 结构。未复制其中的旧版本 Actions、npm/pip 安装或自动部署示例。
