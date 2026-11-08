"""Bilingual copy for shared structured views; no business-state decisions."""
from __future__ import annotations

from .entry import Entry

MESSAGES: dict[str, Entry] = {
    "structured.action.close": Entry(zh="关闭详情", en="Close details"),
    "structured.action.copied": Entry(zh="已发出复制请求", en="Copy requested"),
    "structured.action.copy": Entry(zh="复制完整值", en="Copy full value"),
    "structured.action.details": Entry(zh="查看详情", en="Details"),
    "structured.callout.impact": Entry(zh="影响", en="Impact"),
    "structured.callout.next": Entry(zh="下一步", en="Next step"),
    "structured.callout.problem": Entry(zh="问题", en="Problem"),
    "structured.detail.title": Entry(zh="技术详情", en="Technical details"),
    "structured.fact.details": Entry(zh="更多信息", en="More details"),
    "structured.fact.details_hint": Entry(zh="Enter 查看全文", en="Enter details"),
    "structured.fact.hint": Entry(zh="Enter 查看全文 · C 复制", en="Enter details · C copy"),
    "structured.reason.access.impact": Entry(
        zh="当前操作无法读取所需资源。", en="The current operation cannot read the required resource.",
    ),
    "structured.reason.access.next": Entry(
        zh="检查模型或文件的访问权限后重新验证。",
        en="Check model or file access permissions, then validate again.",
    ),
    "structured.reason.access.problem": Entry(
        zh="所需资源的访问被拒绝。", en="Access to a required resource was denied.",
    ),
    "structured.reason.artifact_format.impact": Entry(
        zh="当前版本无法读取这些产物。", en="This version cannot read these artifacts.",
    ),
    "structured.reason.artifact_format.next": Entry(
        zh="使用兼容版本读取产物，或重新运行到新目录。",
        en="Read the artifacts with a compatible version, or rerun into a new directory.",
    ),
    "structured.reason.artifact_format.problem": Entry(
        zh="实验产物格式不受支持。", en="The experiment artifact format is unsupported.",
    ),
    "structured.reason.dependency_incompatible.impact": Entry(
        zh="当前依赖组合无法可靠执行该任务。",
        en="The current dependency combination cannot reliably run this task.",
    ),
    "structured.reason.dependency_incompatible.next": Entry(
        zh="检查版本约束并修复对应运行环境后重新验证。",
        en="Review version constraints and repair the runtime before validating again.",
    ),
    "structured.reason.dependency_incompatible.problem": Entry(
        zh="运行依赖版本不兼容。", en="Runtime dependency versions are incompatible.",
    ),
    "structured.reason.dependency_missing.impact": Entry(
        zh="相关运行组件无法加载。", en="The affected runtime component cannot load.",
    ),
    "structured.reason.dependency_missing.next": Entry(
        zh="按该运行环境的依赖说明修复后重新验证。",
        en="Repair dependencies using the runtime requirements, then validate again.",
    ),
    "structured.reason.dependency_missing.problem": Entry(
        zh="运行环境缺少依赖。", en="The runtime is missing a dependency.",
    ),
    "structured.reason.evidence.impact": Entry(
        zh="相关状态和产物暂不能作为可靠结果。",
        en="The affected state and artifacts cannot yet be treated as reliable results.",
    ),
    "structured.reason.evidence.next": Entry(
        zh="展开技术详情核对来源与完整性，再重新检查。",
        en="Inspect sources and integrity in the technical details, then check again.",
    ),
    "structured.reason.evidence.problem": Entry(
        zh="实验记录未通过校验。", en="The recorded experiment evidence failed validation.",
    ),
    "structured.reason.inference.impact": Entry(
        zh="该请求未取得可用的完整输出。",
        en="A usable complete output was not obtained for this request.",
    ),
    "structured.reason.inference.next": Entry(
        zh="核对任务、输入和输出要求，并根据技术详情修复后重试。",
        en="Review task, input and output requirements; use the technical details to fix the cause before retrying.",
    ),
    "structured.reason.inference.problem": Entry(
        zh="推理流程未能完成。", en="The inference workflow did not complete.",
    ),
    "structured.reason.initialization.impact": Entry(
        zh="该运行环境尚未完成准备。", en="This runtime has not finished preparation.",
    ),
    "structured.reason.initialization.next": Entry(
        zh="查看技术详情中的异常，修复对应配置或环境后重试。",
        en="Inspect the technical error, repair the relevant configuration or environment, then retry.",
    ),
    "structured.reason.initialization.problem": Entry(
        zh="运行环境初始化失败。", en="Runtime initialization failed.",
    ),
    "structured.reason.input_budget.impact": Entry(
        zh="超出预算的输入未能完成处理。",
        en="The input exceeding the budget could not be processed.",
    ),
    "structured.reason.input_budget.next": Entry(
        zh="减小输入规模，或检查并调整输入预算后重新验证。",
        en="Reduce input size, or review and adjust the input budget before validating again.",
    ),
    "structured.reason.input_budget.problem": Entry(
        zh="输入超过当前预算限制。", en="The input exceeds the current budget.",
    ),
    "structured.reason.resources.impact": Entry(
        zh="受影响的任务未能在当前资源限制内完成。",
        en="The affected task did not complete within the current resource limits.",
    ),
    "structured.reason.resources.next": Entry(
        zh="检查内存和显存限制，调整资源或输入规模后重新验证。",
        en="Check memory limits, then adjust resources or input size and validate again.",
    ),
    "structured.reason.resources.problem": Entry(
        zh="可用资源不足。", en="Available resources are insufficient.",
    ),
    "structured.reason.state_format.impact": Entry(
        zh="无法可靠判断运行状态或续跑条件。",
        en="Run status and resume requirements cannot be verified reliably.",
    ),
    "structured.reason.state_format.next": Entry(
        zh="使用与该实验兼容的版本检查记录，或创建新实验。",
        en="Inspect the record with a compatible version, or create a new experiment.",
    ),
    "structured.reason.state_format.problem": Entry(
        zh="实验状态记录格式不受支持。", en="The experiment state format is unsupported.",
    ),
    "structured.reason.task.impact": Entry(
        zh="所选任务无法在该运行环境中执行。",
        en="The selected task cannot run in this runtime.",
    ),
    "structured.reason.task.next": Entry(
        zh="选择支持该任务的运行环境，或调整任务选择。",
        en="Select a runtime supporting this task, or change the selected task.",
    ),
    "structured.reason.task.problem": Entry(
        zh="当前运行环境不支持所选任务。",
        en="The current runtime does not support the selected task.",
    ),
    "structured.reason.timeout.impact": Entry(
        zh="该请求未取得完整结果。", en="A complete result was not obtained for this request.",
    ),
    "structured.reason.timeout.next": Entry(
        zh="检查任务进度与资源；必要时调整超时或输入规模后重试。",
        en="Check task progress and resources; adjust the timeout or input size if needed before retrying.",
    ),
    "structured.reason.timeout.problem": Entry(
        zh="请求未在限定时间内完成。", en="The request did not complete within its time limit.",
    ),
    "structured.reason.unknown.impact": Entry(
        zh="当前信息不足以确定具体原因。",
        en="The available information does not establish the specific cause.",
    ),
    "structured.reason.unknown.next": Entry(
        zh="展开技术详情并核对相关日志，再决定下一步操作。",
        en="Expand technical details and review the related logs before choosing the next action.",
    ),
    "structured.reason.unknown.problem": Entry(
        zh="实验记录需要进一步检查。", en="The experiment record needs further inspection.",
    ),
    "structured.status.checking": Entry(zh="检查中", en="Checking"),
    "structured.status.incomplete": Entry(zh="未完成", en="Incomplete"),
    "structured.status.pending": Entry(zh="等待处理", en="Pending"),
    "structured.status.preparing": Entry(zh="准备中", en="Preparing"),
    "structured.status.present": Entry(zh="产物已生成", en="Artifact created"),
}
