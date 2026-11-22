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
    "structured.fact.details_hint": Entry(zh="Enter / 双击 查看全文", en="Enter / double-click: details"),
    "structured.fact.hint": Entry(zh="单击 / C 复制 · Enter / 双击 查看全文", en="Click / C: copy · Enter / double-click: details"),
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
    "structured.reason.artifact_incomplete.impact": Entry(
        zh="当前快照缺少已报告的配置、权重分片或组件文件。",
        en="The snapshot is missing the reported configuration, weight shard or component file.",
    ),
    "structured.reason.artifact_incomplete.next": Entry(
        zh="根据技术详情补齐缺失文件，或选择完整的固定 revision。",
        en="Use the technical details to supply the missing files or select a complete pinned revision.",
    ),
    "structured.reason.artifact_incomplete.problem": Entry(
        zh="模型制品不完整。", en="The model artifacts are incomplete.",
    ),
    "structured.reason.artifact_unknown.impact": Entry(
        zh="现有文件证据不足以确认加载兼容性。",
        en="The available file evidence does not establish loader compatibility.",
    ),
    "structured.reason.artifact_unknown.next": Entry(
        zh="取得固定 revision 的文件清单，再检查所选加载器。",
        en="Obtain the pinned revision's file list, then check the selected loader.",
    ),
    "structured.reason.artifact_unknown.problem": Entry(
        zh="模型制品布局尚未确认。", en="The model artifact layout is unverified.",
    ),
    "structured.reason.backend_unsupported.impact": Entry(
        zh="当前登记的实现不能执行这个后端选项。",
        en="The registered implementations cannot execute this backend choice.",
    ),
    "structured.reason.backend_unsupported.next": Entry(
        zh="选择已实现且兼容模型制品的后端，或登记经过验证的适配。",
        en="Select an implemented backend compatible with the artifacts, or register a validated adapter.",
    ),
    "structured.reason.backend_unsupported.problem": Entry(
        zh="所选推理后端尚未受支持。", en="The selected inference backend is unsupported.",
    ),
    "structured.reason.compatibility_budget.impact": Entry(
        zh="模型在该设备和配置下的兼容性仍未确认。",
        en="Model compatibility remains unverified on this device with the recorded configuration.",
    ),
    "structured.reason.compatibility_budget.next": Entry(
        zh="查看验证日志，必要时延长时限并重新运行兼容性验证。",
        en="Review the probe log. If needed, retry the probe with a longer time limit.",
    ),
    "structured.reason.compatibility_budget.problem": Entry(
        zh="独立兼容性验证已达到时限。", en="The compatibility probe reached its time limit.",
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
    "structured.reason.metadata_unavailable.impact": Entry(
        zh="当前证据不足以确认模型接口兼容性。", en="The available evidence does not establish interface compatibility.",
    ),
    "structured.reason.metadata_unavailable.next": Entry(
        zh="检查该文件的缓存和仓库连接，再用相同的固定版本重试。",
        en="Check the cached file and repository connection, then retry the same pinned revision.",
    ),
    "structured.reason.metadata_unavailable.problem": Entry(
        zh="未能读取固定版本的模型元数据。", en="The pinned model metadata could not be read.",
    ),
    "structured.reason.model_artifact_format.impact": Entry(
        zh="模型标签或配置文件不能代替兼容的权重制品。",
        en="Model tags and configuration files cannot replace compatible weight artifacts.",
    ),
    "structured.reason.model_artifact_format.next": Entry(
        zh="选择与制品格式匹配的已登记后端或模型版本。",
        en="Select a registered backend or model revision matching the artifact format.",
    ),
    "structured.reason.model_artifact_format.problem": Entry(
        zh="模型制品格式不受所选加载器支持。",
        en="The selected loader does not support this model artifact format.",
    ),
    "structured.reason.model_configuration.impact": Entry(
        zh="所需输入或执行方式尚未确定，不能开始正式采集。",
        en="Required inputs or execution semantics are unresolved, so measurement cannot begin.",
    ),
    "structured.reason.model_configuration.next": Entry(
        zh="补充界面要求的配置，或通过 --model-spec 提供模型声明。",
        en="Supply the requested configuration or provide a model declaration with --model-spec.",
    ),
    "structured.reason.model_configuration.problem": Entry(
        zh="模型接口配置尚不完整。", en="The model interface configuration is incomplete.",
    ),
    "structured.reason.model_conflict.impact": Entry(
        zh="当前任务、后端或模型声明不能共同构成一致的执行配置。",
        en="The current task, backend and model declarations do not form a consistent execution configuration.",
    ),
    "structured.reason.model_conflict.next": Entry(
        zh="核对冲突来源，再选择兼容的任务、后端或模型声明。",
        en="Review the conflicting sources and select a compatible task, backend or model declaration.",
    ),
    "structured.reason.model_conflict.problem": Entry(
        zh="模型识别证据存在冲突。", en="The model identification evidence contains conflicts.",
    ),
    "structured.reason.model_metadata.impact": Entry(
        zh="接口解析已停止，尚未进入模型加载或正式采集。",
        en="Interface resolution stopped before model loading or measurement.",
    ),
    "structured.reason.model_metadata.next": Entry(
        zh="根据技术详情中的文件和字段路径修正元数据，或选择有效 revision。",
        en="Review the file and field paths in the technical details, then correct the metadata or select a valid revision.",
    ),
    "structured.reason.model_metadata.problem": Entry(
        zh="模型元数据无效。", en="The model metadata is invalid.",
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
    "structured.reason.runtime_unavailable.impact": Entry(
        zh="任务虽已识别，当前运行环境仍无法执行它。",
        en="The task was identified, but the available runtimes cannot execute it.",
    ),
    "structured.reason.runtime_unavailable.next": Entry(
        zh="选择兼容的已登记后端或运行环境，或补齐适配后验证。",
        en="Select a compatible registered backend or runtime, or add and validate its adapter.",
    ),
    "structured.reason.runtime_unavailable.problem": Entry(
        zh="没有兼容的已登记运行环境。", en="No compatible registered runtime is available.",
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
    "structured.reason.task_selection.impact": Entry(
        zh="已发现可选推理路径，但采集范围尚未确定。",
        en="Candidate inference paths were found, but the collection scope is unresolved.",
    ),
    "structured.reason.task_selection.next": Entry(
        zh="选择一个或多个推理配置组；CLI 使用 --tasks，--task 只指定接口。",
        en="Select one or more inference profiles with --tasks; --task specifies the interface.",
    ),
    "structured.reason.task_selection.problem": Entry(
        zh="尚未选择有效的推理配置组。", en="A valid inference profile has not been selected.",
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
