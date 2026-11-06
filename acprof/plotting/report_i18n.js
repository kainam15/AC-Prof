/* UI copy only. Evidence, identifiers and imported user text are never translated. */
"use strict";
window.ACProfI18n = (() => {
  const labels = {
    "实验管理 / Experiments":"Experiments", "添加实验 / Add experiment":"Add experiment", "取消 / Cancel":"Cancel",
    "另存自包含 HTML / Save HTML":"Save self-contained HTML", "恢复默认基线 / Reset baseline":"Reset baseline",
    "实验 / Experiment":"Experiment", "模型 / Model":"Model", "来源 / Source":"Source", "配置 / Configurations":"Configurations",
    "质量 / Quality":"Quality", "操作 / Actions":"Actions", "删除 / Remove":"Remove", "原始内嵌 / Embedded":"Embedded",
    "基线实验 / Baseline experiment":"Baseline experiment", "对照实验 / Comparison experiment":"Comparison experiment",
    "全部 / All":"All", "实验名称 / Experiment":"Experiment name",
    "/ 实验分析":"/ Experiment analysis", "离线报告":"Offline report", "性能、资源与能耗":"Performance, resources and energy",
    "比较已有实验，查看配置取舍、窗口统计与数据质量。":"Compare experiments, configuration trade-offs, window statistics and data quality.",
    "筛选实验":"Filter experiments", "重置筛选":"Reset filters", "环境":"Environment", "质量状态":"Quality status", "全部":"All",
    "比较用途":"Comparison purpose", "单次实验资源比较":"Within-experiment resources", "相同硬件":"Same hardware", "跨硬件":"Cross hardware",
    "资源扩容（CPU / 内存）":"Resource scaling (CPU / memory)", "质量范围":"Quality filter", "全部观测":"All observations",
    "满足质量要求的候选":"Quality-eligible candidates", "指标预设":"Metric preset", "自定义指标":"Custom metrics",
    "总览":"Overview", "窗口统计":"Window statistics", "实验对照 / 跨模型":"Experiments / Cross-model", "数据质量":"Data quality",
    "导出当前视图及筛选数据；浏览器描述性结果与 Python 统计分别标记":"Export this view and selection; browser descriptions and Python statistics are labeled separately.",
    "导出 JSON":"Export JSON", "导出 CSV":"Export CSV", "所选配置的质量证据":"Selected configuration evidence",
    "选择配置后查看状态、原因和原始证据。":"Select a configuration to inspect status, reasons and recorded evidence.",
    "点击表格行或图中数据点，可在各视图中定位同一配置。":"Select a table row or chart point to locate that configuration across views.",
    "模型、硬件与采集规模":"Models, hardware and collection size", "配置比较":"Configuration comparison",
    "点击列名排序 · 浅红 → 浅绿：由差到好 · 灰色：无方向或缺失":"Click column headers to sort. Red to green: worse to better. Gray: neutral or missing.",
    "没有匹配的配置。请调整筛选条件。":"No matching configurations. Adjust the filters.", "指标取舍":"Metric trade-offs",
    "空心菱形：当前筛选下的 Pareto frontier":"Open diamonds: Pareto frontier for the current selection", "预设":"Preset",
    "X 指标":"X metric", "Y 指标":"Y metric", "点大小":"Point size", "固定大小":"Fixed size", "资源扩容趋势":"Resource scaling",
    "按 model / runtime / 输入 / 环境分面":"Facets by model / runtime / input / environment", "资源轴":"Resource axis", "观察指标":"Metric",
    "每条线固定其余资源、device、GPU 与 run；空值处断线。点表示窗口汇总，连线仅用于观察趋势。":"Each line holds other resources, device, GPU and run fixed. Missing values break lines. Points summarize windows; lines only illustrate trends.",
    "统计指标":"Statistics metric", "配置 / 状态":"Configuration / Status", "指标 / 单位":"Metric / Unit", "均值":"Mean", "标准差":"Standard deviation",
    "有效 / 缺失窗口":"Valid / Missing windows", "缺失与排除原因":"Missing and exclusion reasons", "全部统计指标":"All statistics metrics",
    "没有匹配的窗口统计。请调整筛选条件。":"No matching window statistics. Adjust the filters.", "数据质量与来源":"Data quality and provenance",
    "失败、缺失、不适用及未记录分别保留；未知运行身份不计作独立运行。审计状态描述完整来源，表格遵循当前筛选。":"Failed, missing, inapplicable and unrecorded values remain distinct. Unknown run identities are not independent runs. Audit status describes complete sources; tables follow filters.",
    "独立实验对照":"Independent experiment comparison", "复用 Python 独立运行统计":"Reuse Python independent-run statistics",
    "基线实验 / 组":"Baseline experiment / group", "对照实验 / 组":"Comparison experiment / group", "比较指标":"Comparison metric",
    "全部比较指标":"All comparison metrics", "资源 / 输入":"Resources / Input", "基线均值":"Baseline mean", "对照均值":"Comparison mean",
    "差值":"Difference", "比值 / 变化":"Ratio / Change", "差值 CI / 比值 CI":"Difference CI / Ratio CI", "有效独立运行":"Valid independent runs",
    "原因 / 失败与缺失窗口":"Reasons / Failed and missing windows", "实验身份、兼容性及统计参数":"Experiment identity, compatibility and statistics parameters",
    "跨模型并列观测":"Side-by-side model observations", "数据来源与分析口径":"Sources and analysis contract", "Plotly.js 许可":"Plotly.js license",
    "AC-Prof · 只读分析 · 报告生成后不连接服务器":"AC-Prof · Read-only analysis · No server connection after generation",
    "无 Baseline · 仅显示绝对值":"No baseline · Absolute values only", "Core · 六项比较":"Core · Six metrics", "Summary · 核心指标":"Summary · Key metrics",
    "缺失字段与排除窗口明细":"Missing fields and excluded windows", "质量与完整性原始证据":"Recorded quality and completeness evidence",
    "复用统计 JSON":"Reused statistics JSON", "从原始测量窗口重新计算":"Computed from original measurement windows",
    "需要至少两个来源；可使用 acprof report --left RUN --right RUN 指定独立组。":"At least two sources are needed. Use acprof report --left RUN --right RUN to specify independent groups.",
    "窗口不足":"Insufficient windows", "无正式成功窗口":"No successful formal windows", "已估计":"Estimated", "无指标缺失":"No missing metrics",
    "没有有效观测值":"No valid observations", "无方向指标不标注 frontier。":"Neutral metrics do not define a frontier.",
  };
  Object.assign(labels, {
  "选择或拖放 result_all.csv、AC-Prof HTML、分析 JSON。CSV 缺少完整元数据时仅作描述性展示；严格统计未计算。 / Drop CSV, AC-Prof HTML or analysis JSON. CSV-only comparisons are descriptive; strict statistics are not computed.": "Choose or drop result_all.csv, AC-Prof HTML or analysis JSON. CSV files without complete metadata support descriptive comparisons only; strict statistics are not computed.",
  "仅汇总正式成功窗口；缺失值显示 —，失败与推断配置保留状态。P50/P95 为窗口分位数的均值，CPU/GPU 占用率无默认优劣方向。条件检查与严格比较共用规则；变化值仍是描述性比较，不含独立实验置信区间。自动优选暂停使用质量 blocked/unknown 或条件证据不足的配置；告警与证据保留，passed 只表示已记录的检查无问题，不代表模型准确率合格。": "Only successful formal windows are summarized. Missing values display as —; failed and inferred configurations retain their status. P50/P95 are means of window quantiles. CPU/GPU utilization has no default better direction. Changes are descriptive and have no independent-run confidence intervals. Quality blocked/unknown or incomplete condition evidence excludes automatic selection. Passed means recorded checks found no issues, not that model accuracy is acceptable.",
  "均值与样本标准差（ddof=1）按正式成功窗口等权计算；CI 描述本实验内窗口均值的不确定性，置信水平以各来源参数为准。 窗口数不是请求数或独立实验数，区间不证明跨实验重复性。窗口不足或连续块不完整时保留原因，空值显示 —。": "Means and sample standard deviations (ddof=1) weight successful formal windows equally. CI describes uncertainty in this run, at each source confidence level. Windows are not requests or independent experiments; these intervals do not establish repeatability. Insufficient windows or incomplete blocks retain their reasons and missing values.",
  "先对各独立运行的成功窗口求均值，再对运行均值等权统计。每组至少 3 次有效独立运行才估计 Bootstrap CI； 窗口数不能代替独立运行数。筛选只控制整组展示，不改变组内独立样本。不同模型的可比性不代表模型质量相等。": "Successful windows are averaged within each independent run, then run means receive equal weight. Each group needs at least 3 valid independent runs for a Bootstrap CI. Window counts cannot replace run counts. Filters control whole-group visibility without changing independent samples. Model comparability does not establish equal model quality.",
  "沿用上方统一筛选。不同任务、输入单位、生成长度或推理语义的数据仅并列展示； 原始延迟差异不能解释为模型优劣。可直接比较的条件及指标由上方严格检查单独说明。": "The shared filters also apply here. Different tasks, input units, output lengths or inference semantics are only shown side by side. Raw latency differences do not establish model superiority. The condition checks above identify comparable conditions and metrics.",
  "配置状态 ok 只描述已写入的正式窗口，不证明整个实验计划已完成；完成度仍以原始 run state 和 audit 验收为准。": "Configuration status ok describes recorded formal windows. It does not prove that the experiment plan completed; completion is determined by the original run state and audit.",
  "原始 CSV 保持完整。CPU package + GPU 总能量仅覆盖成功请求窗口；CPU 配置只计 package，GPU 配置须同时具有 package 与所选 GPU 的记录。没有加入 DRAM、启动或 idle 时段，也不等同于整机墙插能耗。": "Original CSV files remain intact. CPU package + GPU energy covers successful request windows only. CPU configurations count package energy; GPU configurations require both package and selected-GPU records. DRAM, startup and idle periods are excluded. This is not whole-system wall power.",
  "窗口总能量按每行 energy/request × 实际请求数求和；energy/request 按请求数加权。其他均值等权汇总窗口，峰值取最大值。Cold start 按启动时间去重；缺少启动身份时仅接受一致的复用值。": "Window energy is summed as energy/request times actual request count. Energy/request is request-weighted. Other means weight windows equally; peaks take the maximum. Cold start is deduplicated by startup identity; missing identities require consistent reused values.",
  "QPS 仅展示显式记录的 request/s；不会从 samples/s 推断。颜色与 frontier 按已核验条件分组；历史单来源比较仅为描述性对比。窗口统计展示 within-run CI；实验对照另行展示独立运行 CI，不重新计算合并请求分位数。": "QPS uses explicit request/s records, never inferred samples/s. Colors and frontiers group verified conditions. Historical single-source comparisons are descriptive. Window statistics show within-run CI, while experiment comparisons show independent-run CI; pooled request quantiles are not re-estimated.",
  "关键指标 · 按配置展示，不合并不同资源条件": "Key metrics by configuration; resource conditions are not pooled",
  "计划覆盖：unknown（缺少可核验计划；已写入窗口成功不代表实验完整）": "Plan coverage: unknown. Recorded successful windows do not establish a complete experiment.",
  "元数据缺失，严格统计未计算 / Missing metadata; strict statistics not computed": "Missing metadata; strict statistics not computed"
});
  const originals = new WeakMap();
  function translate(value) {
    const stripped = value.trim();
    if (labels[stripped.replace(/\s+/g, " ")]) return labels[stripped.replace(/\s+/g, " ")];
    if (labels[stripped]) return value.replace(stripped, labels[stripped]);
    // Bilingual controls are intentionally identical in both modes.
    if (stripped.includes(" / ") && /[a-zA-Z]/.test(stripped)) return value;
    return value.replace(/(\d+(?:\.\d+)?(?: \/ \d+(?:\.\d+)?)*)% 置信区间/g, "$1% confidence interval")
      .replace(/^\/ (\d+) 配置$/, "/ $1 configurations")
      .replace(/^用途：/, "Purpose: ")
      .replace(/。允许 CPU \/ memory 配额变化；双轴同时变化不能归因某一个轴。/g, ". CPU / memory quotas may differ; changing both axes does not isolate either effect. ")
      .replace(/同一来源、任务及输入条件的描述性比较；历史元数据缺失不证明实验独立性。多资源变化不能单独归因。/g, "Descriptive comparison within source, task and input conditions. Missing historical metadata does not establish independence or causality. ")
      .replace(/比值=当前\/基线；性能比按指标方向计算，>1 表示改善，neutral 不评级。/g, "Ratio=current/baseline. Direction-aware performance ratios above 1 indicate improvement; neutral metrics are not ranked. ")
      .replace(/选择 baseline 后显示成对检查原因。/g, "Choose a baseline to see pairwise condition checks. ").replace(/变化资源：/, "Changed resources: ")
      .replace(/^可比性 /, "Comparability ").replace(/ · 质量 /g, " · Quality ")
      .replace(/^运行 /, "Run ").replace(/ · 完整性 /g, " · Completeness ")
      .replace(/^优选暂停：/, "Selection withheld: ").replace(/^质量可参与优选$/, "Quality eligible")
      .replace(/^排除：/, "Excluded: ").replace(/^已选：/, "Selected: ")
      .replace(/（当前筛选隐藏）$/, " (hidden by filters)");
  }
  function apply(language) {
    document.documentElement.lang = language === "en" ? "en" : "zh-CN";
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (node.parentElement.closest("script, style, pre, #sources, #experiment-list td:not(:last-child), #import-status, .js-plotly-plot")) continue;
      const previous = originals.get(node), original = previous && node.nodeValue === previous.rendered ? previous.original : node.nodeValue;
      const rendered = language === "en" ? translate(original) : original;
      originals.set(node, {original, rendered});
      if (node.nodeValue !== rendered) node.nodeValue = rendered;
    }
  }
  return {apply};
})();
