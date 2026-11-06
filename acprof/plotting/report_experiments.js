/* Render Python's independent-run estimates; no statistical calculations here. */
"use strict";
window.ACProfExperiments = ({data, $, V, element, option, fmt, metricTitle, plot, axis, schedule, filtered, escape}) => {
  const comparisons = data.experiment_comparisons || [];
  const sourceKey = sources => JSON.stringify(sources);
  const reasons = {
    missing_recorded_run_identity: "缺少 run_state.json 中可核验的运行身份；仅提供描述性分析",
    duplicate_independent_run: "重复运行身份不能作为独立样本",
    insufficient_independent_runs: "独立运行不足（每组至少 3 次），不估计置信区间",
    missing_comparable_runs: "缺少可比较的完整运行",
    nonpositive_baseline_for_ratio: "基线非正，比值区间不可用",
  };
  for (const [key, entry] of new Map(comparisons.map(item => [sourceKey(item.left), item]))) {
    option($("experiment-left"), key, entry.left_label);
  }
  function selectRight() {
    const previous = $("experiment-right").value;
    $("experiment-right").replaceChildren();
    for (const item of comparisons.filter(item => sourceKey(item.left) === $("experiment-left").value)) {
      option($("experiment-right"), sourceKey(item.right), item.right_label);
    }
    if ([...$("experiment-right").options].some(item => item.value === previous)) $("experiment-right").value = previous;
  }
  selectRight();
  $("experiment-left").onchange = () => { selectRight(); schedule(); };
  $("experiment-right").onchange = schedule;
  option($("experiment-metric"), "", "全部比较指标");
  for (const name of new Set(comparisons.flatMap(item => (item.report?.groups || []).map(group => group.metric)))) {
    option($("experiment-metric"), name, data.registry[name]?.label || name);
  }
  $("experiment-metric").onchange = schedule;
  for (const name of Object.keys(data.registry)) option($("cross-model-metric"), name, metricTitle(name));
  $("cross-model-metric").value = "latency_app_s";
  $("cross-model-metric").onchange = schedule;
  const interval = bounds => bounds ? `[${bounds.map(fmt).join(", ")}]` : "—";
  const current = () => comparisons.find(item => sourceKey(item.left) === $("experiment-left").value &&
    sourceKey(item.right) === $("experiment-right").value);
  function visible(group, entry, configs) {
    return ["left", "right"].every(side => configs.some(config => entry[side].includes(config.source) &&
      config.cpu === (group.resource_coordinates?.[side]?.cpu_cores ?? group.cpu_cores) &&
      config.memory === (group.resource_coordinates?.[side]?.mem_cap_gb ?? group.mem_cap_gb) &&
      config.input_scale === group.input_scale && config.device === (group.gpu_mode === "on" ? "GPU" : "CPU")));
  }
  function groups() {
    const entry = current(), metric = $("experiment-metric").value;
    return entry?.report?.groups.filter(group => (!metric || group.metric === metric) && visible(group, entry, filtered())) || [];
  }
  async function draw(configs) {
    const entry = current(), report = entry?.report;
    $("experiments").tBodies[0].replaceChildren();
    $("experiment-status").textContent = report ? `${report.status} · ${entry.display_only ? "仅并列展示，差值和比值暂停" : "条件可比；统计范围见逐指标样本数"}` :
      entry ? reasons[entry.reason] || entry.reason : "需要至少两个来源；可使用 acprof report --left RUN --right RUN 指定独立组。";
    $("experiment-evidence").textContent = JSON.stringify(report || entry || {}, null, 2);
    for (const group of groups()) {
      const row = element("tr"), left = group.left, right = group.right;
      const coordinates = side => `${group.resource_coordinates?.[side]?.cpu_cores ?? group.cpu_cores}c / ${group.resource_coordinates?.[side]?.mem_cap_gb ?? group.mem_cap_gb} GiB`;
      const cells = [`${coordinates("left")} → ${coordinates("right")} / GPU ${group.gpu_mode} / scale ${group.input_scale}`,
        `${data.registry[group.metric]?.label || group.metric} (${group.unit})`, fmt(left.mean), fmt(right.mean),
        fmt(group.difference), `${fmt(group.ratio)} / ${fmt(group.percent)}%`,
        `${interval(group.difference_ci)} / ${interval(group.ratio_ci)} (${report.confidence * 100}%)`,
        `${left.n_runs} / ${right.n_runs}`,
        `${reasons[group.reason] || group.reason || group.comparability} · 失败 ${left.failed_windows}/${right.failed_windows} · 缺失 ${left.missing_windows}/${right.missing_windows}`];
      for (const cell of cells) row.append(element("td", cell));
      $("experiments").tBodies[0].append(row);
    }
    for (const chart of $("cross-model-charts").querySelectorAll(".scaling-chart")) Plotly.purge(chart);
    $("cross-model-charts").replaceChildren();
    const metric = $("cross-model-metric").value;
    const facets = new Map();
    for (const config of configs) {
      const key = JSON.stringify([config.task, config.input_case, config.environment_class]);
      if (!facets.has(key)) facets.set(key, []);
      facets.get(key).push(config);
    }
    for (const group of facets.values()) {
      const article = element("article"), chart = element("div", undefined, "scaling-chart");
      article.append(element("h3", `${group[0].task} · ${group[0].input_case}`), chart);
      $("cross-model-charts").append(article);
      await plot(chart, [{type:"bar", x:group.map(config => escape(`${config.model} · ${config.device} · ${config.cpu}c/${config.memory}G · ${config.config_id.slice(-6)}`)),
        y:group.map(config => V.value(config, metric)), customdata:group.map(config => config.config_id),
        marker:{color:"#087f8c"}}], {height:380, yaxis:axis(metric), xaxis:{automargin:true}, showlegend:false});
    }
  }
  return {draw, current, groups};
};
