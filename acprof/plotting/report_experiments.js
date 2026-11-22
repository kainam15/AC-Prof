/* Render the current independent-run estimates and retained Python condition checks. */
"use strict";
window.ACProfExperiments = ({data, $, V, element, option, fmt, metricTitle, metricComparisonReason,
  unreportedNote, plot, axis, schedule, filtered, escape, color, selectConfig}) => {

  const sourceKey = sources => JSON.stringify(sources);
  function coordinates(config) {
    let scale = config.input_scale;
    try {
      const raw = JSON.parse(config.input_case)?.scale;
      if (typeof raw === "string") {
        const exact = ACProfCoordinates.decimalCoordinate(raw)?.key;
        if (exact && exact !== ACProfCoordinates.decimalCoordinate(String(scale))?.key) scale = exact;
      }
    } catch { /* Legacy descriptive input labels have no separate scale token. */ }
    return [config.exact_resources?.cpu ?? config.cpu, config.exact_resources?.memory ?? config.memory, scale, config.device];
  }
  function groupCoordinates(group, side, exact = true) {
    if (group.resource_coordinates?.[side] === null) return null;
    const resources = group.resource_coordinates?.[side] ?? group;
    const axes = exact ? resources.exact_coordinates : null, scale = exact ? group.exact_coordinates : null;
    return [axes?.cpu_cores ?? resources.cpu_cores, axes?.mem_cap_gb ?? resources.mem_cap_gb,
      scale?.input_scale ?? group.input_scale, group.gpu_mode === "on" ? "GPU" : "CPU"];
  }
  const reasons = {
    missing_recorded_run_identity: "缺少 run_state.json 中可核验的运行身份；仅提供描述性分析",
    duplicate_independent_run: "重复运行身份不能作为独立样本",
    insufficient_independent_runs: "独立运行不足（每组至少 3 次），不估计置信区间",
    missing_comparable_runs: "缺少可比较的完整运行",
    nonpositive_baseline_for_ratio: "基线非正，比值区间不可用",
  };
  function choices() {
    const result = new Map();
    for (const item of data.experiment_comparisons) {
      for (const side of ["left", "right"]) result.set(sourceKey(item[side]), item[side + "_label"]);
    }
    for (const source of data.sources) result.set(sourceKey([source.path]),
      (source.display_name || source.path) + " · " + source.run_id);
    return result;
  }
  function selectRight() {
    const previous = $("experiment-right").value;
    $("experiment-right").replaceChildren();
    const left = JSON.parse($("experiment-left").value || "[]");
    for (const [key, label] of choices()) if (!JSON.parse(key).some(path => left.includes(path))) option($("experiment-right"), key, label);
    if ([...$("experiment-right").options].some(item => item.value === previous)) $("experiment-right").value = previous;
  }
  function refresh() {
    const previous = $("experiment-left").value, metric = $("experiment-metric").value;
    $("experiment-left").replaceChildren();
    for (const [key, label] of choices()) option($("experiment-left"), key, label);
    if ([...$("experiment-left").options].some(item => item.value === previous)) $("experiment-left").value = previous;
    selectRight();
    $("experiment-metric").replaceChildren(); option($("experiment-metric"), "", "全部比较指标");
    for (const name of Object.keys(data.registry)) option($("experiment-metric"), name, data.registry[name].label);
    $("experiment-metric").value = metric;
  }
  refresh();
  function syncFromActive() {
    const paths = side => [...new Set(data.configs.filter(config => config.experiment_id === $("active-experiment-" + side).value).map(config => config.source))];
    const left = paths("left"), right = paths("right");
    if (left.length) { $("experiment-left").value = sourceKey(left); selectRight(); }
    if (right.length) $("experiment-right").value = sourceKey(right);
  }
  function syncActive() {
    const left = JSON.parse($("experiment-left").value || "[]"), right = JSON.parse($("experiment-right").value || "[]");
    for (const [side, paths] of [["left", left], ["right", right]]) {
      $("active-experiment-" + side).value = left.length === 1 && right.length === 1 ?
        data.configs.find(config => config.source === paths[0])?.experiment_id || "" : "";
    }
    if (left.length === 1) $("baseline").value = data.configs.find(config => config.source === left[0])?.config_id || "";
  }
  $("experiment-left").onchange = () => { selectRight(); syncActive(); schedule(); };
  $("experiment-right").onchange = () => { syncActive(); schedule(); };
  $("experiment-metric").onchange = schedule;
  for (const name of Object.keys(data.registry)) option($("cross-model-metric"), name, metricTitle(name));
  $("cross-model-metric").value = "latency_app_s";
  $("cross-model-metric").onchange = schedule;
  const interval = bounds => bounds ? `[${bounds.map(fmt).join(", ")}]` : "—";
  function current() {
    const paths = side => {
      const active = $("active-experiment-" + side).value;
      return active ? [...new Set(data.configs.filter(config => config.experiment_id === active).map(config => config.source))] :
        JSON.parse($("experiment-" + side).value || "[]");
    };
    const left = paths("left"), right = paths("right");
    if (!left.length || !right.length || left.some(path => right.includes(path))) return null;
    const purpose = $("comparison-purpose").value === "within-experiment" ? "same-hardware" : $("comparison-purpose").value;
    return data.experiment_comparisons.find(item => sourceKey(item.left) === sourceKey(left) && sourceKey(item.right) === sourceKey(right) &&
      (!item.report || item.report.purpose === purpose)) || {left, right, report:null, reason:"statistics_not_computed", display_only:true};
  }
  function descriptions() {
    const entry = current(), configs = filtered(), purpose = $("comparison-purpose").value;
    if (!entry || entry.report) return [];
    const metrics = $("experiment-metric").value ? [$("experiment-metric").value] : data.default_matrix;
    const rows = [], axes = new Map(configs.map(config => [config, coordinates(config)]));
    for (const left of configs.filter(config => entry.left.includes(config.source))) {
      for (const right of configs.filter(config => entry.right.includes(config.source))) {
        // Match resource coordinates; input or semantic conflicts remain visible in the verdict.
        const [lc, lm, ls] = axes.get(left), [rc, rm, rs] = axes.get(right);
        if (ls !== rs || left.device !== right.device || purpose !== "resource-scaling" && (lc !== rc || lm !== rm)) continue;
        const comparison = V.comparisonContext(left, purpose, data.baseline_comparisons, data.registry);
        for (const metric of metrics) rows.push({left_config_id:left.config_id, config_id:right.config_id,
          left_source:left.source, source:right.source, left_experiment:left.experiment_name, experiment:right.experiment_name,
          model:right.model, run_id:right.run_id, cpu:rc, memory:rm, input_scale:rs, device:right.device,
          left_cpu:lc, left_memory:lm, metric, unit:data.registry[metric].unit,
          baseline_value:V.value(left, metric), value:V.value(right, metric), ...comparison.delta(right, metric),
          comparison:comparison.qualification(right), scope:"browser_descriptive", statistics_status:"not_computed"});
      }
    }
    return rows;
  }
  function groupFilter(entry, configs) {
    const legacy = !entry.report.coordinate_identity, selected = new Set(configs);
    const ambiguous = new Set(), seen = new Map();
    if (legacy) for (const group of entry.report.groups) {
      const key = JSON.stringify([group.metric, groupCoordinates(group, "left"), groupCoordinates(group, "right")]);
      if (seen.has(key)) { ambiguous.add(seen.get(key)); ambiguous.add(group); }
      else seen.set(key, group);
    }
    const sides = ["left", "right"].map(side => {
      const visible = new Set(), candidates = new Map(), paths = new Set(entry[side]);
      for (const config of data.configs) {
        if (!paths.has(config.source)) continue;
        const axes = coordinates(config), key = JSON.stringify(axes);
        if (selected.has(config)) visible.add(key);
        if (legacy) {
          const projection = JSON.stringify([config.cpu, config.memory, config.input_scale, config.device]);
          if (!candidates.has(projection)) candidates.set(projection, new Map());
          candidates.get(projection).set(key, axes);
        }
      }
      return group => {
        const axes = groupCoordinates(group, side);
        if (!axes) return null;
        if (!legacy) return visible.has(JSON.stringify(axes)) ? axes : null;
        const candidatesAtPoint = candidates.get(JSON.stringify(groupCoordinates(group, side, false)));
        if (!candidatesAtPoint) return null;
        // A legacy group may record an exact token for only some axes.
        const matches = [...candidatesAtPoint].filter(([, actual]) => axes.every((value, i) =>
          typeof value !== "string" || value === actual[i]));
        return matches.length === 1 && visible.has(matches[0][0]) ? matches[0][1] : null;
      };
    });
    return group => {
      if (ambiguous.has(group)) return false;
      const [left, right] = sides.map(match => match(group));
      return left && right && left[2] === right[2] && (entry.report.purpose === "resource-scaling" ||
        left[0] === right[0] && left[1] === right[1]);
    };
  }
  function groups() {
    const entry = current(), metric = $("experiment-metric").value;
    if (!entry?.report) return [];
    const matches = groupFilter(entry, filtered());
    return entry.report.groups.filter(group => (!metric || group.metric === metric) && matches(group));
  }
  async function draw(configs) {
    const entry = current(), report = entry?.report;
    $("experiments").tBodies[0].replaceChildren();
    $("experiment-status").textContent = report ? `${report.status} · ${entry.display_only ? "仅并列展示，差值和比值暂停" : "条件可比；统计范围见逐指标样本数"}` :
      entry ? "浏览器描述性比较 · 严格统计未计算 / Browser descriptive comparison; strict statistics not computed. " + (reasons[entry.reason] || entry.reason) : "需要至少两个来源；可使用 acprof report --left RUN --right RUN 指定独立组。";
    if (report) $("experiment-status").textContent += ` · ${report.engine === "browser" ? "浏览器重新计算 / Browser recalculation" : "Python 预计算 / Python precomputed"} · confidence=${report.confidence} · resamples=${report.resamples} · seed=${report.seed}`;
    if (report && !report.coordinate_identity) $("experiment-status").textContent += " · 历史比较仅显示可唯一匹配的坐标 / Legacy comparisons require unambiguous coordinates";
    if (report?.unavailable_metrics?.length) $("experiment-status").textContent += " · 缺少 Python 比较证据 / Missing Python comparison evidence: " + report.unavailable_metrics.join(", ");
    $("experiment-evidence").textContent = JSON.stringify(report || entry || {}, null, 2);
    for (const group of groups()) {
      const row = element("tr"), left = group.left, right = group.right;
      const leftAxes = groupCoordinates(group, "left"), rightAxes = groupCoordinates(group, "right");
      const label = ([cpu, mem]) => `${cpu ?? "inherited"}c / ${mem ?? "inherited"} GiB`;
      const cells = [`${label(leftAxes)} → ${label(rightAxes)} / GPU ${group.gpu_mode} / scale ${leftAxes[2]}`,
        `${data.registry[group.metric]?.label || group.metric} (${group.unit})`, fmt(left.mean), fmt(right.mean),
        fmt(group.difference), `${fmt(group.ratio)} / ${fmt(group.percent)}%`,
        `${interval(group.difference_ci)} / ${interval(group.ratio_ci)} (${report.confidence * 100}%)`,
        `${left.n_runs} / ${right.n_runs}`,
        `${reasons[group.reason] || group.reason || group.comparability} · 失败 ${left.failed_windows}/${right.failed_windows} · 缺失 ${left.missing_windows}/${right.missing_windows} · 推断 ${left.inferred_windows || 0}/${right.inferred_windows || 0}`];
      for (const cell of cells) row.append(element("td", cell));
      $("experiments").tBodies[0].append(row);
    }
    for (const group of descriptions()) {
      const row = element("tr"); row.dataset.configId = group.config_id;
      row.onclick = () => {
        $("baseline").value = group.left_config_id;
        selectConfig(data.configs.find(config => config.config_id === group.config_id));
        document.querySelector('[data-view="matrix"]').click();
      };
      row.tabIndex = 0; row.onkeydown = event => { if (event.key === "Enter") row.click(); };
      const cells = [`${group.left_experiment} ${group.left_cpu}c/${group.left_memory}G → ${group.experiment} ${group.cpu}c/${group.memory}G · scale ${group.input_scale}`,
        metricTitle(group.metric), fmt(group.baseline_value), fmt(group.value), fmt(group.delta), `${fmt(group.ratio)} / ${fmt(group.percent)}%`,
        "未计算 / Not computed", "未核验 / Not verified",
        [group.comparison.status, group.comparison.reasons.join(", "), metricComparisonReason(group)].filter(Boolean).join(" · ")];
      for (const cell of cells) row.append(element("td", cell));
      $("experiments").tBodies[0].append(row);
    }
    if (entry && !report && !descriptions().length) $("experiment-status").textContent += " · 没有匹配的配置 / No matching configurations";
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
      article.append(element("h3", `${group[0].task} · ${V.inputLabel(group[0].input_case, true)}`), chart);
      const note = unreportedNote(group, [metric]);
      if (note) article.append(element("small", note));
      $("cross-model-charts").append(article);
      await plot(chart, [{type:"bar", x:group.map(config => escape(`${config.model} · ${config.device} · ${config.cpu}c/${config.memory}G · ${config.config_id.slice(-6)}`)),
        y:group.map(config => V.value(config, metric)), customdata:group.map(config => config.config_id),
        marker:{color:group.map(color)}}], {height:380, yaxis:axis(metric), xaxis:{automargin:true}, showlegend:false});
    }
  }
  return {draw, current, groups, refresh, descriptions, syncFromActive};
};
