"use strict";
(() => {
  const data = JSON.parse(document.getElementById("report-data").textContent);
  const registry = data.registry, V = window.ACProfViews;
  const $ = id => document.getElementById(id);
  const state = {view: "matrix", selected: null, sort: data.default_matrix[0], descending: false};
  const filterFields = [
    ["model", "Model"], ["runtime", "Runtime"], ["device", "CPU / GPU"],
    ["input_case", "Input case"], ["environment_class", "环境"], ["task", "Task"],
    ["cpu", "CPU cores"], ["memory", "Memory (GiB)"], ["concurrency", "Concurrency"],
    ["experiment_batch", "Experiment batch"], ["quality_status", "质量状态"], ["gpu", "GPU identity"],
  ];
  const text = value => value == null ? "unknown" : String(value);
  const escape = value => text(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const fmt = value => V.finite(value) ? value.toLocaleString("en-US", {maximumSignificantDigits: 5}) : "—";
  const signed = value => (value > 0 ? "+" : "") + fmt(value);
  const direction = metric => ({higher: "↑", lower: "↓", neutral: "·"})[registry[metric].direction];
  const metricTitle = metric => `${registry[metric].label} (${registry[metric].unit})`;
  const statistics = data.window_statistics || [];
  const reasonLabels = {
    insufficient_windows: "窗口不足", no_successful_windows: "无正式成功窗口",
    missing_windows_break_blocks: "缺失窗口打断连续块", nonconsecutive_windows: "窗口序号不连续",
    invalid_window_identity: "缺少有效窗口身份", not_recorded: "未记录字段",
    unavailable_unspecified: "不可用（未记录具体原因）", not_applicable: "不适用",
    measurement_failed: "测量失败", unsupported_by_process_runtime: "Process Runtime 不支持",
    unavailable_on_wsl2: "WSL2 不可用", not_requested: "未请求采集",
    warmup: "预热", error: "失败", warn: "告警", inferred_not_measured: "推断、未实测",
    source_changed: "原始数据已变化", audit_changed: "质量或完整性状态已变化",
    incompatible_statistics_contract: "统计版本或口径不匹配",
    configuration_groups_changed: "配置分组不匹配", statistics_summary_changed: "统计摘要不匹配",
    invalid_confidence_interval: "置信区间无效", invalid_statistics_groups: "统计分组无效",
    statistics_parameters_changed: "统计指标或 Bootstrap 参数不匹配",
  };
  const reasonText = reason => reasonLabels[reason] || reason;
  const reasonCounts = reasons => Object.entries(reasons || {}).map(([reason, count]) => `${reasonText(reason)} ${count}`).join(" · ");
  const inputLabel = (input, detailed = false) => {
    try {
      const parsed = JSON.parse(input);
      return `scale=${text(parsed.scale)} · batch=${text(parsed.batch_size)} · ${text(parsed.type)}` +
        (detailed ? ` · param=${text(parsed.task_param)}` : "");
    } catch { return input; }
  };
  const label = config => `${config.model} · ${config.runtime} · ${config.device} · CPU ${text(config.cpu)} · ${text(config.memory)} GiB`;
  const fullLabel = config => `${label(config)} · ${inputLabel(config.input_case, true)} · ${config.run_id} · ${config.config_id.slice(-8)}`;
  function element(tag, content, className) {
    const item = document.createElement(tag);
    if (content !== undefined) item.textContent = content;
    if (className) item.className = className;
    return item;
  }
  function option(select, value, title) {
    const item = new Option(title, value);
    item.title = title;
    select.add(item);
  }
  function selectConfig(config) {
    state.selected = state.selected === config.config_id ? null : config.config_id;
    schedule();
  }
  for (const [field, title] of filterFields) {
    const wrapper = element("label", title), select = element("select");
    select.id = "filter-" + field;
    if (field === "model") { select.multiple = true; select.size = 3; }
    select.setAttribute("aria-label", title);
    option(select, "", "全部");
    const values = [...new Set(data.configs.map(config => text(config[field])))];
    values.sort((a, b) => a.localeCompare(b, "zh-CN", {numeric: true}));
    for (const value of values) option(select, value, field === "input_case" ? inputLabel(value, true) : value);
    select.onchange = schedule;
    wrapper.append(select);
    $("filters").append(wrapper);
  }
  option($("baseline"), "", "无 Baseline · 仅显示绝对值");
  for (const config of data.configs) option($("baseline"), config.config_id, fullLabel(config));
  $("baseline").value = data.baseline || "";
  $("baseline").onchange = schedule;
  $("comparison-purpose").value = data.comparison_purpose;
  $("comparison-purpose").onchange = schedule;
  $("quality-filter").onchange = schedule;
  option($("statistics-metric"), "", "全部统计指标");
  for (const metric of new Set(statistics.flatMap(report => report.groups.map(group => group.metric)))) {
    option($("statistics-metric"), metric, registry[metric]?.label || metric);
  }
  $("statistics-metric").onchange = schedule;
  option($("preset"), "Core", "Core · 六项比较");
  option($("preset"), "Summary", "Summary · 核心指标");
  for (const group of new Set(Object.values(registry).map(metric => metric.group))) option($("preset"), group, group);
  option($("preset"), "Custom", "Custom");
  for (const id of ["metrics", "x-metric", "y-metric", "size-metric", "scaling-metric"]) {
    if (id === "size-metric") option($(id), "", "固定大小");
    for (const [name, metric] of Object.entries(registry)) option($(id), name, `${metric.label} (${metric.unit}) ${direction(name)}`);
  }
  function setMetrics(metrics) {
    for (const item of $("metrics").options) item.selected = metrics.includes(item.value);
  }
  setMetrics(data.default_matrix);
  $("preset").onchange = () => {
    const preset = $("preset").value;
    if (preset !== "Custom") setMetrics(preset === "Core" ? data.default_matrix :
      Object.keys(registry).filter(name => preset === "Summary" ? registry[name].summary : registry[name].group === preset));
    schedule();
  };
  $("metrics").onchange = () => { $("preset").value = "Custom"; schedule(); };
  $("x-metric").value = "latency_app_p95_s";
  $("y-metric").value = "observed_energy_per_request_j";
  $("scaling-metric").value = "latency_app_p95_s";
  for (const id of ["x-metric", "y-metric", "size-metric", "scaling-axis", "scaling-metric"]) $(id).onchange = schedule;
  $("tradeoff-preset").onchange = () => {
    [$("x-metric").value, $("y-metric").value] = $("tradeoff-preset").value.split(",");
    schedule();
  };
  $("reset").onclick = () => {
    $("quality-filter").value = "all";
    $("statistics-metric").value = "";
    for (const [field] of filterFields) $("filter-" + field).value = "";
    state.selected = null;
    schedule();
  };
  for (const button of document.querySelectorAll("[data-view]")) button.onclick = () => {
    state.view = button.dataset.view;
    for (const item of document.querySelectorAll("[data-view]")) item.setAttribute("aria-pressed", String(item === button));
    schedule();
  };
  for (const source of data.sources) {
    $("sources").append(element("li", `${source.path} · run=${source.run_id} · state=${source.run_state} · quality=${source.quality_status} · SHA256 ${source.sha256}`));
  }
  function filtered() {
    return data.configs.filter(config => ($("quality-filter").value !== "eligible" || config.auto_selection_eligible === true) && filterFields.every(([field]) => {
      const chosen = [...$("filter-" + field).selectedOptions].map(option => option.value);
      return !chosen.length || chosen.includes("") || chosen.includes(text(config[field]));
    }));
  }
  function drawMatrix(configs, comparison) {
    const {baseline, purpose} = comparison;
    const metrics = $("preset").value === "Core" ? data.default_matrix :
      [...$("metrics").selectedOptions].map(item => item.value);
    const head = $("matrix").tHead, body = $("matrix").tBodies[0];
    head.replaceChildren(); body.replaceChildren();
    const header = element("tr");
    header.append(element("th", "配置 / 状态"));
    for (const name of metrics) {
      const cell = element("th"), button = element("button", registry[name].label + " " + direction(name));
      button.title = `排序 · ${name}`;
      cell.setAttribute("aria-sort", state.sort === name ? (state.descending ? "descending" : "ascending") : "none");
      button.onclick = () => { state.descending = state.sort === name ? !state.descending : false; state.sort = name; schedule(); };
      cell.append(button, element("small", registry[name].unit)); header.append(cell);
    }
    head.append(header);
    const ranges = V.colorRanges(configs, metrics, registry, purpose);
    for (const config of V.sort(configs, state.sort, state.descending)) {
      const qualification = comparison.qualification(config);
      const row = element("tr"); row.dataset.configId = config.config_id;
      row.classList.toggle("selected", state.selected === config.config_id);
      row.onclick = () => selectConfig(config);
      row.tabIndex = 0;
      row.onkeydown = event => { if (event.key === "Enter") selectConfig(config); };
      const identity = element("td");
      identity.title = fullLabel(config) + ` · GPU=${config.gpu}`;
      identity.append(element("strong", `${config.device} · CPU ${text(config.cpu)} · ${text(config.memory)} GiB`),
        element("span", config.status, "status"),
        element("small", `运行 ${config.run_status} · 完整性 ${config.measurement_status}`),
        element("small", `可比性 ${qualification.status} · 质量 ${config.quality_status}`),
        element("small", qualification.reasons.join(", ")),
        element("small", qualification.changed_resources ? `变化资源：${qualification.changed_resources.join(" + ") || "无"} · 描述性比较，不推断因果` : ""),
        element("small", config.auto_selection_eligible ? "质量可参与优选" : `优选暂停：${config.quality_reasons.join(", ")}`),
        element("small", `${config.model.split("/").pop()} · ${config.runtime}`),
        element("small", `${inputLabel(config.input_case)} · n=${config.valid_windows}`));
      row.append(identity);
      for (const name of metrics) {
        const value = V.value(config, name), cell = element("td", fmt(value));
        const score = config.auto_selection_eligible && comparison.automaticQualification(config).status === "compatible" ?
          V.score(value, ranges.get(V.cohort(config, purpose))?.[name], registry[name]) : null;
        if (score != null) cell.style.backgroundColor = `hsl(${12 + 140 * score} 43% 91%)`;
        const stats = config.metrics[name];
        cell.title = `${name} · ${registry[name].unit} · ${stats.aggregation} · n=${stats.n} · missing=${stats.missing}`;
        if (baseline) {
          const change = comparison.delta(config, name);
          const content = change.delta == null ? (change.reasons ? `${change.reason}: ${change.reasons.join(", ")}` : "— vs baseline") :
            `Δ ${signed(change.delta)} ${registry[name].unit} · ${change.percent == null ? "% —" : signed(change.percent) + "%"}` +
            ` · 比值 ${fmt(change.ratio)} · 性能比 ${fmt(change.performance_ratio)}`;
          cell.append(element("small", content));
        }
        row.append(cell);
      }
      body.append(row);
    }
    $("matrix-empty").hidden = configs.length > 0;
  }
  const palette = ["#087f8c", "#c57532", "#7161ab", "#2673a8", "#a54967", "#4e824d", "#986833"];
  const colorKeys = [...new Set(data.configs.map(c => `${c.runtime} / ${c.device}`))];
  const color = config => palette[colorKeys.indexOf(`${config.runtime} / ${config.device}`) % palette.length];
  const axis = metric => ({title: {text: escape(metricTitle(metric))}, type: registry[metric].scale,
    automargin: true, gridcolor: "#e8eef2", zeroline: false});
  const layout = {paper_bgcolor: "white", plot_bgcolor: "white", font: {family: "system-ui", color: "#29475d"},
    margin: {l: 70, r: 25, t: 35, b: 65}, legend: {orientation: "h", y: 1.14}, hovermode: "closest"};
  const plotOptions = {responsive: true, displaylogo: false, scrollZoom: false,
    modeBarButtonsToRemove: ["sendDataToCloud"], toImageButtonOptions: {format: "svg", filename: "acprof-comparison"}};
  async function plot(container, traces, specific) {
    await Plotly.react(container, traces, {...layout, ...specific}, plotOptions);
    container.removeAllListeners("plotly_click");
    container.on("plotly_click", event => {
      const id = event.points[0]?.customdata;
      const config = data.configs.find(c => c.config_id === id);
      if (config) selectConfig(config);
    });
  }
  function tooltip(config) {
    return `${escape(label(config))}<br>${escape(inputLabel(config.input_case, true))}<br>${escape(config.environment_class)} · ${escape(config.status)} · n=${config.valid_windows}<br>${escape(config.config_id)}`;
  }
  async function drawPareto(configs, comparison) {
    const x = $("x-metric").value, y = $("y-metric").value, size = $("size-metric").value;
    const visible = configs.filter(c => V.finite(V.value(c, x)) && V.finite(V.value(c, y)) &&
      (registry[x].scale !== "log" || V.value(c, x) > 0) && (registry[y].scale !== "log" || V.value(c, y) > 0));
    const frontier = new Set(V.pareto(visible, x, y, registry, comparison));
    const neutral = registry[x].direction === "neutral" || registry[y].direction === "neutral";
    const sizes = visible.map(c => V.value(c, size)).filter(v => V.finite(v) && v >= 0);
    const maxSize = Math.max(0, ...sizes);
    const markerSize = config => {
      const v = V.value(config, size);
      return size && V.finite(v) && v >= 0 && maxSize > 0 ? 8 + 25 * Math.sqrt(v / maxSize) : 11;
    };
    const traces = [];
    for (const key of colorKeys) {
      const group = visible.filter(c => `${c.runtime} / ${c.device}` === key);
      if (!group.length) continue;
      traces.push({type: "scatter", mode: "markers", name: escape(key),
        x: group.map(c => V.value(c, x)), y: group.map(c => V.value(c, y)),
        customdata: group.map(c => c.config_id), text: group.map(tooltip),
        hovertemplate: `%{text}<br>${escape(registry[x].label)}: %{x} ${escape(registry[x].unit)}<br>${escape(registry[y].label)}: %{y} ${escape(registry[y].unit)}<extra></extra>`,
        marker: {color: color(group[0]), size: group.map(markerSize), opacity: .82,
          symbol: group.map(c => c.device === "GPU" ? "square" : "circle"),
          line: {color: group.map(c => c.config_id === state.selected ? "#9c42c9" : "white"),
            width: group.map(c => c.config_id === state.selected ? 4 : 1)}}});
    }
    const front = visible.filter(c => frontier.has(c.config_id));
    if (front.length) traces.push({type: "scatter", mode: "markers", name: "Pareto frontier",
      x: front.map(c => V.value(c, x)), y: front.map(c => V.value(c, y)),
      customdata: front.map(c => c.config_id), text: front.map(tooltip),
      hovertemplate: "%{text}<br>Pareto frontier<extra></extra>",
      marker: {symbol: "diamond-open", color: "#122f42", size: front.map(c => markerSize(c) + 9), line: {width: 2}}});
    $("pareto-note").textContent = `${visible.length} 个有效点 · ${configs.length - visible.length} 个配置缺少坐标或不适用于当前刻度。` +
      (neutral ? "所选指标包含 neutral，仅展示 scatter。" : `${front.length} 个非支配点；按已核验条件分组，partial/failed、质量 blocked/unknown 和不满足 baseline 条件的配置不进入 frontier。`) +
      (size ? " 点面积随第三指标增加；该指标缺失时使用固定小点。" : "");
    await plot($("pareto-chart"), traces, {height: 510, xaxis: axis(x), yaxis: axis(y),
      annotations: visible.length ? [] : [{text: "所选指标没有成对的有效观测值", showarrow: false, xref: "paper", yref: "paper", x: .5, y: .5}]});
  }
  async function drawScaling(configs) {
    const resource = $("scaling-axis").value, metric = $("scaling-metric").value;
    for (const chart of $("scaling-charts").querySelectorAll(".scaling-chart")) Plotly.purge(chart);
    $("scaling-charts").replaceChildren();
    const facets = V.scaling(configs, resource, metric);
    if (!facets.length) $("scaling-charts").append(element("p", "没有记录该资源轴的配置；不会为历史数据推断 concurrency。"));
    for (const facet of facets) {
      const article = element("article"), chart = element("div", undefined, "scaling-chart");
      article.title = fullLabel(facet.config);
      article.append(element("h3", `${facet.config.model.split("/").pop()} / ${facet.config.runtime} · ${facet.config.environment_class} · ${inputLabel(facet.config.input_case)}`), chart);
      $("scaling-charts").append(article);
      const runs = [...new Set(facet.series.map(s => s.config.run_id))];
      const devices = [...new Set(facet.series.map(s => s.config.gpu))];
      const seriesLabel = series => {
        const held = series.held.filter(k => series.config[k] != null).map(k =>
          `${({cpu: "CPU", memory: "Mem", concurrency: "C"})[k]} ${series.config[k]}${k === "memory" ? " GiB" : ""}`);
        return `${series.config.device} · ${held.join(" · ")}` +
          (runs.length > 1 ? ` · run ${runs.indexOf(series.config.run_id) + 1}` : "") +
          (devices.filter(d => d !== "n/a").length > 1 && series.config.device === "GPU" ? ` · GPU ${devices.indexOf(series.config.gpu) + 1}` : "");
      };
      const traces = facet.series.map((series, index) => ({type: "scatter", mode: "lines+markers",
        name: escape(seriesLabel(series)),
        x: series.points.map(p => p.x), y: series.points.map(p => p.y), connectgaps: false,
        customdata: series.points.map(p => p.config.config_id), text: series.points.map(p => tooltip(p.config)),
        hovertemplate: `%{text}<br>${escape(resource)}: %{x}<br>${escape(metricTitle(metric))}: %{y}<extra></extra>`,
        line: {color: palette[index % palette.length], dash: series.config.device === "GPU" ? "solid" : "dot"},
        marker: {size: series.points.map(p => p.config.config_id === state.selected ? 13 : 7),
          symbol: series.config.device === "GPU" ? "square" : "circle"}}));
      const hasValues = facet.series.some(s => s.points.some(p => V.finite(p.y)));
      await plot(chart, traces, {height: 400, showlegend: true,
        margin: {l: 65, r: 20, t: 75, b: 60},
        legend: {orientation: "h", y: 1.22, font: {size: 10}},
        annotations: hasValues ? [] : [{text: "没有有效观测值", showarrow: false, xref: "paper", yref: "paper", x: .5, y: .5}],
        xaxis: {title: {text: $("scaling-axis").selectedOptions[0].text}, automargin: true, gridcolor: "#e8eef2"},
        yaxis: axis(metric)});
    }
  }
  function statisticsEvidence(report) {
    const article = element("article", undefined, "statistics-source");
    const parameters = report.parameters, audit = report.audit;
    article.append(element("h3", report.origin === "cached" ? "复用统计 JSON" : "从原始测量窗口重新计算"),
      element("p", report.source, "muted"));
    if (report.report_path) article.append(element("p", report.report_path, "muted"));
    article.append(element("p", `Bootstrap · ${parameters.method} · confidence=${parameters.confidence} · ` +
      `resamples=${parameters.resamples} · seed=${parameters.seed} · block_size=${parameters.block_size}`),
      element("p", `抽样单位：${parameters.resampling_unit} · 筛选：${parameters.filter}；推断行排除。${parameters.assumption}`),
      element("p", `数据质量：${audit.quality_status} · 运行状态：${audit.run_status} · 实验完整性：${audit.measurement_status}`));
    article.append(element("p", `结果审计：${audit.valid === true ? "通过" : audit.valid === false ? "存在问题" : "unknown"}` +
      ` · 诊断 ${(audit.issues || []).length} 项`));
    const coverage = audit.coverage;
    article.append(element("p", coverage ?
      `实验计划覆盖（全配置，含预热）：${coverage.actual} / ${coverage.expected} · 未写入 ${coverage.missing} · 计划外 ${coverage.unexpected}` :
      "计划覆盖：unknown（缺少可核验计划；已写入窗口成功不代表实验完整）"));
    for (const diagnostic of report.cache_diagnostics) {
      article.append(element("p", `未复用 ${diagnostic.path}：${reasonText(diagnostic.reason)}`, "muted"));
    }
    const details = element("details", undefined, "evidence");
    details.append(element("summary", "质量与完整性原始证据"), element("pre", JSON.stringify({
      result_sha256: report.result_sha256, ...audit,
    }, null, 2)));
    article.append(details);
    return article;
  }
  function drawStatistics(configs) {
    const visible = new Map(configs.map(config => [config.config_id, config]));
    const body = $("statistics").tBodies[0], evidence = $("statistics-evidence");
    body.replaceChildren(); evidence.replaceChildren();
    const metric = $("statistics-metric").value;
    $("statistics-confidence").textContent = [...new Set(statistics.map(report => report.parameters.confidence * 100))].join(" / ") + "% 置信区间";
    for (const report of statistics) {
      const groups = report.groups.filter(group => visible.has(group.config_id) && (!metric || group.metric === metric));
      if (!groups.length) continue;
      for (const group of groups) {
        const config = visible.get(group.config_id), row = element("tr");
        row.dataset.statConfigId = group.config_id;
        row.classList.toggle("selected", state.selected === group.config_id);
        row.onclick = () => selectConfig(config);
        row.tabIndex = 0;
        row.onkeydown = event => { if (event.key === "Enter") selectConfig(config); };
        const identity = element("td");
        identity.append(element("strong", label(config)), element("small", inputLabel(config.input_case, true)),
          element("small", `run=${config.run_id} · ${group.config_id.slice(-8)}`),
          element("small", `质量 ${config.quality_status} · 完整性 ${config.measurement_status}`));
        const name = element("td", registry[group.metric]?.label || group.metric);
        name.append(element("small", `${group.metric} · ${group.unit}`));
        const interval = V.finite(group.ci_low) && V.finite(group.ci_high) ? `[${fmt(group.ci_low)}, ${fmt(group.ci_high)}]` : "—";
        const reasons = element("td", reasonText(group.reason) || "已估计");
        reasons.append(element("small", reasonCounts(group.missing_reasons) || "无指标缺失"),
          element("small", `排除：${reasonCounts(group.excluded_windows) || "无"}`));
        if (group.detail) reasons.append(element("small", group.detail));
        row.append(identity, name, element("td", fmt(group.mean)), element("td", fmt(group.std)),
          element("td", interval), element("td", `${group.n_windows} / ${group.missing_windows}`), reasons);
        body.append(row);
      }
      evidence.append(statisticsEvidence(report));
    }
    $("statistics-empty").hidden = body.rows.length > 0;
  }
  function drawOverview(configs) {
    $("overview-content").replaceChildren();
    for (const source of data.overview || []) {
      const matches = configs.filter(config => config.source === source.source);
      if (!matches.length) continue;
      const article = element("article", undefined, "statistics-source");
      article.append(element("h3", matches[0].model), element("p", source.source),
        element("p", `${matches.length} 个配置 · ${matches.reduce((sum, config) => sum + config.valid_windows, 0)} 个有效窗口`));
      for (const [key, title] of [["model_revision", "Revision"], ["runtime_backend", "Runtime"],
        ["pipeline_tag", "任务"], ["input_scale_type", "输入单位"], ["cpu_model", "CPU"], ["gpu", "GPU"]]) {
        article.append(element("p", `${title}: ${text(source.identity[key])}`));
      }
      article.append(element("h3", "关键指标 · 按配置展示，不合并不同资源条件"));
      const wrap = element("div", undefined, "table-scroll"), table = element("table");
      const head = element("thead"), titles = element("tr"), body = element("tbody");
      const metrics = ["latency_app_p95_s", "throughput_samples_per_s", "observed_energy_per_request_j"];
      titles.append(element("th", "配置 / 输入"));
      for (const metric of metrics) titles.append(element("th", metricTitle(metric)));
      head.append(titles);
      for (const config of matches) {
        const row = element("tr"); row.dataset.configId = config.config_id;
        row.append(element("td", fullLabel(config)));
        for (const metric of metrics) row.append(element("td", fmt(V.value(config, metric))));
        row.onclick = () => selectConfig(config); body.append(row);
      }
      table.append(head, body); wrap.append(table); article.append(wrap);
      $("overview-content").append(article);
    }
  }
  function drawQuality(configs) {
    $("quality-content").replaceChildren();
    const visible = new Set(configs.map(config => config.config_id));
    for (const report of statistics) {
      const groups = report.groups.filter(group => visible.has(group.config_id));
      if (!groups.length) continue;
      $("quality-content").append(statisticsEvidence(report));
      const details = element("details", undefined, "evidence");
      details.append(element("summary", "缺失字段与排除窗口明细"), element("pre", JSON.stringify(groups.map(group => ({
        config_id:group.config_id, metric:group.metric, missing_reasons:group.missing_reasons,
        excluded_windows:group.excluded_windows, reason:group.reason, detail:group.detail,
      })), null, 2)));
      $("quality-content").append(details);
    }
  }
  async function draw() {
    const configs = filtered();
    const purpose = $("comparison-purpose").value;
    const baseline = data.configs.find(config => config.config_id === $("baseline").value);
    const comparison = V.comparisonContext(baseline, purpose, data.baseline_comparisons);
    const qualifications = configs.map(config => comparison.qualification(config));
    $("comparison-note").textContent = `用途：${purpose} · ` + ["descriptive", "compatible", "incompatible", "unknown"]
      .map(status => `${status}: ${qualifications.filter(item => item.status === status).length}`).join(" · ") +
      (purpose === "resource-scaling" ? "。允许 CPU / memory 配额变化；双轴同时变化不能归因某一个轴。" : "。") +
      (purpose === "within-experiment" ? "同一来源、任务及输入条件的描述性比较；历史元数据缺失不证明实验独立性。多资源变化不能单独归因。" : "") +
      (baseline ? "比值=当前/基线；性能比按指标方向计算，>1 表示改善，neutral 不评级。" : "选择 baseline 后显示成对检查原因。") +
      [...new Set(qualifications.flatMap(item => item.reasons))].join(", ");
    $("counts").replaceChildren(element("strong", String(configs.length)), document.createTextNode(`/ ${data.configs.length} 配置`));
    const selected = data.configs.find(c => c.config_id === state.selected);
    $("quality-detail").textContent = selected ? JSON.stringify({run_status: selected.run_status,
      measurement_status: selected.measurement_status, comparison: comparison.qualification(selected),
      comparison_profile: selected.comparison_profiles[purpose],
      quality_status: selected.quality_status, auto_selection_eligible: selected.auto_selection_eligible,
      quality_reasons: selected.quality_reasons, quality_checks: selected.quality_checks}, null, 2) : "选择配置后查看状态、原因和原始证据。";
    $("selection").textContent = selected ? `已选：${fullLabel(selected)}${configs.includes(selected) ? "" : "（当前筛选隐藏）"}` :
      "点击表格行或图中数据点，可在各视图中定位同一配置。";
    for (const view of ["overview", "matrix", "pareto", "scaling", "statistics", "experiments", "quality"]) $(view + "-panel").hidden = state.view !== view;
    if (state.view === "matrix") drawMatrix(configs, comparison);
    else if (state.view === "pareto") await drawPareto(configs, comparison);
    else if (state.view === "scaling") await drawScaling(configs);
    else if (state.view === "statistics") drawStatistics(configs);
    else if (state.view === "experiments") await experiments.draw(configs);
    else if (state.view === "overview") drawOverview(configs);
    else drawQuality(configs);
    document.body.dataset.ready = "true";
  }
  let pending = Promise.resolve();
  function schedule() {
    document.body.dataset.busy = "true";
    pending = pending.then(draw).catch(error => {
      $("error").hidden = false;
      $("error").textContent = "视图生成失败：" + error.message;
    }).finally(() => { document.body.dataset.busy = "false"; });
  }
  const experiments = window.ACProfExperiments({data, $, V, element, option, fmt, metricTitle, plot, axis, schedule, filtered, escape});
  window.ACProfExport({data, $, V, state, filtered, experiments});
  schedule();
})();
