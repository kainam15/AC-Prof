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
    ["experiment_batch", "Experiment batch"], ["quality_status", "质量状态"],
  ];
  const text = value => value == null ? "unknown" : String(value);
  const escape = value => text(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const fmt = value => V.finite(value) ? value.toLocaleString("en-US", {maximumSignificantDigits: 5}) : "—";
  const signed = value => (value > 0 ? "+" : "") + fmt(value);
  const direction = metric => ({higher: "↑", lower: "↓", neutral: "·"})[registry[metric].direction];
  const metricTitle = metric => `${registry[metric].label} (${registry[metric].unit})`;
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
      const chosen = $("filter-" + field).value;
      return !chosen || chosen === text(config[field]);
    }));
  }
  function drawMatrix(configs) {
    const purpose = $("comparison-purpose").value;
    const metrics = $("preset").value === "Core" ? data.default_matrix :
      [...$("metrics").selectedOptions].map(item => item.value);
    const baseline = data.configs.find(config => config.config_id === $("baseline").value);
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
      const qualification = V.qualification(config, baseline, purpose);
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
        element("small", config.auto_selection_eligible ? "质量可参与优选" : `优选暂停：${config.quality_reasons.join(", ")}`),
        element("small", `${config.model.split("/").pop()} · ${config.runtime}`),
        element("small", `${inputLabel(config.input_case)} · n=${config.valid_windows}`));
      row.append(identity);
      for (const name of metrics) {
        const value = V.value(config, name), cell = element("td", fmt(value));
        const score = config.auto_selection_eligible && qualification.status === "compatible" ?
          V.score(value, ranges.get(V.cohort(config, purpose))?.[name], registry[name]) : null;
        if (score != null) cell.style.backgroundColor = `hsl(${12 + 140 * score} 43% 91%)`;
        const stats = config.metrics[name];
        cell.title = `${name} · ${registry[name].unit} · ${stats.aggregation} · n=${stats.n} · missing=${stats.missing}`;
        if (baseline) {
          const change = V.delta(config, baseline, name, purpose);
          const content = change.delta == null ? (change.reasons ? `${change.reason}: ${change.reasons.join(", ")}` : "— vs baseline") :
            `Δ ${signed(change.delta)} ${registry[name].unit} · ${change.percent == null ? "% —（baseline=0）" : signed(change.percent) + "%"}`;
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
  async function drawPareto(configs) {
    const x = $("x-metric").value, y = $("y-metric").value, size = $("size-metric").value;
    const visible = configs.filter(c => V.finite(V.value(c, x)) && V.finite(V.value(c, y)) &&
      (registry[x].scale !== "log" || V.value(c, x) > 0) && (registry[y].scale !== "log" || V.value(c, y) > 0));
    const purpose = $("comparison-purpose").value;
    const baseline = data.configs.find(config => config.config_id === $("baseline").value);
    const frontier = new Set(V.pareto(visible, x, y, registry, purpose, baseline));
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
  async function draw() {
    const configs = filtered();
    const purpose = $("comparison-purpose").value;
    const baseline = data.configs.find(config => config.config_id === $("baseline").value);
    const qualifications = configs.map(config => V.qualification(config, baseline, purpose));
    $("comparison-note").textContent = `用途：${purpose} · ` + ["compatible", "incompatible", "unknown"]
      .map(status => `${status}: ${qualifications.filter(item => item.status === status).length}`).join(" · ") +
      (purpose === "resource-scaling" ? "。允许 CPU / memory 配额变化；双轴同时变化不能归因某一个轴。" : "。") +
      (baseline ? "以所选 baseline 检查条件；不兼容或证据不足时不展示改善比例。" : "选择 baseline 后显示成对检查原因。") +
      [...new Set(qualifications.flatMap(item => item.reasons))].join(", ");
    $("counts").replaceChildren(element("strong", String(configs.length)), document.createTextNode(`/ ${data.configs.length} 配置`));
    const selected = data.configs.find(c => c.config_id === state.selected);
    $("quality-detail").textContent = selected ? JSON.stringify({run_status: selected.run_status,
      measurement_status: selected.measurement_status, comparison: V.qualification(selected, baseline, purpose),
      comparison_profile: selected.comparison_profiles[purpose],
      quality_status: selected.quality_status, auto_selection_eligible: selected.auto_selection_eligible,
      quality_reasons: selected.quality_reasons, quality_checks: selected.quality_checks}, null, 2) : "选择配置后查看状态、原因和原始证据。";
    $("selection").textContent = selected ? `已选：${fullLabel(selected)}${configs.includes(selected) ? "" : "（当前筛选隐藏）"}` :
      "点击表格行或图中数据点，可在三个视图中定位同一配置。";
    for (const view of ["matrix", "pareto", "scaling"]) $(view + "-panel").hidden = state.view !== view;
    if (state.view === "matrix") drawMatrix(configs);
    else if (state.view === "pareto") await drawPareto(configs);
    else await drawScaling(configs);
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
  schedule();
})();
