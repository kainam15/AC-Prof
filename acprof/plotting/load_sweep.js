/* Offline point evidence only; no interpolation, pooled statistics or capacity ranking. */
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("sweep-data").textContent);
  const byId = id => document.getElementById(id);
  const metrics = {
    successful_requests_per_s: ["Successful requests / s", "req/s"],
    goodput_requests_per_s: ["SLO Goodput / s", "req/s"],
    output_tokens_per_s: ["Complete output tokens / s", "token/s"],
    ttft_p99_s: ["P99 TTFT", "s"],
    tpot_p99_s: ["P99 TPOT", "s/token"],
    e2e_p99_s: ["P99 E2E", "s"],
  };
  const states = {valid: "完整运行", partial: "部分证据", not_run: "未运行", missing: "文件缺失", invalid: "证据无效"};
  const colors = ["#176a77", "#b45628", "#6750a4", "#2f7c4c", "#b33f67", "#357abd"];
  const groupNames = new Map(data.conditions.map((condition, index) => [condition.condition_id, `G${index + 1}`]));
  const fmt = value => value == null ? "不可用" : Number(value).toLocaleString("en-US", {maximumSignificantDigits: 5});
  const percent = value => value == null ? "不可用" : `${fmt(value * 100)}%`;
  function element(tag, text, parent) {
    const item = document.createElement(tag);
    item.textContent = text;
    if (parent) parent.append(item);
    return item;
  }
  function option(select, value, label) {
    const item = element("option", label, select);
    item.value = value;
  }
  for (const [key, [label]] of Object.entries(metrics)) option(byId("metric"), key, label);
  for (const condition of data.conditions) {
    const name = groupNames.get(condition.condition_id);
    option(byId("condition"), condition.condition_id, name);
    const box = element("details", "", byId("conditions"));
    const probe = condition.service_evidence.probe;
    element("summary", `${name} · ${condition.identity.model || "模型未声明"} · probe ${probe.status || "unknown"}`, box);
    element("pre", JSON.stringify(condition, null, 2), box);
  }
  byId("sources").textContent = JSON.stringify(data.sources, null, 2);
  byId("summary").textContent = `${data.point_count} 个负载点 · ${data.valid_points} 个完整运行 · ${data.partial_points} 个部分证据 · ${data.unavailable_points} 个无可用指标 · manifest ${data.status}`;
  byId("diagnostics").textContent = data.diagnostics.join("\n");
  if (data.points.some(point => point.load_config.rate != null)) byId("axis").value = "rate";

  function selectedPoints() {
    const condition = byId("condition").value;
    return data.points.filter(point => condition === "all" || point.condition_id === condition || point.condition_id == null);
  }
  function renderTable() {
    byId("points").replaceChildren();
    for (const point of selectedPoints()) {
      const tr = element("tr", "", byId("points"));
      tr.dataset.evidence = point.evidence_status;
      const config = point.load_config, m = point.metrics, c = point.counts;
      element("td", `${point.point_id} / ${groupNames.get(point.condition_id) || "未确认"}`, tr);
      element("td", `concurrency ${config.concurrency}\nrate ${fmt(config.rate)}`, tr);
      element("td", `${states[point.evidence_status]}\nmanifest: ${point.status}\nreport: ${point.report_status || "不可用"}`, tr);
      element("td", c ? `${c.succeeded} / ${c.offered} (${fmt(point.success_rate * 100)}%)\nerror ${c.failed} · drop ${c.dropped} · cancel ${c.cancelled}` : "不可用", tr);
      for (const name of ["successful_requests_per_s", "goodput_requests_per_s", "output_tokens_per_s"]) element("td", fmt(m[name]), tr);
      element("td", [m.ttft_p99_s, m.tpot_p99_s, m.e2e_p99_s].map(fmt).join(" / "), tr);
      const samples = point.latency_samples;
      element("td", samples ? `TTFT ${samples.ttft_s} · TPOT ${samples.tpot_s} · E2E ${samples.e2e_s}\ntokens ${percent(point.output_token_coverage)}\nduration ${fmt(point.duration_s)} s` : "不可用", tr);
      const detail = element("td", point.diagnostics.join("\n"), tr);
      if (point.slo?.configured) element("div", `SLO: met ${point.slo.counts.satisfied} · violated ${point.slo.counts.violated} · unevaluable ${point.slo.counts.unevaluable}`, detail);
      if (point.report_href) {
        const link = element("a", "load.json", detail);
        link.href = point.report_href;
      }
    }
  }

  function renderChart() {
    const axis = byId("axis").value, metric = byId("metric").value;
    const fixed = axis === "concurrency" ? "rate" : "concurrency";
    const selection = byId("condition").value;
    const series = new Map();
    for (const point of data.points) {
      if (!point.condition_id || point.load_config[axis] == null || (selection !== "all" && point.condition_id !== selection)) continue;
      const key = JSON.stringify([point.condition_id, point.load_config[fixed]]);
      if (!series.has(key)) series.set(key, {condition: point.condition_id, fixed: point.load_config[fixed], points: []});
      series.get(key).points.push(point);
    }
    const traces = [];
    let plotted = 0;
    for (const seriesData of series.values()) {
      // Include coordinates from every condition and unavailable point. A missing or changed
      // condition interrupts the line at that coordinate instead of bridging evidence gaps.
      const coordinates = [...new Set(data.points.filter(point => point.load_config[fixed] === seriesData.fixed)
        .map(point => point.load_config[axis]).filter(value => value != null))].sort((a, b) => a - b);
      const points = coordinates.map(value => seriesData.points.find(point => point.load_config[axis] === value));
      const ys = points.map(point => point?.curve_eligible ? (point.metrics[metric] ?? null) : null);
      const group = groupNames.get(seriesData.condition);
      const color = colors[(Number(group.slice(1)) - 1) % colors.length];
      const label = `${group} · ${fixed}=${seriesData.fixed == null ? "n/a" : seriesData.fixed}`;
      plotted += ys.filter(value => value != null).length;
      const hover = point => point ? `${point.point_id} · ${states[point.evidence_status]}<br>success ${fmt(point.success_rate * 100)}% · error ${point.counts.failed} · drop ${point.counts.dropped}<br>TTFT/TPOT/E2E samples ${point.latency_samples.ttft_s}/${point.latency_samples.tpot_s}/${point.latency_samples.e2e_s}` : "";
      traces.push({type: "scatter", mode: "lines+markers", name: label, legendgroup: label,
        x: coordinates, y: ys, connectgaps: false, text: points.map(hover),
        marker: {color, size: 9, symbol: points.map(point => point?.counts?.failed || point?.counts?.dropped ? "x" : "circle")},
        line: {color, width: 2}, hovertemplate: "%{text}<br>x=%{x}<br>y=%{y}<extra>%{fullData.name}</extra>"});
      const partial = seriesData.points.filter(point => !point.curve_eligible && point.metrics[metric] != null);
      if (partial.length) {
        plotted += partial.length;
        traces.push({type: "scatter", mode: "markers", name: `${label} · partial`, legendgroup: label,
          x: partial.map(point => point.load_config[axis]), y: partial.map(point => point.metrics[metric]),
          marker: {color, size: 11, symbol: "diamond-open"}, text: partial.map(hover),
          hovertemplate: "%{text}<br>x=%{x}<br>y=%{y}<extra>%{fullData.name}</extra>"});
      }
    }
    byId("chart-status").textContent = `${plotted} 个点具有所选指标。${plotted ? "" : "检查指标覆盖、负载轴或条件组；缺失值不会填成 0。"}`;
    const layout = {paper_bgcolor: "#fff", plot_bgcolor: "#fff", font: {family: "system-ui, sans-serif", color: "#172b3b"},
      margin: {l: 72, r: 25, t: 18, b: 110}, hovermode: "closest", showlegend: true,
      xaxis: {title: {text: axis === "rate" ? "Target rate (req/s)" : "Client concurrency"}, gridcolor: "#e6edf2"},
      yaxis: {title: {text: `${metrics[metric][0]} (${metrics[metric][1]})`}, rangemode: "tozero", gridcolor: "#e6edf2"},
      legend: {orientation: "h", y: -0.25, x: 0},
      annotations: plotted ? [] : [{text: "所选指标暂无可绘制证据", x: 0.5, y: 0.5, xref: "paper", yref: "paper", showarrow: false}]};
    return Plotly.react(byId("chart"), traces, layout, {responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ["sendDataToCloud", "select2d", "lasso2d"]});
  }
  function render() {
    document.documentElement.dataset.reportReady = "false";
    renderTable();
    renderChart().then(() => { document.documentElement.dataset.reportReady = "true"; }).catch(error => {
      byId("chart-status").textContent = `图表渲染失败：${error.message}`;
    });
  }
  for (const id of ["axis", "metric", "condition"]) byId(id).addEventListener("change", render);
  render();
})();
