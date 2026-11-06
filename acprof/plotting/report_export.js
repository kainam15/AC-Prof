/* Package already-computed values; CSV/JSON export never estimates statistics. */
"use strict";
window.ACProfExport = ({data, $, V, state, filtered, experiments, workspace}) => {
  function selectedStatistics(configs) {
    const visible = new Set(configs.map(config => config.config_id));
    return (data.window_statistics || []).map(report => ({schema_version:1, ...report.parameters, ...report.audit,
      result_csv:report.source, result_sha256:report.result_sha256,
      groups:report.groups.filter(group => visible.has(group.config_id) &&
        (!$("statistics-metric").value || group.metric === $("statistics-metric").value))})).filter(report => report.groups.length);
  }
  function current() {
    const configs = filtered();
    if (state.view === "experiments" && experiments.current()?.report) {
      const payload = {...experiments.current().report, groups:experiments.groups()};
      return {payload, rows:payload.groups.map(group => ({...group, analysis:{confidence:payload.confidence,
        resamples:payload.resamples, seed:payload.seed, resampling_unit:payload.resampling_unit, source_sha256:payload.source_sha256}}))};
    }
    if (state.view === "experiments" && !experiments.current()?.report) {
      const rows = experiments.descriptions();
      return {payload:{kind:"browser_descriptive_comparison", statistics_status:"not_computed", groups:rows}, rows};
    }
    if (state.view === "statistics") {
      const reports = selectedStatistics(configs);
      return {payload:reports.length === 1 ? reports[0] : {kind:"window_statistics_collection", reports},
        rows:reports.flatMap(report => report.groups.map(group => ({...group,
          analysis:Object.fromEntries(Object.entries(report).filter(([key]) => key !== "groups"))})))};
    }
    const baseline = data.configs.find(config => config.config_id === $("baseline").value);
    const comparison = V.comparisonContext(baseline, $("comparison-purpose").value, data.baseline_comparisons, data.registry);
    const metrics = state.view === "pareto" ? [...new Set([$("x-metric").value, $("y-metric").value, $("size-metric").value].filter(Boolean))] :
      state.view === "scaling" ? [$("scaling-metric").value] :
      $("preset").value === "Core" ? data.default_matrix : [...$("metrics").selectedOptions].map(option => option.value);
    const rows = configs.flatMap(config => metrics.map(metric => ({config_id:config.config_id, model:config.model,
      run_id:config.run_id, statistics_status:config.statistics_status || "python_computed", source:config.source, experiment:config.experiment_name, input_case:config.input_case, cpu:config.cpu, memory:config.memory,
      device:config.device, metric, unit:data.registry[metric].unit, evidence:data.registry[metric].source,
      ...config.metrics[metric], baseline:baseline?.config_id || null, ...comparison.delta(config, metric),
      comparison:comparison.qualification(config)})));
    return {payload:{kind:"resource_configuration_analysis", comparison_purpose:comparison.purpose, groups:rows,
      sources:data.sources.filter(source => configs.some(config => config.source === source.path))}, rows};
  }
  function flatten(record, prefix = "", output = {}) {
    for (const [key, value] of Object.entries(record)) {
      if (value && typeof value === "object" && !Array.isArray(value)) flatten(value, prefix + key + ".", output);
      else output[prefix + key] = Array.isArray(value) ? JSON.stringify(value) : value;
    }
    return output;
  }
  function download(format) {
    const {payload, rows} = current();
    payload.analysis_bundle = workspace.bundle(filtered());
    payload.uncomputed_statistics = filtered().filter(config => config.statistics_status === "not_computed").map(config => config.config_id);
    if (state.view === "statistics" && payload.uncomputed_statistics.length) payload.statistics_status = "not_computed_for_csv";
    let content = JSON.stringify(payload, null, 2);
    if (format === "csv") {
      const flat = rows.map(row => flatten(row)), fields = [...new Set(flat.flatMap(row => Object.keys(row)))];
      const quote = value => {
        let text = String(value ?? "");
        if (typeof value === "string" && /^[=+\-@\t\r\n]/.test(text)) text = "'" + text;
        return '"' + text.replace(/"/g, '""') + '"';
      };
      content = [fields.map(quote).join(","), ...flat.map(row => fields.map(field => quote(row[field])).join(","))].join("\r\n");
    }
    window.ACProfDownload(content, format === "json" ? "application/json" : "text/csv;charset=utf-8", `acprof-${state.view}.${format}`);
  }
  $("export-json").onclick = () => download("json");
  $("export-csv").onclick = () => download("csv");
};

window.ACProfDownload = (content, type, name) => {
  const url = URL.createObjectURL(new Blob([content], {type})), link = document.createElement("a");
  link.href = url; link.download = name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
