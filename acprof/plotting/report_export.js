/* Package already-computed values; CSV/JSON export never estimates statistics. */
"use strict";
window.ACProfExport = ({data, $, V, state, filtered, experiments}) => {
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
    if (state.view === "statistics") {
      const reports = selectedStatistics(configs);
      return {payload:reports.length === 1 ? reports[0] : {kind:"window_statistics_collection", reports},
        rows:reports.flatMap(report => report.groups.map(group => ({...group,
          analysis:Object.fromEntries(Object.entries(report).filter(([key]) => key !== "groups"))})))};
    }
    const baseline = data.configs.find(config => config.config_id === $("baseline").value);
    const comparison = V.comparisonContext(baseline, $("comparison-purpose").value, data.baseline_comparisons);
    const metrics = $("preset").value === "Core" ? data.default_matrix : [...$("metrics").selectedOptions].map(option => option.value);
    const rows = configs.flatMap(config => metrics.map(metric => ({config_id:config.config_id, model:config.model,
      run_id:config.run_id, source:config.source, input_case:config.input_case, cpu:config.cpu, memory:config.memory,
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
    let content = JSON.stringify(payload, null, 2);
    if (format === "csv") {
      const flat = rows.map(row => flatten(row)), fields = [...new Set(flat.flatMap(row => Object.keys(row)))];
      const quote = value => '"' + String(value ?? "").replace(/"/g, '""') + '"';
      content = [fields.map(quote).join(","), ...flat.map(row => fields.map(field => quote(row[field])).join(","))].join("\r\n");
    }
    const url = URL.createObjectURL(new Blob([content], {type:format === "json" ? "application/json" : "text/csv;charset=utf-8"}));
    const link = document.createElement("a");
    link.href = url; link.download = `acprof-${state.view}.${format}`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  $("export-json").onclick = () => download("json");
  $("export-csv").onclick = () => download("csv");
};
