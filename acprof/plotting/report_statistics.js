/* Apply statistical parameters atomically; cancelled jobs never replace displayed results. */
"use strict";
window.ACProfStatisticsControls = ({workspace, $, option, refresh, schedule}) => {
  const data = workspace.data, engine = window.ACProfStatistics, importedInputs = new Map();
  const defaults = {confidence:.95, resamples:5000, seed:0, block_size:1,
    metrics:["latency_app_s", "latency_s", "container_attributed_energy_eff_j"]};
  const key = report => JSON.stringify([report.source, report.result_sha256]);
  let controller = null;
  function operation(message) {
    for (const prefix of ["statistics", "experiment"]) $(prefix + "-operation").textContent = message;
  }
  function progress(value) {
    for (const prefix of ["statistics", "experiment"]) $(prefix + "-progress").value = value;
  }
  const catalog = () => Object.assign({}, ...data.window_statistics.map(report => report.recalculation?.metrics ||
    Object.fromEntries(report.groups.map(group => [group.metric, {label:group.metric, unit:group.unit}]))));
  function selection() {
    const result = {metrics:[...$("statistics-metrics").selectedOptions].map(item => item.value)};
    for (const [field, id] of [["confidence", "level"], ["resamples", "resamples"], ["seed", "seed"], ["block_size", "block-size"]]) {
      if (!$("statistics-" + id).value.trim()) throw new Error("请填写统计参数 / Complete the statistics parameters");
      result[field] = Number($("statistics-" + id).value);
    }
    engine.validate(result);
    if (!result.metrics.length) throw new Error("至少选择一个分析指标 / Select at least one metric");
    return result;
  }
  function setParameters(parameters) {
    for (const item of $("statistics-metrics").options) item.selected = parameters.metrics.includes(item.value);
    for (const [field, id] of [["confidence", "level"], ["resamples", "resamples"], ["seed", "seed"], ["block_size", "block-size"]]) {
      const input = $("statistics-" + id), value = parameters[field];
      if (input.tagName === "SELECT") {
        const existing = [...input.options].find(item => Number(item.value) === value);
        if (!existing) option(input, String(value), field === "confidence" ? `${value * 100}%` : String(value));
        input.value = existing?.value || String(value);
      } else input.value = String(value);
    }
  }
  function initialParameters() {
    const reports = workspace.initialStatistics(), first = reports[0]?.parameters;
    const uniform = first && reports.every(report => ["confidence", "resamples", "seed", "block_size"].every(name => report.parameters[name] === first[name]));
    return {...defaults, ...(uniform ? first : {}), metrics:reports.length ? [...new Set(reports.flatMap(report => report.groups.map(group => group.metric)))] : defaults.metrics};
  }
  function update() {
    const previous = [...$("statistics-metrics").selectedOptions].map(item => item.value);
    $("statistics-metrics").replaceChildren();
    for (const [name, metric] of Object.entries(catalog())) option($("statistics-metrics"), name, `${metric.label} (${metric.unit})`);
    for (const item of $("statistics-metrics").options) item.selected = previous.includes(item.value);
    $("statistics-input-status").textContent = data.window_statistics.filter(report =>
      report.recalculation?.status === "external_required" && !importedInputs.has(key(report))).map(report =>
      `窗口数据超过内嵌预算；重算前导入 ${report.recalculation.file || "windows.json"} (${(report.recalculation.bytes / 1024**2).toFixed(1)} MiB)。 / Import the window JSON before recalculating.`).join("\n");
  }
  function busy(value) {
    document.body.dataset.calculating = String(value);
    for (const prefix of ["statistics", "experiment"]) {
      $(prefix + "-progress").hidden = !value; $(prefix + "-cancel").hidden = !value;
    }
    for (const id of ["statistics-recalculate", "experiment-recalculate", "statistics-defaults", "statistics-restore", "statistics-input-file",
      "statistics-metrics", "statistics-level", "statistics-resamples", "statistics-seed", "statistics-block-size", "export-json", "export-csv", "export-html"]) $(id).disabled = value;
  }
  function matches(report, options, independent = false) {
    const parameters = independent ? report : report.parameters;
    return ["confidence", "resamples", "seed", ...(independent ? [] : ["block_size"])].every(name => parameters[name] === options[name]) &&
      options.metrics.every(name => report.groups.some(group => group.metric === name));
  }
  async function calculate() {
    if (controller) return;
    const job = new AbortController(), revision = workspace.revision();
    controller = job; busy(true); progress(0);
    try {
      const options = selection(), windows = [], comparisons = [];
      const jobs = Math.max(1, data.window_statistics.reduce((n, report) => n + (report.recalculation?.groups?.length || report.groups.length), 0) +
        data.experiment_comparisons.reduce((n, entry) => n + (entry.report?.groups.length || 0), 0));
      let done = 0;
      const hooks = {signal:job.signal, onProgress:fraction => {
        progress(Math.min(.99, (done + fraction) / jobs));
        operation(`正在计算 / Calculating: ${Math.round($("statistics-progress").value * 100)}%`);
      }};
      operation("正在计算 / Calculating…");
      for (const current of data.window_statistics) {
        const initial = workspace.initialStatistics().find(report => key(report) === key(current));
        if (initial && initial.engine !== "browser" && matches(initial, options)) {
          windows.push({...initial, groups:initial.groups.filter(group => options.metrics.includes(group.metric))});
          continue;
        }
        const inputs = importedInputs.get(key(current)) || current.recalculation?.groups;
        if (!inputs) throw new Error(`缺少重算窗口，请导入窗口 JSON 或重新生成 HTML / Missing recalculation inputs: ${current.source}`);
        const groups = [];
        for (const input of inputs.filter(group => options.metrics.includes(group.metric))) {
          groups.push(await engine.summarize(input, options, hooks)); done++;
          if (done % 8 === 0) await new Promise(resolve => setTimeout(resolve, 0));
        }
        windows.push({...current, groups, origin:"browser", engine:engine.engine, rng:engine.rng, report_path:null,
          parameters:{...current.parameters, confidence:options.confidence, resamples:options.resamples,
            seed:options.seed, block_size:options.block_size}});
      }
      for (const entry of workspace.initialComparisons()) {
        if (!entry.report) { comparisons.push(entry); continue; }
        const initial = entry.report, selected = initial.groups.filter(group => options.metrics.includes(group.metric));
        const reuse = initial.engine !== "browser" && matches(initial, options, true);
        const groups = [];
        for (const input of selected) {
          groups.push(reuse ? input : await engine.independent(input, options, hooks)); done++;
        }
        comparisons.push({...entry, report:{...initial, groups,
          confidence:options.confidence, resamples:options.resamples, seed:options.seed,
          engine:reuse ? "python" : engine.engine, origin:reuse ? (initial.origin || "computed") : "browser", rng:engine.rng,
          unavailable_metrics:options.metrics.filter(metric => !initial.groups.some(group => group.metric === metric))}});
      }
      if (job.signal.aborted) throw new DOMException("已取消 / Cancelled", "AbortError");
      if (workspace.revision() !== revision) throw new Error("实验列表已变化，请重新计算 / Experiments changed; recalculate");
      workspace.setStatistics(windows, comparisons);
      progress(1);
      operation("计算完成；表格、图表与导出已更新。 / Calculation complete; tables, charts and exports updated.");
      refresh(); schedule();
    } catch (error) {
      operation(error.name === "AbortError" ? "计算已取消，保留上次结果。 / Cancelled; previous results retained." :
        "计算失败，保留上次结果 / Calculation failed; previous results retained: " + error.message);
    } finally { controller = null; busy(false); }
  }
  $("statistics-recalculate").onclick = calculate;
  $("experiment-recalculate").onclick = calculate;
  for (const prefix of ["statistics", "experiment"]) $(prefix + "-cancel").onclick = () => controller?.abort();
  $("experiment-parameters").onclick = () => {
    document.querySelector('[data-view="statistics"]').click(); $("statistics-parameters").open = true;
    $("statistics-parameters").scrollIntoView({block:"start"});
  };
  $("statistics-defaults").onclick = () => {
    setParameters(defaults);
    operation("已恢复默认参数；点击重新计算以应用。 / Defaults restored; recalculate to apply.");
  };
  $("statistics-restore").onclick = () => {
    workspace.restoreStatistics(); refresh(); setParameters(initialParameters());
    operation("已恢复 HTML 初始统计结果。 / Initial statistics restored.");
    schedule();
  };
  for (const id of ["metrics", "level", "resamples", "seed", "block-size"]) $("statistics-" + id).onchange = () => {
    operation("参数已修改；点击重新计算。表格仍显示上次应用的结果。 / Parameters changed; recalculate to apply. Showing previous results.");
  };
  $("statistics-input-file").onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      if (file.size > 128 * 1024**2) throw new Error("窗口 JSON 超过 128 MiB，请使用 Python CLI / Input exceeds 128 MiB; use Python CLI");
      const content = await file.text(), digest = sha256(content);
      const reports = data.window_statistics.filter(report => report.recalculation?.sha256 === digest);
      if (!reports.length) throw new Error("窗口文件 SHA256 不匹配 / Window file SHA256 mismatch");
      const input = JSON.parse(content);
      if (input.kind !== "acprof_window_inputs" || input.schema_version !== 1 || !Array.isArray(input.groups)) throw new Error("窗口格式无效 / Invalid window format");
      for (const report of reports) {
        if (input.result_sha256 !== report.result_sha256) throw new Error("来源不匹配 / Source mismatch");
        const configs = data.configs.filter(config => config.source === report.source);
        const groups = input.groups.flatMap(group => {
          const config = configs.find(config => (config.original_config_id || config.config_id) === group.config_id);
          return config ? [{...group, config_id:config.config_id}] : [];
        });
        if (!groups.length && report.groups.length) throw new Error("配置不匹配 / Configuration mismatch");
        importedInputs.set(key(report), groups);
      }
      update(); operation("窗口文件校验通过，可重新计算。 / Window file verified; ready to recalculate.");
    } catch (error) { operation("导入失败 / Import failed: " + error.message); }
    finally { event.target.value = ""; }
  };
  update(); setParameters(initialParameters()); busy(false);
  return {update, parameters:selection};
};
