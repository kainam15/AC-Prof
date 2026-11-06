/* Local files only. CSV summaries are descriptive; all uncertainty stays in Python. */
"use strict";
window.ACProfImport = (() => {
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const number = value => value == null || String(value).trim() === "" ? null :
    (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(String(value).trim()) && Number.isFinite(Number(value)) ? Number(value) : null);
  const text = value => value == null || ["", "none", "null", "nan"].includes(String(value).trim().toLowerCase()) ? "unknown" : String(value).trim();
  const canonical = value => "{" + Object.keys(value).sort().map(key => JSON.stringify(key) + ": " + JSON.stringify(value[key])).join(", ") + "}";
  const fail = message => { throw new Error(message); };
  const checkCancelled = signal => { if (signal?.aborted) throw new DOMException("已取消 / Cancelled", "AbortError"); };
  const clone = value => JSON.parse(JSON.stringify(value));
  function accumulator(registry) {
    return Object.fromEntries(Object.keys(registry).map(name => [name, {n:0, sum:0, correction:0, max:null, weight:0, weighted:0, badWeight:false, startups:new Map(), conflict:false}]));
  }
  async function readCsv(file, registry, {signal, onProgress = () => {}, chunkSize = 256 * 1024} = {}) {
    const groups = new Map(), seen = new Set(), hash = sha256.create(), decoder = new TextDecoder("utf-8", {fatal:true});
    let fields, tail = "", newline, rowNumber = 0;
    const required = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "repeat_idx", "warmup", "status"];
    function row(values) {
      if (values.every(value => !value.trim())) return;
      rowNumber++;
      if (!fields) {
        fields = values.map(value => value.trim().replace(/^\uFEFF/, ""));
        if (fields.some(name => !name) || new Set(fields).size !== fields.length) fail("CSV 表头为空或重复 / Empty or duplicate CSV columns");
        const missing = required.filter(name => !fields.includes(name));
        if (missing.length) fail("CSV 缺失字段 / Missing fields: " + missing.join(", "));
        if (!fields.some(name => registry[name] || ["cpu_energy_total_j", "gpu_energy_total_j"].includes(name))) fail("CSV 缺失已知指标 / No supported metrics");
        return;
      }
      if (values.length !== fields.length) fail(`CSV 第 ${rowNumber} 行列数不匹配 / Column count mismatch`);
      const raw = Object.fromEntries(fields.map((field, index) => [field, values[index].trim()]));
      const nums = Object.fromEntries(["cpu_cores", "mem_cap_gb", "input_scale", "repeat_idx", "warmup"].map(name => [name, number(raw[name])]));
      if (Object.entries(nums).some(([name, value]) => value === null && !(raw.runtime_kind === "process" &&
          ["cpu_cores", "mem_cap_gb"].includes(name) && ["", "none", "nan"].includes(raw[name].toLowerCase()))) ||
          ["cpu_cores", "mem_cap_gb", "input_scale"].some(name => nums[name] !== null && nums[name] <= 0) || ![0,1].includes(nums.warmup) || nums.repeat_idx < 0 || !Number.isInteger(nums.repeat_idx)) fail(`CSV 第 ${rowNumber} 行配置或窗口身份无效 / Invalid configuration or window`);
      const mode = raw.gpu_mode.toLowerCase();
      const device = ["on", "1", "1.0", "true", "yes", "gpu"].includes(mode) ? "GPU" :
        ["off", "0", "0.0", "false", "no", "cpu"].includes(mode) ? "CPU" : fail(`CSV gpu_mode 无效 / Invalid gpu_mode: ${raw.gpu_mode}`);
      const input = {scale:nums.input_scale, type:"unknown", task_param:text(raw.task_param),
        batch_size:number(raw.batch_size), input_units:number(raw.input_units_per_request)};
      if (raw.input_case && !["none", "null", "nan"].includes(raw.input_case.toLowerCase())) input.name = raw.input_case;
      const identity = {run_id:text(raw.run_id), model:text(raw.model || raw.model_name), runtime:text(raw.runtime || raw.runtime_backend),
        device, cpu:nums.cpu_cores, memory:nums.mem_cap_gb, gpu:device === "GPU" ? text(raw.gpu_device_uuid) : "n/a",
        concurrency:number(raw.concurrency), environment_class:["native_linux", "wsl2", "vm", "cloud", "container_host"].includes(raw.environment_class) ? raw.environment_class : "unknown",
        task:text(raw.task), input_case:canonical(input), experiment_batch:text(raw.experiment_batch), runtime_kind:text(raw.runtime_kind),
        runtime_id:text(raw.runtime_id), resource_scope:text(raw.resource_scope)};
      const stable = {...identity}; delete stable.experiment_batch;
      const key = JSON.stringify(stable);
      if (!groups.has(key)) groups.set(key, {identity, idKey:key, index:groups.size, input_scale:nums.input_scale, total:0, formal:0, valid:0, inferred:false, metrics:accumulator(registry)});
      const group = groups.get(key);
      const measurement = JSON.stringify([identity.run_id === "unknown" ? group.index : identity.run_id,
        nums.cpu_cores, nums.mem_cap_gb, device, nums.input_scale, nums.warmup, nums.repeat_idx]);
      if (seen.has(measurement)) fail(`CSV 重复或冲突的测量窗口 / Duplicate or conflicting window: ${rowNumber}`);
      seen.add(measurement);
      const eligible = raw.status.toLowerCase() === "ok" && nums.warmup === 0 && raw.result_origin !== "inferred_not_measured";
      group.total++;
      if (nums.warmup === 0) { group.formal++; group.inferred ||= raw.result_origin === "inferred_not_measured"; }
      if (!eligible) return;
      group.valid++;
      const weight = number(raw.repeat_in_window), cpu = number(raw.cpu_energy_total_j), gpu = number(raw.gpu_energy_total_j);
      const energy = cpu !== null && cpu >= 0 && (device === "CPU" || gpu !== null && gpu >= 0) ? cpu + (device === "GPU" ? gpu : 0) : null;
      const derived = {observed_energy_per_request_j:energy, observed_energy_j:energy !== null && weight > 0 ? energy * weight : null};
      for (const [name, metric] of Object.entries(registry)) {
        const value = Object.hasOwn(derived, name) ? derived[name] : number(raw[name]);
        if (!finite(value)) continue;
        const acc = group.metrics[name];
        acc.n++;
        const total = acc.sum + value;
        acc.correction += Math.abs(acc.sum) >= Math.abs(value) ? (acc.sum - total) + value : (value - total) + acc.sum;
        acc.sum = total; acc.max = acc.max === null ? value : Math.max(acc.max, value);
        if (weight > 0) { acc.weight += weight; acc.weighted += value * weight; } else acc.badWeight = true;
        if (metric.aggregation === "lifecycle") {
          const id = ["", "none", "null", "nan", "nat", "n/a", "unknown", "<na>"].includes((raw.cold_start_started_at || "").toLowerCase()) ? null : raw.cold_start_started_at;
          if (acc.startups.size && (id === null) !== acc.startups.has(null)) acc.conflict = true;
          if (acc.startups.has(id) && acc.startups.get(id) !== value) acc.conflict = true;
          acc.startups.set(id, value);
        }
      }
    }
    for (let offset = 0; offset < file.size; offset += chunkSize) {
      checkCancelled(signal);
      const bytes = new Uint8Array(await file.slice(offset, offset + chunkSize).arrayBuffer());
      checkCancelled(signal); hash.update(bytes);
      const done = offset + chunkSize >= file.size;
      tail += decoder.decode(bytes, {stream:!done});
      if (!newline && tail.length > 8 * 1024 * 1024) fail("CSV 表头超过 8 MiB / CSV header exceeds 8 MiB");
      if (!newline) {
        const match = tail.match(/\r\n|\n|\r/);
        if (!match && !done) continue;
        // The header cannot contain a quoted newline in the AC-Prof contract.
        newline = match?.[0] || "\n";
        if (newline === "\r" && tail.endsWith("\r") && !done) { newline = null; continue; }
      }
      const parsed = new Papa.Parser({delimiter:",", newline, step: result => {
        if (result.errors.length) fail("CSV 损坏 / Malformed CSV: " + result.errors[0].message);
        row(result.data[0]);
      }}).parse(tail, 0, !done);
      if (parsed.errors.length) fail("CSV 损坏 / Malformed CSV: " + parsed.errors[0].message);
      tail = tail.slice(parsed.meta.cursor);
      if (tail.length > 8 * 1024 * 1024) fail("CSV 单行超过 8 MiB / CSV row exceeds 8 MiB");
      onProgress(Math.min(1, (offset + bytes.length) / file.size));
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    checkCancelled(signal);
    if (!groups.size) fail("CSV 没有测量数据 / No measurement rows");
    const digest = hash.hex(), path = `csv:${digest}/${file.name}`, quality = {quality_status:"unknown", auto_selection_eligible:false,
      quality_reasons:["csv_metadata_missing"], quality_checks:[], run_status:"unknown", measurement_status:"unknown"};
    const configs = [...groups.values()].map(group => {
      const metrics = Object.fromEntries(Object.entries(registry).map(([name, metric]) => {
        const acc = group.metrics[name], missing = group.valid - acc.n;
        let value = null, n = acc.n, aggregation = metric.aggregation;
        if (aggregation === "lifecycle") {
          n = acc.startups.size;
          if (acc.conflict) { n = 0; aggregation = "lifecycle_conflict"; }
          else if (n) value = [...acc.startups.values()].reduce((a,b) => a+b, 0) / n;
        } else if (n) {
          if (aggregation === "sum") value = missing ? null : acc.sum;
          else if (aggregation === "max") value = acc.max;
          else if (aggregation === "request_weighted") value = !missing && !acc.badWeight ? acc.weighted / acc.weight : null;
          else value = (acc.sum + acc.correction) / n;
        }
        return [name, {value:finite(value) ? value : null, n, missing:acc.conflict ? 0 : missing, aggregation}];
      }));
      return {...group.identity, ...quality, source:path, config_id:`csv-${digest.slice(0,16)}:${sha256(group.idKey).slice(0,16)}`,
        input_scale:group.input_scale, valid_windows:group.valid, excluded_rows:group.total - group.valid,
        status:group.valid ? (group.valid === group.formal ? "ok" : "partial") : group.inferred ? "inferred_not_measured" : group.formal ? "failed" : "no_formal_windows",
        comparison_profiles:{}, comparability:"unknown", missing_metrics:[], metrics, statistics_status:"not_computed"};
    });
    const runs = [...new Set(configs.map(config => config.run_id))];
    return {kind:"acprof_analysis_bundle", schema_version:1, analysis_contract:{version:1, statistics_engine:"not_computed"},
      configs, sources:[{path, original_path:file.name, sha256:digest, run_id:runs.length === 1 ? runs[0] : "unknown", run_state:"unknown", display_name:file.name, ...quality}],
      registry, window_statistics:[], experiment_comparisons:[], baseline_comparisons:{pairs:{}, qualifications:[]},
      overview:[{source:path, identity:{}}]};
  }
  function validateBundle(input, registry) {
    const data = input?.analysis_bundle || input;
    if (!data || data.schema_version !== 1 || data.kind && data.kind !== "acprof_analysis_bundle" ||
        data.analysis_contract && data.analysis_contract.version !== 1) fail("不兼容的数据包版本 / Unsupported analysis bundle version");
    for (const name of ["configs", "sources", "window_statistics", "experiment_comparisons", "overview"]) {
      if (!Array.isArray(data[name])) fail("数据包缺失字段 / Missing bundle field: " + name);
    }
    if (!data.configs.length || !data.sources.length || !data.registry || !data.baseline_comparisons) fail("数据包缺失配置或来源 / Missing configurations or sources");
    const configSources = new Map(), sources = new Set();
    const object = value => value && typeof value === "object" && !Array.isArray(value);
    const count = value => Number.isInteger(value) && value >= 0;
    const numeric = value => value === null || finite(value);
    const interval = value => value === null || Array.isArray(value) && value.length === 2 && value.every(finite) && value[0] <= value[1];
    for (const source of data.sources) {
      if (typeof source.path !== "string" || !/^[a-f0-9]{64}$/i.test(source.sha256) || sources.has(source.path)) fail("来源指纹缺失或重复 / Invalid or duplicate source fingerprint");
      sources.add(source.path);
    }
    for (const config of data.configs) {
      if (typeof config.config_id !== "string" || configSources.has(config.config_id) || !sources.has(config.source)) fail("配置身份缺失或重复 / Invalid or duplicate configuration identity");
      configSources.set(config.config_id, config.source);
      for (const name of ["model", "run_id", "runtime", "device", "input_case", "quality_status", "status"]) {
        if (typeof config[name] !== "string") fail("配置缺失字段 / Missing configuration field: " + name);
      }
      if (!config.comparison_profiles || !Array.isArray(config.quality_reasons) || !Array.isArray(config.quality_checks)) fail("配置缺失质量或比较证据 / Missing quality or comparison evidence");
      if (![config.cpu, config.memory, config.concurrency, config.input_scale].every(numeric) || !count(config.valid_windows) ||
          !count(config.excluded_rows) || typeof config.auto_selection_eligible !== "boolean") fail("配置数值无效 / Invalid configuration values");
      for (const [purpose, profile] of Object.entries(config.comparison_profiles)) {
        if (!["same-hardware", "cross-hardware", "resource-scaling"].includes(purpose) || profile.purpose !== purpose ||
            typeof profile.valid !== "boolean" || !object(profile.checks) || !Object.keys(profile.checks).length ||
            !["compatible", "incompatible", "unknown"].includes(profile.status) || typeof profile.cohort !== "string") fail("比较条件损坏 / Invalid comparison profile");
        for (const check of Object.values(profile.checks)) if (!object(check) ||
            check.value !== null && typeof check.value !== "string" || !["incompatible", "expected_difference"].includes(check.difference)) fail("比较条件损坏 / Invalid condition token");
        if (profile.workload !== null) {
          if (!object(profile.workload)) fail("输入条件损坏 / Invalid workload evidence");
          for (const item of Object.values(profile.workload)) {
            if (item.ambiguous) continue;
            if (!Array.isArray(item.variants) || !item.variants.length) fail("输入条件缺失 / Missing workload variants");
            for (const variant of item.variants) if (!object(variant.fields) || !Array.isArray(variant.controlled) ||
                !/^\d+(?:\/[1-9]\d*)?$/.test(variant.proportion)) fail("输入权重无效 / Invalid workload proportion");
          }
        }
      }
      for (const [name, metric] of Object.entries(registry)) {
        const actual = data.registry[name], values = config.metrics?.[name];
        if (!actual || ["unit", "aggregation", "direction"].some(key => actual[key] !== metric[key])) fail("指标口径不兼容 / Incompatible metric contract: " + name);
        if (!values || values.value !== null && !finite(values.value) || !Number.isInteger(values.n) || values.n < 0 ||
            !Number.isInteger(values.missing) || values.missing < 0 || ![metric.aggregation, "lifecycle_conflict"].includes(values.aggregation)) fail("指标值损坏或缺失 / Invalid metric values: " + name);
      }
    }
    if (new Set(configSources.values()).size !== sources.size) fail("来源没有配置 / Source has no configurations");
    for (const report of data.window_statistics) {
      if (!sources.has(report.source) || report.result_sha256 !== data.sources.find(s => s.path === report.source).sha256 || !Array.isArray(report.groups) || !report.parameters || !report.audit || !Array.isArray(report.cache_diagnostics)) fail("统计来源不匹配 / Invalid statistics provenance");
      for (const group of report.groups) {
        if (configSources.get(group.config_id) !== report.source) fail("统计配置来源不匹配 / Statistics configuration source mismatch");
        if (!registry[group.metric] || group.unit !== registry[group.metric].unit ||
            !count(group.n_windows) || !count(group.missing_windows) ||
            [group.mean, group.std, group.ci_low, group.ci_high].some(value => value !== null && !finite(value)) ||
            (group.ci_low === null) !== (group.ci_high === null) || group.ci_low > group.ci_high ||
            group.ci_low !== null && group.n_windows < 3) fail("统计字段或区间无效 / Invalid statistics or interval");
      }
    }
    for (const entry of data.experiment_comparisons) {
      if (![entry.left, entry.right].every(side => Array.isArray(side) && side.every(path => sources.has(path)))) fail("实验对照来源不匹配 / Invalid comparison sources");
      if (entry.report) {
        const report = entry.report;
        if (report.kind !== "independent_experiment_comparison" || report.schema_version !== 1 || !Array.isArray(report.groups) ||
            report.resampling_unit !== "independent_run_mean" || !object(report.source_sha256) ||
            [...entry.left, ...entry.right].some(path => data.sources.find(source => source.path === path)?.sha256 !== report.source_sha256[path]) ||
            Object.values(report.source_sha256).some(hash => hash !== null && !/^[a-f0-9]{64}$/i.test(hash))) fail("独立统计来源不匹配 / Invalid independent statistics provenance");
        for (const group of report.groups) {
          if (![group.left, group.right].every(side => object(side) && count(side.n_runs) && numeric(side.mean)) ||
              !interval(group.difference_ci) || !interval(group.ratio_ci) ||
              (group.difference_ci !== null || group.ratio_ci !== null) && Math.min(group.left.n_runs, group.right.n_runs) < 3) fail("独立运行数或区间无效 / Invalid independent run count or interval");
        }
      }
    }
    const pairs = data.baseline_comparisons;
    if (!object(pairs.pairs) || !Array.isArray(pairs.qualifications)) fail("基线比较缺失 / Missing baseline comparisons");
    for (const verdict of pairs.qualifications) if (!object(verdict) || !["compatible", "incompatible", "unknown", "descriptive"].includes(verdict.status) || !Array.isArray(verdict.reasons)) fail("基线比较状态无效 / Invalid baseline qualification");
    for (const [id, others] of Object.entries(pairs.pairs)) {
      if (!configSources.has(id) || !object(others)) fail("基线身份无效 / Invalid baseline identity");
      for (const [other, pair] of Object.entries(others)) {
        if (!configSources.has(other) || !object(pair.policies) || !object(pair.metrics) ||
            [pair.resource, ...Object.values(pair.policies)].some(index => !count(index) || !pairs.qualifications[index])) fail("基线证据索引无效 / Invalid baseline evidence reference");
        for (const change of Object.values(pair.metrics)) if (![change.delta, change.percent, change.ratio, change.performance_ratio].every(numeric)) fail("基线差值无效 / Invalid baseline change");
      }
    }
    for (const overview of data.overview) if (!sources.has(overview.source) || !object(overview.identity)) fail("来源元数据无效 / Invalid source metadata");
    return clone(data);
  }
  function validate(input, registry) {
    try { return validateBundle(input, registry); }
    catch (error) {
      if (error instanceof TypeError) fail("数据包结构损坏或缺失字段 / Malformed bundle or missing field: " + error.message);
      throw error;
    }
  }
  function extract(html, id, required = true) {
    // Extract JSON text only. Never create a DOM or execute anything from an imported HTML.
    const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)].filter(match =>
      new RegExp(`\\bid=["']${id}["']`, "i").test(match[1]) && /\btype=["']application\/json["']/i.test(match[1]));
    if (scripts.length !== 1) { if (!required && !scripts.length) return null; fail("HTML 缺少或重复 AC-Prof 数据 / Missing or duplicate AC-Prof data"); }
    return JSON.parse(scripts[0][2]);
  }
  async function read(file, registry, options) {
    if (/\.csv$/i.test(file.name)) return [await readCsv(file, registry, options)];
    if (!/\.(json|html?)$/i.test(file.name)) fail("仅支持 CSV / JSON / HTML / Unsupported file type");
    checkCancelled(options.signal);
    if (file.size > 128 * 1024 * 1024) fail("JSON / HTML 超过 128 MiB / JSON or HTML exceeds 128 MiB");
    const content = await file.text(); checkCancelled(options.signal);
    const html = /\.html?$/i.test(file.name), data = html ? extract(content, "report-data") : JSON.parse(content);
    const session = html ? extract(content, "report-session", false) : null;
    if (session && (session.schema_version !== 1 || !Array.isArray(session.imports))) fail("会话版本不兼容 / Unsupported session version");
    return [data, ...(session?.imports || []).map(item => item.bundle)].map(item => validate(item, registry));
  }
  return {read, readCsv, validate, clone, checkCancelled};
})();
