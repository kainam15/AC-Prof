/* Local files only. CSV summaries are descriptive; all uncertainty stays in Python. */
"use strict";
window.ACProfImport = (() => {
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const number = value => value == null || String(value).trim() === "" ? null :
    (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(String(value).trim()) && Number.isFinite(Number(value)) ? Number(value) : null);
  const resourceAxes = ["cpu_cores", "mem_cap_gb", "input_scale"];
  const {decimalCoordinate, repeatSequence} = ACProfCoordinates;
  const text = value => value == null || ["", "none", "null", "nan"].includes(String(value).trim().toLowerCase()) ? "unknown" : String(value).trim();
  const canonical = value => "{" + Object.keys(value).sort().map(key => JSON.stringify(key) + ": " + JSON.stringify(value[key])).join(", ") + "}";
  const fail = message => { throw new Error(message); };
  const checkCancelled = signal => { if (signal?.aborted) throw new DOMException("已取消 / Cancelled", "AbortError"); };
  const clone = value => JSON.parse(JSON.stringify(value));
  const record = value => value !== null && typeof value === "object" && !Array.isArray(value);
  const qualities = new Set(["complete", "partial", "unsupported", "permission_denied", "not_requested", "not_applicable", "tool_missing", "error", "unavailable", "unknown"]);
  const sampledGroups = new Set(["gpu_power", "cpu_energy", "dram_energy", "cpu_instructions", "cgroup"]);
  const {stableJson, observationSignature} = ACProfViews;
  function accumulator(registry) {
    return Object.fromEntries(Object.keys(registry).map(name => [name, {n:0, sum:0, correction:0, max:null, weight:0, weighted:0, badWeight:false, startups:new Map(), conflict:false}]));
  }
  function collectorDiagnostic(state, field, name, present) {
    if (!Object.hasOwn(state, field)) return null;
    const entries = state[field];
    if (!record(entries)) return {error:`collector ${field}: expected an object`};
    if (!Object.hasOwn(entries, name)) return null;
    const entry = entries[name], path = `collector ${field}.${name}`;
    if (!record(entry)) return {error:`${path}: expected an object`};
    if (typeof entry.detail !== "string") return {error:`${path}.detail: expected text`};
    const quality = present ? "partial" : entry.status;
    if (!qualities.has(quality)) return {error:`${path}.status: invalid metric quality`};
    return {quality, reason:entry.detail};
  }
  function metricObservations(raw, registry) {
    let evidence = {}, collectors = {}, invalid = false;
    const text = (raw.resource_metric_quality || "").trim();
    if (text && !["nan", "null", "none"].includes(text.toLowerCase())) {
      try {
        evidence = JSON.parse(text);
        if (!record(evidence)) throw new Error();
        if (Object.hasOwn(evidence, "schema_version")) {
          if (evidence.schema_version !== 1) throw new Error();
          collectors = Object.hasOwn(evidence, "collectors") ? evidence.collectors : {};
          if (!record(collectors)) throw new Error();
          evidence = evidence.metrics;
        }
        if (!record(evidence) || Object.values(evidence).some(item => !record(item))) throw new Error();
      } catch { invalid = true; evidence = {}; collectors = {}; }
    }
    const runtime = raw.runtime_kind === "process" ? "process" : "docker";
    const hasCollectors = Object.keys(collectors).length > 0;
    return Object.fromEntries(Object.entries(registry).filter(([name]) => Object.hasOwn(raw, name)).map(([name, metric]) => {
      const contract = (runtime === "process" && metric.process_contract) || metric.observation_contract || {};
      const error = reason => [name, {...contract, quality:"error", reason}];
      if (invalid) return error("invalid_metric_quality_evidence");
      let item = evidence[name];
      const group = metric.collectors?.[runtime];
      const state = group && Object.hasOwn(collectors, group) ? collectors[group] : undefined;
      if (hasCollectors && metric.runtimes && !metric.runtimes.includes(runtime))
        item = {quality:"not_applicable", reason:"runtime_not_implemented"};
      else if (state !== undefined && !record(state)) return error(`collector ${group}: expected an object`);
      else if (state?.status === "not_applicable") item = {quality:"not_applicable", reason:Object.hasOwn(state, "reason") ? state.reason : ""};
      if (!item && state) {
        let {status:quality, reason = ""} = state;
        let sampled = false;
        const sampling = Object.hasOwn(state, "sampling") ? state.sampling : {};
        if (group === "packet_latency" && quality === "available" && raw.packet_status !== "ok") {
          quality = raw.packet_status === "partial" ? "partial" : "unavailable";
          reason = "incomplete_packet_request_correlation";
        }
        if (quality === "available") {
          const direction = name.includes("output_megapixel") || name.startsWith("output_pixels") ? "output" : "input";
          const needsBaseline = ["gpu_power", "cpu_energy", "dram_energy"].includes(group) &&
            (metric.window === "control_window" || ["_eff_", "_effective_", "_idle_"].some(part => name.includes(part)));
          if (needsBaseline && !record(sampling)) return error("collector sampling: expected an object");
          if (needsBaseline && sampling.idle_seconds === 0) {
            quality = "not_requested"; reason = "idle_window_disabled";
          } else if (needsBaseline && state.control_error) {
            quality = "error"; reason = state.control_error;
          } else if ((["gpu", "gpu_idle_debug"].includes(metric.applicability) && raw.gpu_mode === "off") ||
                     (metric.applicability === "cpu" && raw.gpu_mode === "on")) {
            quality = "not_applicable"; reason = "device_not_selected";
          } else if (["cpu_cores", "mem_cap_gb"].includes(name) && raw.resource_limit_source === "inherited") {
            quality = "not_applicable"; reason = "inherited_limit_not_controlled";
          } else if (metric.applicability === "audio" && !["audio", "multimodal"].includes(state.task_family) && number(raw[name]) == null) {
            quality = "not_applicable"; reason = "no_audio_samples";
          } else if (metric.applicability === "pixel_geometry" && number(raw[direction + "_pixels_per_request"]) == null) {
            quality = "not_applicable"; reason = "no_pixel_geometry";
          } else if (name.endsWith("_slow_ratio") && !state.latency_slo) {
            quality = "not_requested"; reason = "no_latency_slo";
          } else if (name.endsWith("_per_cpu_core") && number(raw.cpu_cores) == null) {
            quality = "not_applicable"; reason = "no_controlled_cpu_quota";
          } else if (["idle_debug", "gpu_idle_debug"].includes(metric.applicability) && !state.idle_debug) {
            quality = "not_requested"; reason = "idle_debug_disabled";
          }
        }
        if (quality === "available") {
          const present = number(raw[name]) != null;
          quality = present ? "complete" : "unavailable";
          const diagnostic = collectorDiagnostic(state, "reader_errors", name, present) ??
            (present ? null : collectorDiagnostic(state, "preflight_missing", name, false));
          if (diagnostic?.error) return error(diagnostic.error);
          if (diagnostic) ({quality, reason} = diagnostic);
          else reason = present ? "" : reason || "missing_window_value";
          sampled = sampledGroups.has(group);
          if (sampled && !record(sampling)) return error("collector sampling: expected an object");
          if (quality === "complete" && group === "cpu_instructions" && !(number(raw.cpu_perf_running_pct) > 0 && number(raw.cpu_perf_running_pct) <= 100)) {
            quality = "partial"; reason = "unknown_pmu_running_fraction";
          }
        }
        if (quality === "complete" && metric.applicability === "text_tokenizer" &&
            evidence.output_token_count_avg?.completed_requests == null) {
          quality = "unknown"; reason = "output_token_coverage_unrecorded";
        }
        item = {quality, reason};
        if (name.startsWith("throughput_samples_per_s") && ["complete", "unavailable"].includes(quality)) {
          item.sampling = {latency_denominator:["ok", "partial"].includes(raw.packet_status) ? "packet" : "application"};
          if (raw.packet_status === "partial") Object.assign(item, {quality:"partial", reason:"incomplete_packet_request_correlation"});
        }
        if (sampled) item.sampling = {...sampling};
        if (group === "gpu_power" && item.sampling) item.sampling.energy_method = raw.gpu_energy_source;
      }
      const bad = item && (!qualities.has(item.quality) ||
        ["source", "scope", "window", "derivation"].some(key => Object.hasOwn(item, key) && item[key] !== contract[key]));
      if (bad) return error("invalid_metric_quality_evidence");
      if (item && Object.hasOwn(item, "reason") && typeof item.reason !== "string") return error("metric reason must be text");
      if (item?.quality === "complete" && metric.kind === "number" && number(raw[name]) === null) {
        return [name, {...contract, ...item, quality:"unavailable", reason:"missing_window_value"}];
      }
      return [name, {...contract, quality:"unknown", ...(item ? {reason:"", ...item} : {})}];
    }));
  }
  async function readCsv(file, registry, {signal, onProgress = () => {}, chunkSize = 256 * 1024} = {}) {
    const groups = new Map(), seen = new Set(), hash = sha256.create(), decoder = new TextDecoder("utf-8", {fatal:true});
    const maxPendingChars = 8 * 1024 * 1024;
    let fields, tail = "", newline, rowNumber = 0, nextParseLength = 0;
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
      const invalidCoordinate = () => fail(`CSV 第 ${rowNumber} 行配置或窗口身份无效 / Invalid configuration or window`);
      const nums = {}, axisKeys = {}, exact = {};
      for (const name of resourceAxes) {
        const value = number(raw[name]);
        nums[name] = value;
        if (value === null && raw.runtime_kind === "process" && name !== "input_scale" &&
            ["", "none", "nan"].includes(raw[name].toLowerCase())) {
          axisKeys[name] = "inherited";
          continue;
        }
        const coordinate = decimalCoordinate(raw[name]);
        if (value === null || value <= 0 || !coordinate) invalidCoordinate();
        axisKeys[name] = coordinate.key;
        if (raw[name] !== String(value) && coordinate.key !== decimalCoordinate(String(value)).key) exact[name] = coordinate.key;
      }
      const phase = decimalCoordinate(raw.warmup), repeat = decimalCoordinate(raw.repeat_idx);
      if (!phase || !["0", "1"].includes(phase.key) || !repeat?.integer) invalidCoordinate();
      const formal = phase.key === "0";
      const mode = raw.gpu_mode.toLowerCase();
      const device = ["on", "1", "1.0", "true", "yes", "gpu"].includes(mode) ? "GPU" :
        ["off", "0", "0.0", "false", "no", "cpu"].includes(mode) ? "CPU" : fail(`CSV gpu_mode 无效 / Invalid gpu_mode: ${raw.gpu_mode}`);
      const input = {scale:exact.input_scale ?? nums.input_scale, type:"unknown", task_param:text(raw.task_param),
        batch_size:number(raw.batch_size), input_units:number(raw.input_units_per_request)};
      if (raw.input_case && !["none", "null", "nan"].includes(raw.input_case.toLowerCase())) input.name = raw.input_case;
      const identity = {run_id:text(raw.run_id), model:text(raw.model || raw.model_name), runtime:text(raw.runtime || raw.runtime_backend),
        device, cpu:nums.cpu_cores, memory:nums.mem_cap_gb, gpu:device === "GPU" ? text(raw.gpu_device_uuid) : "n/a",
        concurrency:number(raw.concurrency), environment_class:["native_linux", "wsl2", "vm", "cloud", "container_host"].includes(raw.environment_class) ? raw.environment_class : "unknown",
        task:text(raw.task), input_case:canonical(input), experiment_batch:text(raw.experiment_batch), runtime_kind:text(raw.runtime_kind),
        runtime_id:text(raw.runtime_id), resource_scope:text(raw.resource_scope)};
      if (exact.cpu_cores || exact.mem_cap_gb) {
        identity.exact_resources = {};
        if (exact.cpu_cores) identity.exact_resources.cpu = exact.cpu_cores;
        if (exact.mem_cap_gb) identity.exact_resources.memory = exact.mem_cap_gb;
      }
      const stable = {...identity}; delete stable.experiment_batch;
      const key = JSON.stringify(stable);
      if (!groups.has(key)) groups.set(key, {identity, idKey:key, index:groups.size, input_scale:nums.input_scale, total:0, formal:0, valid:0, inferred:false, metrics:accumulator(registry), observations:{}});
      const group = groups.get(key);
      const observations = metricObservations(raw, registry);
      if (formal) for (const [name, observation] of Object.entries(observations)) {
        group.observations[name] ||= new Map();
        group.observations[name].set(stableJson(observation), observation);
      }
      const measurement = JSON.stringify([identity.run_id === "unknown" ? group.index : identity.run_id,
        axisKeys.cpu_cores, axisKeys.mem_cap_gb, device, axisKeys.input_scale, phase.key, repeat.key]);
      if (seen.has(measurement)) fail(`CSV 重复或冲突的测量窗口 / Duplicate or conflicting window: ${rowNumber}`);
      seen.add(measurement);
      const eligible = raw.status.toLowerCase() === "ok" && formal && raw.result_origin !== "inferred_not_measured";
      group.total++;
      if (formal) { group.formal++; group.inferred ||= raw.result_origin === "inferred_not_measured"; }
      if (!eligible) return;
      group.valid++;
      const weight = number(raw.repeat_in_window), cpu = number(raw.cpu_energy_total_j), gpu = number(raw.gpu_energy_total_j);
      const energy = cpu !== null && cpu >= 0 && (device === "CPU" || gpu !== null && gpu >= 0) ? cpu + (device === "GPU" ? gpu : 0) : null;
      const derived = {observed_energy_per_request_j:energy, observed_energy_j:energy !== null && weight > 0 ? energy * weight : null};
      for (const [name, metric] of Object.entries(registry)) {
        if (observations[name] && !["complete", "unknown"].includes(observations[name].quality)) continue;
        const value = Object.hasOwn(derived, name) ? derived[name] : number(raw[name]);
        if (!finite(value)) continue;
        const acc = group.metrics[name];
        if (observations[name] && !acc.mixedContract) {
          const signature = observationSignature(observations[name]);
          acc.signature ??= signature;
          acc.mixedContract = acc.signature !== signature;
        }
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
    function parsePending(done) {
      if (!newline) {
        const match = tail.match(/\r\n|\n|\r/);
        // The header cannot contain a quoted newline in the AC-Prof contract.
        if ((match?.index ?? tail.length) > maxPendingChars) fail("CSV 表头超过 8 MiB / CSV header exceeds 8 MiB");
        if (!done && (!match || (match[0] === "\r" && match.index === tail.length - 1))) return;
        newline = match?.[0] || "\n";
      }
      // Papa needs both bytes of a CRLF after a closing quote to recognize a record boundary.
      const holdCR = !done && newline === "\r\n" && tail.endsWith("\r");
      const parsed = new Papa.Parser({delimiter:",", newline, step: result => {
        if (result.errors.length) fail("CSV 损坏 / Malformed CSV: " + result.errors[0].message);
        row(result.data[0]);
      }}).parse(holdCR ? tail.slice(0, -1) : tail, 0, !done);
      if (parsed.errors.length) fail("CSV 损坏 / Malformed CSV: " + parsed.errors[0].message);
      tail = tail.slice(parsed.meta.cursor);
      if (tail.length - Number(holdCR) > maxPendingChars) fail("CSV 单行超过 8 MiB / CSV row exceeds 8 MiB");
    }
    for (let offset = 0; offset < file.size; offset += chunkSize) {
      checkCancelled(signal);
      const bytes = new Uint8Array(await file.slice(offset, offset + chunkSize).arrayBuffer());
      checkCancelled(signal); hash.update(bytes);
      const done = offset + chunkSize >= file.size;
      tail += decoder.decode(bytes, {stream:!done});
      if (done || tail.length >= nextParseLength) {
        parsePending(done);
        // Papa reparses its unfinished record. Double the buffered prefix between retries
        // so long quoted fields cost linear work, and check the limit as soon as it is crossed.
        nextParseLength = Math.min(maxPendingChars + 1, 2 * tail.length);
      }
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
        return [name, {value:finite(value) && !acc.mixedContract ? value : null, n, missing:acc.conflict ? 0 : missing, aggregation}];
      }));
      return {...group.identity, ...quality, source:path, config_id:`csv-${digest.slice(0,16)}:${sha256(group.idKey).slice(0,16)}`,
        input_scale:group.input_scale, valid_windows:group.valid, excluded_rows:group.total - group.valid,
        status:group.valid ? (group.valid === group.formal ? "ok" : "partial") : group.inferred ? "inferred_not_measured" : group.formal ? "failed" : "no_formal_windows",
        comparison_profiles:{}, comparability:"unknown", missing_metrics:[], metrics, statistics_status:"not_computed",
        metric_observations:Object.fromEntries(Object.entries(group.observations).map(([name, items]) => [name, [...items.values()]]))};
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
      if (Object.hasOwn(config, "exact_resources") && (!record(config.exact_resources) ||
          Object.entries(config.exact_resources).some(([axis, value]) => !["cpu", "memory"].includes(axis) ||
            typeof value !== "string" || decimalCoordinate(value)?.key !== value || !finite(config[axis]) ||
            Number(value) !== config[axis] || decimalCoordinate(String(config[axis]))?.key === value))) {
        fail("资源坐标无效 / Invalid exact resource coordinate");
      }
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
      const parameters = report.parameters, inputs = report.recalculation;
      if (!(parameters.confidence > 0 && parameters.confidence < 1) || !Number.isInteger(parameters.resamples) || parameters.resamples < 1 ||
          !Number.isInteger(parameters.seed) || !Number.isInteger(parameters.block_size) || parameters.block_size < 1 ||
          parameters.method !== "circular_block_percentile_bootstrap" || parameters.resampling_unit !== "csv_request_window") fail("统计参数或方法无效 / Invalid statistics contract");
      if (inputs) {
        if (inputs.schema_version !== 1 || !["embedded", "external_required"].includes(inputs.status) ||
            !object(inputs.metrics) || !/^[a-f0-9]{64}$/.test(inputs.sha256) || !count(inputs.bytes) ||
            inputs.status === "embedded" && !Array.isArray(inputs.groups)) fail("重算数据格式无效 / Invalid recalculation inputs");
        for (const group of inputs.groups || []) {
          if (configSources.get(group.config_id) !== report.source) fail("重算窗口来源不匹配 / Recalculation source mismatch");
          if (!inputs.metrics[group.metric] ||
              group.unit !== inputs.metrics[group.metric].unit || !Array.isArray(group.values) || !Array.isArray(group.repeat_indices) ||
              group.values.length !== group.repeat_indices.length || !group.values.every(numeric) ||
              repeatSequence(group.repeat_indices) === "invalid") fail("重算窗口损坏 / Invalid recalculation windows");
        }
      }
      for (const group of report.groups) {
        if (configSources.get(group.config_id) !== report.source) fail("统计配置来源不匹配 / Statistics configuration source mismatch");
        const metric = registry[group.metric] || inputs?.metrics[group.metric];
        if (!metric || group.unit !== metric.unit ||
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
        if (Object.hasOwn(pair, "metric_reasons") && (!object(pair.metric_reasons) ||
            Object.entries(pair.metric_reasons).some(([name, reason]) =>
              Object.hasOwn(pair.metrics, name) || typeof reason !== "string" || !reason.trim())))
          fail("指标比较原因无效 / Invalid metric comparison reason");
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
    const bundles = [data, ...(session?.imports || []).map(item => item.bundle)].map(item => validate(item, registry));
    if (!session?.statistics) return bundles;
    const saved = session.statistics;
    if (!Array.isArray(saved.windows) || !Array.isArray(saved.comparisons)) fail("统计会话损坏 / Invalid statistics session");
    return bundles.map(bundle => validate({...bundle,
      window_statistics:bundle.window_statistics.map(report => ({...report, ...saved.windows.find(item =>
        item.source === report.source && item.result_sha256 === report.result_sha256)})),
      experiment_comparisons:bundle.experiment_comparisons.map(entry => saved.comparisons.find(item =>
        JSON.stringify([item.left, item.right, item.report?.purpose, item.report?.source_sha256]) ===
        JSON.stringify([entry.left, entry.right, entry.report?.purpose, entry.report?.source_sha256])) || entry),
    }, registry));
  }
  return {read, readCsv, validate, clone, checkCancelled};
})();
