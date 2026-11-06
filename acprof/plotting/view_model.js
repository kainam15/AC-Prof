/* Pure view calculations shared by the offline UI and browser regression tests. */
"use strict";
window.ACProfViews = (() => {
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const value = (config, metric) => config.metrics[metric]?.value ?? null;
  const key = values => JSON.stringify(values);
  function inputLabel(input, detailed = false) {
    const text = value => value == null ? "unknown" : String(value);
    try {
      const parsed = JSON.parse(input);
      return (parsed.name ? `${text(parsed.name)} · ` : "") +
        `scale=${text(parsed.scale)} · batch=${text(parsed.batch_size)} · ${text(parsed.type)}` +
        (detailed ? ` · param=${text(parsed.task_param)} · units=${text(parsed.input_units)}` : "");
    } catch { return input; }
  }
  function cohort(config, purpose = "same-hardware") {
    if (purpose === "within-experiment") purpose = "resource-scaling";
    const profile = config.comparison_profiles?.[purpose];
    return profile?.status === "compatible" ? profile.cohort : config.config_id;
  }
  function workloadStatus(left, right) {
    if (!left || !right) return "unknown";
    if (key(Object.keys(left).sort()) !== key(Object.keys(right).sort())) return "incompatible";
    let status = "compatible";
    // Fractions are normalized with integer arithmetic; no statistical threshold
    // or floating point tolerance can silently change the workload policy.
    const gcd = (a, b) => b ? gcd(b, a % b) : a;
    function distribution(caseEvidence, known) {
      const result = new Map();
      for (const item of caseEvidence.variants) {
        const projection = key(item.controlled.filter(name => known.has(name)).sort().map(name => [name, item.fields[name]]));
        const fraction = item.proportion.split("/").map(BigInt), prior = result.get(projection) || [0n, 1n];
        const divisor = fraction[1] || 1n;
        const numerator = prior[0] * divisor + fraction[0] * prior[1], denominator = prior[1] * divisor;
        const common = gcd(numerator, denominator);
        result.set(projection, [numerator / common, denominator / common]);
      }
      return key([...result].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
        .map(([name, value]) => [name, value.map(String)]));
    }
    for (const name of Object.keys(left)) {
      const a = left[name], b = right[name];
      if (a.ambiguous || b.ambiguous) { status = "unknown"; continue; }
      const variants = [...a.variants, ...b.variants];
      const known = new Set(Object.keys(variants[0].fields).filter(field => variants.every(v => Object.hasOwn(v.fields, field))));
      if (distribution(a, known) !== distribution(b, known)) return "incompatible";
      if (!a.complete || !b.complete || a.observed !== b.observed) status = "unknown";
    }
    return status;
  }
  function compareProfiles(left, right) {
    if (!left || !right || left.purpose !== right.purpose) return {status:"unknown", reasons:["comparison_evidence_missing"]};
    const checks = {};
    for (const name of new Set([...Object.keys(left.checks), ...Object.keys(right.checks)])) {
      const a = left.checks[name], b = right.checks[name];
      checks[name] = !a || !b || a.value === null || b.value === null ? "unknown" :
        a.value === b.value ? "compatible" : a.difference;
    }
    checks.actual_workload = workloadStatus(left.workload, right.workload);
    if (!left.valid || !right.valid) checks.invalid_artifacts = "incompatible";
    const values = Object.values(checks);
    return {status: values.includes("incompatible") ? "incompatible" : values.includes("unknown") ? "unknown" : "compatible",
      reasons: [...new Set(Object.keys(checks).filter(name => ["incompatible", "unknown"].includes(checks[name]))
        .map(name => name.split(".")[0]))].sort()};
  }
  function descriptiveChange(reference, current, direction = "neutral") {
    const result = {delta:null, percent:null, ratio:null, performance_ratio:null, reason:"missing"};
    if (!finite(reference) || !finite(current)) return result;
    const safe = value => finite(value) ? value : null;
    return {delta:safe(current - reference), percent:reference ? safe(100 * ((current - reference) / Math.abs(reference))) : null,
      ratio:reference > 0 ? safe(current / reference) : null,
      performance_ratio:direction === "lower" && reference > 0 && current > 0 ? safe(reference / current) :
        direction === "higher" && reference > 0 && current >= 0 ? safe(current / reference) : null,
      reason:reference > 0 ? "" : "nonpositive_baseline_for_ratio"};
  }
  function descriptiveQualification(left, right, purpose) {
    const policy = purpose === "within-experiment" ? "resource-scaling" : purpose;
    if (left.comparison_profiles?.[policy] && right.comparison_profiles?.[policy]) {
      return compareProfiles(left.comparison_profiles[policy], right.comparison_profiles[policy]);
    }
    const fields = ["task", "runtime", "runtime_kind", "resource_scope", "input_scale", "concurrency", "device"];
    if (policy !== "resource-scaling") fields.push("cpu", "memory");
    const known = value => value != null && value !== "unknown";
    const reasons = fields.filter(name => known(left[name]) && known(right[name]) && left[name] !== right[name]);
    try {
      const a = JSON.parse(left.input_case), b = JSON.parse(right.input_case);
      if (Object.keys(a).some(name => known(a[name]) && known(b[name]) && key(a[name]) !== key(b[name]))) reasons.push("input_case");
    } catch {
      if (known(left.input_case) && known(right.input_case) && left.input_case !== right.input_case) reasons.push("input_case");
    }
    if (policy !== "cross-hardware" && known(left.gpu) && known(right.gpu) && left.gpu !== right.gpu) reasons.push("gpu");
    if (known(left.environment_class) && known(right.environment_class) && left.environment_class !== right.environment_class) reasons.push("environment_class");
    // CSV cannot establish hardware, input semantics or independent run identity.
    return {status:reasons.length ? "incompatible" : "unknown", reasons:reasons.length ? reasons : ["comparison_evidence_missing", "descriptive_only"]};
  }
  function comparisonContext(baseline = null, purpose = "same-hardware", pairs = {}, registry = {}) {
    // Owned by one draw: summary, cells and frontier share checks, while a new
    // baseline, purpose or refresh always starts with fresh evidence.
    const comparisons = new Map();
    const pair = config => pairs.pairs?.[baseline?.config_id || config.config_id]?.[config.config_id];
    const verdict = index => pairs.qualifications?.[index];
    function qualification(config) {
      if (!comparisons.has(config)) comparisons.set(config,
        config.experiment_id && baseline?.experiment_id && !pair(config) ? descriptiveQualification(baseline, config, purpose) :
        purpose === "within-experiment" ? verdict(pair(config)?.resource) || {status:"unknown", reasons:["comparison_evidence_missing"]} :
          verdict(pair(config)?.policies[purpose]) || compareProfiles(
            config.comparison_profiles?.[purpose], (baseline || config).comparison_profiles?.[purpose]));
      return comparisons.get(config);
    }
    function delta(config, metric) {
      if (!baseline) return {delta: null, percent: null, reason: "no_baseline"};
      const comparison = qualification(config);
      if (config.experiment_id && baseline.experiment_id && !pair(config)) {
        if (comparison.status === "incompatible") return {delta:null, percent:null, ratio:null, performance_ratio:null, reason:"incompatible", reasons:comparison.reasons};
        return {...descriptiveChange(value(baseline, metric), value(config, metric), registry[metric]?.direction),
          scope:"browser_descriptive", statistics_status:"not_computed"};
      }
      if (!["compatible", "descriptive"].includes(comparison.status)) return {delta: null, percent: null, reason: comparison.status, reasons: comparison.reasons};
      return pair(config)?.metrics[metric] || {delta:null, percent:null, ratio:null, performance_ratio:null, reason:"missing"};
    }
    function automaticQualification(config) {
      return purpose === "within-experiment" ? verdict(pair(config)?.policies["resource-scaling"]) ||
        {status:"unknown", reasons:["comparison_evidence_missing"]} : qualification(config);
    }
    return Object.freeze({baseline, purpose, qualification, automaticQualification, delta});
  }
  const transform = (value, metric) => finite(value) ?
    (metric.scale === "log" ? (value > 0 ? Math.log(value) : null) : value) : null;
  function range(values, metric) {
    let low = null, high = null;
    for (const value of values) {
      const transformed = transform(value, metric);
      if (!finite(transformed)) continue;
      low = low === null ? transformed : Math.min(low, transformed);
      high = high === null ? transformed : Math.max(high, transformed);
    }
    return {low, high};
  }
  function colorRanges(configs, metrics, registry, purpose = "same-hardware") {
    if (purpose === "within-experiment") purpose = "resource-scaling";
    const groups = new Map(), ranges = new Map();
    for (const config of configs) {
      if (!config.auto_selection_eligible || config.comparison_profiles?.[purpose]?.status !== "compatible") continue;
      const key = cohort(config, purpose);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(config);
    }
    for (const [key, group] of groups) {
      ranges.set(key, Object.fromEntries(metrics.map(name => [name, range(group.map(config => value(config, name)), registry[name])])));
    }
    return ranges;
  }
  function score(current, range, metric) {
    if (!range || !finite(current) || metric.direction === "neutral") return null;
    const transformed = transform(current, metric), {low, high} = range;
    if (!finite(transformed) || !finite(low) || !finite(high)) return null;
    const position = high === low ? 0.5 : (transformed - low) / (high - low);
    return metric.direction === "higher" ? position : 1 - position;
  }
  function sort(configs, metric, descending = false) {
    return [...configs].sort((a, b) => {
      const left = value(a, metric), right = value(b, metric);
      if (!finite(left)) return !finite(right) ? a.config_id.localeCompare(b.config_id) : 1;
      if (!finite(right)) return -1;
      return (left - right) * (descending ? -1 : 1) || a.config_id.localeCompare(b.config_id);
    });
  }
  function pareto(configs, xMetric, yMetric, registry, comparison = comparisonContext()) {
    const {purpose} = comparison;
    const directions = [registry[xMetric].direction, registry[yMetric].direction];
    if (directions.includes("neutral")) return [];
    const eligible = configs.filter(c => c.status === "ok" && c.auto_selection_eligible === true &&
      comparison.automaticQualification(c).status === "compatible" && finite(value(c, xMetric)) && finite(value(c, yMetric)));
    const groups = new Map(), frontier = new Set();
    const xSign = directions[0] === "higher" ? -1 : 1, ySign = directions[1] === "higher" ? -1 : 1;
    for (const config of eligible) {
      const group = cohort(config, purpose);
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push({config, x:value(config, xMetric) * xSign, y:value(config, yMetric) * ySign});
    }
    for (const points of groups.values()) {
      points.sort((a, b) => a.x - b.x || a.y - b.y);
      let bestX = null, bestY = Infinity;
      for (const point of points) {
        // After lexicographic sorting, only a new minimum y or an exact
        // frontier tie survives. An equal y at a larger x is dominated.
        if (point.y < bestY || (point.x === bestX && point.y === bestY)) {
          bestX = point.x; bestY = point.y;
          frontier.add(point.config);
        }
      }
    }
    // Preserve the caller's order and every tied configuration without sorting its input.
    return eligible.filter(config => frontier.has(config)).map(config => config.config_id);
  }
  function scaling(configs, axis, metric) {
    const facets = new Map();
    for (const config of configs) {
      if (!finite(config[axis])) continue;
      const facet = key([config.model, config.runtime, config.environment_class,
        config.task, config.input_case]);
      // Vary only the selected resource. In particular, CPU curves never join different memory caps.
      const held = ["cpu", "memory", "concurrency"].filter(name => name !== axis);
      const seriesKey = key([config.experiment_id, config.run_id, config.device, config.gpu, ...held.map(name => config[name])]);
      if (!facets.has(facet)) facets.set(facet, {config, series: new Map()});
      const series = facets.get(facet).series;
      if (!series.has(seriesKey)) series.set(seriesKey, {config, held, points: []});
      series.get(seriesKey).points.push({x: config[axis], y: value(config, metric), config});
    }
    return [...facets.values()].map(facet => ({...facet,
      series: [...facet.series.values()].map(series => ({...series,
        points: series.points.sort((a, b) => a.x - b.x)}))}));
  }
  return {finite, value, inputLabel, cohort, compareProfiles, descriptiveChange, descriptiveQualification, comparisonContext, range, colorRanges, score, sort, pareto, scaling};
})();
