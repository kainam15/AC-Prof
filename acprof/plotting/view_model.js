/* Pure view calculations shared by the offline UI and browser regression tests. */
"use strict";
window.ACProfViews = (() => {
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const value = (config, metric) => config.metrics[metric]?.value ?? null;
  const key = values => JSON.stringify(values);
  function cohort(config, purpose = "same-hardware") {
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
  function qualification(config, baseline, purpose = "same-hardware") {
    return compareProfiles(config.comparison_profiles?.[purpose], (baseline || config).comparison_profiles?.[purpose]);
  }
  function delta(config, baseline, metric, purpose = "same-hardware") {
    const current = value(config, metric), reference = baseline && value(baseline, metric);
    if (!baseline) return {delta: null, percent: null, reason: "no_baseline"};
    const comparison = qualification(config, baseline, purpose);
    if (comparison.status !== "compatible") return {delta: null, percent: null, reason: comparison.status, reasons: comparison.reasons};
    if (!finite(current) || !finite(reference)) return {delta: null, percent: null, reason: "missing"};
    return {delta: current - reference,
      percent: reference === 0 ? null : 100 * (current - reference) / Math.abs(reference),
      reason: reference === 0 ? "zero_baseline" : ""};
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
  function pareto(configs, xMetric, yMetric, registry, purpose = "same-hardware", baseline = null) {
    const directions = [registry[xMetric].direction, registry[yMetric].direction];
    if (directions.includes("neutral")) return [];
    const eligible = configs.filter(c => c.status === "ok" && c.auto_selection_eligible === true &&
      qualification(c, baseline, purpose).status === "compatible" && finite(value(c, xMetric)) && finite(value(c, yMetric)));
    const objectives = config => [xMetric, yMetric].map((metric, i) =>
      value(config, metric) * (directions[i] === "higher" ? -1 : 1));
    // Equal points both survive. Missing and unsuccessful observations cannot dominate.
    return eligible.filter(candidate => {
      const a = objectives(candidate);
      return !eligible.some(other => {
        if (cohort(candidate, purpose) !== cohort(other, purpose)) return false;
        const b = objectives(other);
        return b[0] <= a[0] && b[1] <= a[1] && (b[0] < a[0] || b[1] < a[1]);
      });
    }).map(c => c.config_id);
  }
  function scaling(configs, axis, metric) {
    const facets = new Map();
    for (const config of configs) {
      if (!finite(config[axis])) continue;
      const facet = key([config.model, config.runtime, config.environment_class,
        config.task, config.input_case]);
      // Vary only the selected resource. In particular, CPU curves never join different memory caps.
      const held = ["cpu", "memory", "concurrency"].filter(name => name !== axis);
      const seriesKey = key([config.run_id, config.device, config.gpu, ...held.map(name => config[name])]);
      if (!facets.has(facet)) facets.set(facet, {config, series: new Map()});
      const series = facets.get(facet).series;
      if (!series.has(seriesKey)) series.set(seriesKey, {config, held, points: []});
      series.get(seriesKey).points.push({x: config[axis], y: value(config, metric), config});
    }
    return [...facets.values()].map(facet => ({...facet,
      series: [...facet.series.values()].map(series => ({...series,
        points: series.points.sort((a, b) => a.x - b.x)}))}));
  }
  return {finite, value, cohort, compareProfiles, qualification, delta, range, colorRanges, score, sort, pareto, scaling};
})();
