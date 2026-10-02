/* Pure view calculations shared by the offline UI and browser regression tests. */
"use strict";
window.ACProfViews = (() => {
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const value = (config, metric) => config.metrics[metric]?.value ?? null;
  const key = values => JSON.stringify(values);
  function cohort(config) {
    return key([config.environment_class, config.task, config.input_case,
      config.environment_class === "unknown" ? config.run_id : null]);
  }
  function delta(config, baseline, metric) {
    const current = value(config, metric), reference = baseline && value(baseline, metric);
    if (!baseline) return {delta: null, percent: null, reason: "no_baseline"};
    if (cohort(config) !== cohort(baseline)) return {delta: null, percent: null, reason: "different_conditions"};
    if (!finite(current) || !finite(reference)) return {delta: null, percent: null, reason: "missing"};
    return {delta: current - reference,
      percent: reference === 0 ? null : 100 * (current - reference) / Math.abs(reference),
      reason: reference === 0 ? "zero_baseline" : ""};
  }
  function score(current, values, metric) {
    if (!finite(current) || metric.direction === "neutral") return null;
    const transform = x => metric.scale === "log" ? (x > 0 ? Math.log(x) : null) : x;
    const transformed = transform(current);
    const numbers = values.filter(finite).map(transform).filter(finite);
    if (!finite(transformed) || !numbers.length) return null;
    const low = Math.min(...numbers), high = Math.max(...numbers);
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
  function pareto(configs, xMetric, yMetric, registry) {
    const directions = [registry[xMetric].direction, registry[yMetric].direction];
    if (directions.includes("neutral")) return [];
    const eligible = configs.filter(c => c.status === "ok" && finite(value(c, xMetric)) && finite(value(c, yMetric)));
    const objectives = config => [xMetric, yMetric].map((metric, i) =>
      value(config, metric) * (directions[i] === "higher" ? -1 : 1));
    // Equal points both survive. Missing and unsuccessful observations cannot dominate.
    return eligible.filter(candidate => {
      const a = objectives(candidate);
      return !eligible.some(other => {
        if (cohort(candidate) !== cohort(other)) return false;
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
  return {finite, value, cohort, delta, score, sort, pareto, scaling};
})();
