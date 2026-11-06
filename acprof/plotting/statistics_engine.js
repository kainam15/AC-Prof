/* Browser implementation of acprof.analysis.uncertainty and independent_comparison.
 * Python defines selection, grouping, eligibility, units and statistical semantics.
 * Each resample stores one estimate; yielding also occurs inside large samples.
 */
"use strict";
window.ACProfStatistics = (() => {
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const check = signal => { if (signal?.aborted) throw new DOMException("已取消 / Cancelled", "AbortError"); };
  function validate(options) {
    if (!(options.confidence > 0 && options.confidence < 1) ||
        !Number.isInteger(options.resamples) || options.resamples < 1 || options.resamples > 10000 ||
        !Number.isSafeInteger(options.seed) || !Number.isSafeInteger(options.block_size) || options.block_size < 1) {
      throw new Error("参数无效：置信水平 0–1、Bootstrap 1–10000、整数 Seed、正整数块大小 / Invalid statistics parameters");
    }
  }
  function sum(values) {
    let total = 0, correction = 0;
    for (const value of values) {
      const next = total + value;
      correction += Math.abs(total) >= Math.abs(value) ? total - next + value : value - next + total;
      total = next;
    }
    return total + correction;
  }
  const mean = values => values.length ? sum(values) / values.length : null;
  function deviation(values) {
    if (values.length < 2) return null;
    // Center first to retain small variation on a large offset, as statistics.stdev does.
    const deltas = values.map(value => value - values[0]), center = mean(deltas);
    return Math.sqrt(sum(deltas.map(value => (value - center) ** 2)) / (values.length - 1));
  }
  function interval(values, confidence) {
    values.sort((a, b) => a - b);
    const quantile = probability => {
      const index = (values.length - 1) * probability, lower = Math.floor(index), fraction = index - lower;
      return values[lower] * (1 - fraction) + values[Math.min(lower + 1, values.length - 1)] * fraction;
    };
    const tail = (1 - confidence) / 2;
    return [quantile(tail), quantile(1 - tail)];
  }
  function scheduler({signal, onProgress = () => {}} = {}) {
    let deadline = performance.now() + 12;
    return async progress => {
      check(signal);
      if (performance.now() >= deadline || progress === 1) {
        onProgress(progress);
        await new Promise(resolve => setTimeout(resolve, 0));
        check(signal);
        deadline = performance.now() + 12;
      }
    };
  }
  async function sampledMean(values, blockSize, rng, choices, tick, progress) {
    let total = 0, correction = 0, start = 0;
    for (let index = 0; index < values.length; index++) {
      if (index % blockSize === 0) start = choices ? Math.floor(rng.random() * values.length) : rng.index(values.length);
      const value = values[(start + index % blockSize) % values.length], next = total + value;
      correction += Math.abs(total) >= Math.abs(value) ? total - next + value : value - next + total;
      total = next;
      if ((index & 4095) === 4095) await tick(progress);
    }
    return (total + correction) / values.length;
  }
  async function summarize(input, options, hooks = {}) {
    validate(options); check(hooks.signal);
    const {values:raw, repeat_indices:indices, ...identity} = input;
    if (!Array.isArray(raw) || !Array.isArray(indices) || raw.length !== indices.length ||
        raw.some(value => value !== null && !finite(value)) || indices.some(value => !finite(value))) {
      throw new Error("窗口数据损坏 / Invalid window inputs");
    }
    const values = raw.filter(finite), count = values.length;
    const result = {...identity, n_windows:count, missing_windows:raw.length - count,
      mean:mean(values), std:deviation(values), ci_low:null, ci_high:null, reason:""};
    if (["invalid_window_identity", "no_successful_windows"].includes(input.reason)) result.reason = input.reason;
    else if (count < Math.max(3, 3 * options.block_size)) result.reason = "insufficient_windows";
    else if (options.block_size > 1 && count !== raw.length) result.reason = "missing_windows_break_blocks";
    else if (options.block_size > 1 && indices.some((value, index) => index > 0 && value !== indices[index - 1] + 1)) result.reason = "nonconsecutive_windows";
    else {
      const rng = window.ACProfRandom(options.seed), estimates = [], tick = scheduler(hooks);
      for (let index = 0; index < options.resamples; index++) {
        estimates.push(await sampledMean(values, options.block_size, rng, false, tick, index / options.resamples));
        if ((index & 31) === 31) await tick(index / options.resamples);
      }
      [result.ci_low, result.ci_high] = interval(estimates, options.confidence);
      await tick(1);
    }
    if ([result.mean, result.std, result.ci_low, result.ci_high].some(value => value !== null && !finite(value))) {
      throw new Error("统计数值溢出 / Statistics numeric overflow");
    }
    check(hooks.signal);
    return result;
  }
  async function independent(input, options, hooks = {}) {
    validate({...options, block_size:1}); check(hooks.signal);
    const result = {...input, difference_ci:null, ratio_ci:null};
    // Eligibility and comparability remain Python decisions. Never promote unavailable comparisons.
    if (input.comparability !== "comparable" || !["", "nonpositive_baseline_for_ratio"].includes(input.reason)) return result;
    const left = input.left.runs.map(run => run.mean).filter(finite), right = input.right.runs.map(run => run.mean).filter(finite);
    if (Math.min(left.length, right.length) < 3) return {...result, reason:"insufficient_independent_runs"};
    const rng = window.ACProfRandom(options.seed), differences = [], ratios = [], tick = scheduler(hooks);
    for (let index = 0; index < options.resamples; index++) {
      const a = await sampledMean(left, 1, rng, true, tick, index / options.resamples);
      const b = await sampledMean(right, 1, rng, true, tick, index / options.resamples);
      if (![a, b, b - a, ...(a > 0 ? [b / a] : [])].every(finite)) {
        throw new Error("统计数值溢出 / Statistics numeric overflow");
      }
      differences.push(b - a);
      if (a > 0) ratios.push(b / a);
      if ((index & 31) === 31) await tick(index / options.resamples);
    }
    result.difference_ci = interval(differences, options.confidence);
    result.ratio_ci = ratios.length === options.resamples ? interval(ratios, options.confidence) : null;
    result.reason = result.ratio_ci ? "" : "nonpositive_baseline_for_ratio";
    await tick(1);
    return result;
  }
  return {summarize, independent, validate, engine:"browser", rng:"python_random_mt19937"};
})();
