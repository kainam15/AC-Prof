/* One live report dataset; original embedded payload and user files remain untouched. */
"use strict";
window.ACProfWorkspace = (original, restored = null) => {
  const I = window.ACProfImport, imports = [];
  const data = {...original}, palette = ["#087f8c", "#c57532", "#7161ab", "#2673a8", "#a54967", "#4e824d", "#986833"];
  const colors = new Map((restored?.colors || []).filter(([, color]) => palette.includes(color)));
  let entries = [];
  const known = value => value && value !== "unknown";
  const runsFor = (bundle, source) => [...new Set(bundle.configs.filter(config => config.source === source.path).map(config => config.run_id).filter(known))].sort();
  const experimentId = (bundle, source) => `experiment-${source.sha256}-${sha256(JSON.stringify(runsFor(bundle, source))).slice(0,12)}`;
  function subset(bundle, configs) {
    const ids = new Set(configs.map(config => config.config_id)), paths = new Set(configs.map(config => config.source));
    return {...bundle, kind:"acprof_analysis_bundle", schema_version:1,
      analysis_contract:{...bundle.analysis_contract, version:1, statistics_engine:
        configs.every(config => config.statistics_status === "not_computed") ? "not_computed" :
        configs.some(config => config.statistics_status === "not_computed") ? "mixed" : "python"}, configs,
      sources:bundle.sources.filter(source => paths.has(source.path)),
      overview:bundle.overview.filter(source => paths.has(source.source)),
      window_statistics:bundle.window_statistics.map(report => ({...report, groups:report.groups.filter(group => ids.has(group.config_id))})).filter(report => report.groups.length),
      experiment_comparisons:bundle.experiment_comparisons.filter(entry => [...entry.left, ...entry.right].every(path =>
        paths.has(path) && bundle.configs.filter(config => config.source === path).every(config => ids.has(config.config_id)))),
      baseline_comparisons:{qualifications:bundle.baseline_comparisons.qualifications,
        pairs:Object.fromEntries(Object.entries(bundle.baseline_comparisons.pairs).filter(([id]) => ids.has(id)).map(([id, pairs]) =>
          [id, Object.fromEntries(Object.entries(pairs).filter(([other]) => ids.has(other)))]))}};
  }
  function rebuild() {
    entries = [];
    for (const name of ["configs", "sources", "overview", "window_statistics", "experiment_comparisons"]) data[name] = [];
    data.baseline_comparisons = {pairs:{}, qualifications:[]};
    for (const [index, bundle] of [original, ...imports.map(item => item.bundle)].entries()) {
      const offset = data.baseline_comparisons.qualifications.length;
      data.baseline_comparisons.qualifications.push(...bundle.baseline_comparisons.qualifications);
      for (const [id, pairs] of Object.entries(bundle.baseline_comparisons.pairs)) {
        data.baseline_comparisons.pairs[id] = Object.fromEntries(Object.entries(pairs).map(([other, pair]) => [other, {
          ...pair, resource:pair.resource + offset, policies:Object.fromEntries(Object.entries(pair.policies).map(([p, n]) => [p, n + offset]))}]));
      }
      for (const source of bundle.sources) {
        const id = experimentId(bundle, source), configs = bundle.configs.filter(config => config.source === source.path);
        if (!colors.has(id)) colors.set(id, palette[colors.size % palette.length]);
        const name = source.display_name || source.path.split(/[\\/]/).slice(-2).join("/");
        const entry = {id, name, source, original:index === 0, color:colors.get(id), configs:[]};
        const hardware = bundle.overview.find(item => item.source === source.path)?.identity || {};
        entry.configs = configs.map(config => ({...config, experiment_id:id, experiment_name:name,
          hardware:hardware.cpu_model || hardware.platform_fingerprint || "unknown"}));
        entries.push(entry); data.configs.push(...entry.configs);
      }
      for (const name of ["sources", "overview", "window_statistics", "experiment_comparisons"]) data[name].push(...bundle[name]);
    }
  }
  function namespace(bundle) {
    const paths = new Map(bundle.sources.map(source => [source.path, `import:${source.sha256}/${source.original_path || source.path}`]));
    const ids = new Map(bundle.configs.map(config => [config.config_id, `import:${bundle.sources.find(source => source.path === config.source).sha256.slice(0,16)}/${config.original_config_id || config.config_id}`]));
    const copy = I.clone(bundle);
    copy.sources.forEach(source => { source.original_path ||= source.path; source.path = paths.get(source.path); });
    copy.configs.forEach(config => { config.original_config_id ||= config.config_id; config.config_id = ids.get(config.config_id); config.source = paths.get(config.source); });
    copy.overview.forEach(source => { source.source = paths.get(source.source); });
    copy.window_statistics.forEach(report => { report.original_source ||= report.source; report.source = paths.get(report.source); report.groups.forEach(group => { group.config_id = ids.get(group.config_id); }); });
    copy.experiment_comparisons.forEach(entry => {
      for (const side of ["left", "right"]) entry[side] = entry[side].map(path => paths.get(path));
      if (entry.report) {
        entry.report.original_source_sha256 ||= entry.report.source_sha256;
        entry.report.source_sha256 = Object.fromEntries(Object.entries(entry.report.source_sha256).map(([path, hash]) => [paths.get(path) || path, hash]));
      }
    });
    copy.baseline_comparisons.pairs = Object.fromEntries(Object.entries(copy.baseline_comparisons.pairs).map(([id, pairs]) =>
      [ids.get(id), Object.fromEntries(Object.entries(pairs).map(([other, pair]) => [ids.get(other), pair]))]));
    return copy;
  }
  function add(bundles, name, restoring = false) {
    const fingerprints = data.sources.map(source => ({hash:source.sha256, runs:runsFor(data, source)}));
    const runs = new Set(data.configs.map(config => config.run_id).filter(known));
    const candidates = [];
    for (const raw of bundles) {
      const bundle = I.validate(raw, original.registry);
      for (const source of bundle.sources) {
        const recordedRuns = runsFor(bundle, source);
        if (fingerprints.some(previous => previous.hash === source.sha256 &&
            (!previous.runs.length || !recordedRuns.length || recordedRuns.some(run => previous.runs.includes(run))))) {
          throw new Error("重复导入 / Duplicate source: " + (source.original_path || source.path));
        }
        fingerprints.push({hash:source.sha256, runs:recordedRuns});
        for (const run of recordedRuns) {
          if (runs.has(run)) throw new Error("重复或冲突的 run ID / Duplicate or conflicting run ID: " + run);
          runs.add(run);
        }
      }
      candidates.push({name, bundle:restoring ? bundle : namespace(bundle)});
    }
    imports.push(...candidates); rebuild();
  }
  function remove(id) {
    if (entries.find(entry => entry.id === id)?.original) return;
    for (let index = imports.length - 1; index >= 0; index--) {
      const bundle = imports[index].bundle;
      const paths = new Set(bundle.sources.filter(source => experimentId(bundle, source) === id).map(source => source.path));
      imports[index].bundle = subset(bundle, bundle.configs.filter(config => !paths.has(config.source)));
      if (!imports[index].bundle.configs.length) imports.splice(index, 1);
    }
    colors.delete(id); rebuild();
  }
  rebuild();
  if (restored) {
    if (restored.schema_version !== 1 || !Array.isArray(restored.imports)) throw new Error("会话版本不兼容 / Unsupported session version");
    add(restored.imports.map(item => item.bundle), "restored", true);
  }
  return {data, add, remove, entries:() => entries, color:config => colors.get(config.experiment_id) || palette[0],
    snapshot:state => ({schema_version:1, imports, state, colors:[...colors]}), bundle:configs => subset(data, configs)};
};

window.ACProfExperimentManager = ({workspace, $, element, option, refresh, schedule, state, getState, template}) => {
  let controller = null;
  const data = workspace.data;
  function render() {
    $("experiment-list").tBodies[0].replaceChildren();
    const left = $("active-experiment-left").value, right = $("active-experiment-right").value;
    for (const id of ["active-experiment-left", "active-experiment-right"]) {
      $(id).replaceChildren(); option($(id), "", "全部 / All");
    }
    for (const entry of workspace.entries()) {
      const row = element("tr"), identity = element("td", entry.name);
      identity.style.borderLeft = `4px solid ${entry.color}`;
      row.dataset.experimentId = entry.id;
      row.append(identity, element("td", [...new Set(entry.configs.map(config => config.model))].join(", ")),
        element("td", [...new Set(entry.configs.map(config => config.run_id))].join(", ")),
        element("td", entry.source.original_path || entry.source.path), element("td", entry.configs.length),
        element("td", [...new Set(entry.configs.map(config => config.quality_status))].join(", ")));
      const action = element("td");
      if (entry.original) action.append(element("span", "原始内嵌 / Embedded"));
      else {
        const remove = element("button", "删除 / Remove"); remove.dataset.remove = entry.id;
        remove.onclick = () => { workspace.remove(entry.id); if (!data.configs.some(config => config.config_id === state.selected)) state.selected = null; render(); refresh(); schedule(); };
        action.append(remove);
      }
      row.append(action); $("experiment-list").tBodies[0].append(row);
      for (const id of ["active-experiment-left", "active-experiment-right"]) option($(id), entry.id, entry.name + " · " + entry.configs[0].run_id);
    }
    $("active-experiment-left").value = workspace.entries().some(entry => entry.id === left) ? left : "";
    $("active-experiment-right").value = workspace.entries().some(entry => entry.id === right) ? right : "";
  }
  async function importFiles(files) {
    if (controller || !files.length) return;
    controller = new AbortController(); document.body.dataset.importing = "true";
    $("cancel-import").hidden = false; $("add-experiment").disabled = true; $("import-progress").hidden = false;
    const messages = [];
    try {
      for (const file of files) {
        try {
          window.ACProfImport.checkCancelled(controller.signal);
          $("import-status").textContent = "正在导入 / Importing: " + file.name;
          $("import-progress").value = 0;
          const bundles = await window.ACProfImport.read(file, data.registry, {signal:controller.signal,
            onProgress: progress => { $("import-progress").value = progress; }});
          window.ACProfImport.checkCancelled(controller.signal);
          workspace.add(bundles, file.name);
          messages.push("已添加 / Added: " + file.name);
          render(); refresh(); schedule();
        } catch (error) {
          messages.push(file.name + ": " + error.message);
          if (error.name === "AbortError") break;
        }
      }
    } finally {
      controller = null; document.body.dataset.importing = "false";
      $("cancel-import").hidden = true; $("add-experiment").disabled = false; $("import-progress").hidden = true;
      $("import-status").textContent = messages.join("\n"); $("experiment-files").value = "";
    }
  }
  $("add-experiment").onclick = () => $("experiment-files").click();
  $("experiment-files").onchange = event => importFiles([...event.target.files]);
  $("cancel-import").onclick = () => controller?.abort();
  $("experiment-manager").ondragover = event => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; };
  $("experiment-manager").ondrop = event => { event.preventDefault(); importFiles([...event.dataTransfer.files]); };
  $("active-experiment-left").onchange = () => {
    const entry = workspace.entries().find(entry => entry.id === $("active-experiment-left").value);
    $("baseline").value = entry?.configs[0].config_id || ""; schedule();
  };
  $("active-experiment-right").onchange = schedule;
  $("export-html").onclick = () => {
    const json = JSON.stringify(workspace.snapshot(getState())).replace(/[<>&]/g, character => ({"<":"\\u003c", ">":"\\u003e", "&":"\\u0026"})[character]);
    const html = template.replace(/(<script id="report-session" type="application\/json">)[\s\S]*?(<\/script>)/,
      (_match, before, after) => before + json + after);
    window.ACProfDownload(html, "text/html;charset=utf-8", "acprof-comparison.html");
  };
  render(); document.body.dataset.importing = "false";
  return {render};
};
