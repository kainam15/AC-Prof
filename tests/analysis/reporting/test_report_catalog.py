"""Published view catalogs remain distinct from the independently stored statistics metrics."""
import pytest

from acprof.analysis.model import load_analysis
from acprof.plotting.report import report_payload
from tests.analysis.reporting.test_report_import import (
    browser as browser,
    page as page,
    source as source,
)

APP_RATE = ["throughput_app_samples_per_s"]
HTTP_SIZES = ["request_payload_bytes", "response_payload_bytes"]
GPU_SENSORS = ["gpu_memory_clock_mhz", "gpu_sm_clock_mhz", "gpu_temp_c"]
PROCESS_COUNTERS = [
    "process_count_end", "process_cpu_system_time_s", "process_cpu_user_time_s", "process_fds_end",
    "process_io_read_bytes", "process_io_read_ops", "process_io_write_bytes", "process_io_write_ops",
    "process_leader_involuntary_ctx_switches", "process_leader_voluntary_ctx_switches",
    "process_major_faults", "process_minor_faults", "process_rss_growth_bytes", "process_swap_bytes_end",
    "process_threads_end", "service_cgroup_cpu_time_s", "service_cgroup_io_read_bytes",
    "service_cgroup_io_write_bytes", "service_cgroup_mem_usage_avg_bytes", "service_cgroup_mem_usage_peak_bytes",
]


@pytest.mark.parametrize(("size", "unreported"), [
    (45, APP_RATE + HTTP_SIZES + GPU_SENSORS + PROCESS_COUNTERS),
    (65, APP_RATE + HTTP_SIZES + GPU_SENSORS),
    (68, APP_RATE + HTTP_SIZES),
    (70, APP_RATE),
    (71, []),
], ids=["initial", "runtime-counters", "gpu-sensors", "http-sizes", "current"])
def test_published_view_catalog_round_trip_without_inventing_summaries(page, size, unreported):
    result = page.evaluate("""async omitted => {
      const base = JSON.parse(document.getElementById("report-data").textContent), raw = structuredClone(base);
      if (omitted.includes("process_fds_end")) delete raw.kind; // Original schema v1 HTML.
      for (const name of omitted) {
        delete raw.registry[name];
        for (const c of raw.configs) { delete c.metrics[name]; delete c.metric_observations[name]; }
      }
      const file = new File([JSON.stringify({analysis_bundle: raw})], "prior-export.json");
      const [read] = await ACProfImport.read(file, base.registry, {});
      const again = ACProfImport.validate(read, base.registry);
      return {size:Object.keys(raw.registry).length, canonicalRegistry:Object.keys(again.registry).sort(),
        expectedRegistry:Object.keys(base.registry).sort(), marked:again.sources[0].unreported_view_metrics ?? null,
        configsUnchanged:JSON.stringify(again.configs) === JSON.stringify(raw.configs),
        inputUnchanged:!Object.hasOwn(raw.sources[0], "unreported_view_metrics"),
        roundTrip:JSON.stringify(read) === JSON.stringify(again)};
    }""", unreported)
    assert result["size"] == size
    assert result["canonicalRegistry"] == result["expectedRegistry"]
    assert result["marked"] == (sorted(unreported) if unreported else None)
    assert result["configsUnchanged"]
    assert result["inputUnchanged"]
    assert result["roundTrip"]


def test_source_catalog_checks_each_config_and_baseline_pair(page):
    result = page.evaluate("""() => {
      const raw = JSON.parse(document.getElementById("report-data").textContent), I = ACProfImport;
      const next = "throughput_app_samples_per_s", latency = "latency_app_s";
      function check(edit) {
        const data = structuredClone(raw);
        edit(data, data.configs[0], data.sources[0]);
        try { I.validate(data, raw.registry); return "accepted"; } catch (e) { return e.message; }
      }
      const historical = (d,c,s) => { s.unreported_view_metrics = [next]; delete c.metrics[next]; };
      function cross(d,c,s) {
        const path = s.path + "-old", id = c.config_id + "-old";
        const old = {...structuredClone(c), source:path, config_id:id};
        d.sources.push({...s, path, sha256:"f".repeat(64), unreported_view_metrics:[next]});
        delete old.metrics[next]; d.configs.push(old);
        d.baseline_comparisons.qualifications.push({status:"descriptive",reasons:[]});
        const p = {resource:0,policies:{},metrics:{}};
        (d.baseline_comparisons.pairs[c.config_id] ??= {})[id] = p;
        return p;
      }
      return {
        validMixed:check(cross),
        completeGap:check(historical),
        unmarkedMissing:check((d,c) => { delete c.metrics[next]; }),
        unknownMarker:check((d,c,s) => { s.unreported_view_metrics = ["unknown"]; }),
        duplicateMarker:check((d,c,s) => { historical(d,c,s); s.unreported_view_metrics.push(next); }),
        invalidMarkerType:check((d,c,s) => { s.unreported_view_metrics = next; }),
        partialCatalog:check((d,c) => { delete d.registry.request_payload_bytes; delete c.metrics.request_payload_bytes; }),
        registryMarkerConflict:check((d,c,s) => { historical(d,c,s); delete d.registry[next]; s.unreported_view_metrics = []; }),
        declaredButMissing:check((d,c,s) => { historical(d,c,s); delete c.metrics[latency]; }),
        unitMismatch:check((d,c,s) => { historical(d,c,s); d.registry[latency].unit = "ms"; }),
        hiddenSummary:check((d,c,s) => { s.unreported_view_metrics = [next]; }),
        hiddenObservation:check((d,c,s) => { historical(d,c,s); c.metric_observations[next] = [{quality:"complete"}]; }),
        hiddenDelta:check((d,c,s) => { cross(d,c,s).metrics[next] = {delta:42,percent:0,ratio:1,performance_ratio:1}; }),
        hiddenReason:check((d,c,s) => { cross(d,c,s).metric_reasons = {[next]:"partial"}; })
      };
    }""")
    assert result.pop("validMixed") == "accepted"
    assert result.pop("completeGap") == "accepted"
    assert all(message != "accepted" for message in result.values()), result


def test_historical_view_omission_preserves_separate_statistics_inputs(page, tmp_path):
    metric = "request_payload_bytes"
    path = tmp_path / "body-size.csv"
    path.write_text(f"cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,{metric}\n"
                    "2,4,off,32,0,0,ok,100\n2,4,off,32,1,0,ok,200\n")
    payload = report_payload(load_analysis([path]), statistics_options={"metrics": [metric], "resamples": 10})
    config, = payload["configs"]
    for name in HTTP_SIZES + APP_RATE:
        del payload["registry"][name]
        del config["metrics"][name]
        config["metric_observations"].pop(name, None)
        for pairs in payload["baseline_comparisons"]["pairs"].values():
            for pair in pairs.values():
                pair["metrics"].pop(name, None)
    report, = payload["window_statistics"]
    group, = report["groups"]
    assert (group["metric"], group["mean"]) == (metric, 150)
    assert report["recalculation"]["groups"][0]["values"] == [100, 200]
    result = page.evaluate("""raw => {
      const {registry} = JSON.parse(document.getElementById("report-data").textContent);
      const checked = ACProfImport.validate(raw, registry);
      return {statisticsUnchanged:JSON.stringify(raw.window_statistics) === JSON.stringify(checked.window_statistics),
        reported:Object.hasOwn(checked.configs[0].metrics, "request_payload_bytes")};
    }""", payload)
    assert result == {"statisticsUnchanged": True, "reported": False}
