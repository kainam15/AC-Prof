"""Window quality is applied to the pairs used by the figure before taking medians."""
import json

import pandas as pd
import pytest

from acprof.analysis.latency_model import latency_model_frame
from acprof.plotting import diagnostics, tail_latency

P50, P90, P95 = ("latency_p50_s", "latency_p90_s", "latency_p95_s")
ENERGY = "container_attributed_energy_eff_j"


def observation(quality="complete", method="a"):
    return {"quality": quality, "sampling": {"method": method}}


def window(value, evidence=None, **kwargs):
    return {"cpu_cores": 2, "mem_cap_gb": 4, "gpu_mode": "off", "input_scale": 1,
            "packet_status": "ok", "latency_s": value, P50: value, P90: value, P95: value, ENERGY: value,
            "resource_metric_quality": json.dumps({"schema_version": 1, "metrics": evidence or {},
                                                   "collectors": {"packet_latency": {"status": "available"}}}),
            **kwargs}


def points(monkeypatch, frame):
    seen: list[tuple[float, float]] = []
    frontier = diagnostics._pareto_frontier_mask

    def collect(latency, energy):
        seen.extend(zip(latency.tolist(), energy.tolist(), strict=True))
        return frontier(latency, energy)

    monkeypatch.setattr(diagnostics, "_pareto_frontier_mask", collect)
    diagnostics.plot_latency_energy_pareto(frame, xlabel="size", out_png=None)
    return seen


def test_partial_packet_retains_unaffected_fields_and_excludes_its_entire_pair(monkeypatch):
    frame = pd.DataFrame([window(1), window(101, packet_status="partial")], index=[7, 7])
    frame["latency_app_p50_s"] = [8., 10.]
    frame["latency_app_p95_s"] = [8., 10.]
    tail = tail_latency._aggregate_tail_latency(frame).iloc[0]
    assert tail[P95] == 1
    assert tail["latency_app_p95_s"] == 9
    assert latency_model_frame(frame)["latency_s"].tolist() == [1]
    assert points(monkeypatch, frame.drop(columns="latency_app_p95_s")) == [(1, 1)]


def test_mixed_signatures_drop_only_affected_scale_and_family(monkeypatch):
    fields = (P50, P95, "latency_s")
    frame = pd.DataFrame([window(1, {name: observation() for name in fields}),
                          window(101, {name: observation(method="b") for name in fields}),
                          window(2, {name: observation() for name in fields}, input_scale=2)])
    tail = tail_latency._aggregate_tail_latency(frame)
    assert pd.isna(tail.iloc[0][P95])
    assert tail.iloc[1][P95] == 2
    assert latency_model_frame(frame)["input_scale"].tolist() == [2]
    assert points(monkeypatch, frame) == [(2, 2)]


@pytest.mark.parametrize(("optional", "expected"), [(observation("partial"), 2.), (observation(method="b"), None)])
def test_optional_p90_does_not_drop_an_eligible_pair(optional, expected):
    frame = pd.DataFrame([window(1, {P90: observation()}, **{P90: 2., P95: 4.}),
                          window(3, {P90: optional}, **{P90: 4., P95: 6.})])
    tail = tail_latency._aggregate_tail_latency(frame).iloc[0]
    assert (tail[P50], tail[P95]) == (2, 5)
    if expected is None:
        assert pd.isna(tail[P90])
    else:
        assert tail[P90] == expected


def test_only_participating_pairs_can_contribute_conflicting_signatures(monkeypatch):
    frame = pd.DataFrame([window(7, {P50: observation(), P95: observation(), ENERGY: observation()}),
                          window(99, {P50: observation("partial"), P95: observation(method="b"),
                                      ENERGY: observation("partial")})])
    assert tail_latency._aggregate_tail_latency(frame)[P95].tolist() == [7]
    assert points(monkeypatch, frame) == [(7, 7)]


def test_pareto_selects_one_usable_latency_definition_for_the_whole_figure(monkeypatch):
    frame = pd.DataFrame([window(1, {"latency_app_p95_s": observation("partial")}, latency_app_p95_s=90.),
                          window(3, {"latency_app_p95_s": observation("partial")}, latency_app_p95_s=99.)])
    assert points(monkeypatch, frame) == [(2, 2)]


def test_legacy_unpaired_energy_is_not_moved_to_another_latency_definition(monkeypatch):
    frame = pd.DataFrame([window(1, latency_app_p95_s=5., **{ENERGY: float("nan")}),
                          window(2, latency_app_p95_s=float("nan"))]).drop(columns="resource_metric_quality")
    assert points(monkeypatch, frame) == []
