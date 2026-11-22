"""Rolling samples select distinct repositories from the Hub's ranked pages."""

import json

import httpx
import pytest

from acprof.cli.main import main


def candidate(repo, *, library="transformers", downloads=7, sha="a" * 40):
    return {"id": f"fixture/{repo}", "sha": sha, "downloads": downloads,
            "library_name": library, "pipeline_tag": "feature-extraction"}


@pytest.fixture
def ranked_pages(monkeypatch):
    pages = []
    requests = []

    def send(client, request, **kwargs):
        assert request.url.path == "/api/models"
        page = int(request.url.params.get("page", 0))
        requests.append(request.url.params)
        headers = {}
        if page + 1 < len(pages):
            next_url = request.url.copy_set_param("page", str(page + 1))
            headers["Link"] = f'<{next_url}>; rel="next"'
        return httpx.Response(200, request=request, json=pages[page], headers=headers)

    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", lambda *_: None)
    monkeypatch.setattr("acprof.host.preparation.model_coverage.hf_token_for_url", lambda *_: False)
    monkeypatch.setattr(httpx.Client, "send", send)
    return pages, requests


def snapshot(path, limit):
    return main(["coverage", "snapshot", "--stratum", "feature-extraction:transformers",
                 "--limit", str(limit), "--output", str(path)])


def test_exact_library_can_first_match_beyond_twenty_candidates(tmp_path, ranked_pages):
    pages, requests = ranked_pages
    pages.extend([[candidate(str(i), library="sentence-transformers") for i in range(20)],
                  [candidate("first", downloads=0)], [candidate("unused", downloads=0)]])
    output = tmp_path / "sample.json"
    assert snapshot(output, 1) == 0
    models = json.loads(output.read_text())["models"]
    assert [(m["model_id"], m["weight"]) for m in models] == [("fixture/first", 0)]
    assert [r.get("page") for r in requests] == [None, "1"]


@pytest.mark.parametrize("second_sha", ["a" * 40, "b" * 40])
def test_repeated_repository_between_pages_keeps_first_rank(tmp_path, ranked_pages, second_sha):
    pages, requests = ranked_pages
    pages.extend([[candidate("first")], [candidate("first", sha=second_sha, downloads=6),
                                         candidate("second", sha="c" * 40, downloads=5)]])
    output = tmp_path / "sample.json"
    assert snapshot(output, 2) == 0
    models = json.loads(output.read_text())["models"]
    assert [(m["model_id"], m["revision"], m["weight"]) for m in models] == [
        ("fixture/first", "a" * 40, 7), ("fixture/second", "c" * 40, 5)]
    assert [r.get("page") for r in requests] == [None, "1"]


def test_unknown_downloads_cannot_be_frozen_as_zero(tmp_path, ranked_pages, capsys):
    pages, _requests = ranked_pages
    pages.append([candidate("known"), candidate("unweighted", downloads=None), candidate("other")])
    output = tmp_path / "sample.json"
    assert snapshot(output, 2) == 2
    error = capsys.readouterr().err
    assert "fixture/unweighted" in error
    assert "downloads" in error
    assert not output.exists()


def test_exhausted_stratum_keeps_only_available_exact_matches(tmp_path, ranked_pages):
    pages, requests = ranked_pages
    pages.extend([[candidate("wrong-task") | {"pipeline_tag": "fill-mask"}], [candidate("only")]])
    output = tmp_path / "sample.json"
    assert snapshot(output, 3) == 0
    assert [m["model_id"] for m in json.loads(output.read_text())["models"]] == ["fixture/only"]
    assert [r.get("page") for r in requests] == [None, "1"]


def test_small_snapshot_requests_a_bounded_page_of_ranking_fields(tmp_path, ranked_pages):
    pages, requests = ranked_pages
    pages.append([candidate("first")])
    assert snapshot(tmp_path / "sample.json", 1) == 0
    params, = requests
    assert 0 < int(params.get("limit", "1000")) <= 100
    assert set(params.get_list("expand")) == {"library_name", "pipeline_tag", "sha", "downloads"}
    assert params["filter"] == "transformers"
    assert params["pipeline_tag"] == "feature-extraction"
    assert params["sort"] == "downloads"
    assert "full" not in params


def test_repeating_a_stratum_observes_the_ranking_once(tmp_path, ranked_pages):
    pages, requests = ranked_pages
    pages.append([candidate("first")])
    output = tmp_path / "sample.json"
    stratum = "feature-extraction:transformers"
    assert main(["coverage", "snapshot", "--stratum", stratum, "--stratum", stratum,
                 "--limit", "1", "--output", str(output)]) == 0
    assert len(requests) == 1
    assert json.loads(output.read_text())["strata"] == [stratum]


@pytest.mark.parametrize("detail_sha", ["c" * 40, None])
def test_missing_list_revision_uses_only_the_detail_sha(tmp_path, ranked_pages, monkeypatch, detail_sha, capsys):
    from huggingface_hub import ModelInfo

    pages, _requests = ranked_pages
    pages.append([candidate("unresolved", sha=None)])

    def model_info(api, repo_id, *, expand):
        assert repo_id == "fixture/unresolved"
        assert expand == ["sha"]
        return ModelInfo(id=repo_id, sha=detail_sha)

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", model_info)
    output = tmp_path / "sample.json"
    if detail_sha is None:
        assert snapshot(output, 1) == 2
        assert "fixture/unresolved" in capsys.readouterr().err
        assert not output.exists()
    else:
        assert snapshot(output, 1) == 0
        model, = json.loads(output.read_text())["models"]
        assert (model["revision"], model["weight"]) == (detail_sha, 7)
