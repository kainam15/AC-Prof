"""PyPI release selection must inspect files, not the info.version shortcut."""
import httpx
import pytest
from packaging.version import Version

from acprof.tui.updates import check_pypi_version
from acprof.updates.releases import select_releases, version_type


@pytest.mark.parametrize(("releases", "expected"), [
    ({"0.3.0": [{"yanked": False, "requires_python": ">=3.12"}],
      "0.4.0a1": [{"yanked": False}]}, "0.3.0"),
    ({"0.3.0": [{"yanked": False}], "0.4.0": [{"yanked": True}]}, "0.3.0"),
    ({"0.3.0": [{"yanked": False}],
      "0.4.0": [{"yanked": False, "requires_python": ">=99"}]}, "0.3.0"),
])
async def test_files_determine_stable_candidate(monkeypatch, releases, expected):
    async def respond(_transport, request):
        return httpx.Response(200, json={
            "info": {"version": "0.4.0a1", "yanked": False}, "releases": releases,
        })

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    result = await check_pypi_version("0.2.0")
    assert str(result.latest) == expected


@pytest.mark.parametrize(("version", "kind"), [
    ("1.0.0a1", "Alpha"), ("1.0.0b2", "Beta"), ("1.0.0rc3", "RC"),
    ("1.0.0", "Stable"), ("1.0.0.post1", "Stable"), ("1.0.0.dev1", "Dev"),
])
def test_release_type(version, kind):
    assert version_type(Version(version)) == kind


@pytest.mark.parametrize(("current", "versions", "channel", "target", "upgrade"), [
    ("1.0.0", ["1.0.0", "1.1.0a1"], "stable", "1.0.0", False),
    ("1.0.0", ["1.0.0", "1.1.0a1"], "preview", "1.1.0a1", True),
    ("1.1.0a1", ["1.0.0", "1.1.0a1"], "stable", "1.0.0", False),
    ("1.1.0a1", ["1.1.0b1", "1.1.0rc1"], "preview", "1.1.0rc1", True),
    ("1.1.0rc1", ["1.1.0rc1", "1.1.0"], "preview", "1.1.0", True),
    ("1.1.0", ["1.1.0rc1", "1.1.0"], "preview", "1.1.0", False),
    ("1.0.0", ["1.1.0a1"], "stable", None, False),
    ("1.0.0", [], "preview", None, False),
    ("1.0.0", ["99.0.0.dev1", "99.0.0+local", "invalid"], "preview", None, False),
])
def test_channel_candidate_sets(current, versions, channel, target, upgrade):
    result = select_releases({"releases": {v: [{"yanked": False}] for v in versions}}, current, channel)
    assert (str(result.latest) if result.latest else None) == target
    assert result.update_available == upgrade


def test_file_eligibility_any_compatible_non_yanked_file():
    releases = {
        "1.0.0": [{"yanked": False, "requires_python": ">=3.12,<3.13"}],
        "1.1.0": [{"yanked": True}, {"yanked": False, "requires_python": ">=99"}],
        "1.2.0": [{"yanked": False, "requires_python": "invalid"}],
        "1.3.0": [], "1.4.0": [{}],
        "1.5.0": [{"yanked": True}, {"yanked": False, "requires_python": ">=3.12"}],
        "2.0.0rc1": [{"yanked": False, "requires_python": ">=3.12"}],
    }
    result = select_releases({"releases": releases}, "1.0.0", "preview", python="3.12.5")
    assert str(result.latest_stable) == "1.5.0"
    assert str(result.latest_preview) == "2.0.0rc1"


def test_binary_only_updates_check_the_published_wheel_identity_and_tags():
    def wheel(filename, **fields):
        return {"yanked": False, "packagetype": "bdist_wheel", "filename": filename, **fields}

    releases = {
        "1.0": [wheel("acprof-1.0-py2.py3-none-any.whl")],
        "1.1": [wheel("acprof-1.1-py3-none-any.whl", yanked=True), wheel("acprof-1.1-py3-none-acme.whl")],
        "1.2": [wheel("acprof-1.2-py3-none-any.whl", requires_python=">=99")],
        "1.3": [wheel("another-1.3-py3-none-any.whl")],
        "1.4": [wheel("acprof-2.0-py3-none-any.whl")],
        "1.5": [wheel("malformed.whl"), wheel(None)],
        "1.6b1": [wheel("acprof-1.6b1-py3-none-any.whl")],
        "1.7": [{"yanked": False, "packagetype": "sdist", "filename": "acprof-1.7.tar.gz"}],
    }
    result = select_releases({"releases": releases}, "0.9", "preview", wheel_only=True)
    assert result.latest_stable == Version("1.0")
    assert result.latest == Version("1.6b1")
    assert select_releases({"releases": releases}, "0.9").latest == Version("1.7")
