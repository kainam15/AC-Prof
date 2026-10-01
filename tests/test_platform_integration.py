"""Real local identity checks; these do not certify hardware measurement semantics."""
try:
    import pytest
except ImportError:
    pytest = None

from acprof.platform import detect_environment


@pytest.mark.wsl if pytest else lambda test: test
def test_actual_wsl_identity_stays_partial():
    identity = detect_environment().metadata()
    assert identity["platform"]["native"] is False
    assert identity["collection_tier"] == "partial"
    assert identity["comparability_class"] == "wsl2"
    assert identity["platform"]["kernel"]
    assert identity["platform"]["wsl"]["generation"] == 2


@pytest.mark.native_linux if pytest else lambda test: test
def test_actual_native_environment_identity():
    identity = detect_environment().metadata()
    assert identity["platform"]["native"] is True
    assert identity["collection_tier"] == "full"
    assert identity["comparability_class"] == "native_linux"
