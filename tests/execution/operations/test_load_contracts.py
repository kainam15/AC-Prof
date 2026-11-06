"""Legacy execution imports expose the canonical shared contract objects."""
from acprof.experiments import load as load_contract, streaming as request_contract
from acprof.host.measurement import load, request_inputs


def test_existing_contract_imports_keep_the_shared_objects():
    assert load.LoadConfig is load_contract.LoadConfig
    assert load.MAX_SWEEP_POINTS == load_contract.MAX_SWEEP_POINTS
    assert request_inputs.request_digest is request_contract.request_digest
    assert request_inputs.validate_request_credentials is request_contract.validate_request_credentials
