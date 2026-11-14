"""Registered non-text task parameter consumers shared by host and runtime."""
from acprof.extensions import get_extension
from acprof.extensions.parameters import validate_parameter_values


def task_consumer_specs(*, family: str, backend: str, adapter: str, task: str, values: dict) -> dict:
    declaration = get_extension(family, backend, adapter=adapter, task=task)
    specs = {spec.name: spec for spec in declaration.parameters if spec.scope == "inference"
             and spec.validation_method in {"generate.kwargs", "pipeline.predict.kwargs", "pipeline.call.kwargs"}}
    if not values or values.keys() - specs.keys():
        raise ValueError("task inference parameters have no registered observable consumer")
    validate_parameter_values(tuple(specs.values()), values, explicit_names=set(values))
    selected = {name: specs[name] for name in values}
    if len({spec.validation_method for spec in selected.values()}) != 1:
        raise ValueError("task inference parameters require different consumer boundaries")
    return selected
