"""Registry isolation and actionable selected-backend failures."""
import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from acprof.container import handlers


class TestHandlerRegistry:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.stack = ExitStack()
        self._request.addfinalizer(partial(self.stack.close))
        self.stack.enter_context(patch.dict(handlers.HandlerRegistry._handlers, clear=True))
        self.stack.enter_context(patch.dict(handlers.HandlerRegistry._adapters, clear=True))
        self.stack.enter_context(patch.dict(os.environ, {"ACPROF_MODEL_ADAPTER": "family-default"}))
        monkeypatch.delenv("ACPROF_RUNTIME_PROFILE", raising=False)

    def test_normal_registration_and_duplicate_rejected_before_construction(self):
        class First:
            pass

        class Replacement:
            def __init__(self):
                self.fail = "must not be constructed on duplicate"

        handlers.HandlerRegistry.register("test", "runtime", First)
        with pytest.raises(ValueError, match="duplicate handler key") as caught:
            handlers.HandlerRegistry.register("test", "runtime", Replacement)
        assert (type(caught.value).__name__) == ("DuplicateHandlerRegistrationError")
        for detail in ("test:runtime", "First", "Replacement", __name__):
            assert (detail) in (str(caught.value))
        assert isinstance(handlers.HandlerRegistry.get("test", "runtime"), First)

    def test_explicit_override_and_adapter_duplicates(self):
        class First:
            pass

        class Second:
            pass

        handlers.HandlerRegistry.register("test", "runtime", First)
        original = handlers.HandlerRegistry.get("test", "runtime")
        handlers.HandlerRegistry.register("test", "runtime", Second, override=True)
        replacement = handlers.HandlerRegistry.get("test", "runtime")
        assert isinstance(replacement, Second)
        assert replacement is not original
        handlers.HandlerRegistry.register_adapter("a", "test", "runtime", First)
        handlers.HandlerRegistry.register_adapter("b", "test", "runtime", First)
        first = handlers.HandlerRegistry.get("test", "runtime", adapter="a")
        other = handlers.HandlerRegistry.get("test", "runtime", adapter="b")
        assert first is not other
        with pytest.raises(ValueError, match="duplicate handler key") as caught:
            handlers.HandlerRegistry.register_adapter("a", "test", "runtime", Second)
        assert (type(caught.value).__name__) == ("DuplicateHandlerRegistrationError")
        assert handlers.HandlerRegistry.get("test", "runtime", adapter="a") is first
        handlers.HandlerRegistry.register_adapter("a", "test", "runtime", Second, override=True)
        overridden = handlers.HandlerRegistry.get("test", "runtime", adapter="a")
        assert isinstance(overridden, Second)
        assert overridden is not first
        assert handlers.HandlerRegistry.get("test", "runtime", adapter="a") is overridden
        assert handlers.HandlerRegistry.get("test", "runtime", adapter="b") is other

    def test_explicit_adapter_and_default_do_not_follow_later_environment_changes(self, monkeypatch):
        handlers.HandlerRegistry.register("test", "runtime", object)
        handlers.HandlerRegistry.register_adapter("a", "test", "runtime", object)
        handlers.HandlerRegistry.register_adapter("b", "test", "runtime", object)
        default = handlers.HandlerRegistry.get("test", "runtime")
        first = handlers.HandlerRegistry.get("test", "runtime", adapter="a")
        second = handlers.HandlerRegistry.get("test", "runtime", adapter="b")
        monkeypatch.setenv("ACPROF_MODEL_ADAPTER", "b")

        assert handlers.HandlerRegistry.get("test", "runtime") is default
        assert handlers.HandlerRegistry.get("test", "runtime", adapter="a") is first
        assert handlers.HandlerRegistry.get("test", "runtime", adapter="b") is second
        assert len({id(default), id(first), id(second)}) == 3
        with pytest.raises(handlers.HandlerNotRegisteredError, match="adapter=unknown"):
            handlers.HandlerRegistry.get("test", "runtime", adapter="unknown")

    def test_package_import_does_not_import_optional_handlers(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import acprof.container.handlers; "
             "assert 'acprof.container.handlers.nlp' not in sys.modules; "
             "assert 'acprof.container.handlers.audio' not in sys.modules; "
             "assert not {'torch', 'transformers', 'diffusers', 'onnxruntime'} & sys.modules.keys()"],
            capture_output=True, text=True,
        )
        assert (result.returncode) == (0), result.stderr

    def _module(self, source, *, adapter="family-default"):
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        name = "acprof_registry_fixture"
        Path(directory, name + ".py").write_text(source, encoding="utf-8")
        self.stack.enter_context(patch.object(sys, "path", [directory, *sys.path]))
        self._request.addfinalizer(partial(sys.modules.pop, name, None))
        handlers.HandlerRegistry.register_lazy("test", "optional", name + ":Handler", adapter=adapter)
        assert (name) not in (sys.modules)
        return name

    def test_lazy_load_constructs_once(self):
        self._module("class Handler: pass\n")
        first = handlers.HandlerRegistry.get("test", "optional")
        assert (first) is (handlers.HandlerRegistry.get("test", "optional"))

    def test_unselected_missing_dependency_does_not_block_registration(self):
        self._module(
            "import acprof_missing_optional_dependency\nclass Handler: pass\n", adapter="unselected",
        )
        handlers.HandlerRegistry.register("test", "optional", object)
        assert isinstance(handlers.HandlerRegistry.get("test", "optional"), object)

    def test_selected_missing_dependency_preserves_original_exception(self):
        name = self._module("import acprof_missing_optional_dependency\nclass Handler: pass\n")
        with pytest.raises(ValueError, match="missing dependency: acprof_missing_optional_dependency") as caught:
            handlers.HandlerRegistry.get("test", "optional")
        assert (type(caught.value).__name__) == ("HandlerDependencyMissingError")
        for detail in ("optional", name, "acprof_missing_optional_dependency", "ModuleNotFoundError"):
            assert (detail) in (str(caught.value))
        assert isinstance(caught.value.__cause__, ModuleNotFoundError)
        handlers.HandlerRegistry.register_adapter("available", "test", "optional", object)
        other = handlers.HandlerRegistry.get("test", "optional", adapter="available")
        self.stack.enter_context(patch.dict(
            sys.modules, {"acprof_missing_optional_dependency": ModuleType("acprof_missing_optional_dependency")},
        ))
        recovered = handlers.HandlerRegistry.get("test", "optional")
        assert handlers.HandlerRegistry.get("test", "optional") is recovered
        assert handlers.HandlerRegistry.get("test", "optional", adapter="available") is other

    def test_internal_module_failure_is_distinct_from_missing_dependency(self):
        self._module("raise RuntimeError('module defect')\n")
        with pytest.raises(ValueError, match="RuntimeError: module defect") as caught:
            handlers.HandlerRegistry.get("test", "optional")
        assert (type(caught.value).__name__) == ("HandlerModuleImportError")
        assert isinstance(caught.value.__cause__, RuntimeError)

    def test_handler_constructor_failure_is_distinct(self):
        self._module("class Handler:\n def __init__(self): raise RuntimeError('constructor defect')\n")
        with pytest.raises(ValueError, match="RuntimeError: constructor defect") as caught:
            handlers.HandlerRegistry.get("test", "optional")
        assert (type(caught.value).__name__) == ("HandlerInitializationError")
        assert isinstance(caught.value.__cause__, RuntimeError)

    def test_constructor_failure_can_retry_without_affecting_another_adapter(self):
        attempts = []
        cause = RuntimeError("constructor interrupted")

        class Flaky:
            def __init__(self):
                attempts.append("construct")
                if len(attempts) == 1:
                    raise cause

        handlers.HandlerRegistry.register_adapter("a", "test", "runtime", Flaky)
        handlers.HandlerRegistry.register_adapter("b", "test", "runtime", object)
        with pytest.raises(handlers.HandlerInitializationError, match="constructor interrupted") as caught:
            handlers.HandlerRegistry.get("test", "runtime", adapter="a")
        assert caught.value.__cause__ is cause
        other = handlers.HandlerRegistry.get("test", "runtime", adapter="b")
        recovered = handlers.HandlerRegistry.get("test", "runtime", adapter="a")
        assert isinstance(recovered, Flaky)
        assert handlers.HandlerRegistry.get("test", "runtime", adapter="a") is recovered
        assert handlers.HandlerRegistry.get("test", "runtime", adapter="b") is other
        assert attempts == ["construct", "construct"]

    def test_unknown_handler_is_distinct(self):
        with pytest.raises(ValueError, match="Handler not registered") as caught:
            handlers.HandlerRegistry.get("test", "unknown")
        assert (type(caught.value).__name__) == ("HandlerNotRegisteredError")

    def test_selected_handler_load_missing_dependency_is_actionable(self):
        class Handler:
            def load(self, *_args, adapter="family-default"):
                import acprof_missing_runtime_dependency  # noqa: F401 -- 验证缺失依赖的失败路径。

        with pytest.raises(ValueError, match="missing dependency: acprof_missing_runtime_dependency") as caught:
            handlers.load_handler(Handler(), "/model", "text-generation", "transformers_pipeline", "cpu")
        assert (type(caught.value).__name__) == ("HandlerDependencyMissingError")
        assert ("acprof_missing_runtime_dependency") in (str(caught.value))
        assert ("transformers_pipeline") in (str(caught.value))
        assert isinstance(caught.value.__cause__, ModuleNotFoundError)

    def test_selected_handler_load_failure_preserves_exception_chain(self):
        class Handler:
            def load(self, *_args, adapter="family-default"):
                raise RuntimeError("corrupt model")

        with pytest.raises(ValueError, match="RuntimeError: corrupt model") as caught:
            handlers.load_handler(Handler(), "/model", "text-generation", "transformers_pipeline", "cpu")
        assert (type(caught.value).__name__) == ("HandlerInitializationError")
        assert isinstance(caught.value.__cause__, RuntimeError)

    def test_model_loading_failure_does_not_leave_shared_context(self):
        attempts = []
        cause = RuntimeError("incomplete model load")

        class Handler:
            def load(self, model_source, *_args, adapter="family-default"):
                attempts.append(model_source)
                if len(attempts) == 1:
                    raise cause
                return {"model": object(), "source": model_source}

        handlers.HandlerRegistry.register("nlp", "transformers_pipeline", Handler)
        selected = handlers.HandlerRegistry.get("nlp", "transformers_pipeline")
        with pytest.raises(handlers.HandlerInitializationError, match="incomplete model load") as caught:
            handlers.load_handler(selected, "fixture/failed", "text-generation", "transformers_pipeline", "cpu")
        assert caught.value.__cause__ is cause
        assert handlers.HandlerRegistry.get("nlp", "transformers_pipeline") is selected
        recovered = handlers.load_handler(
            selected, "fixture/recovered", "text-generation", "transformers_pipeline", "cpu",
        )
        assert recovered["source"] == "fixture/recovered"
        assert "model" in recovered
        assert vars(selected) == {}
        assert attempts == ["fixture/failed", "fixture/recovered"]

    def test_manifest_validator_is_invoked_and_original_task_error_preserved(self):
        self._module("def validate(*args):\n return {'sentinel': args[1]['value']}\n"
                     "def fail(*args):\n raise ValueError('task sanity failed')\n"
                     "not_callable = 7\n")
        context = {"_validation_entrypoint": "acprof_registry_fixture:validate"}
        result = handlers.BaseHandler.validate_output(None, context, {"value": 42}, {}, {}, {})
        assert (result) == ({"sentinel": 42})
        context["_validation_entrypoint"] = "acprof_registry_fixture:fail"
        with pytest.raises(ValueError, match="task sanity failed"):
            handlers.BaseHandler.validate_output(None, context, {}, {}, {}, {})
        context["_validation_entrypoint"] = "acprof_registry_fixture:not_callable"
        with pytest.raises(TypeError, match="callable"):
            handlers.BaseHandler.validate_output(None, context, {}, {}, {}, {})
