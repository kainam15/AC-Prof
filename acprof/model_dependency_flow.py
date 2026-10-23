"""Bounded abstract interpretation of local loader calls, never repository execution.

Only JSON values, explicit local bindings and selected Transformers entrypoints
are understood. Unknown Python stays unknown; inactive calls remain in evidence.
"""
from __future__ import annotations

import ast
import copy
from pathlib import PurePosixPath
from typing import Any

from acprof.model_dependency_domain import MAX_CANDIDATES, UNKNOWN, Instance, Symbol
from acprof.model_dependency_eval import DependencyEvaluationMixin
from acprof.model_source_analysis import parse_source


class DependencyAnalyzer(DependencyEvaluationMixin):
    def __init__(self, sources, config, model_id, *, files=(), metadata=None, transformers_version=None):
        self.config = {**copy.deepcopy(config), "_name_or_path": model_id}
        self.model_id = model_id
        self.files = set(files)
        self.metadata = metadata or {}
        self.version = transformers_version
        self.trees = {name: parse_source(source, name) for name, source in sources.items()}
        self.functions = {}
        self.classes = {}
        self.modules = {}
        self.visited = set()
        self.stack = []
        self.candidates = []
        self.contexts = 0
        self.steps = 0
        self.transparent_calls = set()
        self.entries = {}
        for name, tree in self.trees.items():
            module = self.module_name(name)
            env: dict[str, Any] = {"config": copy.deepcopy(self.config)}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = module + "." + node.name
                    self.functions[key] = (name, None, node)
                    env[node.name] = Symbol(key)
                elif isinstance(node, ast.ClassDef):
                    key = module + "." + node.name
                    self.classes[key] = (name, node)
                    env[node.name] = Symbol(key)
                    for method in node.body:
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.functions[key + "." + method.name] = (name, key, method)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self.import_names(name, node, env)
            self.modules[name] = env
        for key, reference in config.get("auto_map", {}).items():
            if isinstance(reference, str) and reference in self.classes:
                self.entries[reference] = "config" if key == "AutoConfig" else "processor" if key == "AutoProcessor" else "model" if key.startswith("AutoModel") else "other"
        for pipeline in config.get("custom_pipelines", {}).values():
            if isinstance(pipeline, dict) and pipeline.get("impl") in self.classes:
                self.entries[pipeline["impl"]] = "pipeline"

    @staticmethod
    def module_name(filename):
        return filename.removesuffix(".py").replace("/", ".").removesuffix(".__init__")

    def import_names(self, filename, node, env):
        if isinstance(node, ast.Import):
            for alias in node.names:
                env[alias.asname or alias.name.split(".")[0]] = Symbol(alias.name if alias.asname else alias.name.split(".")[0])
            return
        prefix = node.module or ""
        if node.level:
            parent = PurePosixPath(filename).parent
            for _ in range(node.level - 1):
                parent = parent.parent
            prefix = ".".join([*(() if str(parent) == "." else parent.parts), *prefix.split(".")]).strip(".")
        for alias in node.names:
            env[alias.asname or alias.name] = Symbol((prefix + "." + alias.name).strip("."))

    def run(self):
        for filename, tree in self.trees.items():
            self.block(tree.body, self.modules[filename], filename, "active", [], None)
        for key, kind in self.entries.items():
            source_file, class_node = self.classes[key]
            if class_node.decorator_list:
                self.unresolved(source_file, class_node, "unknown", [], None, "declared loader class is decorated")
            obj = Instance(key, {"config": copy.deepcopy(self.config)})
            if kind == "config":
                self.invoke(key + ".__init__", [obj], copy.deepcopy(self.config), "active", [], None)
            elif kind == "pipeline":
                main = next((name for name, role in self.entries.items() if role == "model"), "main_model")
                model = Instance(main, {"config": copy.deepcopy(self.config)})
                self.invoke(key + ".__init__", [obj], {"model": model}, "active", [], None)
                for method in ("preprocess", "_forward", "postprocess", "_sanitize_parameters"):
                    self.invoke(key + "." + method, [obj], {}, "active", [], None)
            elif kind in {"model", "processor"}:
                self.invoke(key + ".from_pretrained", [Symbol(key), self.model_id], {}, "active", [], None)
                if kind == "model":
                    # Pinned Transformers 4.57.6 has two loader phases: construction
                    # under no_init_weights, then its registered initialization hook.
                    # These are API summaries, never arbitrary training-mode guesses.
                    supported = self.version == "4.57.6" and self.hf_model_class(key)
                    self.invoke(key + ".__init__", [obj, copy.deepcopy(self.config)], {}, "active", [], None,
                                init_weights=False if supported else UNKNOWN)
                    obj.attributes["_is_hf_initialized"] = False
                    if key + ".initialize_weights" in self.functions:
                        hook, hook_args = ".initialize_weights", [obj]
                    elif key + "._initialize_weights" in self.functions:
                        hook, hook_args = "._initialize_weights", [obj, obj]
                    else:
                        hook, hook_args = "._init_weights", [obj, obj]
                    self.invoke(key + hook, hook_args, {}, "active" if supported else "unknown", [], None,
                                init_weights=True if supported else UNKNOWN)
        # Unbound loader methods are not silently discarded. They remain review
        # candidates, with unknown arguments and reachability.
        for key, (_, owner, node) in self.functions.items():
            if key not in self.visited and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "from_pretrained" for n in ast.walk(node)):
                args: list[Any] = [Instance(owner)] if owner else []
                self.invoke(key, args, {}, "unknown", [], None, use_defaults=False)
        return self.candidates

    def hf_model_class(self, key):
        filename, node = self.classes[key]
        if node.decorator_list:
            return False
        for base in node.bases:
            value = self.value(base, self.modules[filename], filename, "inactive", [], None)
            if isinstance(value, Symbol) and value.name.startswith("transformers.") and value.name.endswith("PreTrainedModel"):
                return True
        return False

    @staticmethod
    def transparent_wrapper(node):
        if node.decorator_list or isinstance(node, ast.AsyncFunctionDef):
            return False
        body = [item for item in node.body if not (isinstance(item, ast.Expr)
                and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str))]
        if len(body) == 1 and isinstance(body[0], ast.Return):
            call = body[0].value
        elif (len(body) == 2 and isinstance(body[0], ast.Assign) and len(body[0].targets) == 1
              and isinstance(body[0].targets[0], ast.Name) and isinstance(body[1], ast.Return)
              and isinstance(body[1].value, ast.Name) and body[0].targets[0].id == body[1].value.id):
            call = body[0].value
        else:
            return False
        return isinstance(call, ast.Call) and not any(isinstance(part, ast.Call)
                   for arg in [*call.args, *(kw.value for kw in call.keywords)] for part in ast.walk(arg))

    def main_tokenizer_files_complete(self):
        metadata = self.metadata.get("tokenizer_config.json", {})
        return ({"tokenizer.json", "tokenizer_config.json"} <= self.files and not metadata.get("auto_map")
                and metadata.get("tokenizer_class") in {"PreTrainedTokenizer", "PreTrainedTokenizerFast"})

    def complete_primary(self, statements, candidates, filename):
        if len(statements) != 1 or not isinstance(statements[0], (ast.Assign, ast.AnnAssign, ast.Expr)):
            return False
        node = statements[0].value
        if not isinstance(node, ast.Call):
            return False
        if len(candidates) != 1:
            return False
        item = candidates[0]
        location = f"{filename}:{node.lineno}"
        if item["source"] != location and location not in self.transparent_calls:
            return False
        # Every wrapper below this try's call site must consist solely of
        # forwarding and returning the loader result, without extra operations.
        path = item["call_chain"]
        if location not in path:
            return False
        tail = path[path.index(location) + 1:]
        definitions = {f"{name}:{method.lineno}": method for name, _, method in self.functions.values()}
        if any(not self.transparent_wrapper(definitions[source]) for source in tail if source in definitions):
            return False
        return (item["dependency_kind"] == "main_model" and item["role"] == "tokenizer"
                and item["activation"] == "active" and item["requested_revision"] == "main"
                and item.get("file_availability") == "complete")

    def append(self, candidate):
        if len(self.candidates) >= MAX_CANDIDATES:
            raise ValueError("dependency candidate limit exceeded")
        if candidate not in self.candidates:
            self.candidates.append(candidate)

    def unresolved(self, filename, node, state, chain, alternative, reason):
        item = {"repo_id": None, "role": "unknown", "loader": None, "required": "candidate",
                "activation": state, "dependency_kind": "external", "state": "unresolved",
                "source": f"{filename}:{node.lineno}", "expression": reason, "call_chain": chain,
                "requested_revision": None}
        if alternative:
            item["alternative"] = alternative
        self.append(item)


def analyze_dependencies(sources, config, model_id, *, files=(), metadata=None, transformers_version=None):
    try:
        return DependencyAnalyzer(sources, config, model_id, files=files, metadata=metadata,
                                  transformers_version=transformers_version).run()
    except RecursionError as exc:
        raise ValueError("dependency expression nesting exceeds analysis limit") from exc
