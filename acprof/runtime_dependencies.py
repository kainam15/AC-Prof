"""Bounded, non-executing dependency checks against a selected runtime lock."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import PurePosixPath

from acprof.dependency_locks import normalized_name, package_versions, read_python_lock
from acprof.failures import Failure, RuntimeFailure
from acprof.installation import resource_root
from acprof.model_evidence import pinned_revision
from acprof.model_spec import custom_code_files

# Import names and distribution names are different namespaces. Unknown names
# remain unresolved; in particular, never suggest installing an import name.
IMPORT_DISTRIBUTIONS = {
    "PIL": "pillow", "skimage": "scikit-image", "sklearn": "scikit-learn",
    "cv2": "opencv-python", "yaml": "pyyaml", "fugashi": "fugashi",
    "detectron2": "detectron2", "flash_attn": "flash-attn", "timm": "timm",
    "einops": "einops", "sentencepiece": "sentencepiece", "unidic_lite": "unidic-lite",
}
MAX_FILES = 64
RUNTIME_REQUIREMENTS = {"requirements.txt", "requirements-inference.txt", "requirements-runtime.txt"}


def dependency_preflight(task, profile, *, read_source=None) -> dict:
    """Read only pinned source/config, without importing or installing anything."""
    from packaging.markers import default_environment
    from packaging.requirements import InvalidRequirement, Requirement

    installed = package_versions(read_python_lock(resource_root() / profile.environment.requirements_lock))
    distributions = {name.replace("-", "_"): name for name in installed}
    distributions.update(IMPORT_DISTRIBUTIONS)
    files = set(task.repository_files)
    metadata = task.repository_metadata or {}
    configs = [task.model_config, *(value for value in metadata.values() if isinstance(value, dict))]
    pending = set().union(*(custom_code_files(config) for config in configs))
    records, sources, requirements = [], {}, []
    tokenizer = metadata.get("tokenizer_config.json", {})
    if tokenizer.get("tokenizer_class") in {"BertJapaneseTokenizer", "MecabTokenizer"} and tokenizer.get("word_tokenizer_type", "mecab") == "mecab":
        records.append({"module": "fugashi", "source": "tokenizer_config.json", "conditional": False})
    if pending or files & RUNTIME_REQUIREMENTS:
        if not pinned_revision(task.model_revision):
            records.append({"status": "unknown", "detail": "source dependencies require a pinned commit SHA"})
            pending.clear()
        else:
            if read_source is None:
                from acprof.host.detect import read_model_source
                def read_source(name):
                    if name not in task.repository_sources:
                        task.repository_sources[name] = read_model_source(task.model_id, name, task.model_revision)
                    return task.repository_sources[name]
            environment = {**default_environment(), "python_version": "3.10", "python_full_version": profile.environment.platform.python_version,
                           "sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64", "extra": ""}
            for name in sorted(files):
                if name not in RUNTIME_REQUIREMENTS:
                    continue
                try:
                    source = read_source(name)
                    if len(source.encode()) > 512 * 1024:
                        raise ValueError("requirements exceed dependency analysis budget")
                    sources[name] = hashlib.sha256(source.encode()).hexdigest()
                    for line in source.splitlines():
                        line = line.split("#", 1)[0].strip()
                        if not line:
                            continue
                        requirement = Requirement(line)
                        if requirement.url or requirement.extras:
                            raise ValueError("URLs and extras require a reviewed locked environment")
                        if requirement.marker is None or requirement.marker.evaluate(environment):
                            requirements.append((name, requirement))
                except (OSError, ValueError, InvalidRequirement) as exc:
                    records.append({"status": "unknown", "source": name, "detail": str(exc)})
    for name, requirement in requirements:
        distribution = normalized_name(requirement.name)
        version = installed.get(distribution)
        records.append({"distribution": distribution, "required": str(requirement.specifier), "installed": version,
                        "source": name, "status": "missing" if version is None else
                        "present" if requirement.specifier.contains(version, prereleases=True) else "incompatible"})

    def visit(nodes, name, conditional=False):
        for node in nodes:
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                visit(node.orelse, name, conditional)
                continue
            if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
                visit(node.body if node.test.value else node.orelse, name, conditional)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if isinstance(node, ast.ImportFrom) and node.level:
                    base = PurePosixPath(name).parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    modules = [node.module] if node.module else [alias.name for alias in node.names]
                    for module in modules:
                        stem = str(base / module.replace(".", "/"))
                        candidates = [stem + ".py", stem + "/__init__.py"]
                        matched = next((path for path in candidates if path in files), None)
                        if matched:
                            pending.add(matched)
                        else:
                            records.append({"status": "unknown", "source": name, "detail": f"unresolved relative import {module}"})
                else:
                    for module in modules:
                        root = module.split(".")[0]
                        local = next((p for p in (root + ".py", root + "/__init__.py") if p in files), None)
                        if local:
                            pending.add(local)
                        elif root and root not in sys.stdlib_module_names:
                            records.append({"module": root, "source": name, "conditional": conditional})
            elif isinstance(node, ast.Call) and (isinstance(node.func, ast.Name) and node.func.id == "__import__" or
                    isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"):
                records.append({"status": "unknown", "source": name, "detail": "dynamic import requires review"})
            else:
                children = list(ast.iter_child_nodes(node))
                visit(children, name, conditional or isinstance(node, (ast.If, ast.Try)))

    while pending:
        name = min(pending)
        pending.remove(name)
        if name in sources:
            continue
        if len(sources) >= MAX_FILES:
            records.append({"status": "unknown", "detail": "source dependency file budget exceeded"})
            break
        try:
            text = read_source(name)
            if len(text.encode()) > 512 * 1024:
                raise ValueError("source exceeds dependency analysis budget")
            sources[name] = hashlib.sha256(text.encode()).hexdigest()
            visit(ast.parse(text, filename=name).body, name)
        except (OSError, SyntaxError, ValueError, RecursionError) as exc:
            records.append({"status": "unknown", "source": name, "detail": str(exc)})
    for record in records:
        if "status" in record:
            continue
        distribution = distributions.get(record["module"])
        record.update(distribution=distribution, installed=installed.get(distribution))
        record["status"] = "present" if record["installed"] else (
            "unknown" if record["conditional"] or distribution is None else "missing")
    report = {"schema_version": 1, "revision": task.model_revision, "runtime_profile": profile.profile_id,
              "lock": profile.environment.requirements_lock, "source_sha256": sources, "dependencies": records}
    task.model_resolution["dependency_preflight"] = report
    for status, code in (("missing", "runtime_dependency_missing"), ("incompatible", "runtime_dependency_incompatible"),
                         ("unknown", "runtime_dependency_unknown")):
        problems = [record for record in records if record["status"] == status]
        if problems:
            raise RuntimeFailure(Failure("dependency_preflight", code,
                "; ".join(str(item.get("distribution") or item.get("module") or item.get("detail")) for item in problems),
                runtime_profile=profile.profile_id, retryability="after_configuration",
                evidence={**report, "dependencies": problems}))
    return report
