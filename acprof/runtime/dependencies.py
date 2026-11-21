"""Bounded, non-executing dependency checks against a selected runtime lock."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from acprof.failures import Failure, RuntimeFailure
from acprof.installation import resource_root
from acprof.models.evidence import pinned_revision
from acprof.models.inspection.source_imports import RepositoryImports
from acprof.models.spec import custom_code_files
from acprof.runtime.dependency_locks import normalized_name, package_versions, read_python_lock
from acprof.runtime.stdlib import STDLIB_MODULES

# Import names and distribution names are different namespaces. Unknown names
# remain unresolved; in particular, never suggest installing an import name.
IMPORT_DISTRIBUTIONS = {
    "PIL": "pillow", "skimage": "scikit-image", "sklearn": "scikit-learn",
    "cv2": "opencv-python", "yaml": "pyyaml", "fugashi": "fugashi",
    "detectron2": "detectron2", "flash_attn": "flash-attn", "timm": "timm",
    "einops": "einops", "sentencepiece": "sentencepiece", "unidic_lite": "unidic-lite",
}


class DependencyRecord(TypedDict, total=False):
    """One dependency observation; source and resolution phases set different keys."""

    module: str
    source: str
    conditional: bool
    distribution: str | None
    installed: str | None
    required: str
    status: Literal["unknown", "present", "missing", "incompatible", "stdlib"]
    detail: str
    python_version: str


class DependencyPreflightReport(TypedDict):
    """Stable persisted dependency-preflight protocol, including failure evidence."""

    schema_version: int
    revision: str
    runtime_profile: str
    lock: str
    source_sha256: dict[str, str]
    dependencies: list[DependencyRecord]


MAX_FILES = 64
RUNTIME_REQUIREMENTS = {"requirements.txt", "requirements-inference.txt", "requirements-runtime.txt"}


class _FileLimitExceeded(ValueError):
    """No more repository reads are allowed in this preflight."""


class _PreflightSources:
    def __init__(self, read_source: Callable[[str], str]):
        self._read_source = read_source
        self._remaining = MAX_FILES
        self._trees: dict[str, ast.Module | str] = {}
        self.sha256: dict[str, str] = {}

    def _read(self, name: str) -> tuple[str, str]:
        if not self._remaining:
            raise _FileLimitExceeded("source dependency file budget exceeded")
        self._remaining -= 1  # Failed reads and invalid documents cost a slot too.
        text = self._read_source(name)
        data = text.encode()
        if len(data) > 512 * 1024:
            subject = "requirements exceed" if name in RUNTIME_REQUIREMENTS else "source exceeds"
            raise ValueError(f"{subject} dependency analysis budget")
        return text, hashlib.sha256(data).hexdigest()

    def requirements(self, name: str) -> str:
        text, self.sha256[name] = self._read(name)
        return text

    def tree(self, name: str) -> ast.Module:
        if name not in self._trees:
            try:
                text, digest = self._read(name)
                self._trees[name] = ast.parse(text, filename=name)
            except (OSError, SyntaxError, ValueError, RecursionError) as exc:
                # Remember the first diagnostic, not a traceback retaining invalid source.
                self._trees[name] = str(exc)
                raise
            self.sha256[name] = digest
        observed = self._trees[name]
        if isinstance(observed, str):
            raise ValueError(observed)
        return observed


def dependency_preflight(task, profile, *, read_source) -> DependencyPreflightReport:
    """Read only pinned source/config, without importing or installing anything."""
    from packaging.markers import default_environment
    from packaging.requirements import InvalidRequirement, Requirement

    target_python = profile.environment.platform.python_version
    stdlib = STDLIB_MODULES.get(target_python)
    installed = package_versions(read_python_lock(resource_root() / profile.environment.requirements_lock))
    distributions = {name.replace("-", "_"): name for name in installed}
    distributions.update(IMPORT_DISTRIBUTIONS)
    files = set(task.repository_files)
    metadata = task.repository_metadata or {}
    configs = [task.model_config, *(value for value in metadata.values() if isinstance(value, dict))]
    pending = set().union(*(custom_code_files(config) for config in configs))
    records: list[DependencyRecord] = []
    sources = _PreflightSources(read_source)
    requirements: list[Any] = []
    processed: set[str] = set()
    tokenizer = metadata.get("tokenizer_config.json", {})
    if tokenizer.get("tokenizer_class") in {"BertJapaneseTokenizer", "MecabTokenizer"} and tokenizer.get("word_tokenizer_type", "mecab") == "mecab":
        records.append({"module": "fugashi", "source": "tokenizer_config.json", "conditional": False})
    if pending or files & RUNTIME_REQUIREMENTS:
        if not pinned_revision(task.model_revision):
            records.append({"status": "unknown", "detail": "source dependencies require a pinned commit SHA"})
            pending.clear()
        else:
            environment = {**default_environment(),
                           "python_version": ".".join(target_python.split(".")[:2]),
                           "python_full_version": target_python,
                           "implementation_version": target_python,
                           "sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64", "extra": ""}
            for name in sorted(files):
                if name not in RUNTIME_REQUIREMENTS:
                    continue
                try:
                    source = sources.requirements(name)
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
        # The obsolete, unpinned typing backport is supplied by target Python.
        # Versioned requirements and typing-extensions remain real distributions.
        if (distribution == "typing" and not requirement.specifier
                and tuple(map(int, target_python.split(".")[:2])) >= (3, 5)):
            records.append({"distribution": distribution, "required": "", "source": name,
                            "status": "stdlib", "python_version": ".".join(target_python.split(".")[:2])})
            continue
        version = installed.get(distribution)
        records.append({"distribution": distribution, "required": str(requirement.specifier), "installed": version,
                        "source": name, "status": "missing" if version is None else
                        "present" if requirement.specifier.contains(version, prereleases=True) else "incompatible"})

    imports = RepositoryImports(files, sources.tree)
    guarded_files: set[str] = set()

    def visit(nodes, name, conditional=False):
        for node in nodes:
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                visit(node.orelse, name, conditional)
                continue
            if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
                visit(node.body if node.test.value else node.orelse, name, conditional)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                targets = imports.resolve(name, node)
                (guarded_files if conditional else pending).update(targets.sources)
                guarded_files.update(targets.possible_sources)
                for missing in targets.missing:
                    kind = "relative" if isinstance(node, ast.ImportFrom) and node.level else "local"
                    records.append({"status": "unknown", "source": name, "detail": f"unresolved {kind} import {missing}"})
                for root in targets.external_roots:
                    if stdlib is None:
                        records.append({"module": root, "source": name, "status": "unknown",
                                        "python_version": target_python,
                                        "detail": "no stdlib inventory for the target Python"})
                    elif root not in stdlib:
                        records.append({"module": root, "source": name, "conditional": conditional})
            elif isinstance(node, ast.Call) and (isinstance(node.func, ast.Name) and node.func.id == "__import__" or
                    isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"):
                records.append({"status": "unknown", "source": name, "detail": "dynamic import requires review"})
            else:
                children = list(ast.iter_child_nodes(node))
                visit(children, name, conditional or isinstance(node, (ast.If, ast.Try)))

    # Drain mandatory paths first. A module reached by both paths is graded once,
    # as mandatory; a guarded import cannot make a transitive dependency mandatory.
    while pending or guarded_files:
        guarded = not pending
        queue = guarded_files if guarded else pending
        name = min(queue)
        queue.remove(name)
        if name in processed:
            continue
        processed.add(name)
        try:
            tree = sources.tree(name)
            visit(tree.body, name, guarded)
            queue.update(imports.package_initializers(name))
        except _FileLimitExceeded as exc:
            records.append({"status": "unknown", "source": name, "detail": str(exc)})
            break
        except (OSError, SyntaxError, ValueError, RecursionError) as exc:
            records.append({"status": "unknown", "source": name, "detail": str(exc)})
    for record in records:
        if "status" in record:
            continue
        resolved_distribution = distributions.get(record["module"])
        record["distribution"] = resolved_distribution
        record["installed"] = (
            installed.get(resolved_distribution) if resolved_distribution is not None else None
        )
        record["status"] = "present" if record["installed"] else (
            "unknown" if record["conditional"] or resolved_distribution is None else "missing")
    report: DependencyPreflightReport = {
        "schema_version": 1, "revision": task.model_revision, "runtime_profile": profile.profile_id,
        "lock": profile.environment.requirements_lock, "source_sha256": sources.sha256, "dependencies": records,
    }
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
