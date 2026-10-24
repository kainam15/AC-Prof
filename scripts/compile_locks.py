"""生成并核验 Docker 专用 wheel 锁；项目环境由 uv.lock 管理。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.dependency_locks import (  # noqa: E402 -- 脚本先设置仓库导入路径。
    normalized_name,
    package_versions,
    python_lock_text,
    read_python_lock,
)
from acprof.runtime_profiles import (  # noqa: E402 -- 脚本先设置仓库导入路径。
    ENVIRONMENTS,
    PLATFORMS,
    PROFILES,
    environment_id,
    environment_identity,
)


def target_markers(platform):
    return {"python_version": ".".join(platform.python_version.split(".")[:2]),
            "python_full_version": platform.python_version, "sys_platform": "linux",
            "platform_machine": "x86_64", "platform_system": "Linux", "os_name": "posix",
            "implementation_name": "cpython", "implementation_version": platform.python_version,
            "platform_python_implementation": "CPython", "extra": ""}


def target_tags(platform):
    from packaging.tags import compatible_tags, cpython_tags
    if platform.architecture != "linux/amd64" or platform.python_target != "x86_64-manylinux_2_28":
        raise ValueError("尚未注册此目标平台的 wheel 选择规则")
    version = tuple(int(value) for value in platform.python_version.split(".")[:2])
    interpreter = "cp" + "".join(str(value) for value in version)
    # 与声明的 manylinux 2.28 下限一致；不使用主机 Python/ABI/体系结构。
    platforms = [f"manylinux_2_{v}_x86_64" for v in range(28, 4, -1)]
    platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    tags = list(cpython_tags(version, abis=[interpreter], platforms=platforms))
    tags += list(compatible_tags(version, interpreter=interpreter, platforms=platforms))
    return tags


def wheel_records(data, platform):
    from packaging.markers import Marker
    from packaging.utils import parse_wheel_filename
    rank = {tag: index for index, tag in enumerate(target_tags(platform))}
    records = []
    for package in data["packages"]:
        if package.get("marker") and not Marker(package["marker"]).evaluate(target_markers(platform)):
            continue
        ranked = []
        for wheel in package.get("wheels", []):
            parsed = parse_wheel_filename(unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[-1]))
            matches = [rank[tag] for tag in parsed[3] if tag in rank]
            if matches:
                ranked.append((min(matches), wheel["url"], wheel))
        if not ranked:
            raise ValueError(f"没有目标平台 wheel；不会自动从源码构建：{package['name']}")
        wheel = min(ranked, key=lambda item: (item[0], item[1]))[2]
        records.append({"name": package["name"], "version": package["version"],
                        "url": wheel["url"], "sha256": wheel["hashes"]["sha256"],
                        **({"size": wheel["size"]} if type(wheel.get("size")) is int else {})})
    return records


def check_sources(paths, packages, root, visited=None):
    from packaging.requirements import Requirement
    visited = set() if visited is None else visited
    for relative in paths:
        path = (root / relative).resolve()
        if path in visited:
            continue
        visited.add(path)
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("-r "):
                check_sources([str(path.parent / line[3:].strip())], packages, root, visited)
                continue
            requirement = Requirement(line)
            name = normalized_name(requirement.name)
            if name not in packages or not requirement.specifier.contains(packages[name], prereleases=True):
                raise ValueError(f"依赖输入与 lock 不符：{path}: {line}")


def check_runtime_lock(path, platform):
    """Check wheel compatibility and the generated inventory without network access."""
    from packaging.utils import parse_wheel_filename

    records = read_python_lock(path)
    compatible = set(target_tags(platform))
    for record in records:
        filename = unquote(urlsplit(record["url"]).path.rsplit("/", 1)[-1])
        if not compatible.intersection(parse_wheel_filename(filename)[3]):
            raise ValueError(f"wheel 与目标平台不兼容：{path}: {filename}")
    inventory_path = path.with_suffix(".artifacts.json")
    try:
        inventory = json.loads(inventory_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"无法读取 artifact 清单：{inventory_path}") from error
    if (not isinstance(inventory, dict)
            or type(inventory.get("schema_version")) is not int
            or inventory["schema_version"] != 1
            or inventory.get("python_version") != platform.python_version
            or inventory.get("python_target") != platform.python_target):
        raise ValueError(f"artifact 清单平台或 schema 不符：{inventory_path}")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(item, dict) for item in artifacts):
        raise ValueError(f"无效 artifact 条目：{inventory_path}")
    fields = ("name", "version", "url", "sha256")
    if any(not all(isinstance(item.get(key), str) for key in fields) for item in artifacts):
        raise ValueError(f"artifact 清单缺少制品身份：{inventory_path}")
    actual = sorted(({key: item[key] for key in fields} for item in artifacts),
                    key=lambda item: item["name"])
    if actual != records or any(
        "size" in item and (type(item["size"]) is not int or item["size"] <= 0)
        for item in artifacts
    ):
        raise ValueError(f"artifact 清单与依赖锁不符：{inventory_path}")


def check_catalog(root=ROOT, variants=None):
    environments = {}
    selected = set(variants or PLATFORMS)
    for key in sorted(selected):
        platform = PLATFORMS[key]
        check_runtime_lock(root / platform.requirements_lock, platform)
    for key, environment in ENVIRONMENTS.items():
        if environment.platform.platform_id not in selected:
            continue
        identity = environment_identity(environment, root)
        check_runtime_lock(root / environment.requirements_lock, environment.platform)
        check_sources(environment.requirements_inputs, package_versions(identity["packages"]), root)
        identifier = environment_id(environment, root)
        if identifier in environments:
            raise ValueError(f"重复依赖环境应合并声明：{key} / {environments[identifier]['environment_key']}")
        environments[identifier] = {"environment_key": key, "platform_id": environment.platform.platform_id,
                                    "profiles": []}
    for profile in PROFILES.values():
        if profile.environment.platform.platform_id not in selected:
            continue
        if profile.environment not in ENVIRONMENTS.values():
            raise ValueError(f"profile 引用了未登记的依赖环境：{profile.profile_id}")
        environments[environment_id(profile.environment, root)]["profiles"].append(profile.profile_id)
    return {"profiles": sum(len(item["profiles"]) for item in environments.values()),
            "environments": environments, "platforms": sorted(selected)}


def resolve(uv, inputs, output, platform, pins, *, index_url=None, torch_index_url=None):
    with tempfile.TemporaryDirectory(prefix="acprof-resolve-") as directory:
        directory = Path(directory)
        constraint = directory / "constraints.txt"
        constraint.write_text("".join(f"{name}=={version}\n" for name, version in sorted(pins.items())))
        target = directory / "pylock.toml"
        command = [uv, "pip", "compile", *map(str, inputs), "--constraint", str(constraint),
                        "--python-version", platform.python_version, "--python-platform", platform.python_target,
                        "--only-binary", ":all:",
                        "--format", "pylock.toml", "--no-header", "--output-file", str(target)]
        if index_url:
            command += ["--default-index", index_url]
        if torch_index_url and platform.torch_version:
            command += ["--index", torch_index_url]
        elif platform.torch_version:
            command += ["--torch-backend", platform.platform_id]
        subprocess.run(command,
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        records = wheel_records(tomllib.loads(target.read_text()), platform)
        for name, version in package_versions(records).items():
            if name in pins and pins[name] != version:
                raise ValueError(f"解析改变了锁定版本：{name}")
        text = python_lock_text(records)
        # 先核验生成格式再替换，失败时保留原锁。
        generated = directory / "requirements.txt"
        generated.write_text(text)
        read_python_lock(generated)
        if output.exists() and (index_url or torch_index_url):
            old = {entry["name"]: entry for entry in read_python_lock(output)}
            for record in records:
                prior = old.get(record["name"])
                if prior and prior["version"] == record["version"] and prior["sha256"] != record["sha256"]:
                    raise ValueError(f"镜像解析改变了锁定 artifact hash：{record['name']}；不会替换原锁")
        output.write_text(text)
        output.with_suffix(".artifacts.json").write_text(json.dumps({
            "schema_version": 1, "python_target": platform.python_target, "python_version": platform.python_version,
            "index_url": index_url, "torch_index_url": torch_index_url,
            "artifacts": records}, sort_keys=True, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    required_uv = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["required-version"]
    parser.add_argument("--uv", default="uv", help=f"uv 可执行文件，要求 {required_uv}")
    parser.add_argument("--variant", action="append", choices=PLATFORMS)
    parser.add_argument("--check", action="store_true", help="不访问 Docker/网络、不写文件；核验容器锁")
    parser.add_argument("--upgrade", action="store_true", help="显式更新环境包；平台 Torch 和安装工具仍保持锁定")
    parser.add_argument("--index-url", help="Resolve artifacts from this PEP 503 index; never rewrite lock URLs")
    parser.add_argument("--torch-index-url", help="Explicit target Torch wheel index; exact versions/platform/hashes remain checked")
    args = parser.parse_args(argv)
    for url in (args.index_url, args.torch_index_url):
        if url:
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                parser.error("镜像索引必须是无凭据的 HTTPS URL")
    if args.check:
        if args.upgrade or args.index_url or args.torch_index_url:
            parser.error("--check 不与 --upgrade、--index-url 或 --torch-index-url 同用")
        result = check_catalog(variants=args.variant)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    version = subprocess.check_output([args.uv, "--version"], text=True).split()
    from packaging.specifiers import SpecifierSet
    if len(version) < 2 or version[0] != "uv" or version[1] not in SpecifierSet(required_uv):
        parser.error(f"锁生成工具必须是 uv {required_uv}")
    for key in args.variant or PLATFORMS:
        platform = PLATFORMS[key]
        platform_pins = package_versions(read_python_lock(ROOT / platform.requirements_lock))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "platform.in"
            base_packages = ("pip", "wheel", "packaging", "setuptools")
            if platform.torch_version:
                base_packages = ("torch", *base_packages)
            declarations = [f"{name}=={platform_pins[name]}" for name in base_packages]
            if args.upgrade and platform.torch_version:
                declarations.append("filelock>=4.2.4")
            source.write_text("\n".join(declarations) + "\n")
            # Upgrades retain only the intentionally fixed platform packages;
            # transitive dependencies must be allowed to receive security fixes.
            protected = ({name: platform_pins[name] for name in base_packages}
                         if args.upgrade else platform_pins)
            resolve(args.uv, [source], ROOT / platform.requirements_lock, platform, protected,
                    index_url=args.index_url, torch_index_url=args.torch_index_url)
        # A refreshed platform may have updated transitive packages; use its
        # *new* lock for all child environments, never stale pre-upgrade pins.
        platform_pins = package_versions(read_python_lock(ROOT / platform.requirements_lock))
        for environment in ENVIRONMENTS.values():
            if environment.platform.platform_id != key:
                continue
            pins = {} if args.upgrade else package_versions(read_python_lock(ROOT / environment.requirements_lock))
            pins.update(package_versions(read_python_lock(ROOT / platform.requirements_lock)))
            # 平台完整闭包与安装工具也是环境的显式依赖。
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "platform.in"
                source.write_text("".join(f"{name}=={version}\n" for name, version in platform_pins.items()))
                resolve(args.uv, [*(ROOT / name for name in environment.requirements_inputs), source],
                        ROOT / environment.requirements_lock, platform, pins,
                        index_url=args.index_url, torch_index_url=args.torch_index_url)
    check_catalog(variants=args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
