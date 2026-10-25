"""Build and verify shared dependency images; publish only with explicit --push."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.dependency_locks import content_digest  # noqa: E402
from acprof.host.images.environment import (  # noqa: E402 -- 源码发布脚本先设置仓库导入路径。
    DEFAULT_RUNTIME_REGISTRY,
    platform_fingerprint,
    prepare_environment_image,
    prepare_platform_image,
    registry_reference,
    runtime_fingerprint,
    verify_labels,
)
from acprof.runtime_profiles import (  # noqa: E402 -- 同上。
    ENVIRONMENTS,
    PLATFORMS,
    environment_identity,  # noqa: E402
)


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"GHCR {field} is not a SHA256 digest")
    return value


@dataclass(frozen=True)
class RemoteImage:
    """Registry identity fetched without downloading filesystem layers."""

    image_id: str
    manifest_digest: str
    diff_ids: tuple[str, ...]


def _inspect(reference: str, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", reference, *args],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot inspect GHCR image {reference}: {error}") from error


def inspect_remote_image(reference: str, expected: dict[str, str], kind: str,
                         *, parent: RemoteImage | None = None) -> RemoteImage | None:
    """Read OCI manifest/config only; fail closed on auth, network or identity errors.

    A missing tag may be built. A present but invalid tag must never be overwritten.
    The full installed-package manifest is verified by the existing local builder
    for newly published images; remote reuse verifies OCI content identity.
    """
    metadata = _inspect(reference, "--format", "{{json .}}")
    if metadata.returncode:
        error = (metadata.stderr or metadata.stdout).strip()
        lowered = error.lower()
        missing = ("manifest unknown" in lowered or "name unknown" in lowered
                   or re.search(r": not found\b|\b404 not found\b", lowered) is not None)
        if missing and not any(term in lowered for term in ("unauthorized", "denied", "forbidden")):
            return None
        raise RuntimeError(f"GHCR lookup failed for {reference}: {error}")
    try:
        details = json.loads(metadata.stdout)
        descriptor, image = details["manifest"], details["image"]
        manifest_digest = _require_sha256(descriptor["digest"], field="manifest digest")
        if descriptor["mediaType"] not in (
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
        ):
            raise RuntimeError("GHCR image is not a single-platform OCI/Docker manifest")
        if image["os"] != "linux" or image["architecture"] != "amd64":
            raise RuntimeError("GHCR image is not linux/amd64")
        verify_labels({"labels": image["config"]["Labels"]}, expected, kind)
        rootfs = image["rootfs"]
        if rootfs["type"] != "layers" or not isinstance(rootfs["diff_ids"], list) or not rootfs["diff_ids"]:
            raise RuntimeError("GHCR image has no valid filesystem layer chain")
        diff_ids = tuple(_require_sha256(item, field="filesystem layer") for item in rootfs["diff_ids"])
        if parent is not None and diff_ids[:len(parent.diff_ids)] != parent.diff_ids:
            raise RuntimeError("GHCR environment does not inherit the verified platform layers")
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid GHCR image metadata for {reference}") from error

    # Docker Buildx exposes labels and layer chain from the image config above.
    # Read the raw manifest as well to obtain its immutable config digest (image ID).
    raw = _inspect(reference, "--raw")
    if raw.returncode:
        raise RuntimeError(f"GHCR manifest changed during verification: {reference}: {raw.stderr.strip()}")
    if "sha256:" + hashlib.sha256(raw.stdout.encode("utf-8")).hexdigest() != manifest_digest:
        raise RuntimeError(f"GHCR manifest changed during verification: {reference}")
    try:
        manifest = json.loads(raw.stdout)
        if manifest["schemaVersion"] != 2 or manifest["mediaType"] != descriptor["mediaType"]:
            raise RuntimeError("GHCR returned an unexpected manifest type")
        if manifest["config"]["mediaType"] not in (
            "application/vnd.docker.container.image.v1+json",
            "application/vnd.oci.image.config.v1+json",
        ):
            raise RuntimeError("GHCR returned an unexpected image config type")
        image_id = _require_sha256(manifest["config"]["digest"], field="image ID")
        if not isinstance(manifest["layers"], list) or len(manifest["layers"]) != len(diff_ids):
            raise RuntimeError("GHCR image manifest and filesystem layer counts disagree")
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid GHCR manifest for {reference}") from error
    return RemoteImage(image_id, manifest_digest, diff_ids)


def _platform_labels(platform) -> dict[str, str]:
    return {"platform_id": platform.platform_id,
            "platform_build_fingerprint": platform_fingerprint(platform, ROOT)}


def _environment_labels(environment, platform: RemoteImage) -> dict[str, str]:
    return {**_platform_labels(environment.platform),
            "environment_id": content_digest(environment_identity(environment, ROOT)),
            "environment_build_fingerprint": content_digest({
                "definition": runtime_fingerprint(environment, ROOT),
                "platform_image_id": platform.image_id,
            })}


def _report(args, *, kind: str, image_id: str, reference: str,
            pushed: bool, reused: bool, verified: str,
            manifest_digest: str | None = None) -> None:
    data = {"kind": kind, "image_id": image_id, "reference": reference,
            "verified": verified, "pushed": pushed, "reused": reused}
    if manifest_digest is not None:
        data["manifest_digest"] = manifest_digest
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--matrix", action="store_true", help="只输出 CI matrix，不访问 Docker")
    selection.add_argument("--platform", choices=tuple(PLATFORMS))
    selection.add_argument("--environment", choices=tuple(ENVIRONMENTS))
    parser.add_argument("--registry", default=DEFAULT_RUNTIME_REGISTRY)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--pull-platform", action="store_true",
                        help="环境发布必须复用已发布平台的不可变 ID，不在每个 job 重建父层")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.matrix:
        print(json.dumps({"platforms": list(PLATFORMS), "environments": list(ENVIRONMENTS)}))
        return 0
    if args.push and args.environment and not args.pull_platform:
        parser.error("Publishing an environment requires --pull-platform")

    if args.platform:
        kind = "platform"
        platform = PLATFORMS[args.platform]
        expected = _platform_labels(platform)
        fingerprint = expected["platform_build_fingerprint"]
        reference = registry_reference(kind, fingerprint, args.registry)
        parent = None
    else:
        kind = "environment"
        environment = ENVIRONMENTS[args.environment]
        parent = None
        if args.push:
            parent_expected = _platform_labels(environment.platform)
            platform_reference = registry_reference(
                "platform", parent_expected["platform_build_fingerprint"], args.registry,
            )
            parent = inspect_remote_image(platform_reference, parent_expected, "platform")
            if parent is None:
                raise RuntimeError(f"Verified GHCR platform image is missing: {platform_reference}")
            expected = _environment_labels(environment, parent)
            reference = registry_reference(kind, expected["environment_build_fingerprint"], args.registry)

    if args.push:
        remote = inspect_remote_image(reference, expected, kind, parent=parent)
        if remote is not None:
            print(f"[release] Reusing verified GHCR {kind}: {reference}", flush=True)
            _report(args, kind=kind, image_id=remote.image_id, reference=reference,
                    pushed=False, reused=True, verified="remote_oci_identity",
                    manifest_digest=remote.manifest_digest)
            return 0

    if args.platform:
        # The tag is known absent from GHCR: do not waste time pulling it again.
        built_platform = prepare_platform_image(
            platform, ROOT, image_source="build" if args.push else "auto",
            registry=args.registry,
        )
        fingerprint = built_platform.expected["platform_build_fingerprint"]
        image_id = built_platform.image_id
    else:
        if args.pull_platform:
            local_platform = prepare_platform_image(
                environment.platform, ROOT, image_source="pull", registry=args.registry,
            )
            if parent is not None and local_platform.image_id != parent.image_id:
                raise RuntimeError("GHCR platform changed between remote verification and pull")
        built_environment = prepare_environment_image(
            environment, ROOT, image_source="build",
        )
        fingerprint = built_environment.manifest["environment_build_fingerprint"]
        image_id = built_environment.image_id
        if parent is not None and built_environment.platform_image_id != parent.image_id:
            raise RuntimeError("Environment was built from a different platform image")

    built_reference = registry_reference(kind, fingerprint, args.registry)
    if args.push and built_reference != reference:
        raise RuntimeError("Dependency inputs changed during build; refusing to publish")
    reference = built_reference

    if args.push:
        # Never overwrite an image another publisher placed at the fingerprint tag.
        remote = inspect_remote_image(reference, expected, kind, parent=parent)
        if remote is not None:
            if remote.image_id != image_id:
                raise RuntimeError(f"GHCR tag already contains a different image: {reference}")
            _report(args, kind=kind, image_id=image_id, reference=reference,
                    pushed=False, reused=True, verified="dependency_manifest+remote_oci_identity",
                    manifest_digest=remote.manifest_digest)
            return 0
        subprocess.run(["docker", "tag", image_id, reference], check=True)
        subprocess.run(["docker", "push", reference], check=True)
        verified = inspect_remote_image(reference, expected, kind, parent=parent)
        if verified is None or verified.image_id != image_id:
            raise RuntimeError(f"Published GHCR image ID did not match the validated build: {reference}")
        _report(args, kind=kind, image_id=image_id, reference=reference,
                pushed=True, reused=False, verified="dependency_manifest+remote_oci_identity",
                manifest_digest=verified.manifest_digest)
    else:
        _report(args, kind=kind, image_id=image_id, reference=reference,
                pushed=False, reused=False, verified="dependency_manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
