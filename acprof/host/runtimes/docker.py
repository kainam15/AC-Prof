"""Docker adapter retains the existing native lifecycle and measurement semantics."""
from __future__ import annotations


class DockerRuntime:
    kind = "docker"

    def prepare(self, task_info, project_dir, output_dir, *, rebuild=False, timeout_seconds=300):
        from acprof.host.images.runtime import prepare_image
        from acprof.host.interface_probe import probe_interface
        probe_interface(task_info, output_dir, timeout_seconds=timeout_seconds)
        return prepare_image(task_info, project_dir, reuse_existing=not rebuild)

    def validate(self, **kwargs) -> dict:
        from acprof.host.runtime_validation import validate_runtime
        return dict(validate_runtime(**kwargs))

    def start(self, **kwargs):
        from acprof.host.docker_runtime import start_container_session
        return start_container_session(**kwargs)

    def stop(self, session, *, log_prefix=None) -> None:
        from acprof.host.docker_runtime import stop_container_session
        stop_container_session(session, log_prefix=log_prefix)

    def failure(self, session, memory, returncode) -> str:
        from acprof.host.container_state import container_runtime_oom_error
        return container_runtime_oom_error(session.name, memory, returncode) or ""

    def restore(self, snapshot: dict):
        from acprof.host.images.runtime import ImageInfo, require_image_identity
        image = ImageInfo(**snapshot)
        if not image.tag.startswith("sha256:"):
            raise ValueError("resume requires the original immutable Docker image ID")
        require_image_identity(image.tag, image.runtime_environment)
        return image
