ARG MODEL_IMAGE
FROM ${MODEL_IMAGE}
ARG RUNTIME_PROFILE
ARG MODEL_ADAPTER=family-default
ARG BUILD_FINGERPRINT
ARG DEPENDENCY_LOCK_SHA256
ARG REQUEST_FINGERPRINT
ARG PLATFORM_IMAGE_ID
ARG ENVIRONMENT_IMAGE_ID
ARG MODEL_IMAGE_ID
ARG MODEL_SPEC_B64=
ENV ACPROF_MODEL_SPEC_B64=${MODEL_SPEC_B64}
ENV ACPROF_RUNTIME_PROFILE=${RUNTIME_PROFILE} ACPROF_MODEL_ADAPTER=${MODEL_ADAPTER}
ENV ACPROF_BUILD_FINGERPRINT=${BUILD_FINGERPRINT} ACPROF_DEPENDENCY_LOCK_SHA256=${DEPENDENCY_LOCK_SHA256}
ENV ACPROF_REQUEST_FINGERPRINT=${REQUEST_FINGERPRINT} ACPROF_PLATFORM_IMAGE_ID=${PLATFORM_IMAGE_ID} \
    ACPROF_ENVIRONMENT_IMAGE_ID=${ENVIRONMENT_IMAGE_ID} ACPROF_MODEL_IMAGE_ID=${MODEL_IMAGE_ID}
LABEL org.acprof.build-fingerprint=${BUILD_FINGERPRINT} \
      org.acprof.request-fingerprint=${REQUEST_FINGERPRINT} \
      org.acprof.runtime-profile=${RUNTIME_PROFILE} org.acprof.model-adapter=${MODEL_ADAPTER} \
      org.acprof.image-kind="model"
# runtime_images stages exactly source_identity.service_context_files before build.
# This package contains shared modules, container/, workloads/ and extensions/.
COPY acprof/ /app/acprof/
COPY LICENSE NOTICE /usr/share/licenses/acprof/
COPY licenses/ /usr/share/licenses/acprof/third-party/
RUN python -m acprof.container.runtime_manifest
# Runtime requests and model imports never need UID 0. Fixed IDs make
# container ownership deterministic; /models is a read-only host mount.
RUN groupadd --gid 65532 acprof \
    && useradd --no-log-init --uid 65532 --gid 65532 --home-dir /tmp \
       --no-create-home --shell /usr/sbin/nologin acprof
ENV HOME=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/acprof-torch
USER acprof:acprof
CMD ["python", "-m", "acprof.container.server"]
