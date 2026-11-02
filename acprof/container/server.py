"""AC-Prof HTTP application and explicit isolated-model startup."""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

SERVER_PROCESS_STARTED_AT = time.time()
SERVER_PROCESS_STARTED_PERF = time.perf_counter()

if TYPE_CHECKING:
    from flask import Flask

    from acprof.container.model_session import ModelSession

MAX_REQUEST_BODY_BYTES = 32 * 1024 * 1024


def _extract_probe_metadata(processed: object) -> dict:
    if not isinstance(processed, dict):
        return {}
    return {
        "effective_input_scale": processed.get("_effective_input_scale"),
        "truncated_by_limit": processed.get("_truncated_by_limit"),
        "reason": processed.get("_probe_reason", ""),
    }


def create_app(
    session: ModelSession, *, startup_timing: dict | None = None,
    service_token: str = "", runtime_profile: str = "", app: Flask | None = None,
) -> Flask:
    """Bind HTTP behavior to an already loaded session; never load or infer here."""
    from flask import Flask, jsonify, request

    from acprof.container.execution import complete_prediction
    from acprof.container.handlers import InputLimitError
    from acprof.workloads.contract import workload_contract

    app = Flask(__name__) if app is None else app
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BODY_BYTES
    config, handler, model_ctx = session.config, session.handler, session.context
    execution, device = session.execution, session.device
    timing = dict(startup_timing or {})
    load_time = timing.get("model_load_s")
    load_time_s = round(load_time, 3) if load_time is not None else None

    def request_json_body() -> dict:
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}

    def failure_response(exc, stage, data):
        from acprof.failures import failure_from_exception

        failure = failure_from_exception(exc, stage=stage, device=device,
            runtime_profile=runtime_profile, evidence={
                "request_id": request.headers.get("X-Req-Id"), "input_scale": data.get("input_scale"),
                "request_phase": stage, "model_loaded": True, "service_alive": True})
        if failure.reason_code == "request_timeout":
            from dataclasses import replace

            from acprof.runtime.settings import request_timeout_s

            failure = replace(failure, evidence={**failure.evidence,
                "timeout_seconds": request_timeout_s() if stage == "completion" else None,
                "timeout_scope": "completion_wait" if stage == "completion" else "unknown"})
        return jsonify({"error": str(exc), "failure": failure.to_dict()}), 500

    @app.errorhandler(413)
    def request_body_too_large(_error):
        return jsonify({"error": f"request body exceeds {MAX_REQUEST_BODY_BYTES} bytes",
                        "limit_bytes": MAX_REQUEST_BODY_BYTES}), 413

    @app.route("/ready")
    def ready():
        return jsonify({
            "status": "ok", "service_token": service_token,
            "model_id": config.model_id, "model_revision": config.model_revision,
            "device": device, "load_time_s": load_time_s, "startup_timing": timing,
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request_json_body()
        stage = "preprocess"
        try:
            processed = handler.preprocess(model_ctx, data)
            stage = "predict"
            with execution.inference_context():
                output = handler.predict(model_ctx, processed)
                stage = "completion"
                output = complete_prediction(execution, model_ctx, output)
            stage = "postprocess"
            result = handler.postprocess(model_ctx, output)
            result["workload_contract"] = workload_contract(model_ctx, data, processed, result)
            metadata = _extract_probe_metadata(processed)
            effective_input_scale = metadata.get("effective_input_scale")
            if effective_input_scale is not None:
                result["effective_input_scale"] = effective_input_scale
            return jsonify(result)
        except Exception as exc:
            return failure_response(exc, stage, data)

    @app.route("/probe", methods=["POST"])
    def probe():
        data = request_json_body()
        try:
            processed = handler.preprocess(model_ctx, data)
            return jsonify(_extract_probe_metadata(processed))
        except InputLimitError as exc:
            return jsonify(exc.probe_metadata)
        except Exception as exc:
            return failure_response(exc, "preprocess", data)

    @app.route("/scale_meta", methods=["GET", "POST"])
    def scale_meta():
        data = request_json_body()
        try:
            return jsonify(handler.get_scale_metadata(model_ctx, data))
        except Exception as exc:
            return failure_response(exc, "metadata", data)

    @app.route("/meta")
    def meta():
        runtime_metadata = execution.metadata()
        return jsonify({
            "model_id": config.model_id, "model_revision": config.model_revision,
            "task_family": config.task_family, "task_type": config.task_type,
            "runtime_backend": config.backend,
            "runtime_parameters": model_ctx.get("runtime_parameters", runtime_metadata.get("runtime_parameters", {})),
            "device": device, "load_time_s": load_time_s, "startup_timing": timing,
        })

    return app


def create_runtime_app() -> Flask:
    """Own the existing startup phases; application construction never reloads."""
    # Set offline policy before importing any runtime/model implementation.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from flask import Flask

    # Import execution support before its separately timed device initialization.
    from acprof.container.execution import complete_prediction  # noqa: F401
    from acprof.container.model_session import (
        ModelLoadConfig,
        load_model_session,
        prepare_execution,
        select_handler,
    )

    # Keep HTTP contract imports in setup, before the model-load timing span.
    from acprof.workloads.contract import workload_contract  # noqa: F401

    # Construct Flask before the load timer, as in the original startup path.
    app = Flask(__name__)
    config = ModelLoadConfig.from_environment()
    handler = select_handler(config)
    model_source = config.resolve_source()
    t_cuda_init_start = time.perf_counter()
    execution, device = prepare_execution(config)
    t_cuda_init_end = time.perf_counter()
    cuda_init_s = t_cuda_init_end - t_cuda_init_start if config.use_gpu else 0.0

    print(f"[server] Loading model: {config.model_id}")
    print(f"[server] Revision: {config.model_revision}")
    print(f"[server] Source: {model_source}")
    print(f"[server] Task: {config.task_type} (family={config.task_family}, backend={config.backend})")
    print(f"[server] Device: {device}")

    model_load_started_at = time.time()
    t_load_start = time.perf_counter()
    server_setup_s = max(0.0, t_load_start - SERVER_PROCESS_STARTED_PERF - cuda_init_s)
    session = load_model_session(
        config, execution, device, handler=handler, model_source=model_source,
    )
    t_load_end = time.perf_counter()
    model_load_completed_at = time.time()
    load_time_s = t_load_end - t_load_start
    print(f"[server] Model loaded in {load_time_s:.2f}s")

    return create_app(session, app=app, service_token=os.getenv("ACPROF_SERVICE_TOKEN", ""),
        runtime_profile=os.getenv("ACPROF_RUNTIME_PROFILE", ""), startup_timing={
            "server_process_started_at_epoch_s": SERVER_PROCESS_STARTED_AT,
            "server_setup_s": server_setup_s, "cuda_init_s": cuda_init_s,
            "model_load_started_at_epoch_s": model_load_started_at,
            "model_load_completed_at_epoch_s": model_load_completed_at,
            "model_load_s": load_time_s,
        })


def main() -> None:
    app = create_runtime_app()
    if os.getenv("ACPROF_SERVER_FD"):
        from werkzeug.serving import make_server

        descriptor = int(os.environ["ACPROF_SERVER_FD"])
        server = make_server("127.0.0.1", int(os.environ["ACPROF_SERVER_PORT"]), app,
                             threaded=True, fd=descriptor)
        os.close(descriptor)
        server.serve_forever()
    else:
        app.run(host="0.0.0.0", port=8002)


if __name__ == "__main__":
    main()
