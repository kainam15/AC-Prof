"""Short-form ASR using one eagerly loaded, loopback-only audio.cpp session."""
from __future__ import annotations

import base64
import http.client
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import weakref
from pathlib import Path

from acprof.container.audiocpp_execution import server_binary, server_environment
from acprof.container.handlers import BaseHandler
from acprof.runtime.settings import request_timeout_s, runtime_threads
from acprof.workloads.audio import SAMPLE_RATE, read_pcm16_wav


class NativeASRServer:
    def __init__(self, path: Path, family: str, device: str):
        binary = server_binary(device)
        self.model_id = secrets.token_hex(16)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.threads = runtime_threads(default=1) or 1
        config = {"host": "127.0.0.1", "port": self.port, "backend": device, "device": 0,
                  "threads": self.threads, "lazy_load": False, "idle_unload_ms": 0,
                  "models": [{"id": self.model_id, "family": family, "path": str(path),
                              "task": "asr", "mode": "offline"}]}
        with tempfile.TemporaryFile() as stream:
            stream.write(json.dumps(config).encode())
            stream.seek(0)
            fd = stream.fileno()
            self.process = subprocess.Popen(
                [sys.executable, "-m", "acprof.container.audiocpp_execution", str(os.getpid()),
                 str(binary), "--config", f"/proc/self/fd/{fd}", "--no-ui"],
                env=server_environment(binary), stdin=subprocess.DEVNULL, pass_fds=(fd,),
            )
        self.close = weakref.finalize(self, self._stop, self.process)
        try:
            self.build = self._wait_ready(device)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _request(self, route, *, timeout, body=None, content_type=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request("GET" if body is None else "POST", route, body=body,
                         headers={"Content-Type": content_type} if body is not None else {})
            response = conn.getresponse()
            payload = response.read()
            if response.status != 200:
                raise RuntimeError(f"audio.cpp {route} HTTP {response.status}: {payload.decode(errors='replace')}")
            return json.loads(payload)
        finally:
            conn.close()

    def _wait_ready(self, device):
        timeout = request_timeout_s()
        deadline = time.monotonic() + timeout if timeout is not None else None
        while deadline is None or time.monotonic() < deadline:
            if (code := self.process.poll()) is not None:
                raise RuntimeError(f"audiocpp_server exited during model load ({code}). See service log for native diagnostics.")
            try:
                health = self._request("/health", timeout=1)
            except (ConnectionRefusedError, TimeoutError, http.client.RemoteDisconnected):
                time.sleep(0.05)
                continue
            models = self._request("/v1/models", timeout=2)["data"]
            if (health.get("status") != "ok" or health.get("backend") != device
                    or not any(m.get("id") == self.model_id and m.get("loaded") is True for m in models)):
                raise RuntimeError("audio.cpp readiness does not identify the requested resident model and device")
            return health
        raise TimeoutError("audio.cpp eager model load timed out")

    def transcribe(self, body: bytes, content_type: str):
        if self.process.poll() is not None:
            raise RuntimeError(f"audio.cpp server exited with status {self.process.returncode}")
        try:
            return self._request("/v1/audio/transcriptions", timeout=request_timeout_s(),
                                 body=body, content_type=content_type)
        except (OSError, http.client.HTTPException):
            # The server can continue computing after the client disconnects.
            self.close()
            raise


class AudioCppHandler(BaseHandler):
    def load(self, model_source, task_type, backend, device, model_revision="main",
             load_options=None, *, adapter="family-default"):
        from acprof.models.audiocpp import gguf_candidate

        if backend != "audio_cpp" or task_type != "automatic-speech-recognition" or device not in {"cpu", "cuda"}:
            raise ValueError("audio.cpp requires offline ASR on CPU or CUDA")
        options = dict(load_options or {})
        variant = options.pop("variant", "")
        candidate = gguf_candidate(os.environ.get("MODEL_ID", ""), variant)
        if options or not candidate or not candidate["selectable"] or candidate["task"] != task_type:
            raise ValueError(f"audio.cpp requires a selected, supported ASR GGUF: {variant!r}")
        path = Path(model_source).absolute() / variant
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError("selected audio.cpp artifact has an invalid GGUF header")
        family = candidate["family"]
        server = NativeASRServer(path, family, device)
        return {"server": server, "task_type": task_type, "device": device,
                "artifact": {"format": "gguf", "file": variant, "revision": model_revision,
                             "family": family, "quantization": candidate["quantization"]},
                "runtime_parameters": {"requested": {"backend": device},
                                       "effective": {"backend": server.build["backend"], "threads": server.threads,
                                                     "version": server.build.get("version"), "git": server.build.get("git"),
                                                     "loader": "audiocpp_server", "lazy_load": False}}}

    def preprocess(self, model_ctx, raw_input):
        if raw_input.get("audio_format") != "wav" or raw_input.get("sample_rate") != SAMPLE_RATE:
            raise ValueError("audio.cpp expects mono 16 kHz PCM16 WAV")
        if raw_input.get("batch_size", 1) != 1:
            raise ValueError("audio.cpp ASR requires batch_size=1")
        params = raw_input.get("params", {})
        if (not isinstance(params, dict) or set(params) - {"mode", "asr_task", "language", "return_timestamps", "pipeline_kwargs"}
                or params.get("mode", "short_form") != "short_form" or params.get("asr_task", "transcribe") != "transcribe"
                or params.get("return_timestamps", False) is not False or params.get("pipeline_kwargs", {})):
            raise ValueError("unsupported audio.cpp ASR parameters")
        wav = base64.b64decode(raw_input["audio_base64"], validate=True)
        _, samples, rate = read_pcm16_wav(wav, "audio_base64")
        if not samples:
            raise ValueError("audio.cpp ASR requires non-empty audio")
        from urllib3.filepost import encode_multipart_formdata

        language = params.get("language", "en")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string")
        body, content_type = encode_multipart_formdata({
            "model": model_ctx["server"].model_id, "language": language,
            "file": ("input.wav", wav, "audio/wav"),
        })
        return {"body": body, "content_type": content_type, "_effective_input_scale": samples / rate,
                "_truncated_by_limit": False, "_duration_s": samples / rate,
                "_probe_reason": "Validated mono PCM16 WAV at the native ASR service boundary",
                "_input_num_samples": samples, "sample_rate": rate, "_channels": 1,
                "_workload": {"input": {"audio": {"audio_seconds": samples / rate, "channels": 1,
                                                "sample_rate": rate, "processor_input_duration": samples / rate}}}}

    def predict(self, model_ctx, processed_input):
        return model_ctx["server"].transcribe(processed_input["body"], processed_input["content_type"])

    def postprocess(self, model_ctx, raw_output):
        if not isinstance(raw_output, dict) or not isinstance(raw_output.get("text"), str):
            raise ValueError("audio.cpp returned no transcript text")
        text = raw_output["text"]
        return {"task": model_ctx["task_type"], "output_type": "transcription", "text": text,
                "output_length": len(text), "output_token_count": None}

    def get_scale_metadata(self, model_ctx, raw_input):
        return {"input_scale_type": "duration_s", "required_sampling_rate": SAMPLE_RATE,
                "source_sampling_rate": raw_input.get("sample_rate"), "short_form_fixed_padding": False,
                "model_type": model_ctx["artifact"]["family"],
                "reason": "audio.cpp ASR input scale is the observed WAV duration"}
