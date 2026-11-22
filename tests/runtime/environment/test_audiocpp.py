"""Exercise the native HTTP/child-process boundary without weights or a GPU."""
import base64
import hashlib
import io
import json
import sys
import tarfile
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.container import audiocpp_execution as execution
from acprof.container.handlers import audiocpp
from acprof.host import audiocpp_assets as assets
from acprof.host.input_plan import probe
from acprof.models.audiocpp import ASR, DEFAULT_ASR, REPOSITORY, SPEC_REVISION


@pytest.fixture
def server_executable(tmp_path):
    executable = tmp_path / "cpu" / "audiocpp_server"
    executable.parent.mkdir()
    executable.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
if "--version" in sys.argv:
    print("audio.cpp 0.9.6\\ngit: unknown unknown")
    sys.exit(0)
if "--list-devices" in sys.argv:
    print('{"devices":[{"backend":"cpu"}]}')
    sys.exit(0)
with open(sys.argv[sys.argv.index("--config")+1]) as stream:
    config = json.load(stream)
model = config["models"][0]["id"]
class Handler(BaseHTTPRequestHandler):
    def reply(self, payload, status=200):
        self.send_response(status)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())
    def do_GET(self):
        if self.path == "/health":
            self.reply({"backend": "cpu", "status": "ok", "version": "0.9.6"})
        else:
            self.reply({"data": [{"id": model, "loaded": not config["lazy_load"]}]})
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path != "/v1/audio/transcriptions" or model.encode() not in body or b"RIFF" not in body:
            self.reply({"error":"bad request"}, 400)
        else:
            self.reply({"text": "test transcript"})
HTTPServer((config["host"],config["port"]),Handler).serve_forever()
""")
    executable.chmod(0o755)
    (tmp_path / "manifest.json").write_text(json.dumps({"cpu": {"source": "official"}, "cuda": {"source": "official"}}))
    return executable


def audio_request():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as dest:
        dest.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        dest.writeframes(bytes(32000))
    return {"audio_base64": base64.b64encode(stream.getvalue()).decode(), "sample_rate": 16000,
            "audio_format": "wav", "input_scale": 1, "batch_size": 1}


def test_native_handler_resident_model_request_scale_and_cleanup(server_executable, tmp_path, monkeypatch):
    model = tmp_path / DEFAULT_ASR
    model.parent.mkdir()
    model.write_bytes(b"GGUFfixture")
    monkeypatch.setenv("MODEL_ID", REPOSITORY)
    monkeypatch.setattr(audiocpp, "server_binary", lambda _: server_executable)
    monkeypatch.setattr(execution, "server_binary", lambda _: server_executable)
    assert execution.prepare_device(False) == "cpu"
    with pytest.raises(RuntimeError, match="cuda device unavailable"):
        execution.prepare_device(True)
    handler = audiocpp.AudioCppHandler()
    context = handler.load(str(tmp_path), ASR, "audio_cpp", "cpu", "revision", {"variant": DEFAULT_ASR})
    server = context["server"]
    try:
        payload = audio_request()
        monkeypatch.setattr(probe, "request_scale_meta", lambda _, body: handler.get_scale_metadata(context, body))
        constraints = probe.request_audio_scale_meta(Mock(), payload)
        assert constraints["short_form_fixed_padding"] is False
        assert constraints["model_type"] == "citrinet_asr"
        assert constraints["required_sampling_rate"] == 16000
        assert constraints["max_short_form_duration_s"] is None
        prepared = handler.preprocess(context, payload)
        raw = handler.predict(context, prepared)
        response = handler.postprocess(context, raw)
        assert response["text"] == "test transcript"
        result = handler.validate_output(context, payload, prepared, raw, response)
        assert result["task"]["status"] == "verified"
        assert result["workload_contract"]["input"]["audio"]["processor_input_duration"] == 1
        assert context["artifact"]["file"] == DEFAULT_ASR
        assert context["runtime_parameters"]["effective"]["backend"] == "cpu"
        assert server.transcribe(prepared["body"], prepared["content_type"]) == raw
        with pytest.raises(RuntimeError, match="bad request"):
            server.transcribe(b"invalid", prepared["content_type"])
    finally:
        server.close()
    assert server.process.poll() is not None
    with pytest.raises(RuntimeError, match="exited"):
        handler.predict(context, prepared)


def test_load_rejects_wrong_file_before_process_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ID", REPOSITORY)
    path = tmp_path / DEFAULT_ASR
    path.parent.mkdir()
    path.write_bytes(b"ONNX")
    handler = audiocpp.AudioCppHandler()
    with pytest.raises(ValueError, match="GGUF header"):
        handler.load(str(tmp_path), ASR, "audio_cpp", "cpu", load_options={"variant": DEFAULT_ASR})
    with pytest.raises(ValueError, match="ASR GGUF"):
        handler.load(str(tmp_path), ASR, "audio_cpp", "cpu", load_options={"variant": str(path)})
    with pytest.raises(ValueError, match="offline ASR"):
        handler.load(str(tmp_path), ASR, "onnxruntime", "cpu", load_options={"variant": DEFAULT_ASR})
    with pytest.raises(ValueError, match="no transcript"):
        handler.postprocess({}, {"error": "failed"})


def test_local_cpu_requires_compatible_build_and_stable_sha(server_executable, tmp_path, monkeypatch):
    task = SimpleNamespace(runtime_backend="audio_cpp", model_resolution={"requested_devices": ["off"]})
    monkeypatch.setenv("ACPROF_AUDIOCPP_CPU_SERVER", str(server_executable))
    selection = assets.selected_assets(task)
    assert set(selection) == {"cpu"}
    assert assets.download_sources(selection, process=True) == []
    output = tmp_path / "installed"
    assets.install_assets(selection, output)
    with pytest.raises(ValueError, match="requires v0.9.6"):
        execution.check_binary(output / "cpu/audiocpp_server", "cpu")
    server_executable.write_text(server_executable.read_text().replace("git: unknown", f"git: {SPEC_REVISION[:8]}"))
    with pytest.raises(ValueError, match="changed"):
        assets.install_assets(selection, tmp_path / "changed")
    output = tmp_path / "compatible"
    assets.install_assets(assets.selected_assets(task), output)
    assert execution.check_binary(output / "cpu/audiocpp_server", "cpu")["devices"] == [{"backend":"cpu"}]


def test_download_locks_integrity_and_extraction(tmp_path, monkeypatch):
    archive = tmp_path / "fixture.tar.gz"
    binary = b"fake native binary"
    with tarfile.open(archive, mode="w:gz") as tar:
        member = tarfile.TarInfo("./audiocpp_server")
        member.size = len(binary)
        tar.addfile(member, io.BytesIO(binary))
    source = {"url": archive.as_uri(), "size": archive.stat().st_size, "filename": "release.tar.gz",
              "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    destination = tmp_path / "staging"
    with pytest.raises(ValueError, match="verification"):
        assets.install_assets({"cuda": {**source, "sha256": "0"*64}}, destination)
    assert not assets._archive(source).exists()
    assets.install_assets({"cuda": source}, tmp_path / "ready")
    assert assets.download_sources({"cuda": source}, process=True)[0].cache_status == "hit"
    assert (tmp_path / "ready/cuda/audiocpp_server").read_bytes() == binary
    artifacts = assets.selected_assets(SimpleNamespace(runtime_backend="audio_cpp",
                                                      model_resolution={"requested_devices":["on"]}))
    assert set(artifacts) == {"cuda"}


def test_cancel_during_eager_load_reaps_native_child(server_executable, monkeypatch):
    monkeypatch.setattr(audiocpp, "server_binary", lambda _: server_executable)
    children = []

    def cancel(server, _device):
        children.append(server.process)
        raise KeyboardInterrupt()

    monkeypatch.setattr(audiocpp.NativeASRServer, "_wait_ready", cancel)
    with pytest.raises(KeyboardInterrupt):
        audiocpp.NativeASRServer(server_executable, "citrinet_asr", "cpu")
    assert len(children) == 1
    assert children[0].returncode is not None


def test_cpu_abi_rejects_before_transfer(monkeypatch):
    monkeypatch.delenv("ACPROF_AUDIOCPP_CPU_SERVER", raising=False)
    monkeypatch.setattr(assets.platform, "libc_ver", lambda: ("glibc", "2.39"))
    selection = assets.selected_assets(SimpleNamespace(runtime_backend="audio_cpp",
                                                      model_resolution={"requested_devices":["off"]}))
    for process in (False, True):
        with pytest.raises(RuntimeError, match="ACPROF_AUDIOCPP_CPU_SERVER"):
            assets.download_sources(selection, process=process)
