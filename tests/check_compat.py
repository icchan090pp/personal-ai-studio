"""Additional assertions inside the existing hardware and numerical product cases."""

import io
import json
import os
import queue
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from studio.app import create_app
from studio.hardware import describe
from studio.setup_runtime import installation_plan


def check_hardware_and_setup(root, monkeypatch):
    from studio import app as module
    from studio import hardware
    from tests.check_updates import check_updates

    check_updates(root, monkeypatch)
    assert "8GB未満" in hardware.memory_guidance(4)
    assert "ノーマル" in hardware.memory_guidance(8)
    assert "GPUメモリ" in hardware.memory_guidance(16)
    assert "取得できません" in hardware.memory_guidance(None)

    output = subprocess.run(
        [
            sys.executable,
            "-c",
            "from studio import setup_runtime as s; "
            "s.installation_plan=lambda *a: []; "
            "s.inspect_system=lambda: {'message':'日本語の診断'}; "
            "s.inspect_system.cache_clear=lambda: None; s.main()",
            "--backend",
            "cpu",
        ],
        env={
            **os.environ,
            "PYTHONIOENCODING": "cp1252",
            "PYTHONUTF8": "0",
            "STUDIO_DATA_DIR": str(root / "cli-state"),
        },
        capture_output=True,
        check=True,
    )
    assert "日本語の診断" in output.stdout.decode("utf-8")
    assert "セットアップ完了" in output.stdout.decode("utf-8")
    assert hardware.selected_backend(root / "cli-state") == "cpu"
    for vendor in ("AMD Radeon (TM) Graphics", "Intel Iris Xe", "Unknown GPU"):
        assert describe([vendor], True, False, True, {"ready": False})["code"] == "cpu_ready"
        assert describe([vendor], False, False, False, {"ready": False})["code"] == "setup_required"
        assert describe([vendor], True, False, True, {"ready": True})["code"] == "directml_ready"
    assert describe(["NVIDIA RTX"], True, True, True, {})["code"] == "cuda_ready"
    assert "ドライバ" in describe(["NVIDIA RTX"], True, False, True, {}, "12.8")["message"]
    assert "ライブラリ" in describe(["NVIDIA RTX"], True, False, True, {})["message"]
    for backend in ("cpu", "directml", "cuda"):
        plan = installation_plan(root, backend, voice=True)
        suffix = "cu128" if backend == "cuda" else "cpu"
        assert f"torch==2.7.0+{suffix}" in plan[0]
        assert f"torchaudio==2.7.0+{suffix}" in plan[0]
        assert plan[0][-1] == "https://download.pytorch.org/whl/" + suffix
        assert plan[2] == [str(root / "scripts/install-voice-backend.py")]
        assert plan[-1] == ["-m", "pip", "check"]
    with monkeypatch.context() as patch:
        patch.setattr(
            hardware.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        try:
            hardware.run_json(["unused"])
            raise AssertionError("Empty diagnostic response was accepted")
        except ValueError:
            pass
    folder = root / "desktop-test"
    application = create_app(folder, run_workers=False)
    command = root / "command.json"
    with (
        monkeypatch.context() as patch,
        TestClient(application, client=("127.0.0.1", 40000)) as client,
    ):
        # Exercise the desktop startup handshake with the expanded real catalog.
        from windows.desktop import Application

        desktop = Application.__new__(Application)
        desktop.python = Path(sys.executable)
        desktop.port = 8000
        desktop.url = "http://127.0.0.1:8000"
        desktop.closing = False
        desktop.events = queue.Queue()
        desktop.spawn = lambda args: SimpleNamespace(poll=lambda: None)
        catalog = client.get("/api/models").content
        assert len(json.loads(catalog)) == 7
        with monkeypatch.context() as startup_patch:
            startup_patch.setattr(
                "windows.desktop.urllib.request.urlopen", lambda *a, **k: io.BytesIO(catalog)
            )
            desktop.start()
            try:
                assert desktop.events.get(timeout=2) == ("ready", True)
            finally:
                desktop.closing = True
        patch.setenv("STUDIO_DESKTOP_COMMAND", str(command))
        assert client.post("/api/system/setup", json={"backend": "unknown"}).status_code == 422
        assert (
            client.post("/api/system/setup", json={"backend": "cpu", "voice": True}).status_code
            == 202
        )
        assert json.loads(command.read_text()) == {"backend": "cpu", "voice": True}
        assert client.post("/api/system/setup", json={}).status_code == 409
        command.unlink()
        patch.setattr(
            module,
            "check_update",
            lambda: {"current": "0.1.7", "latest": "0.1.8", "available": True},
        )
        assert client.get("/api/system/update").json()["available"]
        assert client.post("/api/system/update", json={}).status_code == 202
        assert json.loads(command.read_text()) == {"action": "update"}
        assert client.post("/api/system/update", json={}).status_code == 409
        command.unlink()
        pid = client.post("/api/projects", json={"name": "compat", "trigger": "tinytest"}).json()[
            "id"
        ]
        patch.setattr(module, "inspect_system", lambda: {"cpu_ready": False})
        assert (
            client.post(
                f"/api/projects/{pid}/generate", json={"prompt": "bird", "public_model": "tiny"}
            ).status_code
            == 503
        )
        patch.setattr(module, "inspect_system", lambda: {"cpu_ready": True})
        response = client.post(
            f"/api/projects/{pid}/generate", json={"prompt": "bird", "public_model": "tiny"}
        )
        assert (
            response.status_code == 202
            and json.loads(response.json()["config"])["public_model"] == "tiny"
        )
        assert client.post("/api/system/setup", json={}).status_code == 409
        assert client.post("/api/system/update", json={}).status_code == 409
        assert not command.exists()
    with TestClient(application, client=("192.168.1.2", 40000)) as remote:
        assert remote.post("/api/system/setup", json={}).status_code == 403
        assert remote.post("/api/system/update", json={}).status_code == 403
    hardware.inspect_system.cache_clear()


def check_compat_generation(root, monkeypatch):
    from studio import compat, hardware
    from tests.test_lora_numerical import tiny_pipeline

    folder = root / "compat"
    folder.mkdir()
    config = {
        "prompt": "a blue bird",
        "negative_prompt": "blur",
        "seed": 42,
        "inference_steps": 2,
        "resolution": "native",
        "public_model": "tiny",
    }
    database = folder / "studio.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE jobs (id TEXT, config TEXT, status TEXT, progress INTEGER)")
        db.execute("INSERT INTO jobs VALUES ('tiny',?,'running',0)", (json.dumps(config),))
    actual_generate = compat.generate
    with monkeypatch.context() as patch:
        patch.setattr(compat, "pipeline", lambda model=None: tiny_pipeline(folder))
        patch.setattr(compat, "generate", lambda *a, **k: actual_generate(*a, **k, size=32))
        patch.setattr(hardware, "inspect_system", lambda: {"directml": {"ready": False}})
        compat.execute(folder, "tiny")
        output = folder / "jobs/tiny/output.png"
        assert Image.open(output).size == (32, 32)
        assert json.loads((output.parent / "execution.json").read_text())["backend"] == "cpu"
        first = output.read_bytes()
        output.unlink()
        patch.setenv("STUDIO_DML_PYTHON", "directml-test-python")
        patch.setattr(hardware, "inspect_system", lambda: {"directml": {"ready": True, "index": 0}})
        commands = []

        def fail_gpu(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 1)

        patch.setattr(compat.subprocess, "run", fail_gpu)
        compat.execute(folder, "tiny")
        assert commands[0][0] == "directml-test-python"
        safe = folder / "models/tiny-sd-safetensors"
        assert (safe / "text_encoder/model.safetensors").exists()
        assert not list(safe.rglob("*.bin"))
        assert output.read_bytes() == first
        detail = json.loads((output.parent / "execution.json").read_text())
        assert detail["backend"] == "cpu" and detail["fallback_reason"]
        compat.execute(folder, "tiny")
        assert len(commands) == 1  # A known failing GPU is not retried on every image.
        assert output.read_bytes() == first
        attempts = []

        def reject_cuda(*args, **kwargs):
            attempts.append(kwargs.get("device"))
            if kwargs.get("device") == "cuda":
                raise RuntimeError("CUDA out of memory")
            return actual_generate(*args, **kwargs, size=32)

        patch.setattr(compat, "generate", reject_cuda)
        patch.setattr(
            hardware, "inspect_system", lambda: {"cuda_ready": True, "directml": {"ready": False}}
        )
        compat.execute(folder, "tiny")
        assert attempts == ["cuda", "cpu"]
        detail = json.loads((output.parent / "execution.json").read_text())
        assert detail["backend"] == "cpu" and "CUDA" in detail["fallback_reason"]
        with sqlite3.connect(database) as db:
            assert db.execute("SELECT progress FROM jobs").fetchone()[0] == 99
