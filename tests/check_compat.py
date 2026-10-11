"""Additional assertions inside the existing hardware and numerical product cases."""

import json
import sqlite3
import subprocess
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from studio.app import create_app
from studio.hardware import describe
from studio.setup_runtime import installation_plan


def check_hardware_and_setup(root, monkeypatch):
    from studio import app as module
    from studio import hardware

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
        patch.setenv("STUDIO_DESKTOP_COMMAND", str(command))
        assert client.post("/api/system/setup", json={"backend": "unknown"}).status_code == 422
        assert (
            client.post("/api/system/setup", json={"backend": "cpu", "voice": True}).status_code
            == 202
        )
        assert json.loads(command.read_text()) == {"backend": "cpu", "voice": True}
        assert client.post("/api/system/setup", json={}).status_code == 409
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
        assert not command.exists()
    with TestClient(application, client=("192.168.1.2", 40000)) as remote:
        assert remote.post("/api/system/setup", json={}).status_code == 403
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
        with sqlite3.connect(database) as db:
            assert db.execute("SELECT progress FROM jobs").fetchone()[0] == 99
