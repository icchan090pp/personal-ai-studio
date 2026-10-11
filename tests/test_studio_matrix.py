"""Distinct boundary and integration cases, not repeated runs of identical inputs."""

import io
import json
import sqlite3
import time

import pytest
from PIL import Image
from pydantic import ValidationError

from studio.app import GenerationInput, TrainingInput, create_app
from tests.test_studio import png, project


@pytest.mark.parametrize(
    "name,trigger,valid",
    [
        ("モデル", "myconcept", True),
        ("A", "abc", True),
        ("x" * 80, "a" * 40, True),
        ("水彩画 🎨", "Art_123", True),
        ("人物\n作品", "ABC", True),
        ("  trim  ", "Ab0", True),
        ("", "abc", False),
        (" ", "abc", False),
        ("x" * 81, "abc", False),
        ("x", "ab", False),
        ("x", "a" * 41, False),
        ("x", "1abc", False),
        ("x", "日本語", False),
        ("x", "ab-c", False),
        ("x", "a b", False),
        ("x", "../abc", False),
        ("x", "abc!", False),
        ("x", "", False),
    ],
)
def test_project_boundaries(studio, name, trigger, valid):
    client, _root = studio
    response = client.post("/api/projects", json={"name": name, "trigger": trigger})
    assert response.status_code == (201 if valid else 422)
    if valid:
        assert response.json()["name"] == name.strip()
        assert response.json()["trigger"] == trigger
    else:
        assert client.get("/api/projects").json() == []


@pytest.mark.parametrize(
    "field,value,valid",
    [
        ("steps", 9, False),
        ("steps", 10, True),
        ("steps", 500, True),
        ("steps", 5000, True),
        ("steps", 5001, False),
        ("learning_rate", 0.0000009, False),
        ("learning_rate", 0.000001, True),
        ("learning_rate", 0.0001, True),
        ("learning_rate", 0.001, True),
        ("learning_rate", 0.00101, False),
        ("rank", 3, False),
        ("rank", 4, True),
        ("rank", 8, True),
        ("rank", 32, True),
        ("rank", 33, False),
        ("seed", -1, False),
        ("seed", 0, True),
        ("seed", 42, True),
        ("seed", 2**32 - 1, True),
        ("seed", 2**32, False),
    ],
)
def test_training_boundaries(field, value, valid):
    if valid:
        assert getattr(TrainingInput(**{field: value}), field) == value
    else:
        with pytest.raises(ValidationError):
            TrainingInput(**{field: value})


@pytest.mark.parametrize(
    "field,value,valid",
    [
        ("kind", "image", True),
        ("kind", "video", True),
        ("kind", "sound", False),
        ("kind", "", False),
        ("kind", None, False),
        ("prompt", "", False),
        ("prompt", " ", False),
        ("prompt", "a", True),
        ("prompt", "p" * 2000, True),
        ("prompt", "p" * 2001, False),
        ("negative_prompt", "", True),
        ("negative_prompt", "n" * 2000, True),
        ("negative_prompt", "n" * 2001, False),
        ("negative_prompt", None, False),
        ("negative_prompt", "ぼやけた画像", True),
        ("seed", -1, False),
        ("seed", 0, True),
        ("seed", 42, True),
        ("seed", 2**32 - 1, True),
        ("seed", 2**32, False),
    ],
)
def test_generation_boundaries(field, value, valid):
    data = {"prompt": "a photo of myconcept", field: value}
    if valid:
        assert getattr(GenerationInput(**data), field) == value
    else:
        with pytest.raises(ValidationError):
            GenerationInput(**data)


@pytest.mark.parametrize(
    "format,size,raw,status",
    [
        ("PNG", (64, 64), None, 201),
        ("JPEG", (128, 128), None, 201),
        ("WEBP", (128, 128), None, 201),
        ("GIF", (128, 128), None, 422),
        ("BMP", (128, 128), None, 422),
        ("TIFF", (128, 128), None, 422),
        ("PNG", (63, 128), None, 422),
        ("PNG", (128, 63), None, 422),
        ("PNG", (64, 128), None, 201),
        ("PNG", (128, 128), b"corrupt", 422),
        ("PNG", (128, 128), b"", 422),
        ("PNG", (128, 128), b"x" * (20 * 1024 * 1024 + 1), 413),
    ],
    ids=[
        "png-minimum",
        "jpeg",
        "webp",
        "gif-rejected",
        "bmp-rejected",
        "tiff-rejected",
        "width-small",
        "height-small",
        "height-minimum",
        "corrupt",
        "empty",
        "over-20mb",
    ],
)
def test_upload_format_dimension_and_capacity(studio, format, size, raw, status):
    client, root = studio
    pid = project(client)
    if raw is None:
        stream = io.BytesIO()
        Image.new("RGB", size, "orange").save(stream, format=format)
        raw = stream.getvalue()
    response = client.post(
        f"/api/projects/{pid}/images",
        files={"files": ("untrusted-name.png", raw, "application/octet-stream")},
    )
    assert response.status_code == status
    rows = client.get(f"/api/projects/{pid}/images").json()
    assert len(rows) == (1 if status == 201 else 0)
    if rows:
        with Image.open(root / "projects" / pid / "images" / f"{rows[0]['id']}.png") as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("get", "/api/projects/missing/images", None),
        ("get", "/api/projects/missing/jobs", None),
        ("post", "/api/projects/missing/train", {}),
        ("post", "/api/projects/missing/generate", {"prompt": "test"}),
        ("get", "/api/images/missing/file", None),
        ("patch", "/api/images/missing", {"caption": "test"}),
        ("get", "/api/jobs/missing/result", None),
    ],
)
def test_missing_resources(studio, method, path, body):
    client, _root = studio
    response = getattr(client, method)(path, **({"json": body} if body is not None else {}))
    assert response.status_code == 404


@pytest.mark.parametrize(
    "kind,name,mime",
    [
        ("train", "adapter.zip", "application/zip"),
        ("image", "output.png", "image/png"),
        ("video", "output.mp4", "video/mp4"),
    ],
)
def test_output_streaming_and_download_headers(studio, kind, name, mime):
    client, root = studio
    pid = project(client)
    # Test transport of an existing artifact; these fixture bytes are not model outputs.
    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,kind,status,config) "
            "VALUES('artifact',?,?,'completed','{}')",
            (pid, kind),
        )
    folder = root / "jobs/artifact"
    folder.mkdir(parents=True)
    (folder / name).write_bytes(b"transport-test")
    response = client.get("/api/jobs/artifact/result?download=true")
    assert response.status_code == 200
    assert response.content == b"transport-test"
    assert response.headers["content-type"] == mime
    assert name in response.headers["content-disposition"]


def ready_dataset(client):
    pid = project(client)
    client.post(
        f"/api/projects/{pid}/images",
        files=[("files", (f"{i}.png", png(), "image/png")) for i in range(3)],
    )
    return pid


def gpu_readiness_only(monkeypatch):
    # Hardware readiness is the only mock; job configuration and worker code stay real.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_queued_job_freezes_dataset_and_base(studio, monkeypatch):
    client, root = studio
    gpu_readiness_only(monkeypatch)
    monkeypatch.setenv("STUDIO_CHECKPOINT", "/test/model.safetensors")
    pid = ready_dataset(client)
    response = client.post(f"/api/projects/{pid}/train", json={"steps": 10})
    assert response.status_code == 202
    job = response.json()
    config = json.loads(job["config"])
    assert config["steps"] == 10 and len(config["images"]) == 3
    assert config["base_model"] == "/test/model.safetensors"
    iid = config["images"][0]["id"]
    client.patch(f"/api/images/{iid}", json={"caption": "changed after queuing"})
    with sqlite3.connect(root / "studio.sqlite3") as db:
        saved = json.loads(
            db.execute("SELECT config FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
        )
    assert saved["images"][0]["caption"] != "changed after queuing"


def test_second_job_does_not_overlap(studio, monkeypatch):
    client, _root = studio
    gpu_readiness_only(monkeypatch)
    pid = ready_dataset(client)
    assert client.post(f"/api/projects/{pid}/train", json={}).status_code == 202
    assert client.post(f"/api/projects/{pid}/train", json={}).status_code == 409
    assert len(client.get(f"/api/projects/{pid}/jobs").json()) == 1
    before = client.get(f"/api/projects/{pid}/images").json()
    assert client.post(f"/api/projects/{pid}/training/reset", json={}).json()["reset_count"] == 1
    assert client.get(f"/api/projects/{pid}/jobs").json()[0]["status"] == "cancelled"
    assert client.get(f"/api/projects/{pid}/images").json() == before
    assert client.post(f"/api/projects/{pid}/train", json={}).status_code == 202
    assert client.post("/api/projects/missing/training/reset", json={}).status_code == 404


def test_worker_failure_records_error_without_marking_completed(studio, monkeypatch):
    from fastapi.testclient import TestClient

    client, root = studio
    gpu_readiness_only(monkeypatch)
    pid = ready_dataset(client)
    with TestClient(create_app(root, run_workers=True)) as running:
        # Exercise a real failing subprocess; CPU environment cannot run a CUDA job.
        response = running.post(f"/api/projects/{pid}/train", json={"steps": 10})
        assert response.status_code == 202
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            job = running.get(f"/api/projects/{pid}/jobs").json()[0]
            if job["status"] == "failed":
                break
            time.sleep(0.1)
        assert job["status"] == "failed"
        assert job["error"]
        assert running.get(f"/api/jobs/{job['id']}/result").status_code == 404
        assert (root / "jobs" / job["id"] / "worker.log").is_file()


def test_worker_rejects_success_exit_without_output(studio, monkeypatch):
    import subprocess
    import sys
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    client, root = studio
    gpu_readiness_only(monkeypatch)
    pid = ready_dataset(client)
    real_popen = subprocess.Popen
    # Controlled process exit tests the artifact guard, not model behavior.
    monkeypatch.setattr(
        "studio.app.subprocess.Popen",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, wait=lambda **kw: 0, poll=lambda: 0),
    )
    with TestClient(create_app(root, run_workers=True)) as running:
        response = running.post(f"/api/projects/{pid}/train", json={"steps": 10})
        assert response.status_code == 202
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = running.get(f"/api/projects/{pid}/jobs").json()[0]
            if job["status"] == "failed":
                break
            time.sleep(0.05)
        assert job["status"] == "failed"
        assert "出力ファイル" in job["error"]
        assert (
            running.post(f"/api/projects/{pid}/training/reset", json={}).json()["reset_count"] == 1
        )
        children = []

        def sleeper(*args, **kwargs):
            child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            children.append(child)
            return child

        monkeypatch.setattr("studio.app.subprocess.Popen", sleeper)
        response = running.post(f"/api/projects/{pid}/train", json={"steps": 10})
        deadline = time.monotonic() + 5
        while not children and time.monotonic() < deadline:
            time.sleep(0.02)
        assert children
        assert running.post(f"/api/projects/{pid}/training/reset", json={}).status_code == 200
        assert children[0].poll() is not None
        assert running.get(f"/api/projects/{pid}/jobs").json()[0]["status"] == "cancelled"
