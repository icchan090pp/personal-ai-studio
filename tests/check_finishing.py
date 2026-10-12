"""Local restoration, reference ownership and preservation of original images."""

import io
import sqlite3

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from studio import enhancement, finishing
from studio.app import create_app
from tests.test_studio import png, project


def check_finishing(root, monkeypatch):
    from studio import app as app_module

    root = root / "finishing-contract"
    monkeypatch.setattr(app_module, "inspect_system", lambda: {"cpu_ready": True, "ram_gb": 16})
    app = create_app(root, run_workers=False)
    with TestClient(app) as client:
        pid = project(client)
        assert client.get("/api/system/google-images").status_code == 404
        assert (
            client.put("/api/system/google-images", json={"api_key": "unused"}).status_code == 404
        )
        assert (
            client.post(
                f"/api/projects/{pid}/finish/attachments",
                files={"file": ("bad.png", b"bad", "image/png")},
            ).status_code
            == 422
        )
        reply = client.post(
            f"/api/projects/{pid}/finish/attachments",
            files={"file": ("reference.png", png(), "image/png")},
        )
        assert reply.status_code == 201
        reference = reply.json()
        iid = reference["id"]
        assert client.get(reference["url"]).status_code == 200
        other = project(client)
        assert client.get(f"/api/projects/{other}/finish/attachments/{iid}").status_code == 404
        assert client.get(f"/api/projects/{pid}/images").json() == []
        body = {"provider": "realesrgan", "source_attachment_id": iid, "scale": 2}
        assert (
            client.post(
                f"/api/projects/{pid}/finish", json={**body, "provider": "google"}
            ).status_code
            == 422
        )
        assert client.post(f"/api/projects/{other}/finish", json=body).status_code == 404
        assert (
            client.post(
                f"/api/projects/{pid}/finish", json={**body, "source_image_id": "missing"}
            ).status_code
            == 422
        )
        job = client.post(f"/api/projects/{pid}/finish", json=body)
        assert job.status_code == 202, job.text
        jid = job.json()["id"]
        assert client.post(f"/api/projects/{pid}/finish", json=body).status_code == 409
        original = (root / "projects" / pid / "references" / f"{iid}.png").read_bytes()
        calls = []

        def restore(source, data, model, scale, denoise, progress):
            calls.append((source.size, model, scale, denoise))
            assert data == root
            return source.resize((24, 18)), {"model": model}

        monkeypatch.setattr(enhancement, "enhance", restore)
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
        finishing.execute(root, jid)
        assert len(calls) == 1
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='completed' WHERE id=?", (jid,))
        assert (root / "projects" / pid / "references" / f"{iid}.png").read_bytes() == original
        assert client.get(f"/api/jobs/{jid}/result").status_code == 200
        assert client.get(f"/api/jobs/{jid}/before").content == original
        assert Image.open(io.BytesIO(client.get(f"/api/jobs/{jid}/result").content)).size == (
            24,
            18,
        )
        assert client.get(f"/api/projects/{pid}/jobs").json()[0]["finish_provider"] == "realesrgan"
        local = client.post(f"/api/projects/{pid}/finish", json={"source_job_id": jid, "scale": 4})
        assert local.status_code == 202
        assert client.post(f"/api/projects/{pid}/finish/stop").json()["stopped"] == 1
        assert client.get(f"/api/jobs/{jid}/result").status_code == 200
        from tests.check_reference import check_reference

        check_reference(client, root, pid, other, iid, monkeypatch)
    array = np.random.default_rng(42).integers(0, 256, (11, 17, 3), dtype=np.uint8)
    progress = []
    enlarged = enhancement.tiled(
        torch.nn.Upsample(scale_factor=4, mode="nearest"),
        Image.fromarray(array),
        progress.append,
        tile=5,
        padding=2,
    )
    assert np.array_equal(np.asarray(enlarged), np.repeat(np.repeat(array, 4, axis=0), 4, axis=1))
    assert progress[-1] == 95 and len(progress) == 12
    assert set(enhancement.MODELS) == {"anime", "photo"}
    assert len(enhancement.network(6).body) == 6
