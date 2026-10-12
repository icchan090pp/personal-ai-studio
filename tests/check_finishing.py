"""Paid-API contracts, private key storage and real tiled image processing."""

import base64
import io
import json
import os
import sqlite3

import httpx
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from studio import enhancement, finishing, google_images
from studio.app import create_app
from tests.test_studio import png, project


def check_finishing(root, monkeypatch):
    secret = "test-api-key-never-returned"
    root = root / "finishing-contract"
    app = create_app(root, run_workers=False)
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        pid = project(client)
        assert not client.get("/api/system/google-images").json()["configured"]
        response = client.put("/api/system/google-images", json={"api_key": secret})
        assert response.status_code == 200 and response.json()["configured"]
        assert secret not in response.text
        assert google_images.load_key(root) == secret
        if os.name != "nt":
            assert (root / "google-api-key.bin").stat().st_mode & 0o777 == 0o600
        else:
            assert secret.encode() not in (root / "google-api-key.bin").read_bytes()
        assert (
            client.put("/api/system/google-images", json={"api_key": "bad\nkey"}).status_code == 422
        )
        with TestClient(app, client=("192.168.1.4", 50000)) as remote:
            assert (
                remote.put("/api/system/google-images", json={"api_key": "no"}).status_code == 403
            )
        assert (
            client.post(
                f"/api/projects/{pid}/finish/attachments",
                files={"file": ("broken.png", b"broken", "image/png")},
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
        assert client.get(f"/api/projects/{pid}/images").json() == []
        other = project(client)
        body = {
            "provider": "google",
            "source_attachment_id": iid,
            "prompt": "Preserve composition; make the background blue.",
        }
        assert client.post(f"/api/projects/{other}/finish", json=body).status_code == 404
        assert (
            client.post(
                f"/api/projects/{pid}/finish", json={**body, "source_attachment_id": "../no"}
            ).status_code
            == 404
        )
        assert (
            client.post(f"/api/projects/{pid}/finish", json={**body, "prompt": ""}).status_code
            == 422
        )
        job = client.post(f"/api/projects/{pid}/finish", json=body)
        assert job.status_code == 202, job.text
        jid = job.json()["id"]
        assert secret not in job.text
        assert client.post(f"/api/projects/{pid}/finish", json=body).status_code == 409
        calls = []
        output = Image.new("RGB", (24, 18), "blue")
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")

        def transport(request):
            calls.append(request)
            assert request.url == google_images.ENDPOINT
            assert request.headers["x-goog-api-key"] == secret
            payload = json.loads(request.content)
            assert payload["store"] is False
            assert payload["model"] == "gemini-nano-banana-2.1"
            assert payload["input"][0]["text"] == body["prompt"]
            assert len(payload["input"]) == 2 and payload["input"][1]["type"] == "image"
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "steps": [
                        {"type": "thought", "content": [{"type": "text", "text": "ignored"}]},
                        {
                            "type": "model_output",
                            "content": [
                                {
                                    "type": "image",
                                    "mime_type": "image/png",
                                    "data": base64.b64encode(buffer.getvalue()).decode(),
                                }
                            ],
                        },
                    ],
                },
            )

        actual_edit = google_images.edit

        def mocked_edit(*args):
            return actual_edit(
                *args,
                client_factory=lambda **kwargs: httpx.Client(
                    transport=httpx.MockTransport(transport), **kwargs
                ),
            )

        monkeypatch.setattr(google_images, "edit", mocked_edit)
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
        original = (root / "projects" / pid / "references" / f"{iid}.png").read_bytes()
        finishing.execute(root, jid)
        assert len(calls) == 1
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='completed' WHERE id=?", (jid,))
        assert (root / "projects" / pid / "references" / f"{iid}.png").read_bytes() == original
        assert client.get(f"/api/jobs/{jid}/result").status_code == 200
        assert client.get(f"/api/jobs/{jid}/before").status_code == 200
        assert Image.open(io.BytesIO(client.get(f"/api/jobs/{jid}/result").content)).size == (
            24,
            18,
        )
        rows = client.get(f"/api/projects/{pid}/jobs").json()
        assert rows[0]["finish_provider"] == "google" and secret not in json.dumps(rows)
        assert secret not in (root / "jobs" / jid / "execution.json").read_text()
        for status in [400, 401, 403, 404, 429, 500, 302]:
            attempts = []

            def denied(request):
                attempts.append(request)
                return httpx.Response(
                    status, json={"error": secret}, headers={"location": "https://example.org/leak"}
                )

            with pytest.raises(ValueError) as error:
                actual_edit(
                    output,
                    "edit",
                    "gemini-nano-banana-2.1",
                    "1K",
                    secret,
                    client_factory=lambda **kwargs: httpx.Client(
                        transport=httpx.MockTransport(denied), **kwargs
                    ),
                )
            assert secret not in str(error.value) and len(attempts) == 1

        def empty(request):
            return httpx.Response(200, json={"status": "completed", "steps": []})

        with pytest.raises(ValueError, match="画像が返されません"):
            actual_edit(
                output,
                "edit",
                "gemini-nano-banana-2.1",
                "1K",
                secret,
                client_factory=lambda **kwargs: httpx.Client(
                    transport=httpx.MockTransport(empty), **kwargs
                ),
            )

        def timed_out(request):
            raise httpx.ReadTimeout("sensitive response " + secret)

        with pytest.raises(ValueError, match="自動再送はしていません") as error:
            actual_edit(
                output,
                "edit",
                "gemini-nano-banana-2.1",
                "1K",
                secret,
                client_factory=lambda **kwargs: httpx.Client(
                    transport=httpx.MockTransport(timed_out), **kwargs
                ),
            )
        assert secret not in str(error.value)
        assert (
            client.put("/api/system/google-images", json={"api_key": ""}).json()["configured"]
            is False
        )
        assert client.post(f"/api/projects/{pid}/finish", json=body).status_code == 422
        from studio import app as app_module

        monkeypatch.setattr(app_module, "inspect_system", lambda: {"cpu_ready": True, "ram_gb": 8})
        local = client.post(
            f"/api/projects/{pid}/finish",
            json={"provider": "realesrgan", "source_job_id": jid, "scale": 2},
        )
        assert local.status_code == 202
        assert client.post(f"/api/projects/{pid}/finish/stop").json()["stopped"] == 1
        assert client.get(f"/api/jobs/{jid}/result").status_code == 200
    # Overlapping tiles retain all pixels, including right/bottom edges and color ordering.
    array = np.random.default_rng(42).integers(0, 256, (11, 17, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    progress = []
    enlarged = enhancement.tiled(
        torch.nn.Upsample(scale_factor=4, mode="nearest"), image, progress.append, tile=5, padding=2
    )
    assert np.array_equal(np.asarray(enlarged), np.repeat(np.repeat(array, 4, axis=0), 4, axis=1))
    assert progress[-1] == 95 and len(progress) == 12
    assert set(enhancement.MODELS) == {"anime", "photo"}
    assert len(enhancement.network(6).body) == 6
