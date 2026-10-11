import io
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image

from studio.app import create_app


def png(color="green", size=(128, 128)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def project(client):
    response = client.post("/api/projects", json={"name": "私のスタイル", "trigger": "mystyle"})
    assert response.status_code == 201
    return response.json()["id"]


def test_dataset_caption_persistence_and_normalization(studio):
    client, root = studio
    pid = project(client)
    response = client.post(
        f"/api/projects/{pid}/images",
        files=[
            ("files", ("../../escape.png", png(), "image/png")),
            ("files", ("second.png", png("blue"), "image/png")),
        ],
    )
    assert response.status_code == 201
    rows = response.json()
    assert len(rows) == 2
    iid = rows[0]["id"]
    assert client.get(f"/api/images/{iid}/file").headers["content-type"] == "image/png"
    assert (
        client.patch(f"/api/images/{iid}", json={"caption": "a drawing of mystyle"}).status_code
        == 200
    )
    assert client.get(f"/api/projects/{pid}/images").json()[0]["caption"] == "a drawing of mystyle"
    assert not (root.parent / "escape.png").exists()
    with TestClient(create_app(root, run_workers=False)) as fresh:
        assert fresh.get("/api/projects").json()[0]["image_count"] == 2


def test_invalid_upload_batch_is_not_partially_saved(studio):
    client, _root = studio
    pid = project(client)
    response = client.post(
        f"/api/projects/{pid}/images",
        files=[
            ("files", ("valid.png", png(), "image/png")),
            ("files", ("bad.png", b"not an image", "image/png")),
        ],
    )
    assert response.status_code == 422
    assert client.get(f"/api/projects/{pid}/images").json() == []
    assert (
        client.post(
            f"/api/projects/{pid}/images",
            files={"files": ("small.png", png(size=(32, 32)), "image/png")},
        ).status_code
        == 422
    )


def test_validation_and_gpu_readiness(studio, monkeypatch):
    client, _root = studio
    pid = project(client)
    assert client.post("/api/projects", json={"name": " ", "trigger": "valid"}).status_code == 422
    assert client.post("/api/projects", json={"name": "x", "trigger": "../x"}).status_code == 422
    assert client.post(f"/api/projects/{pid}/train", json={"steps": 1}).status_code == 422
    assert client.post(f"/api/projects/{pid}/train", json={}).status_code == 422
    monkeypatch.setattr("studio.app.importlib.util.find_spec", lambda _module: None)
    assert client.get("/api/system").json()["gpu_ready"] is False
    client.post(
        f"/api/projects/{pid}/images",
        files=[("files", (f"{n}.png", png(), "image/png")) for n in range(3)],
    )
    response = client.post(f"/api/projects/{pid}/train", json={})
    assert response.status_code == 503
    assert client.get(f"/api/projects/{pid}/jobs").json() == []
    assert (
        client.post(f"/api/projects/{pid}/generate", json={"prompt": "mystyle"}).status_code == 422
    )
    assert client.get("/api/projects/missing/jobs").status_code == 404


def test_restart_recovery_and_unready_results(studio):
    client, root = studio
    pid = project(client)
    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,kind,status,config) VALUES(?,?,'train','running','{}')",
            ("test-job", pid),
        )
    with TestClient(create_app(root, run_workers=False)) as fresh:
        assert fresh.get(f"/api/projects/{pid}/jobs").json()[0]["status"] == "failed"
        assert fresh.get("/api/jobs/test-job/result").status_code == 404


def test_video_source_must_belong_to_project(studio):
    client, root = studio
    pid = project(client)
    other = project(client)
    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,kind,status,config) "
            "VALUES('trained',?,'train','completed','{}')",
            (pid,),
        )
        db.execute(
            "INSERT INTO jobs(id,project_id,kind,status,config) "
            "VALUES('foreign-image',?,'image','completed','{}')",
            (other,),
        )
    response = client.post(
        f"/api/projects/{pid}/generate",
        json={"kind": "video", "prompt": "gentle movement", "source_id": "foreign-image"},
    )
    assert response.status_code == 422


def test_cross_origin_upload_is_rejected(studio, monkeypatch):
    client, _root = studio
    response = client.post(
        "/api/projects",
        headers={"Origin": "https://unrelated.example"},
        json={"name": "blocked", "trigger": "valid"},
    )
    assert response.status_code == 403
    assert client.get("/api/projects").json() == []

    monkeypatch.setenv("STUDIO_ACCESS_TOKEN", "test-pairing-code")
    with TestClient(create_app(_root / "paired", run_workers=False)) as paired:
        assert paired.get("/api/models").status_code == 401
        assert paired.get("/static/app.js").status_code == 401
        assert paired.get("/connect?token=wrong").status_code == 401
        connected = paired.get("/connect?token=test-pairing-code", follow_redirects=False)
        assert connected.status_code == 303
        assert "HttpOnly" in connected.headers["set-cookie"]
        assert connected.headers["referrer-policy"] == "no-referrer"
        assert len(paired.get("/api/models").json()) == 2
        assert (
            paired.post(
                "/api/projects",
                json={"name": "phone", "trigger": "phone"},
                headers={"Origin": "https://unrelated.example"},
            ).status_code
            == 403
        )
