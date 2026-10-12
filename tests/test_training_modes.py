import json
import socket
import sqlite3

import imageio.v2 as imageio
import numpy as np
import pytest
from diffusers.utils import export_to_video
from fastapi.testclient import TestClient
from PIL import Image

from studio.app import BASE_MODEL, create_app
from studio.collection import MediaPage, collect, extract_media, public_address
from studio.exports import RESOLUTIONS, save_image, save_video
from studio.gpu import prepare_training_rows, reward_weight
from tests.test_studio import png, project
from tests.test_studio_matrix import gpu_readiness_only, ready_dataset


def test_two_model_catalog(studio, monkeypatch):
    client, _root = studio
    models = client.get("/api/models").json()
    assert len(models) == 7
    from tests.check_related_learning import check_related_learning, check_related_video

    with monkeypatch.context() as patch:
        check_related_learning(client, _root, patch)
        check_related_video(client, _root, patch)
    from tests.check_research import check_research

    with monkeypatch.context() as patch:
        check_research(client, _root, patch)
    from tests.check_image_learning import check_image_learning

    check_image_learning(client, _root, monkeypatch)
    from tests.check_agent_video import check_agent_contract, check_video_contract

    check_agent_contract(client, _root, monkeypatch)
    check_video_contract(client, _root, monkeypatch)
    assert {model["kind"] for model in models} == {"image", "video"}
    assert models[0]["trainable"] is True
    assert models[1]["trainable"] is True
    assert all(
        model["weights_url"].endswith(model.get("motion_model", model["id"])) for model in models
    )
    audio_models = client.get("/api/audio-models").json()
    assert {m["key"] for m in audio_models} == {"audioldm2", "kokoro"}
    assert all(m["license"] and m["revision"] and not m["trainable"] for m in audio_models)
    public = client.get("/api/public-models").json()
    assert len(public) == 5 and len({m["id"] for m in public}) == 5
    assert {m["key"] for m in public} == {"tiny", "sdxl", "ssd", "pixart", "ltx"}
    assert {m["kind"] for m in public} == {"image", "video"}
    assert all(not m["trainable"] and m["license"] and m["revision"] for m in public)
    import diffusers

    for model in public:
        assert hasattr(diffusers, model["pipeline"])

    from types import SimpleNamespace

    from studio.public_models import generate_public

    received = []

    class ControlledPipeline:
        vae = SimpleNamespace(enable_tiling=lambda: None)

        def enable_model_cpu_offload(self):
            pass

        def __call__(self, **kwargs):
            received.append(kwargs)
            image = Image.new("RGB", (64, 64), "blue")
            return SimpleNamespace(images=[image], frames=[[image] * 49])

    # Model weights are unavailable on this CPU runner. Exercise all four dispatch
    # contracts and real image/video exports, without claiming quality verification.
    for model in [entry for entry in public if entry["key"] != "tiny"]:

        def load(model_id, revision, torch_dtype):
            assert model_id == model["id"] and revision == model["revision"]
            return ControlledPipeline()

        monkeypatch.setattr(getattr(diffusers, model["pipeline"]), "from_pretrained", load)
        folder = _root / model["key"]
        folder.mkdir()
        generate_public(
            {
                "public_model": model["key"],
                "prompt": "landscape",
                "negative_prompt": "blur",
                "inference_steps": 200,
                "guidance_scale": 8.5,
                "seed": 42,
                "resolution": "native",
            },
            folder,
            lambda *a: None,
        )
        assert received[-1]["num_inference_steps"] == 200
        assert received[-1]["guidance_scale"] in {3.0, 4.5, 5.0}
        assert (folder / ("output.mp4" if model["kind"] == "video" else "output.png")).is_file()
    # All six modes have a functioning validation path, and unknown modes fail.
    pid = project(client)
    from studio import assistant as assistant_module

    calls = []

    class LocalConnection:
        def __init__(self, host, port, timeout):
            assert host == "127.0.0.1" and port == 11434

        def request(self, method, path, body, headers):
            calls.append(json.loads(body))
            assert method == "POST" and path == "/api/chat"

        def getresponse(self):
            return self

        status = 200

        def read(self, limit):
            return b'{"message":{"content":"watercolor texture"}}'

        def close(self):
            pass

    monkeypatch.setattr(assistant_module.http.client, "HTTPConnection", LocalConnection)
    for model in ("gpt-oss:20b", "gpt-oss:120b"):
        response = client.post(
            f"/api/projects/{pid}/assistant",
            json={"prompt": "help", "model": model, "level": "ultra"},
        )
        assert response.json()["answer"] == "watercolor texture"
        assert calls[-1]["think"] == "high" and calls[-1]["model"] == model
        assert calls[-1]["options"]["num_predict"] == 4096
    assert (
        client.post(
            f"/api/projects/{pid}/assistant", json={"prompt": "help", "model": "unknown"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/projects/{pid}/generate",
            json={"prompt": "test", "kind": "video", "public_model": "sdxl"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/projects/{pid}/generate", json={"prompt": "test", "public_model": "unknown"}
        ).status_code
        == 422
    )

    assert client.get("/api/system").json()["hardware"]["cpu_ready"] is True
    gpu_readiness_only(monkeypatch)
    for model in public:
        response = client.post(
            f"/api/projects/{pid}/generate",
            json={
                "prompt": "landscape",
                "public_model": model["key"],
                "kind": model["kind"],
                "level": "ultra",
            },
        )
        assert response.status_code == 202
        config = json.loads(response.json()["config"])
        assert config["inference_steps"] == 200 and config["public_model"] == model["key"]
        with sqlite3.connect(_root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='completed' WHERE id=?", (response.json()["id"],))
    pid = project(client)
    for mode in (
        "supervised",
        "semi_supervised",
        "style",
        "reinforcement",
        "generation_test",
        "imitation",
    ):
        response = client.post(f"/api/projects/{pid}/train", json={"mode": mode})
        assert response.status_code == 422 and isinstance(response.json()["detail"], str)
    initial = client.get(f"/api/projects/{pid}/settings").json()
    assert initial == {"rules_prompt": "", "negative_rules": ""}
    assert (
        client.patch(
            f"/api/projects/{pid}/settings",
            json={"rules_prompt": "watercolor", "negative_rules": "blur"},
        ).status_code
        == 200
    )
    assert client.get(f"/api/projects/{pid}/settings").json()["rules_prompt"] == "watercolor"
    assert (
        client.patch(f"/api/projects/{pid}/settings", json={"rules_prompt": "x" * 1001}).status_code
        == 422
    )
    assert client.patch(f"/api/projects/{pid}/settings", json={}).json() == initial
    modules = client.get("/api/course/modules").json()
    assert len(modules) == 8 and len({item["id"] for item in modules}) == 8
    assert client.get(f"/api/projects/{pid}/course").json() == []
    saved = client.patch(
        f"/api/projects/{pid}/course/foundations", json={"done": True, "notes": "experiment"}
    )
    assert saved.status_code == 200
    assert client.get(f"/api/projects/{pid}/course").json()[0]["notes"] == "experiment"
    assert client.patch(f"/api/projects/{pid}/course/unknown", json={}).status_code == 404
    assert (
        client.patch(
            f"/api/projects/{pid}/course/foundations", json={"notes": "x" * 3001}
        ).status_code
        == 422
    )
    assert client.get(f"/api/projects/{pid}/metrics").json()["points"] == []
    system = client.get("/api/system").json()
    assert set(system["levels"]) == {"normal", "power", "excellent", "ultra"}
    for level in system["levels"]:
        response = client.patch(f"/api/projects/{pid}/level", json={"level": level})
        assert response.status_code == 200 and response.json()["preset"] == system["levels"][level]
        assert client.get("/api/projects").json()[0]["level"] == level
    assert client.patch(f"/api/projects/{pid}/level", json={"level": "unknown"}).status_code == 422
    assert (
        client.post(f"/api/projects/{pid}/collect", json={"url": "http://127.0.0.1"}).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/projects/{pid}/collect", json={"provider": "commons", "query": ""}
        ).status_code
        == 422
    )
    collected = client.post(
        f"/api/projects/{pid}/collect", json={"provider": "commons", "query": "watercolor"}
    )
    assert collected.status_code == 202
    assert client.get(f"/api/projects/{pid}/collections").json()[0]["id"] == collected.json()["id"]
    assert (
        client.post(
            f"/api/projects/{pid}/collect", json={"provider": "commons", "query": "flower"}
        ).status_code
        == 409
    )
    assert client.get("/api/images/missing/source").json() == {}
    assert client.get(f"/api/projects/{pid}/chat/training").json() == []
    assert (
        client.post(f"/api/projects/{pid}/chat/training", json={"instruction": " "}).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/projects/{pid}/chat/training", json={"instruction": "x" * 2001}
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/projects/{pid}/chat/training", json={"instruction": "watercolor texture"}
        ).status_code
        == 201
    )
    assert (
        client.get(f"/api/projects/{pid}/chat/training").json()[0]["content"]
        == "watercolor texture"
    )
    assert client.get(f"/api/projects/{pid}/chat/generation").json() == []
    assert client.get(f"/api/projects/{pid}/chat/invalid").status_code == 422
    invalid = client.post(f"/api/projects/{pid}/train", json={"mode": "invalid"})
    assert invalid.status_code == 422 and isinstance(invalid.json()["detail"], list)


def test_group_filtered_training_and_group_edits(studio, monkeypatch):
    client, _root = studio
    pid = project(client)
    for group in ("subject", "style"):
        response = client.post(
            f"/api/projects/{pid}/images",
            data={"group_name": group},
            files=[("files", (f"{i}.png", png(), "image/png")) for i in range(3)],
        )
        assert response.status_code == 201
    gpu_readiness_only(monkeypatch)
    assert client.post(f"/api/projects/{pid}/train", json={"group": "unknown"}).status_code == 422
    response = client.post(f"/api/projects/{pid}/train", json={"group": "style", "mode": "style"})
    assert response.status_code == 202
    assert {row["group_name"] for row in json.loads(response.json()["config"])["images"]} == {
        "style"
    }
    assert len(json.loads(response.json()["config"])["images"]) == 3
    root = _root
    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.executemany(
            "INSERT INTO training_metrics VALUES(?,?,?,?,?,?)",
            [
                (response.json()["id"], 1, 1.2, 0.0001, 0.8, 3.5),
                (response.json()["id"], 10, 0.7, 0.0001, 0.5, 3.6),
            ],
        )
    metrics = client.get(f"/api/projects/{pid}/metrics").json()
    assert [point["step"] for point in metrics["points"]] == [1, 10]
    assert metrics["points"][1]["loss"] == 0.7
    assert metrics["parameters"]["mode"] == "style"
    iid = client.get(f"/api/projects/{pid}/images").json()[0]["id"]
    assert (
        client.patch(
            f"/api/images/{iid}", json={"caption": "a photo of mystyle", "group_name": "moved"}
        ).status_code
        == 200
    )
    row = client.get(f"/api/projects/{pid}/images").json()[0]
    assert row["group_name"] == "moved"
    # Moving a group does not turn an unchanged default caption into a confirmed label.
    assert row["labeled"] == 0


def test_pseudo_labels_style_labels_and_reward_weighting(tmp_path):
    project = {"id": "model", "trigger": "myconcept"}
    rows = [
        {"id": "manual", "caption": "manual caption", "labeled": 1},
        {"id": "unlabeled", "caption": "default", "labeled": 0},
    ]
    config = {"mode": "semi_supervised", "images": rows}
    # Inject only the teacher text for this unit test; BLIP inference needs a real model download.
    calls = []

    def captioner(path):
        calls.append(path.name)
        return "a red flower"

    prepare_training_rows(tmp_path, project, config, captioner=captioner)
    assert calls == ["unlabeled.png"]
    assert rows[0]["caption"] == "manual caption" and rows[0]["weight"] == 1
    assert rows[1]["caption"] == "a red flower, myconcept" and rows[1]["weight"] == 0.5
    for mode in ("supervised", "style", "imitation", "generation_test", "reinforcement"):
        data = {
            "mode": mode,
            "images": [{"id": "sample", "caption": "default", "labeled": 0, "rating": 5}],
        }
        prepare_training_rows(tmp_path, project, data)
        row = data["images"][0]
        assert row["caption"] == (
            "an artwork in the style of myconcept" if mode == "style" else "default"
        )
        assert row["weight"] == (reward_weight(5) if mode == "reinforcement" else 1)
    assert (
        reward_weight(1) < reward_weight(2) < reward_weight(3) < reward_weight(4) < reward_weight(5)
    )
    assert reward_weight(None) == 1
    data = {
        "mode": "imitation",
        "instruction": "watercolor texture",
        "images": [{"id": "sample", "caption": "manual caption", "labeled": 1}],
    }
    prepare_training_rows(tmp_path, project, data)
    assert data["images"][0]["caption"] == "manual caption, watercolor texture"
    # Public-site crawling exercises HTML discovery, keyword filtering, corrupt media,
    # bounded imports and provenance without depending on a third-party website.
    responses = {
        "https://example.org/gallery": (
            b'<img src="/flower.png" alt="flower"><img src="/corrupt.png" alt="flower"><a href="/next">next</a>',
            "text/html",
            "https://example.org/gallery",
        ),
        "https://example.org/next": (
            b'<img src="/other.png" alt="car">',
            "text/html",
            "https://example.org/next",
        ),
        "https://example.org/flower.png": (png(), "image/png", "https://example.org/flower.png"),
        "https://example.org/corrupt.png": (b"bad", "image/png", "https://example.org/corrupt.png"),
    }

    def fetch(url, limit):
        assert limit <= 20 * 1024 * 1024
        return responses[url]

    rows, errors = collect(
        {"url": "https://example.org/gallery", "query": "flower", "max_files": 10}, fetch
    )
    assert len(rows) == 1 and len(errors) == 1
    assert rows[0]["page"] == "https://example.org/gallery" and rows[0]["image"].size == (128, 128)
    assert extract_media(png(), "image/png", "video") == []
    parser = MediaPage()
    parser.feed(
        '<video src="/video.mp4"></video><source src="/second.webm"><a href="/clip.mp4">clip</a>'
    )
    assert len(parser.media) == 3
    commons_payload = json.dumps(
        {
            "query": {
                "pages": {
                    "1": {
                        "title": "File:flower",
                        "imageinfo": [
                            {
                                "url": "https://example.org/flower.png",
                                "descriptionurl": "https://example.org/license",
                                "extmetadata": {"LicenseShortName": {"value": "CC0"}},
                            }
                        ],
                    }
                }
            }
        }
    ).encode()

    def commons_fetch(url, limit):
        if url.startswith("https://commons.wikimedia.org/w/api.php"):
            assert "gsrsearch=" in url
            return commons_payload, "application/json", url
        return fetch(url, limit)

    rows, errors = collect(
        {"provider": "commons", "query": "flower", "max_files": 1}, commons_fetch
    )
    assert len(rows) == 1 and not errors and rows[0]["license"] == "CC0"
    for url in (
        "file:///etc/passwd",
        "http://user:password@example.org",
        "http://127.0.0.1",
        "http://[::1]",
        "http://10.1.2.3",
        "http://169.254.169.254",
        "https://example.org:8443",
    ):
        with pytest.raises(ValueError):
            public_address(url)
    # A DNS name resolving to an internal address is also blocked.
    original_lookup = socket.getaddrinfo
    try:
        socket.getaddrinfo = lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ]
        with pytest.raises(ValueError):
            public_address("https://rebound.example")
    finally:
        socket.getaddrinfo = original_lookup


def seed_output(root, pid, job_id="output", kind="image", config=None):
    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,kind,status,config) VALUES(?,?,?,'completed',?)",
            (job_id, pid, kind, json.dumps(config or {"prompt": "a photo of mystyle"})),
        )
    folder = root / "jobs" / job_id
    folder.mkdir(parents=True)
    (folder / ("test.png" if kind == "train" else "output.png")).write_bytes(png())


def test_feedback_rating_upsert_and_test_image(studio):
    client, root = studio
    pid = project(client)
    seed_output(root, pid)
    assert client.post("/api/jobs/output/feedback", json={"rating": 0}).status_code == 422
    assert client.post("/api/jobs/output/feedback", json={"rating": 6}).status_code == 422
    assert client.post("/api/jobs/missing/feedback", json={"rating": 5}).status_code == 422
    first = client.post("/api/jobs/output/feedback", json={"rating": 5}).json()
    second = client.post("/api/jobs/output/feedback", json={"rating": 2}).json()
    assert first["image_id"] == second["image_id"]
    assert client.get(f"/api/projects/{pid}/images").json()[0]["rating"] == 2
    assert len(client.get(f"/api/projects/{pid}/images").json()) == 1
    seed_output(root, pid, "training-test", "train")
    assert client.get("/api/jobs/training-test/test-image").status_code == 200
    assert client.post("/api/jobs/training-test/feedback", json={"rating": 4}).status_code == 200
    jobs = client.get(f"/api/projects/{pid}/jobs").json()
    assert jobs[0]["has_test_image"] is True and jobs[0]["feedback_rating"] == 4
    assert client.get("/api/jobs/missing/test-image").status_code == 404


def test_reward_training_resumes_matching_base_and_requires_feedback(studio, monkeypatch):
    client, root = studio
    pid = ready_dataset(client)
    assert (
        client.post(f"/api/projects/{pid}/train", json={"mode": "reinforcement"}).status_code == 422
    )
    seed_output(root, pid, "previous-training", "train", {"base_model": BASE_MODEL})
    seed_output(root, pid, "reward-image")
    assert client.post("/api/jobs/reward-image/feedback", json={"rating": 5}).status_code == 200
    gpu_readiness_only(monkeypatch)
    monkeypatch.setenv("STUDIO_CHECKPOINT", "/different/base.safetensors")
    response = client.post(f"/api/projects/{pid}/train", json={"mode": "reinforcement"})
    assert response.status_code == 202
    config = json.loads(response.json()["config"])
    assert config["base_model"] == BASE_MODEL
    assert config["start_job_id"] == "previous-training"
    assert any(row["rating"] == 5 for row in config["images"])
    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.execute("UPDATE jobs SET status='failed' WHERE status='queued'")
    client.patch(
        f"/api/projects/{pid}/settings",
        json={"rules_prompt": "watercolor", "negative_rules": "blur"},
    )
    image = client.post(
        f"/api/projects/{pid}/generate", json={"prompt": "mystyle", "negative_prompt": "noise"}
    )
    assert image.status_code == 202
    effective = json.loads(image.json()["config"])
    assert effective["prompt"] == "mystyle, watercolor"
    assert effective["negative_prompt"] == "noise, blur"
    assert effective["resolution"] == "720p"
    assert client.get(f"/api/projects/{pid}/chat/generation").json()[0]["content"] == "mystyle"
    assert client.get(f"/api/projects/{pid}/chat/training").json() == []
    assert (
        client.post(
            f"/api/projects/{pid}/generate",
            json={
                "kind": "video",
                "prompt": "test",
                "with_audio": True,
                "source_id": "reward-image",
            },
        ).status_code
        == 422
    )
    for level in ("normal", "power", "excellent", "ultra"):
        for kind in ("image", "video"):
            with sqlite3.connect(root / "studio.sqlite3") as db:
                db.execute("UPDATE jobs SET status='failed' WHERE status='queued'")
            response = client.post(
                f"/api/projects/{pid}/generate",
                json={
                    "kind": kind,
                    "prompt": "test",
                    "source_id": "reward-image",
                    "level": level,
                    "resolution": "4k",
                },
            )
            assert response.status_code == 202
            config = json.loads(response.json()["config"])
            assert config["resolution"] == "4k" and config["level"] == level
            assert (
                config["inference_steps"]
                == {"normal": 20, "power": 35, "excellent": 50, "ultra": 200}[level]
            )
    assert (
        client.post(
            f"/api/projects/{pid}/generate", json={"prompt": "test", "resolution": "8k"}
        ).status_code
        == 422
    )
    assert (
        client.post(f"/api/projects/{pid}/train", json={"instruction": "x" * 2001}).status_code
        == 422
    )
    client.patch(f"/api/projects/{pid}/settings", json={})
    with sqlite3.connect(root / "studio.sqlite3") as db:
        saved = json.loads(
            db.execute("SELECT config FROM jobs WHERE id=?", (image.json()["id"],)).fetchone()[0]
        )
    assert saved["prompt"] == "mystyle, watercolor"


def test_schema_migration_preserves_existing_dataset(tmp_path):
    with sqlite3.connect(tmp_path / "studio.sqlite3") as db:
        db.execute(
            "CREATE TABLE images(id TEXT PRIMARY KEY, project_id TEXT, caption TEXT, created_at TEXT)"
        )
        db.execute("INSERT INTO images VALUES('existing','p','original caption','old')")
    with TestClient(create_app(tmp_path, run_workers=False)):
        with sqlite3.connect(tmp_path / "studio.sqlite3") as db:
            row = db.execute("SELECT caption,labeled,rating,group_name FROM images").fetchone()
            assert row == ("original caption", 0, None, "default")


def test_real_video_encoder_preserves_frames_and_timing(tmp_path):
    frames = [np.full((64, 64, 3), index * 5, dtype=np.uint8) for index in range(49)]
    path = tmp_path / "video.mp4"
    export_to_video(frames, str(path), fps=8)
    with imageio.get_reader(path) as reader:
        assert reader.count_frames() == 49
        metadata = reader.get_meta_data()
        assert metadata["fps"] == 8
        assert abs(metadata["duration"] - 49 / 8) < 0.02
        assert reader.get_data(0).shape == (64, 64, 3)
        assert np.asarray(reader.get_data(48)).mean() > np.asarray(reader.get_data(0)).mean()

    # Production streaming encoder, not a mock: verify advertised dimensions.
    for resolution, size in RESOLUTIONS.items():
        image_path = tmp_path / f"{resolution}.png"
        save_image(Image.new("RGB", (64, 64), "blue"), image_path, resolution)
        with Image.open(image_path) as image:
            assert image.size == (size or (64, 64))
    for resolution in ("720p", "4k", "custom"):
        video_path = tmp_path / f"{resolution}.mp4"
        save_video(frames[:2], video_path, resolution)
        with imageio.get_reader(video_path) as reader:
            assert reader.count_frames() == 2
            assert reader.get_meta_data()["size"] == RESOLUTIONS[resolution]
    video_frames = extract_media(path.read_bytes(), "video/mp4", "video")
    assert len(video_frames) == 3 and video_frames[0].size == (64, 64)
    from tests.check_agent_video import check_motion_optimizer

    check_motion_optimizer(tmp_path)
    # Real tiny CogVideoX attention adapter optimization, preserving base weights.
    import torch
    from diffusers import (
        CogVideoXDDIMScheduler,
        CogVideoXImageToVideoPipeline,
        CogVideoXTransformer3DModel,
    )
    from peft import get_peft_model_state_dict

    from studio.sound import generate_sound, mux_audio, read_audio, train_sound
    from studio.video_learning import optimize_video, read_clip

    transformer = CogVideoXTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=2,
        time_embed_dim=16,
        text_embed_dim=8,
        num_layers=1,
        sample_width=4,
        sample_height=4,
        sample_frames=5,
        patch_size=2,
        temporal_compression_ratio=4,
        max_text_seq_length=4,
        use_rotary_positional_embeddings=False,
    )
    frozen = {
        name: parameter.detach().clone() for name, parameter in transformer.named_parameters()
    }
    cached = [(torch.randn(1, 2, 2, 4, 4), torch.randn(1, 2, 2, 4, 4), torch.randn(1, 4, 8))]
    config = {"rank": 4, "steps": 3, "learning_rate": 0.0001}
    points = []
    optimize_video(
        transformer,
        CogVideoXDDIMScheduler(num_train_timesteps=20),
        cached,
        config,
        lambda _: None,
        points.append,
        device="cpu",
    )
    assert len(points) == 2 and all(np.isfinite(point["loss"]) for point in points)
    adapter = get_peft_model_state_dict(transformer)
    assert any(value.abs().sum() > 0 for name, value in adapter.items() if "lora_B" in name)
    for name, value in frozen.items():
        actual_name = (
            name.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
            if any(key in name for key in ("to_q", "to_k", "to_v", "to_out.0"))
            else name
        )
        assert torch.equal(dict(transformer.named_parameters())[actual_name], value)
    CogVideoXImageToVideoPipeline.save_lora_weights(
        str(tmp_path / "video-adapter"), transformer_lora_layers=adapter
    )
    assert (tmp_path / "video-adapter/pytorch_lora_weights.safetensors").is_file()
    clip = read_clip(path, width=64, height=64, count=9)
    assert clip.shape == (9, 64, 64, 3)
    # Real audio optimizer, reload, deterministic sampling, and AAC/video mux.
    audio_folder = tmp_path / "audio-model"
    signal = np.sin(np.linspace(0, 100, 16000)).astype(np.float32) * 0.2
    audio_points = []
    train_sound(
        [(signal, "sound of mystyle")],
        {"steps": 4, "learning_rate": 0.0001},
        audio_folder,
        lambda _: None,
        audio_points.append,
        device="cpu",
        window=512,
    )
    assert all(np.isfinite(point["loss"]) for point in audio_points)
    sound = tmp_path / "sound.wav"
    generate_sound(
        audio_folder / "sound.pt", "sound of mystyle", 42, sound, device="cpu", samples=16000
    )
    repeat = tmp_path / "repeat.wav"
    generate_sound(
        audio_folder / "sound.pt", "sound of mystyle", 42, repeat, device="cpu", samples=16000
    )
    assert sound.read_bytes() == repeat.read_bytes()
    assert len(read_audio(sound)) == 16000
    mux_audio(path, sound)
    assert path.stat().st_size > 1000
    import subprocess

    import imageio_ffmpeg

    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True
    )
    assert "Audio: aac" in probe.stderr and "Video: h264" in probe.stderr

    # Upload/decode the actual WAV and MP4 via the application's HTTP API.
    from unittest.mock import patch

    with TestClient(create_app(tmp_path / "media-app", run_workers=False)) as client:
        pid = project(client)
        for kind, content, name in (
            ("audio", sound.read_bytes(), "sound.wav"),
            ("video", path.read_bytes(), "clip.mp4"),
        ):
            invalid = client.post(
                f"/api/projects/{pid}/media",
                data={"kind": kind},
                files={"files": (name, b"corrupt", "application/octet-stream")},
            )
            assert invalid.status_code == 422
            added = client.post(
                f"/api/projects/{pid}/media",
                data={"kind": kind},
                files=[("files", (name, content, "application/octet-stream")) for _ in range(3)],
            )
            assert added.status_code == 201
            rows = [row for row in added.json() if row["kind"] == kind]
            assert len(rows) == 3
            assert client.get(f"/api/media/{rows[0]['id']}/file").status_code == 200
            assert (
                client.patch(
                    f"/api/media/{rows[0]['id']}", json={"caption": "waterfall sound"}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    f"/api/projects/{pid}/train", json={"target": kind, "mode": "semi_supervised"}
                ).status_code
                == 422
            )
            with (
                patch("studio.app.importlib.util.find_spec", return_value=True),
                patch("torch.cuda.is_available", return_value=True),
            ):
                queued = client.post(
                    f"/api/projects/{pid}/train",
                    json={"target": kind, "instruction": "waterfall", "steps": 10},
                )
            assert queued.status_code == 202 and queued.json()["kind"] == kind + "_train"
            assert len(json.loads(queued.json()["config"])["media"]) == 3
            with sqlite3.connect(tmp_path / "media-app/studio.sqlite3") as db:
                db.execute("UPDATE jobs SET status='failed' WHERE status='queued'")
        assert len(client.get(f"/api/projects/{pid}/media").json()) == 6
        extracted = client.post(
            f"/api/projects/{pid}/media",
            data={"kind": "video", "extract_audio": "true", "group_name": "voice-reference"},
            files={"files": ("voice-video.mp4", path.read_bytes(), "video/mp4")},
        )
        assert extracted.status_code == 201
        assert len(extracted.json()) == 8
        assert len([r for r in extracted.json() if r["kind"] == "audio"]) == 4
        assert len([r for r in extracted.json() if r["kind"] == "video"]) == 4
        with patch("studio.app.importlib.util.find_spec", return_value=True):
            voice = client.post(
                f"/api/projects/{pid}/train",
                json={
                    "target": "voice",
                    "mode": "imitation",
                    "instruction": "ゆっくり、高めの声",
                    "steps": 10,
                },
            )
        assert voice.status_code == 202 and voice.json()["kind"] == "voice_train"
        voice_config = json.loads(voice.json()["config"])
        assert (
            len(voice_config["media"]) == 4 and voice_config["instruction"] == "ゆっくり、高めの声"
        )
        assert (
            client.post(f"/api/projects/{pid}/training/reset", json={}).json()["reset_count"] == 3
        )
        with patch("studio.app.importlib.util.find_spec", return_value=True):
            speech = client.post(
                f"/api/projects/{pid}/generate",
                json={
                    "kind": "audio",
                    "audio_model": "kokoro",
                    "prompt": "こんにちは。",
                    "level": "ultra",
                },
            )
        assert speech.status_code == 202
        assert json.loads(speech.json()["config"])["inference_steps"] == 200
        assert (
            client.post(
                f"/api/projects/{pid}/generate", json={"kind": "audio", "prompt": "hello"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/projects/{pid}/generate",
                json={
                    "kind": "audio",
                    "audio_model": "kokoro",
                    "use_voice": True,
                    "prompt": "hello",
                },
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/projects/{pid}/generate",
                json={
                    "kind": "audio",
                    "audio_model": "audioldm2",
                    "use_voice": True,
                    "prompt": "hello",
                },
            ).status_code
            == 422
        )
        assert client.get("/api/media/missing/file").status_code == 404
        assert client.patch("/api/media/missing", json={"caption": "x"}).status_code == 404

    from pathlib import Path
    from types import SimpleNamespace

    from diffusers import AudioLDM2Pipeline

    from studio.audio_models import AUDIO_MODELS, generate_audio
    from studio.voice import adjust_pitch, instruction_controls, learn_voice

    audio_calls = []

    class ControlledAudioPipeline:
        def enable_model_cpu_offload(self):
            pass

        def __call__(self, **kwargs):
            audio_calls.append(kwargs)
            return SimpleNamespace(audios=[signal])

    with patch.object(
        AudioLDM2Pipeline, "from_pretrained", return_value=ControlledAudioPipeline()
    ) as loader:
        output = tmp_path / "public-sound.wav"
        generate_audio(
            {
                "audio_model": "audioldm2",
                "prompt": "waterfall",
                "audio_prompt": "",
                "negative_prompt": "noise",
                "inference_steps": 200,
                "seed": 42,
            },
            output,
        )
        assert loader.call_args.kwargs["revision"] == AUDIO_MODELS["audioldm2"]["revision"]
        assert len(read_audio(output)) == 16000
        assert (
            audio_calls[0]["num_inference_steps"] == 200 and audio_calls[0]["prompt"] == "waterfall"
        )
    controls = instruction_controls("ゆっくり、高めの声")
    assert controls == {"speech_speed": 0.8, "pitch_semitones": 2}
    assert instruction_controls("fast lower pitch") == {"speech_speed": 1.2, "pitch_semitones": -2}
    assert instruction_controls("自由な説明") == {"speech_speed": 1.0, "pitch_semitones": 0}
    adjusted = tmp_path / "adjusted.wav"
    adjusted.write_bytes(sound.read_bytes())
    adjust_pitch(adjusted, 2)
    assert 15000 <= len(read_audio(adjusted)) <= 16500
    voice_folder = tmp_path / "voice-model"
    voice_folder.mkdir()

    class FrozenConverter:
        def extract_se(self, paths, destination):
            assert len(paths) == 3 and all(Path(p).is_file() for p in paths)
            torch.save(torch.tensor([0.2, 0.3]), destination)

    with patch("studio.voice.converter", return_value=FrozenConverter()):
        learn_voice([sound] * 3, voice_folder, lambda value: None, "ゆっくり、高めの声")
    card = json.loads((voice_folder / "voice-card.json").read_text())
    assert card["controls"] == controls and card["files"] == 3
    assert (voice_folder / "adapter.zip").is_file()

    # Extract actual WAV and AAC tracks from a bounded public-site collector.
    def site_audio_fetch(url, max_bytes):
        if url.endswith("site"):
            return (
                b'<audio src="/sound.wav"></audio><video src="/voice.mp4"></video>',
                "text/html",
                url,
            )
        return (
            (sound.read_bytes(), "audio/wav", url)
            if url.endswith("wav")
            else (path.read_bytes(), "video/mp4", url)
        )

    audio_rows, errors = collect(
        {"provider": "site", "url": "https://example.org/site", "media": "audio", "max_files": 10},
        site_audio_fetch,
    )
    assert not errors and len(audio_rows) == 2
    assert all(r["image"] is None and len(r["audio_data"]) >= 16000 for r in audio_rows)
    silent = tmp_path / "silent.mp4"
    export_to_video(frames, str(silent), fps=8)
    silent_rows, warnings = collect(
        {
            "provider": "site",
            "url": "https://example.org/silent.mp4",
            "media": "all",
            "max_files": 10,
        },
        lambda url, max_bytes: (silent.read_bytes(), "video/mp4", url),
    )
    assert len(silent_rows) == 3 and warnings
    assert all(row["image"] is not None for row in silent_rows)

    # Background collector persists real video clips, frames and source URLs.
    import time

    with patch(
        "studio.app.collect",
        return_value=(
            [
                {
                    "image": Image.new("RGB", (128, 128), "blue"),
                    "url": "https://example.org/clip.mp4",
                    "page": "https://example.org/gallery",
                    "license": "CC0",
                    "video_data": path.read_bytes(),
                },
                {
                    "image": None,
                    "audio_data": read_audio(sound),
                    "url": "https://example.org/sound.wav",
                    "page": "https://example.org/gallery",
                    "license": "CC0",
                },
            ],
            [],
        ),
    ):
        with TestClient(create_app(tmp_path / "collection-app", run_workers=True)) as client:
            pid = project(client)
            queued = client.post(
                f"/api/projects/{pid}/collect",
                json={"provider": "commons", "query": "flower", "media": "video"},
            )
            assert queued.status_code == 202
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                run = client.get(f"/api/projects/{pid}/collections").json()[0]
                if run["status"] != "running":
                    break
                time.sleep(0.02)
            assert run["status"] == "completed" and run["imported"] == 2
            images = client.get(f"/api/projects/{pid}/images").json()
            assert images[0]["source_page"] == "https://example.org/gallery"
            assert client.get(f"/api/images/{images[0]['id']}/source").json()["license"] == "CC0"
            collected_media = client.get(f"/api/projects/{pid}/media").json()
            assert collected_media[0]["kind"] == "video"
            assert collected_media[1]["kind"] == "audio"
            assert collected_media[1]["source_page"] == "https://example.org/gallery"
