"""Real database/import/weighting contracts with controlled external vision and search."""

import json
import random
import sqlite3

import pytest
from PIL import Image

from studio.gpu import prepare_training_rows
from studio.related_learning import augment, fingerprint, search_words, subjects
from tests.test_studio_matrix import ready_dataset


def check_related_learning(client, root, monkeypatch):
    from studio import app as module

    monkeypatch.setattr(module, "inspect_system", lambda: {"cpu_ready": True, "ram_gb": 16})
    pid = ready_dataset(client)
    project = next(p for p in client.get("/api/projects").json() if p["id"] == pid)
    folder = root / "related-check"
    folder.mkdir()

    def sample(seed, relevant=True):
        image = Image.new("RGB", (80, 80))
        randomizer = random.Random(seed)
        image.putdata([tuple(randomizer.randrange(256) for _ in range(3)) for _ in range(6400)])
        image.putpixel((0, 0), (10 if relevant else 200, 0, 0))
        return image

    good = sample(4)
    unrelated = sample(8, False)

    class Vision:
        def style(self, vector):
            return ""

        def caption(self, image):
            return "a drawing of an anime girl in a blue dress"

        def embed(self, image):
            return [0, 1] if image.getpixel((0, 0))[0] == 200 else [1, 0]

    def collect(options):
        assert options["query"] == "anime girl"
        assert options["provider"] == "commons" and options["max_files"] == 10
        assert options["thumbnail_width"] == 768
        return [
            {
                "url": f"https://example.org/{i}.png",
                "page": f"https://example.org/page/{i}",
                "license": license,
                "image": image,
            }
            for i, (image, license) in enumerate(
                [(good, "CC0"), (good.copy(), "CC0"), (unrelated, "CC0"), (sample(9), "")]
            )
        ], []

    def start(**kwargs):
        reply = client.post(
            f"/api/projects/{pid}/train",
            json={
                "training_model": "tiny",
                "steps": 10,
                "rank": 4,
                "related_images": True,
                **kwargs,
            },
        )
        assert reply.status_code == 202, reply.text
        job = reply.json()
        config = json.loads(job["config"])
        config["job_id"] = job["id"]
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (job["id"],))
        return config

    def complete(config):
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='completed' WHERE id=?", (config["job_id"],))

    assert search_words("a photograph of a cat sitting on the fence") == "cat fence"
    assert subjects("a cat sitting in the mud") & subjects("two cats standing")
    assert not (subjects("a cat in mud") & subjects("a close up of a crab on rocks"))
    assert fingerprint(good)[0] == fingerprint(good.copy())[0]
    for body in (
        {"related_limit": 6},
        {"related_limit": 0},
        {"related_threshold": 0.1},
        {"related_images": True, "target": "video", "video_model": "animatediff"},
        {"related_images": True},
    ):
        assert client.post(f"/api/projects/{pid}/train", json=body).status_code == 422
    config = start()
    augment(root, project, config, folder, Vision, collect)
    assert config["related_report"]["added"] == 1
    assert config["related_report"]["rejected"] == 3
    assert config["related_report"]["queries"] == ["anime girl"]
    assert len(config["images"]) == 4
    prepare_training_rows(root, project, config)
    assert [r["weight"] for r in config["images"]] == [1, 1, 1, 0.5]
    added = config["images"][-1]
    complete(config)
    assert client.get(f"/api/images/{added['id']}/file").status_code == 200
    assert client.get(f"/api/images/{added['id']}/source").json()["license"] == "CC0"
    assert (
        client.patch(f"/api/images/{added['id']}/related", json={"excluded": True}).json()[
            "applies_to"
        ]
        == "next_training"
    )
    assert client.patch("/api/images/missing/related", json={"excluded": True}).status_code == 404
    # Excluded images remain deduplicated, and do not enter future training snapshots.
    config = start(start_job_id=config["job_id"])
    assert len(config["images"]) == 3
    augment(root, project, config, folder, Vision, collect)
    assert config["related_report"]["added"] == 0
    assert config["related_report"]["stage"] == "no_matches"
    assert len(config["images"]) == 3
    complete(config)
    client.patch(f"/api/images/{added['id']}/related", json={"excluded": False})

    # Repeated discoveries must not dominate training when automatic search is disabled.
    with sqlite3.connect(root / "studio.sqlite3") as db:
        for index in range(6):
            iid = f"related-balance-{index}"
            db.execute(
                "INSERT INTO images(id,project_id,caption,group_name,labeled) VALUES(?,?,?,'related',1)",
                (iid, pid, "anime girl"),
            )
            db.execute(
                "INSERT INTO related_images(image_id,job_id,similarity) VALUES(?,?,?)",
                (iid, config["job_id"], 0.9),
            )
    balanced = start(related_images=False)
    assert len(balanced["images"]) == 6
    assert [row["id"] for row in balanced["images"][-3:]] == [
        f"related-balance-{i}" for i in (3, 4, 5)
    ]
    complete(balanced)
    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.execute("DELETE FROM images WHERE id LIKE 'related-balance-%'")
        db.execute("DELETE FROM related_images WHERE image_id LIKE 'related-balance-%'")

    # Search failure is explicitly reported; originals still reach real training.
    def offline(_options):
        raise OSError("offline")

    config = start()
    augment(root, project, config, folder, Vision, offline)
    assert config["related_report"]["failure_type"] == "OSError"
    assert config["related_report"]["warnings"]
    assert len(config["images"]) == 4
    complete(config)
    # Cancellation after discovery must never append rows/files or restart training.
    config = start()

    def cancelled(_options):
        client.post(f"/api/projects/{pid}/training/reset")
        return [
            {
                "url": "https://example.org/new.png",
                "page": "https://example.org/new",
                "license": "CC0",
                "image": sample(12),
            }
        ], []

    before = list((root / "projects" / pid / "images").iterdir())
    with pytest.raises(InterruptedError):
        augment(root, project, config, folder, Vision, cancelled)
    assert list((root / "projects" / pid / "images").iterdir()) == before
    assert len(client.get(f"/api/projects/{pid}/images").json()) == 4
    # Continue to enforce RAM requirements for the real pretrained illustration base.
    monkeypatch.setattr(module, "inspect_system", lambda: {"cpu_ready": True, "ram_gb": 8})
    reply = client.post(f"/api/projects/{pid}/train", json={"training_model": "illustration"})
    assert reply.status_code == 422 and "16GB" in reply.json()["detail"]


def check_related_video(client, root, monkeypatch):
    import imageio.v2 as imageio
    import numpy as np

    from studio import app as module
    from studio.related_learning import frames_at
    from tests.test_studio import project as create_project

    monkeypatch.setattr(module, "inspect_system", lambda: {"cpu_ready": True, "ram_gb": 16})
    pid = create_project(client)
    project = next(p for p in client.get("/api/projects").json() if p["id"] == pid)
    folder = root / "related-video-check"
    folder.mkdir()

    def clip(seed):
        path = folder / f"source-{seed}.mp4"
        rng = np.random.default_rng(seed)
        frame = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
        imageio.mimsave(path, [np.roll(frame, i, axis=1) for i in range(8)], fps=8)
        return path.read_bytes()

    originals = [clip(i) for i in range(3)]
    reply = client.post(
        f"/api/projects/{pid}/media",
        data={"kind": "video"},
        files=[("files", (f"clip{i}.mp4", data, "video/mp4")) for i, data in enumerate(originals)],
    )
    assert reply.status_code == 201, reply.text
    candidate = clip(44)

    class Vision:
        def caption(self, image):
            return "an animated bird flying"

        def style(self, vector):
            return "anime"

        def embed(self, image):
            return [1, 0]

    def collect(options):
        assert options["media"] == "video" and options["thumbnail_width"] is None
        return [
            {
                "video_data": data,
                "page": f"https://example.org/clip/{i}",
                "url": f"https://example.org/clip/{i}.mp4",
                "license": "CC BY 4.0",
            }
            for i, data in enumerate([candidate, candidate, b"corrupt"])
        ], []

    reply = client.post(
        f"/api/projects/{pid}/train",
        json={
            "target": "video",
            "video_model": "animatediff",
            "related_images": True,
            "steps": 10,
            "mode": "style",
            "rank": 4,
        },
    )
    assert reply.status_code == 202, reply.text
    config = json.loads(reply.json()["config"])
    config["job_id"] = reply.json()["id"]
    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.execute("UPDATE jobs SET status='running' WHERE id=?", (config["job_id"],))
    augment(root, project, config, folder, Vision, collect)
    assert config["related_report"]["kind"] == "video"
    assert config["related_report"]["added"] == 1
    assert config["related_report"]["rejected"] == 2
    assert len(config["media"]) == 4
    added = config["media"][-1]
    assert len(frames_at(root / "projects" / pid / "media" / f"{added['id']}.mp4", True)) == 3
    item = next(
        r for r in client.get(f"/api/projects/{pid}/media").json() if r["id"] == added["id"]
    )
    assert item["source_page"] and item["related_job_id"] == config["job_id"]
    assert client.get(f"/api/media/{added['id']}/file").status_code == 200
    assert (
        client.patch(f"/api/media/{added['id']}/related", json={"excluded": True}).status_code
        == 200
    )
    assert client.patch("/api/media/missing/related", json={"excluded": True}).status_code == 404
    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.execute("UPDATE jobs SET status='completed' WHERE id=?", (config["job_id"],))
    reply = client.post(
        f"/api/projects/{pid}/train",
        json={
            "target": "video",
            "video_model": "animatediff",
            "start_job_id": config["job_id"],
            "related_images": True,
            "steps": 10,
        },
    )
    assert reply.status_code == 202
    assert len(json.loads(reply.json()["config"])["media"]) == 3
    client.post(f"/api/projects/{pid}/training/reset")
    # Commons uses its official small video renditions, with bounded rate-limit handling.
    from studio.collection import (
        ACCESS_LIMIT_MESSAGE,
        CollectionAccessLimit,
        discover,
    )
    from studio.collection import (
        collect as fetch_media,
    )

    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "File:Sample.webm",
                    "videoinfo": [
                        {
                            "url": "https://example.org/original.webm",
                            "descriptionurl": "https://example.org/source",
                            "extmetadata": {"LicenseShortName": {"value": "CC0"}},
                            "derivatives": [
                                {
                                    "src": "https://example.org/large.webm",
                                    "type": "video/webm",
                                    "width": 1920,
                                    "height": 1080,
                                },
                                {
                                    "src": "https://example.org/small.webm",
                                    "type": "video/webm",
                                    "width": 640,
                                    "height": 360,
                                },
                            ],
                        }
                    ],
                }
            }
        }
    }
    calls = []

    def fetch(url, limit):
        calls.append(url)
        if "api.php" in url:
            assert "prop=videoinfo" in url and "derivatives" in url
            return json.dumps(payload).encode(), "application/json", url
        raise CollectionAccessLimit(ACCESS_LIMIT_MESSAGE)

    options = {"provider": "commons", "media": "video", "query": "anime"}
    assert discover(options, fetch)[0]["url"] == "https://example.org/small.webm"
    items, warnings = fetch_media(options, fetch)
    assert not items and warnings == [ACCESS_LIMIT_MESSAGE] and len(calls) == 3
