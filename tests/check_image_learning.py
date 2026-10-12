"""CPU training selection, immutable lineage, and failure recovery contracts."""

import json
import sqlite3

from studio.training_models import IMAGE_TRAINING_MODELS
from tests.test_studio import project
from tests.test_studio_matrix import ready_dataset


def check_image_learning(client, root, monkeypatch):
    from studio import app as module
    from studio.image_learning import error_message

    with monkeypatch.context() as patch:
        patch.setattr(
            module, "inspect_system", lambda: {"cpu_ready": True, "cuda_ready": False, "ram_gb": 16}
        )
        import torch

        patch.setattr(torch.cuda, "is_available", lambda: False)
        pid = ready_dataset(client)
        other = project(client)
        assert (
            client.patch(
                f"/api/projects/{pid}/training-model", json={"training_model": "small"}
            ).status_code
            == 200
        )
        assert (
            next(p for p in client.get("/api/projects").json() if p["id"] == pid)["training_model"]
            == "small"
        )
        assert (
            client.patch(
                f"/api/projects/{pid}/training-model", json={"training_model": "invalid"}
            ).status_code
            == 422
        )
        for key in ("tiny", "illustration", "small"):
            reply = client.post(
                f"/api/projects/{pid}/train", json={"training_model": key, "steps": 10, "rank": 4}
            )
            assert reply.status_code == 202, reply.text
            row = reply.json()
            config = json.loads(row["config"])
            assert config["base_model"] == IMAGE_TRAINING_MODELS[key]["id"]
            assert config["base_revision"] == IMAGE_TRAINING_MODELS[key]["revision"]
            with sqlite3.connect(root / "studio.sqlite3") as db:
                db.execute("UPDATE jobs SET status='completed' WHERE id=?", (row["id"],))
            saved = row["id"]
        # Picking an older version must survive a newer training result.
        tiny = next(
            j
            for j in client.get(f"/api/projects/{pid}/jobs").json()
            if j["training_model"] == "tiny"
        )
        for invalid in (
            {"training_model": "tiny", "start_job_id": saved},
            {"training_model": "tiny", "start_job_id": "missing"},
            {"training_model": "bogus"},
            {"target": "video", "training_model": "tiny"},
        ):
            assert client.post(f"/api/projects/{pid}/train", json=invalid).status_code == 422
        assert (
            client.post(f"/api/projects/{other}/train", json={"start_job_id": saved}).status_code
            == 422
        )
        for body in (
            {"trained_job_id": saved, "public_model": "tiny"},
            {"trained_job_id": "missing"},
            {"trained_job_id": saved, "kind": "audio", "audio_model": "kokoro"},
        ):
            assert (
                client.post(
                    f"/api/projects/{pid}/generate", json={"prompt": "circleart", **body}
                ).status_code
                == 422
            )
        assert (
            client.post(
                f"/api/projects/{other}/generate",
                json={"prompt": "circleart", "trained_job_id": saved},
            ).status_code
            == 422
        )
        generated = client.post(
            f"/api/projects/{pid}/generate",
            json={"prompt": "circleart", "trained_job_id": tiny["id"]},
        )
        assert generated.status_code == 202, generated.text
        config = json.loads(generated.json()["config"])
        assert config["training_model"] == "tiny" and config["trained_job_id"] == tiny["id"]
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='completed' WHERE id=?", (generated.json()["id"],))
        continued = client.post(
            f"/api/projects/{pid}/train",
            json={
                "training_model": "small",
                "start_job_id": saved,
                "rank": 32,
                "mode": "imitation",
            },
        )
        assert continued.status_code == 202, continued.text
        config = json.loads(continued.json()["config"])
        assert config["start_job_id"] == saved and config["rank"] == 4
        assert config["base_revision"] == IMAGE_TRAINING_MODELS["small"]["revision"]
        reset = client.post(f"/api/projects/{pid}/training/reset")
        assert reset.json()["reset_count"] == 1
        assert (
            len(
                [
                    j
                    for j in client.get(f"/api/projects/{pid}/jobs").json()
                    if j["kind"] == "train" and j["status"] == "completed"
                ]
            )
            == 3
        )
        for key in ("sd15", "dreamshaper"):
            reply = client.post(f"/api/projects/{pid}/train", json={"training_model": key})
            assert reply.status_code == 503 and "CUDA" in reply.json()["detail"]
        patch.setattr(module, "inspect_system", lambda: {"cpu_ready": False, "ram_gb": 16})
        reply = client.post(f"/api/projects/{pid}/train", json={"training_model": "tiny"})
        assert reply.status_code == 503 and "セットアップ" in reply.json()["detail"]
        patch.setattr(module, "inspect_system", lambda: {"cpu_ready": True, "ram_gb": 8})
        reply = client.post(f"/api/projects/{pid}/train", json={"training_model": "small"})
        assert reply.status_code == 422 and "12GB" in reply.json()["detail"]
    # An adapter saved by v0.1.5 retains its unpinned legacy base when continued.
    from studio.app import BASE_MODEL
    from tests.test_studio_matrix import gpu_readiness_only
    from tests.test_training_modes import seed_output

    with monkeypatch.context() as patch:
        gpu_readiness_only(patch)
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE images SET rating=5 WHERE project_id=?", (pid,))
        for key, model in IMAGE_TRAINING_MODELS.items():
            seed_id = "shared-" + key
            seed_output(
                root,
                pid,
                seed_id,
                "train",
                {
                    "training_model": key,
                    "base_model": model["id"],
                    "base_revision": model["revision"],
                    "rank": 4,
                },
            )
            for mode in (
                "supervised",
                "semi_supervised",
                "style",
                "reinforcement",
                "generation_test",
                "imitation",
            ):
                response = client.post(
                    f"/api/projects/{pid}/train",
                    json={
                        "training_model": key,
                        "start_job_id": seed_id,
                        "mode": mode,
                        "instruction": "soft brush strokes",
                        "steps": 10,
                    },
                )
                assert response.status_code == 202, response.text
                snapshot = json.loads(response.json()["config"])
                assert snapshot["training_model"] == key and snapshot["start_job_id"] == seed_id
                assert snapshot["mode"] == mode and snapshot["instruction"] == "soft brush strokes"
                assert snapshot["base_revision"] == model["revision"] and snapshot["rank"] == 4
                client.post(f"/api/projects/{pid}/training/reset")
    seed_output(root, pid, "legacy-adapter", "train", {"base_model": BASE_MODEL, "rank": 8})
    with monkeypatch.context() as patch:
        gpu_readiness_only(patch)
        response = client.post(
            f"/api/projects/{pid}/train",
            json={"training_model": "sd15", "start_job_id": "legacy-adapter"},
        )
        assert response.status_code == 202, response.text
        config = json.loads(response.json()["config"])
        assert config["training_model"] is None and config["base_model"] == BASE_MODEL
        assert "base_revision" not in config and config["start_job_id"] == "legacy-adapter"
        assert client.post(f"/api/projects/{pid}/training/reset").status_code == 200
    assert "メモリ" in error_message(MemoryError())
    assert "ダウンロード" in error_message(RuntimeError("HTTP timeout token=secret"))
    assert "secret" not in error_message(RuntimeError("HTTP timeout token=secret"))
    assert "学習率" in error_message(RuntimeError("Non-finite training loss"))


def check_comparison_worker(root, adapter, pipeline_factory, monkeypatch):
    """Real small diffusion math verifies matched-condition baseline versus learned output."""
    import shutil

    from fastapi.testclient import TestClient

    from studio import image_learning
    from studio.app import create_app

    with TestClient(create_app(root, run_workers=False)) as client:
        pid = project(client)
        config = {
            "training_model": "tiny",
            "base_model": IMAGE_TRAINING_MODELS["tiny"]["id"],
            "base_revision": IMAGE_TRAINING_MODELS["tiny"]["revision"],
        }
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute(
                "INSERT INTO jobs(id,project_id,kind,status,config) VALUES('learned',?,'train','completed',?)",
                (pid, json.dumps(config)),
            )
        shutil.copytree(adapter, root / "jobs/learned/adapter")
        response = client.post(
            f"/api/projects/{pid}/generate",
            json={
                "prompt": "myconcept",
                "trained_job_id": "learned",
                "compare_before": True,
                "resolution": "native",
                "seed": 42,
            },
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        with sqlite3.connect(root / "studio.sqlite3") as db:
            run_config = json.loads(response.json()["config"])
            run_config["inference_steps"] = 2
            db.execute(
                "UPDATE jobs SET status='running',config=? WHERE id=?",
                (json.dumps(run_config), job_id),
            )
        assert client.get(f"/api/jobs/{job_id}/before").status_code == 404
        with monkeypatch.context() as patch:
            patch.setattr(image_learning, "load_pipeline", lambda _config: pipeline_factory())
            patch.setitem(
                image_learning.IMAGE_TRAINING_MODELS,
                "tiny",
                {**IMAGE_TRAINING_MODELS["tiny"], "resolution": 64},
            )
            image_learning.execute(root, job_id)
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='completed' WHERE id=?", (job_id,))
        before = client.get(f"/api/jobs/{job_id}/before?download=true")
        after = client.get(f"/api/jobs/{job_id}/result")
        assert before.status_code == 200 and before.content != after.content
        assert "before.png" in before.headers["content-disposition"]
        evidence = json.loads((root / "jobs" / job_id / "execution.json").read_text())
        assert (
            evidence["comparison"]
            and evidence["seed"] == 42
            and evidence["adapter_id"] == "learned"
        )
        listed = client.get(f"/api/projects/{pid}/jobs").json()[0]
        assert listed["has_comparison"] and listed["comparison_seed"] == 42
        assert listed["comparison_prompt"] == "myconcept"
        assert client.get("/api/jobs/learned/before").status_code == 404
        assert client.get("/api/jobs/missing/before").status_code == 404
        for invalid in (
            {"public_model": "tiny"},
            {"public_model": "ltx", "kind": "video"},
            {"audio_model": "kokoro", "kind": "audio"},
        ):
            assert (
                client.post(
                    f"/api/projects/{pid}/generate",
                    json={"prompt": "myconcept", "compare_before": True, **invalid},
                ).status_code
                == 422
            )
