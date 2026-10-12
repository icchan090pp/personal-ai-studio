"""Agent memory isolation and laptop-video contracts, within the 100-scenario suite."""

import json
import sqlite3

from tests.test_studio import project


def check_agent_contract(client, root, monkeypatch):
    from studio import assistant

    pid = project(client)
    other = project(client)
    endpoint = f"/api/projects/{pid}/agent"
    payload = {
        "name": "作品アシスタント",
        "instruction": "短く具体的に答える",
        "knowledge": "作品名は青い庭",
        "examples": [{"prompt": "作品名は？", "answer": "青い庭です。"}],
    }
    version = client.post(endpoint + "/versions", json=payload)
    assert version.status_code == 201, version.text
    version_id = version.json()["id"]
    exported = client.get(endpoint + f"/versions/{version_id}/export")
    assert exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
    assert exported.json()["format"] == "deel-agent-memory-v1"
    assert (
        client.get(f"/api/projects/{other}/agent/versions/{version_id}/export").status_code == 404
    )
    assert client.get(endpoint).json()["versions"][0]["config"]["knowledge"] == "作品名は青い庭"
    assert (
        client.post(
            f"/api/projects/{other}/agent/versions", json={**payload, "parent_id": version_id}
        ).status_code
        == 404
    )
    assert (
        client.post(
            endpoint + "/versions",
            json={**payload, "parent_id": version_id, "model": "gpt-oss:120b"},
        ).status_code
        == 422
    )
    assert client.post(endpoint + "/versions", json={**payload, "name": " "}).status_code == 422
    assert (
        client.post(
            endpoint + "/versions",
            json={**payload, "instruction": "a" * 2000, "knowledge": "b" * 4000},
        ).status_code
        == 422
    )
    calls = []

    def ask(model, prompt, level, **kwargs):
        calls.append({"model": model, "prompt": prompt, "level": level, **kwargs})
        return "青い庭です。" if kwargs.get("memory") else "作品名を教えてください。"

    with monkeypatch.context() as patch:
        patch.setattr(assistant, "ask", ask)
        run_payload = {
            "version_id": version_id,
            "prompt": "作品名は？",
            "compare_before": True,
            "level": "ultra",
        }
        response = client.post(endpoint + "/run", json=run_payload)
        assert response.status_code == 200, response.text
        run = response.json()
        assert run["answer"] != run["before_answer"] and run["version_id"] == version_id
        assert len(calls) == 2 and "memory" not in calls[0]
        assert "青い庭" in calls[1]["memory"] and calls[1]["examples"] == payload["examples"]
        assert calls[1]["level"] == "ultra" and calls[1]["model"] == "gpt-oss:20b"
        assert client.post(f"/api/projects/{other}/agent/run", json=run_payload).status_code == 404
        assert (
            client.post(endpoint + "/run", json={**run_payload, "prompt": " "}).status_code == 422
        )
        assert client.post(endpoint + f"/runs/{run['id']}/learn").status_code == 422
        assert client.patch(endpoint + f"/runs/{run['id']}", json={"rating": 2}).status_code == 200
        assert client.post(endpoint + f"/runs/{run['id']}/learn").status_code == 422
        assert (
            client.patch(
                endpoint + f"/runs/{run['id']}", json={"rating": 2, "correction": "作品名：青い庭"}
            ).status_code
            == 200
        )
        child = client.post(endpoint + f"/runs/{run['id']}/learn")
        assert child.status_code == 201, child.text
        assert child.json()["config"]["parent_id"] == version_id
        assert child.json()["config"]["examples"][-1]["answer"] == "作品名：青い庭"
        saved = client.get(endpoint).json()
        assert len(saved["versions"]) == 2 and len(saved["versions"][-1]["config"]["examples"]) == 1
        assert (
            client.patch(
                f"/api/projects/{other}/agent/runs/{run['id']}", json={"rating": 5}
            ).status_code
            == 404
        )
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute(
                "INSERT INTO jobs(id,project_id,kind,status,config) VALUES('agent-busy',?,'train','queued','{}')",
                (pid,),
            )
        assert client.post(endpoint + "/run", json=run_payload).status_code == 409
        client.post(f"/api/projects/{pid}/training/reset")

        def offline(*args, **kwargs):
            raise ConnectionRefusedError("Ollama offline")

        patch.setattr(assistant, "ask", offline)
        assert client.post(endpoint + "/run", json=run_payload).status_code == 503
        assert len(client.get(endpoint).json()["runs"]) == 1
        assert len(client.get(endpoint).json()["versions"]) == 2

        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        entered, release = Event(), Event()

        def slow(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return "done"

        patch.setattr(assistant, "ask", slow)
        with ThreadPoolExecutor() as pool:
            running = pool.submit(
                client.post, endpoint + "/run", json={**run_payload, "compare_before": False}
            )
            try:
                assert entered.wait(5)
                assert client.post(endpoint + "/run", json=run_payload).status_code == 409
                assert (
                    client.post(
                        f"/api/projects/{pid}/assistant", json={"prompt": "hello"}
                    ).status_code
                    == 409
                )
                assert (
                    client.post(
                        f"/api/projects/{pid}/generate",
                        json={"public_model": "tiny", "prompt": "blue bird"},
                    ).status_code
                    == 409
                )
            finally:
                release.set()
            assert running.result().status_code == 200


def check_video_contract(client, root, monkeypatch):
    from studio import app as app_module
    from studio.training_models import VIDEO_TRAINING_MODELS

    models = client.get("/api/video-models").json()
    assert {m["key"] for m in models} == {"animatediff", "cogvideo"}
    assert models[0]["backend"] == "cpu" and models[0]["ram_gb"] == 16
    assert len(models[0]["motion_revision"]) == 40
    pid = project(client)
    other = project(client)
    with sqlite3.connect(root / "studio.sqlite3") as db:
        for index in range(3):
            db.execute(
                "INSERT INTO media(id,project_id,kind,caption) VALUES(?,?,'video','a blue bird moving')",
                (f"motion-{index}", pid),
            )
    body = {
        "target": "video",
        "video_model": "animatediff",
        "steps": 10,
        "rank": 4,
        "instruction": "gentle motion",
    }
    with monkeypatch.context() as patch:
        patch.setattr(app_module, "inspect_system", lambda: {"cpu_ready": True, "ram_gb": 16})
        for mode in ("supervised", "style", "imitation"):
            response = client.post(f"/api/projects/{pid}/train", json={**body, "mode": mode})
            assert response.status_code == 202, response.text
            config = json.loads(response.json()["config"])
            assert config["motion_model"] == VIDEO_TRAINING_MODELS["animatediff"]["motion_model"]
            assert config["instruction"] == "gentle motion" and config["mode"] == mode
            with sqlite3.connect(root / "studio.sqlite3") as db:
                db.execute(
                    "UPDATE jobs SET status='completed' WHERE id=?", (response.json()["id"],)
                )
        parent = response.json()["id"]
        response = client.post(
            f"/api/projects/{pid}/train", json={**body, "start_job_id": parent, "rank": 16}
        )
        assert response.status_code == 202
        assert json.loads(response.json()["config"])["rank"] == 4
        client.post(f"/api/projects/{pid}/training/reset")
        assert (
            client.post(
                f"/api/projects/{pid}/train",
                json={**body, "video_model": "cogvideo", "start_job_id": parent},
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/projects/{other}/train", json={**body, "start_job_id": parent}
            ).status_code
            == 422
        )
        generate = {
            "kind": "video",
            "video_model": "animatediff",
            "trained_job_id": parent,
            "prompt": "a blue bird",
        }
        response = client.post(f"/api/projects/{pid}/generate", json=generate)
        assert response.status_code == 202, response.text
        frozen = json.loads(response.json()["config"])
        assert (
            frozen["trained_job_id"] == parent
            and frozen["motion_revision"] == config["motion_revision"]
        )
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='completed' WHERE id=?", (response.json()["id"],))
        assert client.post(f"/api/projects/{other}/generate", json=generate).status_code == 422
        assert (
            client.post(
                f"/api/projects/{pid}/generate", json={**generate, "compare_before": True}
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/projects/{pid}/generate", json={**generate, "kind": "image"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/projects/{pid}/generate", json={**generate, "with_audio": True}
            ).status_code
            == 422
        )
        patch.setattr(app_module, "inspect_system", lambda: {"cpu_ready": True, "ram_gb": 8})
        assert client.post(f"/api/projects/{pid}/train", json=body).status_code == 422
        patch.setattr(app_module, "inspect_system", lambda: {"cpu_ready": False, "ram_gb": 16})
        assert client.post(f"/api/projects/{pid}/train", json=body).status_code == 503


def check_motion_optimizer(tmp_path):
    import torch
    from diffusers import AnimateDiffPipeline, DDPMScheduler, MotionAdapter, UNetMotionModel
    from peft import get_peft_model_state_dict

    from studio.laptop_video import optimize_motion
    from tests.test_lora_numerical import tiny_pipeline

    torch.set_num_threads(1)

    def pipeline():
        pipe = tiny_pipeline(tmp_path)
        motion = MotionAdapter(
            block_out_channels=(32, 64),
            motion_layers_per_block=1,
            motion_num_attention_heads=4,
            motion_norm_num_groups=8,
        )
        unet = UNetMotionModel.from_unet2d(pipe.unet, motion)
        return AnimateDiffPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            unet=unet,
            scheduler=pipe.scheduler,
            motion_adapter=motion,
        )

    pipe = pipeline()
    original = {k: v.clone() for k, v in pipe.unet.named_parameters()}
    cached = [(torch.randn(1, 4, 4, 8, 8), torch.randn(1, 8, 32))]
    points = []
    config = {"rank": 4, "steps": 3, "learning_rate": 0.001}
    optimize_motion(
        pipe.unet,
        DDPMScheduler(num_train_timesteps=20),
        cached,
        config,
        lambda _: None,
        points.append,
    )
    adapter = {k: v.clone() for k, v in get_peft_model_state_dict(pipe.unet).items()}
    assert all(torch.isfinite(value).all() for value in adapter.values())
    assert any(
        "motion_modules" in k and "lora_B" in k and v.abs().sum() > 0 for k, v in adapter.items()
    )
    for key, value in pipe.unet.named_parameters():
        key = key.replace(".base_layer.", ".")
        if key in original:
            assert torch.equal(value, original[key])
    folder = tmp_path / "motion-adapter"
    AnimateDiffPipeline.save_lora_weights(str(folder), unet_lora_layers=adapter)
    restored = pipeline()
    restored.load_lora_weights(str(folder), adapter_name="default")
    loaded = get_peft_model_state_dict(restored.unet)
    assert adapter.keys() == loaded.keys()
    assert all(torch.equal(adapter[k], loaded[k]) for k in adapter)
    optimize_motion(
        restored.unet,
        DDPMScheduler(num_train_timesteps=20),
        cached,
        config,
        lambda _: None,
        points.append,
    )
    continued = get_peft_model_state_dict(restored.unet)
    assert any(not torch.equal(adapter[k], continued[k]) for k in adapter)
    assert all(point["loss"] >= 0 and point["lora_norm"] > 0 for point in points)
    restored.unet.eval()
    with torch.no_grad():
        sample, text = cached[0]
        text = text.repeat_interleave(sample.shape[2], dim=0)
        before = restored.unet(sample, torch.tensor([1]), encoder_hidden_states=text).sample
        restored.fuse_lora(components=["unet"], safe_fusing=True, adapter_names=["default"])
        restored.unload_lora_weights()
        after = restored.unet(sample, torch.tensor([1]), encoder_hidden_states=text).sample
        assert torch.allclose(before, after, atol=0.0001, rtol=0.0001)
    frames = restored(
        prompt="blue bird",
        height=32,
        width=32,
        num_frames=4,
        num_inference_steps=2,
        guidance_scale=1,
        generator=torch.Generator().manual_seed(42),
    ).frames[0]
    assert len(frames) == 4 and all(frame.size == (32, 32) for frame in frames)
