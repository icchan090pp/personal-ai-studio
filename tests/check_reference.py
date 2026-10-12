"""Actual reference-conditioned diffusion and request ownership checks."""

import json
import sqlite3

import numpy as np
import torch
from PIL import Image

from studio import app as app_module
from studio.reference_images import image_arguments, image_pipeline


def check_reference(client, root, pid, other, iid, monkeypatch):
    monkeypatch.setattr(
        app_module, "inspect_system", lambda: {"cpu_ready": True, "cuda_ready": True, "ram_gb": 32}
    )
    original_cuda_check = torch.cuda.is_available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    body = {
        "kind": "image",
        "public_model": "tiny",
        "prompt": "blue sky",
        "reference_id": iid,
        "reference_strength": 0.55,
    }
    assert client.post(f"/api/projects/{other}/generate", json=body).status_code == 404
    for change in (
        {"reference_id": "../bad"},
        {"reference_strength": 0},
        {"reference_strength": 1},
        {"kind": "video"},
    ):
        assert (
            client.post(f"/api/projects/{pid}/generate", json={**body, **change}).status_code == 422
        )
    for model in ("tiny", "sdxl", "ssd", "pixart"):
        reply = client.post(f"/api/projects/{pid}/generate", json={**body, "public_model": model})
        assert reply.status_code == 202, reply.text
        config = json.loads(reply.json()["config"])
        assert config["reference_path"] == f"projects/{pid}/references/{iid}.png"
        assert config["reference_strength"] == 0.55 and config["public_model"] == model
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET status='cancelled' WHERE id=?", (reply.json()["id"],))
    monkeypatch.setattr(torch.cuda, "is_available", original_cuda_check)
    from tests.test_lora_numerical import tiny_pipeline

    torch.set_num_threads(1)
    numeric = root / "numeric"
    numeric.mkdir()
    pipe = tiny_pipeline(numeric)
    config = {
        "reference_path": f"projects/{pid}/references/{iid}.png",
        "reference_strength": 0.55,
        "inference_steps": 4,
    }
    converted = image_pipeline(pipe, config)
    assert converted.unet is pipe.unet and converted.vae is pipe.vae
    generator = torch.Generator().manual_seed(42)
    args = image_arguments(converted, root, config, 32, generator)
    assert args["image"].size == (32, 32) and args["strength"] == 0.55
    result = converted(
        prompt="blue sky", num_inference_steps=4, guidance_scale=1, generator=generator, **args
    ).images[0]
    assert result.size == (32, 32)
    # Identical prompt/seed with a different source must alter the generated image.
    args["image"] = Image.new("RGB", (32, 32), "red")
    changed = converted(
        prompt="blue sky",
        num_inference_steps=4,
        guidance_scale=1,
        generator=torch.Generator().manual_seed(42),
        **args,
    ).images[0]
    assert not np.array_equal(np.asarray(result), np.asarray(changed))
    from diffusers import DPMSolverMultistepScheduler, PixArtSigmaPipeline, PixArtTransformer2DModel

    pixart = PixArtSigmaPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=pipe.vae,
        transformer=PixArtTransformer2DModel(
            sample_size=16,
            num_layers=1,
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=4,
            out_channels=8,
            cross_attention_dim=32,
            caption_channels=32,
        ),
        scheduler=DPMSolverMultistepScheduler(num_train_timesteps=100),
    )
    config["public_model"] = "pixart"
    kwargs = image_arguments(pixart, root, config, 32, torch.Generator().manual_seed(42))
    assert len(kwargs["timesteps"]) == 2 and torch.isfinite(kwargs["latents"]).all()
    output = pixart(
        prompt_embeds=torch.zeros(1, 8, 32),
        prompt_attention_mask=torch.ones(1, 8),
        guidance_scale=1,
        num_inference_steps=4,
        **kwargs,
    ).images[0]
    assert output.size == (32, 32)
