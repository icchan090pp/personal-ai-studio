"""CPU AnimateDiff personalization, versioned temporal LoRA and sequential audio."""

import argparse
import gc
import json
import os
import random
import sqlite3
import time
import zipfile
from pathlib import Path


def load_pipeline(config):
    import torch
    from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

    motion = MotionAdapter.from_pretrained(
        config["motion_model"],
        revision=config["motion_revision"],
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        config["base_model"],
        revision=config["base_revision"],
        motion_adapter=motion,
        torch_dtype=torch.float32,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        clip_sample=False,
        timestep_spacing="linspace",
        steps_offset=1,
    )
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing("max")
    pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
    return pipe


def optimize_motion(unet, scheduler, cached, config, progress, report):
    """Learn temporal and spatial attention LoRA from genuine ordered video latents."""
    import torch
    from peft import LoraConfig

    unet.requires_grad_(False)
    if getattr(unet, "peft_config", None):
        for name, parameter in unet.named_parameters():
            parameter.requires_grad_("lora_" in name)
    else:
        unet.add_adapter(
            LoraConfig(
                r=config["rank"],
                lora_alpha=config["rank"],
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                init_lora_weights=True,
            )
        )
    unet.enable_gradient_checkpointing()
    unet.train()
    parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"])
    for step in range(config["steps"]):
        index = random.randrange(len(cached))
        latent, text = cached[index]
        noise = torch.randn_like(latent)
        timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,))
        noisy = scheduler.add_noise(latent, noise, timestep)
        optimizer.zero_grad(set_to_none=True)
        # UNetMotionModel flattens batch and frames for spatial cross-attention.
        frame_text = text.repeat_interleave(latent.shape[2], dim=0)
        prediction = unet(noisy, timestep, encoder_hidden_states=frame_text).sample
        target = (
            scheduler.get_velocity(latent, noise, timestep)
            if scheduler.config.prediction_type == "v_prediction"
            else noise
        )
        loss = (prediction.float() - target.float()).square().mean()
        loss = loss * config.get("sample_weights", [1.0] * len(cached))[index]
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite video training loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == config["steps"]:
            norm = sum(p.detach().float().square().sum() for p in parameters).sqrt()
            report(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "learning_rate": config["learning_rate"],
                    "gradient_norm": float(gradient),
                    "lora_norm": float(norm),
                }
            )
        progress((step + 1) * 100 // config["steps"])


def train_motion(pipe, root, project, config, folder, progress, report):
    import torch
    from diffusers import AnimateDiffPipeline, DDPMScheduler
    from peft import get_peft_model_state_dict

    from studio.video_learning import read_clip

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    cached = []
    config["sample_weights"] = [
        0.5 if row.get("related_job_id") else 1.0 for row in config["media"]
    ]
    for row in config["media"]:
        # Four consecutive sampled moments keep back-propagation small; order is retained.
        clip = read_clip(
            root / "projects" / project["id"] / "media" / f"{row['id']}.mp4",
            width=256,
            height=256,
            count=4,
        )
        pixels = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 127.5 - 1
        if config["mode"] == "style" and random.random() < 0.5:
            pixels = pixels.flip(-1)
        caption = ", ".join(filter(None, [row["caption"], config.get("instruction", "")]))
        tokens = pipe.tokenizer(
            caption,
            max_length=pipe.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            latent = pipe.vae.encode(pixels).latent_dist.sample() * pipe.vae.config.scaling_factor
            text = pipe.text_encoder(tokens.input_ids)[0]
        cached.append((latent.permute(1, 0, 2, 3).unsqueeze(0), text))
    optimize_motion(
        pipe.unet,
        DDPMScheduler.from_config(pipe.scheduler.config),
        cached,
        config,
        progress,
        report,
    )
    adapter = folder / "adapter"
    AnimateDiffPipeline.save_lora_weights(
        str(adapter), unet_lora_layers=get_peft_model_state_dict(pipe.unet)
    )
    (adapter / "model-card.json").write_text(
        json.dumps(
            {
                "base_model": config["base_model"],
                "base_revision": config["base_revision"],
                "motion_model": config["motion_model"],
                "motion_revision": config["motion_revision"],
                "start_job_id": config.get("start_job_id"),
                "backend": "cpu",
                "training_frames": 4,
                "training_resolution": 256,
                "target": "spatial and temporal attention LoRA",
                "config": config,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(folder / "adapter.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in adapter.iterdir():
            archive.write(path, path.name)


def add_audio(root, config, folder):
    from studio.audio_models import generate_audio
    from studio.sound import mux_audio

    if config.get("voice_job_id"):
        from studio.voice import adjust_pitch, apply_voice

        card = json.loads(
            (root / "jobs" / config["voice_job_id"] / "voice-card.json").read_text(encoding="utf-8")
        )
        config.update(card.get("controls", {}))
    generate_audio(config, folder / "sound.wav")
    if config.get("voice_job_id"):
        apply_voice(folder / "sound.wav", root / "jobs" / config["voice_job_id"] / "speaker.pt")
        adjust_pitch(folder / "sound.wav", config.get("pitch_semitones", 0))
    # Preserve the entire narration by looping the short visual when it is longer.
    mux_audio(folder / "output.mp4", folder / "sound.wav", loop_video=True)


def execute(root, job_id):
    import torch

    from studio.exports import save_video

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.row_factory = sqlite3.Row
        job = dict(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
        project = dict(
            db.execute("SELECT * FROM projects WHERE id=?", (job["project_id"],)).fetchone()
        )
    config = json.loads(job["config"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    started = time.monotonic()
    folder = root / "jobs" / job_id
    folder.mkdir(parents=True, exist_ok=True)

    def progress(value):
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute(
                "UPDATE jobs SET progress=? WHERE id=? AND status='running'", (value, job_id)
            )

    def report(point):
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute(
                "INSERT OR REPLACE INTO training_metrics(job_id,step,loss,learning_rate,gradient_norm,lora_norm) VALUES(?,?,?,?,?,?)",
                (
                    job_id,
                    point["step"],
                    point["loss"],
                    point["learning_rate"],
                    point["gradient_norm"],
                    point["lora_norm"],
                ),
            )

    if job["kind"] == "video_train" and config.get("related_images"):
        from studio.related_learning import augment

        config["job_id"] = job_id
        augment(root, project, config, folder)
        gc.collect()
    pipe = load_pipeline(config)
    adapter_id = (
        config.get("start_job_id") if job["kind"] == "video_train" else config.get("trained_job_id")
    )
    if adapter_id:
        pipe.load_lora_weights(str(root / "jobs" / adapter_id / "adapter"), adapter_name="default")
    if job["kind"] == "video_train":
        train_motion(pipe, root, project, config, folder, progress, report)
    else:
        if adapter_id:
            # Merge only the in-memory inference copy; saved base and adapter stay separate.
            pipe.fuse_lora(components=["unet"], safe_fusing=True, adapter_names=["default"])
            pipe.unload_lora_weights()

        def callback(_pipe, step, _timestep, values):
            progress(min(95, (step + 1) * 95 // config["inference_steps"]))
            return values

        frames = pipe(
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            height=256,
            width=256,
            num_frames=16,
            num_inference_steps=config["inference_steps"],
            guidance_scale=7.5,
            generator=torch.Generator().manual_seed(config["seed"]),
            decode_chunk_size=1,
            callback_on_step_end=callback,
        ).frames[0]
        save_video(frames, folder / "output.mp4", config["resolution"], fps=8)
        del pipe, frames
        gc.collect()
        if config.get("with_audio"):
            add_audio(root, config, folder)
    (folder / "execution.json").write_text(
        json.dumps(
            {
                "backend": "cpu",
                "model": config["base_model"],
                "motion_model": config["motion_model"],
                "adapter_id": adapter_id,
                "seed": config["seed"],
                "native_resolution": [256, 256],
                "frames": 16,
                "fps": 8,
                "audio_model": config.get("audio_model"),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    try:
        execute(args.data, args.job)
    except Exception:
        folder = args.data / "jobs" / args.job
        folder.mkdir(parents=True, exist_ok=True)
        message = "動画モデルの処理に失敗しました。他のアプリを閉じ、PC環境・AI機能を確認して再実行してください。投稿データと完成モデルは保持しています。"
        (folder / "error.json").write_text(
            json.dumps({"message": message}, ensure_ascii=False), encoding="utf-8"
        )
        raise
