"""Real SD 1.5 LoRA optimization and CogVideoX image-to-video inference."""

import argparse
import json
import math
import random
import sqlite3
import zipfile
from pathlib import Path


def load_pipeline(base, dtype):
    from diffusers import StableDiffusionPipeline

    if Path(base).is_file():
        if not base.endswith(".safetensors"):
            raise ValueError("ローカルモデルはSD 1.5の.safetensorsのみ対応しています")
        pipeline = StableDiffusionPipeline.from_single_file(base, torch_dtype=dtype)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(base, torch_dtype=dtype)
    if pipeline.unet.config.cross_attention_dim != 768:
        raise ValueError("SD 1.5モデルを選択してください。SDXL・Flux・SD 2は対応していません")
    return pipeline


def execute(root, job_id):
    import torch
    from diffusers import CogVideoXImageToVideoPipeline
    from PIL import Image, ImageOps

    from studio.exports import save_image, save_video

    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.row_factory = sqlite3.Row
        job = dict(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
        project = dict(
            db.execute("SELECT * FROM projects WHERE id=?", (job["project_id"],)).fetchone()
        )
        trained = db.execute(
            "SELECT id,config FROM jobs WHERE project_id=? AND kind='train' "
            "AND status='completed' ORDER BY rowid DESC LIMIT 1",
            (job["project_id"],),
        ).fetchone()
    config = json.loads(job["config"])
    config["job_id"] = job_id
    if job["kind"] == "image" and config.get("trained_job_id"):
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.row_factory = sqlite3.Row
            trained = db.execute(
                "SELECT id,config FROM jobs WHERE id=? AND project_id=? AND kind='train' AND status='completed'",
                (config["trained_job_id"], project["id"]),
            ).fetchone()
    folder = root / "jobs" / job_id
    folder.mkdir(parents=True, exist_ok=True)

    def progress(value):
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute("UPDATE jobs SET progress=? WHERE id=?", (int(value), job_id))

    def report_metric(point):
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

    def callback(_pipeline, step, _timestep, callback_kwargs):
        progress(min(99, (step + 1) * 100 // config.get("inference_steps", 50)))
        return callback_kwargs

    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    cpu_voice = job["kind"] == "voice_train" or (
        job["kind"] == "audio" and config.get("audio_model") == "kokoro"
    )
    if cpu_voice and not torch.cuda.is_available():
        torch.set_num_threads(2)
    if not cpu_voice and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    def public_audio(path):
        import gc

        from studio.audio_models import generate_audio

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if config.get("voice_job_id"):
            card = json.loads(
                (root / "jobs" / config["voice_job_id"] / "voice-card.json").read_text(
                    encoding="utf-8"
                )
            )
            config.update(card.get("controls", {}))

        generate_audio(config, path)
        if config.get("voice_job_id"):
            from studio.voice import apply_voice

            apply_voice(path, root / "jobs" / config["voice_job_id"] / "speaker.pt")
            from studio.voice import adjust_pitch

            adjust_pitch(path, config.get("pitch_semitones", 0))

    if job["kind"] == "voice_train":
        from studio.voice import learn_voice

        learn_voice(
            [
                root / "projects" / project["id"] / "media" / f"{row['id']}.wav"
                for row in config["media"]
            ],
            folder,
            progress,
            config.get("instruction", ""),
        )
        return
    if job["kind"] == "audio":
        public_audio(folder / "output.wav")
        return
    if config.get("public_model"):
        from studio.public_models import generate_public

        generate_public(config, folder, callback)
        if config.get("with_audio"):
            from studio.sound import mux_audio

            public_audio(folder / "sound.wav")
            mux_audio(folder / "output.mp4", folder / "sound.wav")
        return
    if job["kind"] == "audio_train":
        from studio.sound import read_audio, train_sound

        samples = [
            (
                read_audio(root / "projects" / project["id"] / "media" / f"{row['id']}.wav"),
                row["caption"]
                + (", " + config["instruction"] if config.get("instruction") else ""),
            )
            for row in config["media"]
        ]
        train_sound(samples, config, folder, progress, report_metric)
        with zipfile.ZipFile(folder / "adapter.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(folder / "sound.pt", "sound.pt")
            archive.write(folder / "sound-card.json", "sound-card.json")
        return
    if job["kind"] == "video_train":
        from studio.video_learning import train_video

        if config.get("related_images"):
            from studio.related_learning import augment

            augment(root, project, config, folder)
        train_video(root, project, config, folder, progress, report_metric)
        return
    if job["kind"] == "video":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
        )
        with sqlite3.connect(root / "studio.sqlite3") as db:
            if config.get("trained_job_id"):
                video_adapter = db.execute(
                    "SELECT id FROM jobs WHERE id=? AND project_id=? AND kind='video_train' AND status='completed'",
                    (config["trained_job_id"], project["id"]),
                ).fetchone()
            else:
                # Legacy requests must never attach an incompatible AnimateDiff adapter.
                candidates = db.execute(
                    "SELECT id,config FROM jobs WHERE project_id=? AND kind='video_train' AND status='completed' ORDER BY rowid DESC",
                    (project["id"],),
                ).fetchall()
                video_adapter = next(
                    (
                        row
                        for row in candidates
                        if json.loads(row[1]).get("video_model", "cogvideo") == "cogvideo"
                    ),
                    None,
                )
        if video_adapter:
            pipe.load_lora_weights(str(root / "jobs" / video_adapter[0] / "adapter"))
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        source = root / "jobs" / config["source_id"] / "output.png"
        image = ImageOps.fit(Image.open(source).convert("RGB"), (720, 480))
        frames = pipe(
            image=image,
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            num_frames=49,
            num_inference_steps=config["inference_steps"],
            guidance_scale=config["guidance_scale"],
            generator=torch.Generator(device="cpu").manual_seed(config["seed"]),
            callback_on_step_end=callback,
        ).frames[0]
        save_video(frames, folder / "output.mp4", config.get("resolution", "720p"), fps=8)
        if config.get("with_audio") and config.get("audio_model"):
            from studio.sound import mux_audio

            del pipe
            torch.cuda.empty_cache()
            public_audio(folder / "sound.wav")
            mux_audio(folder / "output.mp4", folder / "sound.wav")
        elif config.get("with_audio"):
            from studio.sound import generate_sound, mux_audio

            with sqlite3.connect(root / "studio.sqlite3") as db:
                audio_job = db.execute(
                    "SELECT id FROM jobs WHERE project_id=? AND kind='audio_train' AND status='completed' ORDER BY rowid DESC LIMIT 1",
                    (project["id"],),
                ).fetchone()
            if not audio_job:
                raise RuntimeError("学習済み音声モデルが見つかりません")
            # Release the large video model before generating the small sound model.
            del pipe
            torch.cuda.empty_cache()
            generate_sound(
                root / "jobs" / audio_job[0] / "sound.pt",
                config.get("audio_prompt") or f"sound of {project['trigger']}",
                config["seed"],
                folder / "sound.wav",
            )
            mux_audio(folder / "output.mp4", folder / "sound.wav")
        return

    base = (
        config["base_model"]
        if job["kind"] == "train"
        else json.loads(trained["config"])["base_model"]
    )
    # Float32 training avoids half-precision Adam underflow on the trainable adapters.
    dtype = torch.float32 if job["kind"] == "train" else torch.float16
    pipe = load_pipeline(base, dtype)
    if job["kind"] == "image":
        from studio.reference_images import image_arguments, image_pipeline

        pipe = image_pipeline(pipe, config)
        pipe.load_lora_weights(str(root / "jobs" / trained["id"] / "adapter"))
        pipe.enable_model_cpu_offload()
        generator = torch.Generator(device="cpu").manual_seed(config["seed"])
        arguments = image_arguments(pipe, root, config, 512, generator)
        result = pipe(
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            **arguments,
            num_inference_steps=config["inference_steps"],
            guidance_scale=config["guidance_scale"],
            callback_on_step_end=callback,
            generator=generator,
        )
        if result.nsfw_content_detected and any(result.nsfw_content_detected):
            raise RuntimeError("画像の安全性チェックにより出力されませんでした")
        save_image(result.images[0], folder / "output.png", config.get("resolution", "720p"))
        return

    if config.get("start_job_id"):
        pipe.load_lora_weights(
            str(root / "jobs" / config["start_job_id"] / "adapter"), adapter_name="default"
        )
    prepare_training_rows(root, project, config)
    train_adapter(pipe, root, project, config, folder, progress, report=report_metric)
    if config.get("mode") == "generation_test":
        save_test_sample(pipe, project["trigger"], config["seed"], folder / "test.png")


def save_test_sample(pipe, trigger, seed, destination, device="cuda", resolution=512, steps=20):
    import torch

    pipe.unet.eval()
    sample = pipe(
        prompt=f"a photo of {trigger}, natural light",
        height=resolution,
        width=resolution,
        num_inference_steps=steps,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    if sample.nsfw_content_detected and any(sample.nsfw_content_detected):
        raise RuntimeError("学習テスト画像の安全性チェックにより出力されませんでした")
    sample.images[0].save(destination)


def reward_weight(rating):
    """Positive reward-weighted regression: 1..5 ratings emphasize preferred outputs."""
    if rating is None:
        return 1.0
    if not 1 <= rating <= 5:
        raise ValueError("Rating must be between 1 and 5")
    return math.exp((rating - 3) / 2)


def prepare_training_rows(root, project, config, captioner=None):
    """Manual labels stay authoritative; pseudo-labels receive half weight."""
    mode = config.get("mode", "supervised")
    rows = config["images"]
    if mode == "semi_supervised" and any(not row.get("labeled") for row in rows):
        if captioner is None:
            import torch
            from PIL import Image
            from transformers import BlipForConditionalGeneration, BlipProcessor

            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            teacher = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            teacher.eval()

            def captioner(path):
                with Image.open(path) as image, torch.no_grad():
                    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
                    output = teacher.generate(**inputs, max_new_tokens=60)
                    return processor.decode(output[0], skip_special_tokens=True)

        for row in rows:
            if not row.get("labeled"):
                path = root / "projects" / project["id"] / "images" / f"{row['id']}.png"
                description = captioner(path).strip()
                if not description:
                    raise RuntimeError("画像の自動説明を生成できませんでした")
                row["caption"] = f"{description}, {project['trigger']}"
                row["pseudo_label"] = True
    for row in rows:
        if mode == "style" and not row.get("labeled"):
            row["caption"] = f"an artwork in the style of {project['trigger']}"
        if config.get("instruction", "").strip():
            row["caption"] += ", " + config["instruction"].strip()
        row["weight"] = 0.5 if row.get("pseudo_label") else 1.0
        if row.get("related_job_id"):
            row["weight"] *= 0.5
        if mode == "reinforcement":
            row["weight"] *= reward_weight(row.get("rating"))
    # Keep the captions and confidence used for this training run in its immutable snapshot.
    if config.get("job_id"):
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute(
                "UPDATE jobs SET config=? WHERE id=?", (json.dumps(config), config["job_id"])
            )


def train_adapter(
    pipe, root, project, config, folder, progress, device="cuda", resolution=512, report=None
):
    """Shared optimizer/export path; CPU and reduced resolution support numerical tests."""
    import torch
    import torch.nn.functional as functional
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from diffusers.utils import (
        convert_all_state_dict_to_peft,
        convert_state_dict_to_diffusers,
        convert_state_dict_to_kohya,
    )
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from safetensors.torch import save_file
    from torchvision import transforms

    pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    if "default" in getattr(pipe.unet, "peft_config", {}):
        pipe.unet.set_adapter("default")
        for name, parameter in pipe.unet.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
    else:
        pipe.unet.add_adapter(
            LoraConfig(
                r=config["rank"],
                lora_alpha=config["rank"],
                init_lora_weights="gaussian",
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            )
        )
    pipe.unet.train()
    pipe.unet.enable_gradient_checkpointing()
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"])
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    augmentations = [transforms.RandomHorizontalFlip()] if config.get("mode") == "style" else []
    transform = transforms.Compose(
        augmentations
        + [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    rows = config["images"]
    for step in range(config["steps"]):
        row = random.choice(rows)
        path = root / "projects" / project["id"] / "images" / f"{row['id']}.png"
        with Image.open(path) as image:
            pixels = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            latents = pipe.vae.encode(pixels).latent_dist.sample() * pipe.vae.config.scaling_factor
            tokens = pipe.tokenizer(
                row["caption"],
                padding="max_length",
                truncation=True,
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
            conditioning = pipe.text_encoder(tokens)[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (1,), device=device
        ).long()
        noisy = scheduler.add_noise(latents, noise, timesteps)
        prediction = pipe.unet(noisy, timesteps, conditioning).sample
        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError("Unsupported scheduler prediction type")
        loss = functional.mse_loss(prediction.float(), target.float()) * row.get("weight", 1.0)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress((step + 1) * 95 // config["steps"])
        if report is not None and (step % 10 == 0 or step + 1 == config["steps"]):
            lora_norm = (
                sum(parameter.detach().float().square().sum().item() for parameter in params) ** 0.5
            )
            report(
                {
                    "step": step + 1,
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": gradient_norm.item(),
                    "lora_norm": lora_norm,
                }
            )
        if step % 10 == 0:
            print(f"step={step + 1} loss={loss.item():.6f}", flush=True)

    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipe.unet))
    adapter = folder / "adapter"
    adapter.mkdir()
    StableDiffusionPipeline.save_lora_weights(str(adapter), unet_lora_layers=state)
    kohya = convert_state_dict_to_kohya(
        convert_all_state_dict_to_peft({f"unet.{key}": value for key, value in state.items()})
    )
    # Preserve alpha=rank in the Kohya/ComfyUI format.
    save_file(kohya, str(adapter / "stability-matrix-lora.safetensors"))
    metadata = {
        "name": project["name"],
        "trigger": project["trigger"],
        "base_model": config["base_model"],
        "architecture": "SD 1.x derived UNet; use the exact base_model",
        "base_revision": config.get("base_revision"),
        "training_model": config.get("training_model"),
        "training_resolution": resolution,
        "backend": device,
        "start_job_id": config.get("start_job_id"),
        "training_mode": config.get("mode", "supervised"),
        "level": config.get("level", "normal"),
        "group": config.get("group"),
        "reward_method": "reward-weighted regression"
        if config.get("mode") == "reinforcement"
        else None,
        "steps": config["steps"],
        "rank": config["rank"],
        "seed": config["seed"],
        "learning_rate": config["learning_rate"],
        "video": "Image-to-video via CogVideoX; image LoRA is not a video LoRA",
    }
    (adapter / "model-card.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with zipfile.ZipFile(folder / "adapter.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in adapter.iterdir():
            archive.write(file, file.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    execute(args.data, args.job)


if __name__ == "__main__":
    main()
