"""CogVideoX I2V LoRA training; temporal denoising follows Diffusers v0.33.1."""

import json
import random
import zipfile


def read_clip(path, width=720, height=480, count=49):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageOps

    with imageio.get_reader(path) as reader:
        metadata = reader.get_meta_data()
        size = metadata.get("size", (0, 0))
        if not size[0] or size[0] * size[1] > 1920 * 1080:
            raise ValueError("学習動画は1920×1080以下で指定してください")
        duration = min(float(metadata.get("duration", 0)), 6.125)
        if duration < 1:
            raise ValueError("動画は1秒以上必要です")
        fps = float(metadata["fps"])
        indices = np.linspace(0, max(0, int(duration * fps) - 1), count).astype(int)
        return np.stack(
            [
                np.asarray(
                    ImageOps.fit(
                        Image.fromarray(reader.get_data(int(index))),
                        (width, height),
                        method=Image.Resampling.LANCZOS,
                    )
                )
                for index in indices
            ]
        )


def optimize_video(
    transformer, scheduler, cached, config, progress, report, device="cuda", rotary=None
):
    """Shared optimizer path supports a genuine tiny CogVideoX CPU numerical test."""
    import torch
    from peft import LoraConfig

    transformer.requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r=config["rank"],
            lora_alpha=config["rank"],
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=True,
        )
    )
    transformer.enable_gradient_checkpointing()
    parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"])
    dtype = next(transformer.parameters()).dtype
    for step in range(config["steps"]):
        clean, reference, text = (
            tensor.to(device=device, dtype=dtype) for tensor in random.choice(cached)
        )
        noise = torch.randn_like(clean)
        timestep = torch.randint(
            0, scheduler.config.num_train_timesteps, (clean.shape[0],), device=device
        )
        noisy = scheduler.add_noise(clean, noise, timestep)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            output = transformer(
                hidden_states=torch.cat([noisy, reference], dim=2),
                encoder_hidden_states=text,
                timestep=timestep,
                image_rotary_emb=rotary,
                return_dict=False,
            )[0]
            prediction = scheduler.get_velocity(output, noisy, timestep)
            weight = 1 / (1 - scheduler.alphas_cumprod.to(device)[timestep])
            loss = (
                (prediction.float() - clean.float()).square() * weight.view(-1, 1, 1, 1, 1)
            ).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("動画の学習損失が不正です")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == config["steps"]:
            norm = sum(parameter.detach().float().square().sum() for parameter in parameters).sqrt()
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


def train_video(root, project, config, folder, progress, report):
    import torch
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.models.embeddings import get_3d_rotary_pos_embed
    from peft import get_peft_model_state_dict

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
    )
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    cached = []
    pipe.vae.to("cuda")
    for row in config["media"]:
        clip = read_clip(root / "projects" / project["id"] / "media" / f"{row['id']}.mp4")
        video = (
            torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
            / 127.5
            - 1
        )
        with torch.no_grad():
            latent = pipe.vae.encode(video).latent_dist.sample() * pipe.vae.config.scaling_factor
            reference = video[:, :, :1]
            reference += torch.randn_like(reference) * 0.05
            reference = (
                pipe.vae.encode(reference).latent_dist.sample() * pipe.vae.config.scaling_factor
            )
            latent = latent.permute(0, 2, 1, 3, 4)
            reference = reference.permute(0, 2, 1, 3, 4)
            reference = torch.cat(
                [reference, reference.new_zeros((1, latent.shape[1] - 1, *reference.shape[2:]))],
                dim=1,
            )
            cached.append([latent.cpu(), reference.cpu()])
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    pipe.text_encoder.to("cuda")
    for index, row in enumerate(config["media"]):
        caption = row["caption"] + ", " + config.get("instruction", "")
        tokens = pipe.tokenizer(
            caption,
            max_length=pipe.transformer.config.max_text_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to("cuda")
        with torch.no_grad():
            cached[index].append(pipe.text_encoder(tokens)[0].cpu())
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    rotary = (
        get_3d_rotary_pos_embed(
            embed_dim=pipe.transformer.config.attention_head_dim,
            crops_coords=((0, 0), (30, 45)),
            grid_size=(30, 45),
            temporal_size=cached[0][0].shape[1],
            device="cuda",
        )
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )
    optimize_video(
        pipe.transformer, pipe.scheduler, cached, config, progress, report, rotary=rotary
    )
    adapter = folder / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    CogVideoXImageToVideoPipeline.save_lora_weights(
        str(adapter), transformer_lora_layers=get_peft_model_state_dict(pipe.transformer)
    )
    (adapter / "model-card.json").write_text(
        json.dumps(
            {"base": "THUDM/CogVideoX-5b-I2V", "target": "video", "config": config},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(folder / "adapter.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in adapter.iterdir():
            archive.write(path, path.name)


def normalise_video(data, destination):
    """Validate a bounded local clip, then produce browser-compatible H.264 MP4."""
    import subprocess
    import tempfile
    from pathlib import Path

    import imageio_ffmpeg

    with tempfile.TemporaryDirectory(prefix="deel-clip-") as tmp:
        source = Path(tmp) / "source.mp4"
        source.write_bytes(data)
        read_clip(source, width=64, height=64, count=3)
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-t",
                "6.125",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
