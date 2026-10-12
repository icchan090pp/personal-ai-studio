"""Experimental prompt-conditioned waveform diffusion for short sounds, not speech TTS."""

import hashlib
import json
import random
import subprocess
import wave


def read_audio(path, seconds=6, rate=16000):
    import imageio_ffmpeg
    import numpy as np

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-ar",
        str(rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=30, check=True)
    data = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(data) < rate or not np.isfinite(data).all():
        raise ValueError("音声は1秒以上の有効なデータが必要です")
    return data.clip(-1, 1)


def write_audio(path, waveform, rate=16000):
    import numpy as np

    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((np.asarray(waveform).clip(-1, 1) * 32767).astype("<i2").tobytes())


def prompt_id(prompt):
    return (
        int.from_bytes(hashlib.sha256(prompt.strip().casefold().encode()).digest()[:4], "big") % 256
    )


def build_model():
    import torch
    from torch import nn

    class SoundDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt = nn.Embedding(256, 16)
            self.network = nn.Sequential(
                nn.Conv1d(18, 32, 9, padding=4),
                nn.GroupNorm(4, 32),
                nn.SiLU(),
                nn.Conv1d(32, 32, 9, padding=4),
                nn.SiLU(),
                nn.Conv1d(32, 1, 9, padding=4),
            )

        def forward(self, noisy, timestep, labels):
            condition = self.prompt(labels).unsqueeze(-1).expand(-1, -1, noisy.shape[-1])
            time = (timestep.float() / 32).reshape(-1, 1, 1).expand(-1, 1, noisy.shape[-1])
            return self.network(torch.cat([noisy, condition, time], dim=1))

    return SoundDenoiser()


def train_sound(samples, config, folder, progress, report, device="cuda", window=4096):
    import torch

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    beta = torch.linspace(0.0001, 0.02, 32, device=device)
    alpha = (1 - beta).cumprod(0)
    labels = []
    for step in range(config["steps"]):
        waveform, caption = random.choice(samples)
        label = prompt_id(caption)
        labels.append(caption)
        data = torch.as_tensor(waveform, dtype=torch.float32, device=device)
        if len(data) < window:
            data = torch.nn.functional.pad(data, (0, window - len(data)))
        offset = random.randrange(len(data) - window + 1)
        clean = data[offset : offset + window][None, None]
        t = torch.randint(0, 32, (1,), device=device)
        noise = torch.randn_like(clean)
        noisy = alpha[t].sqrt().view(1, 1, 1) * clean + (1 - alpha[t]).sqrt().view(1, 1, 1) * noise
        prediction = model(noisy, t, torch.tensor([label], device=device))
        loss = torch.nn.functional.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise RuntimeError("音声の学習損失が不正です")
        optimizer.zero_grad()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == config["steps"]:
            norm = sum(
                parameter.detach().float().square().sum() for parameter in model.parameters()
            ).sqrt()
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
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": {key: value.cpu() for key, value in model.state_dict().items()},
            "sample_rate": 16000,
            "diffusion_steps": 32,
        },
        folder / "sound.pt",
    )
    (folder / "sound-card.json").write_text(
        json.dumps(
            {
                "type": "experimental-waveform-diffusion",
                "prompts": sorted(set(labels)),
                "limitations": "Short environmental/effect sounds; not speech or voice cloning.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return model


def generate_sound(
    checkpoint, prompt, seed, destination, seconds=6.125, device="cuda", samples=None
):
    import torch

    package = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model().to(device)
    model.load_state_dict(package["state"])
    model.eval()
    count = samples or int(package["sample_rate"] * seconds)
    generator = torch.Generator(device=device).manual_seed(seed)
    beta = torch.linspace(0.0001, 0.02, 32, device=device)
    alpha = 1 - beta
    cumulative = alpha.cumprod(0)
    waveform = torch.randn((1, 1, count), device=device, generator=generator)
    labels = torch.tensor([prompt_id(prompt)], device=device)
    with torch.no_grad():
        for t in reversed(range(32)):
            predicted = model(waveform, torch.tensor([t], device=device), labels)
            waveform = (waveform - beta[t] / (1 - cumulative[t]).sqrt() * predicted) / alpha[
                t
            ].sqrt()
            if t:
                waveform += beta[t].sqrt() * torch.randn(
                    waveform.shape, device=device, generator=generator
                )
    write_audio(destination, waveform[0, 0].cpu().numpy())


def mux_audio(video, audio, *, loop_video=False):
    import imageio_ffmpeg

    temporary = video.with_name("with-sound.mp4")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-nostdin",
            "-y",
            *(["-stream_loop", "-1"] if loop_video else []),
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(temporary),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    temporary.replace(video)
