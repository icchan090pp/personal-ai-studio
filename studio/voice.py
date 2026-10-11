"""OpenVoice V2 speaker embedding extraction, not full model fine-tuning."""

import json
import zipfile

MODEL = "myshell-ai/OpenVoiceV2"
REVISION = "f36e7edfe1684461a8343844af60babc2efbb727"


def converter():
    import torch
    from huggingface_hub import hf_hub_download
    from openvoice.api import ToneColorConverter

    config = hf_hub_download(MODEL, "converter/config.json", revision=REVISION)
    checkpoint = hf_hub_download(MODEL, "converter/checkpoint.pth", revision=REVISION)
    network = ToneColorConverter(config, device="cuda:0" if torch.cuda.is_available() else "cpu")
    network.load_ckpt(checkpoint)
    return network


def learn_voice(paths, folder, progress, instruction=""):
    network = converter()
    progress(20)
    network.extract_se([str(path) for path in paths], str(folder / "speaker.pt"))
    card = {
        "model": MODEL,
        "revision": REVISION,
        "method": "reference speaker embedding mean, frozen pretrained converter",
        "files": len(paths),
        "license": "MIT",
        "limitations": "Not full fine-tuning; fidelity depends on model, recordings and language.",
    }
    card.update(instruction=instruction, controls=instruction_controls(instruction))
    (folder / "voice-card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(folder / "adapter.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(folder / "speaker.pt", "speaker.pt")
        archive.write(folder / "voice-card.json", "voice-card.json")
    progress(100)


def apply_voice(path, speaker):
    import torch

    network = converter()
    source = network.extract_se([str(path)])
    target = torch.load(speaker, map_location=network.device, weights_only=True)
    output = path.with_name("converted.wav")
    network.convert(
        str(path), source, target, output_path=str(output), message="DeeL Pyess synthetic voice"
    )
    output.replace(path)


def instruction_controls(instruction):
    """Translate supported free-text directions into bounded synthesis controls."""
    text = instruction.casefold()
    slow = any(word in text for word in ("ゆっくり", "遅く", "slow"))
    fast = any(word in text for word in ("速く", "早口", "fast"))
    high = any(word in text for word in ("高め", "高い声", "higher pitch"))
    low = any(word in text for word in ("低め", "低い声", "lower pitch"))
    return {
        "speech_speed": 1.0 if slow == fast else 0.8 if slow else 1.2,
        "pitch_semitones": 0 if high == low else 2 if high else -2,
    }


def adjust_pitch(path, semitones):
    if not semitones:
        return
    import subprocess
    import wave

    import imageio_ffmpeg

    with wave.open(str(path), "rb") as recording:
        rate = recording.getframerate()
        samples = recording.getnframes()
    ratio = 2 ** (semitones / 12)
    output = path.with_name("pitch-adjusted.wav")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(path),
            "-af",
            f"asetrate={rate * ratio},aresample={rate},atempo={1 / ratio},apad,atrim=end_sample={samples}",
            str(output),
        ],
        check=True,
        timeout=60,
        capture_output=True,
    )
    output.replace(path)
