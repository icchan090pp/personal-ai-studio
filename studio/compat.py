"""Lightweight image generation on DirectML or CPU; CUDA is never required here."""

import argparse
import gc
import json
import os
import sqlite3
import subprocess
from pathlib import Path

MODEL_ID = "segmind/tiny-sd"
REVISION = "cad0bd7495fa6c4bcca01b19a723dc91627fe84f"


def pipeline(model=None):
    import torch
    from diffusers import StableDiffusionPipeline

    # The primary runtime uses torch >=2.6 for weights_only loading of the pinned
    # upstream BIN weights. DirectML only reads the locally converted safetensors.
    kwargs = {"torch_dtype": torch.float32, "low_cpu_mem_usage": True}
    if model:
        kwargs.update(local_files_only=True, use_safetensors=True)
    else:
        kwargs.update(revision=REVISION, use_safetensors=False)
    result = StableDiffusionPipeline.from_pretrained(str(model or MODEL_ID), **kwargs)
    result.enable_attention_slicing("max")
    result.enable_vae_slicing()
    return result


def generate(config, destination, progress, model=None, device="cpu", device_index=0, size=256):
    import torch

    from studio.exports import save_image

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    from studio.reference_images import image_arguments, image_pipeline

    pipe = image_pipeline(pipeline(model), config)
    if device == "directml":
        import torch_directml

        target = torch_directml.device(device_index)
    else:
        target = device
    pipe.to(target)

    def callback(_pipe, step, _timestep, values):
        progress(min(99, (step + 1) * 100 // config["inference_steps"]))
        return values

    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    arguments = image_arguments(pipe, destination.parents[2], config, size, generator)
    result = pipe(
        prompt=config["prompt"],
        negative_prompt=config.get("negative_prompt", ""),
        **arguments,
        num_inference_steps=config["inference_steps"],
        guidance_scale=7.5,
        generator=generator,
        callback_on_step_end=callback,
    )
    if result.nsfw_content_detected and any(result.nsfw_content_detected):
        raise RuntimeError("画像の安全性チェックにより出力されませんでした")
    save_image(result.images[0], destination, config.get("resolution", "native"))


def prepare_safe_model(destination):
    if (destination / "model_index.json").is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "-building")
    pipe = pipeline()
    pipe.save_pretrained(temporary, safe_serialization=True)
    del pipe
    gc.collect()
    temporary.rename(destination)
    return destination


def execute(root, job_id, directml_model=None, device_index=0):
    with sqlite3.connect(root / "studio.sqlite3") as db:
        row = db.execute("SELECT config FROM jobs WHERE id=?", (job_id,)).fetchone()
    config = json.loads(row[0])
    folder = root / "jobs" / job_id
    folder.mkdir(parents=True, exist_ok=True)

    def progress(value):
        with sqlite3.connect(root / "studio.sqlite3") as db:
            db.execute(
                "UPDATE jobs SET progress=? WHERE id=? AND status='running'", (value, job_id)
            )

    backend = "directml" if directml_model else "cpu"
    if not directml_model:
        from studio.hardware import (
            fallback_state,
            inspect_system,
            remember_fallback,
            selected_backend,
        )

        info = inspect_system()
        previous = fallback_state(root)
        selected = selected_backend(root)
        dml_python = os.environ.get("STUDIO_DML_PYTHON")
        if (
            info.get("cuda_ready")
            and previous.get("backend") != "cuda"
            and selected in ("auto", "cuda")
        ):
            backend = "cuda"
        elif (
            dml_python
            and info["directml"].get("ready")
            and previous.get("backend") != "directml"
            and not info.get("cuda_ready")
            and selected in ("auto", "directml")
        ):
            model = prepare_safe_model(root / "models/tiny-sd-safetensors")
            result = subprocess.run(
                [
                    dml_python,
                    "-m",
                    "studio.compat",
                    "--data",
                    str(root),
                    "--job",
                    job_id,
                    "--directml-model",
                    str(model),
                    "--device-index",
                    str(info["directml"].get("index", 0)),
                ],
                check=False,
            )
            if result.returncode == 0 and (folder / "output.png").is_file():
                return
            print("DirectMLで処理できなかったためCPUで再試行します。", flush=True)
            config["fallback_reason"] = (
                "DirectMLで非対応の演算・メモリ不足などが発生。CPUで生成しました。"
            )
            remember_fallback(root, "directml", config["fallback_reason"])
            progress(0)
        elif previous:
            config["fallback_reason"] = (
                "前回のGPUエラーを記録し、CPUを優先して生成しました。再診断でGPUを再試行できます。"
            )
    try:
        generate(
            config,
            folder / "output.png",
            progress,
            model=directml_model,
            device=backend,
            device_index=device_index,
        )
    except RuntimeError:
        if backend != "cuda":
            raise
        backend = "cpu"
        config["fallback_reason"] = "CUDAの演算・メモリ不足などが発生。CPUで生成しました。"
        remember_fallback(root, "cuda", config["fallback_reason"])
        progress(0)
        generate(config, folder / "output.png", progress, device="cpu")
    (folder / "execution.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "model": MODEL_ID,
                "reference_id": config.get("reference_id"),
                "reference_strength": config.get("reference_strength"),
                "native_resolution": [256, 256],
                "fallback_reason": config.get("fallback_reason"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--directml-model", type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    execute(args.data, args.job, args.directml_model, args.device_index)
