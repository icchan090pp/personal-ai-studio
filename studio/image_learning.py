"""Real LoRA training and inference with the exact base/adapter selected by the user."""

import argparse
import json
import os
import random
import sqlite3
import time
from pathlib import Path

from studio.training_models import IMAGE_TRAINING_MODELS


def load_pipeline(config):
    import torch
    from diffusers import StableDiffusionPipeline

    model = IMAGE_TRAINING_MODELS[config["training_model"]]
    pipe = StableDiffusionPipeline.from_pretrained(
        config["base_model"],
        revision=config["base_revision"],
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=model["safetensors"],
    )
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    return pipe


def execute(root, job_id):
    import torch

    from studio.exports import save_image
    from studio.gpu import prepare_training_rows, save_test_sample, train_adapter

    with sqlite3.connect(root / "studio.sqlite3") as db:
        db.row_factory = sqlite3.Row
        job = dict(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
        project = dict(
            db.execute("SELECT * FROM projects WHERE id=?", (job["project_id"],)).fetchone()
        )
    config = json.loads(job["config"])
    model = IMAGE_TRAINING_MODELS[config["training_model"]]
    device = model["backend"]
    resolution = model["resolution"]
    folder = root / "jobs" / job_id
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

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

    # Captioning is released before loading the diffusion weights on smaller PCs.
    if job["kind"] == "train":
        config["job_id"] = job_id
        if config.get("related_images"):
            from studio.related_learning import augment

            augment(root, project, config, folder)
        prepare_training_rows(root, project, config)
    pipe = load_pipeline(config)
    adapter_id = config.get("start_job_id") if job["kind"] == "train" else config["trained_job_id"]
    if adapter_id and job["kind"] == "train":
        pipe.load_lora_weights(str(root / "jobs" / adapter_id / "adapter"), adapter_name="default")
    if job["kind"] == "train":
        train_adapter(
            pipe,
            root,
            project,
            config,
            folder,
            progress,
            device=device,
            resolution=resolution,
            report=report,
        )
        if config["mode"] == "generation_test":
            save_test_sample(
                pipe,
                project["trigger"],
                config["seed"],
                folder / "test.png",
                device=device,
                resolution=resolution,
            )
    else:
        pipe.to(device)

        stages = (
            [(False, "before.png"), (True, "output.png")]
            if config.get("compare_before")
            else [(True, "output.png")]
        )
        for stage, (learned, filename) in enumerate(stages):
            if learned:
                pipe.load_lora_weights(
                    str(root / "jobs" / adapter_id / "adapter"), adapter_name="default"
                )

            def callback(_pipe, step, _timestep, values):
                progress(
                    min(
                        99,
                        (stage * config["inference_steps"] + step + 1)
                        * 100
                        // (len(stages) * config["inference_steps"]),
                    )
                )
                return values

            result = pipe(
                prompt=config["prompt"],
                negative_prompt=config["negative_prompt"],
                height=resolution,
                width=resolution,
                num_inference_steps=config["inference_steps"],
                guidance_scale=config["guidance_scale"],
                generator=torch.Generator(device="cpu").manual_seed(config["seed"]),
                callback_on_step_end=callback,
            )
            if result.nsfw_content_detected and any(result.nsfw_content_detected):
                raise RuntimeError("画像の安全性チェックにより出力されませんでした")
            save_image(result.images[0], folder / filename, config["resolution"])
    (folder / "execution.json").write_text(
        json.dumps(
            {
                "backend": device,
                "model": config["base_model"],
                "revision": config["base_revision"],
                "adapter_id": adapter_id,
                "comparison": bool(config.get("compare_before")),
                "seed": config["seed"],
                "prompt": config.get("prompt"),
                "inference_steps": config.get("inference_steps"),
                "native_resolution": [resolution, resolution],
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        ),
        encoding="utf-8",
    )


def error_message(error):
    value = str(error).lower()
    if isinstance(error, MemoryError) or any(
        word in value for word in ("out of memory", "allocate memory", "not enough memory")
    ):
        return "メモリが不足しました。他のアプリを閉じ、Tiny-SD・ランク4で再実行してください。投稿画像と完成モデルは保持しています。"
    if any(
        word in value
        for word in ("connection", "timed out", "offline", "http", "localentrynotfound")
    ):
        return "基盤モデルのダウンロードに失敗しました。ネット接続と空き容量を確認して再実行してください。取得済みのファイルは再利用します。"
    if "non-finite" in value:
        return "学習値が不安定になりました。学習率を小さくして再実行してください。以前の完成モデルは保持しています。"
    return "画像モデルの処理に失敗しました。PC環境・AI機能の再セットアップを試してください。投稿画像と完成モデルは保持しています。"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    try:
        execute(args.data, args.job)
    except Exception as exc:
        folder = args.data / "jobs" / args.job
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "error.json").write_text(
            json.dumps({"message": error_message(exc)}, ensure_ascii=False), encoding="utf-8"
        )
        raise
