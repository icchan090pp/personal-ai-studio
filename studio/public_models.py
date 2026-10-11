"""Allowlisted public-weight generation backends, separate from trained LoRA models."""

PUBLIC_MODELS = {
    "sdxl": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "name": "SDXL 1.0",
        "kind": "image",
        "pipeline": "StableDiffusionXLPipeline",
        "license": "OpenRAIL++",
        "memory": "GPU 12〜16GB目安",
        "revision": "462165984030d82259a11f4367a4eed129e94a7b",
    },
    "ssd": {
        "id": "segmind/SSD-1B",
        "name": "SSD-1B",
        "kind": "image",
        "pipeline": "StableDiffusionXLPipeline",
        "license": "Apache-2.0",
        "memory": "GPU 8〜12GB目安",
        "revision": "60987f37e94cd59c36b1cba832b9f97b57395a10",
    },
    "pixart": {
        "id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "name": "PixArt Sigma",
        "kind": "image",
        "pipeline": "PixArtSigmaPipeline",
        "license": "OpenRAIL++",
        "memory": "GPU 12〜24GB目安",
        "revision": "e102b3591cc82e97071b8b4cb90d834d0c487207",
    },
    "ltx": {
        "id": "Lightricks/LTX-Video",
        "name": "LTX-Video",
        "kind": "video",
        "pipeline": "LTXPipeline",
        "license": "LTX独自ライセンス（配布元を確認）",
        "memory": "GPU 16〜24GB目安",
        "revision": "8984fa25007f376c1a299016d0957a37a2f797bb",
    },
}


def catalog():
    return [
        {**v, "key": k, "trainable": False, "weights_url": "https://huggingface.co/" + v["id"]}
        for k, v in PUBLIC_MODELS.items()
    ]


def generate_public(config, folder, callback):
    import diffusers
    import torch

    from studio.exports import save_image, save_video

    model = PUBLIC_MODELS[config["public_model"]]
    cls = getattr(diffusers, model["pipeline"])
    pipe = cls.from_pretrained(
        model["id"],
        revision=model["revision"],
        torch_dtype=torch.bfloat16 if model["kind"] == "video" else torch.float16,
    )
    pipe.enable_model_cpu_offload()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    kwargs = {
        "prompt": config["prompt"],
        "negative_prompt": config["negative_prompt"],
        "num_inference_steps": config["inference_steps"],
        "guidance_scale": 3.0
        if model["kind"] == "video"
        else 4.5
        if config["public_model"] == "pixart"
        else 5.0,
        "generator": torch.Generator(device="cpu").manual_seed(config["seed"]),
    }
    if model["kind"] == "video":
        kwargs.update(
            height=512, width=768, num_frames=49, frame_rate=8, callback_on_step_end=callback
        )
        frames = pipe(**kwargs).frames[0]
        save_video(frames, folder / "output.mp4", config["resolution"], fps=8)
    else:
        kwargs.update(height=1024, width=1024)
        if config["public_model"] != "pixart":
            kwargs["callback_on_step_end"] = callback
        image = pipe(**kwargs).images[0]
        save_image(image, folder / "output.png", config["resolution"])
