"""Use the same uploaded reference and change strength across image backends."""

from PIL import Image, ImageOps


def image_pipeline(pipe, config):
    if not config.get("reference_path") or config.get("public_model") == "pixart":
        return pipe
    from diffusers import AutoPipelineForImage2Image

    return AutoPipelineForImage2Image.from_pipe(pipe)


def image_arguments(pipe, root, config, size, generator):
    if not config.get("reference_path"):
        return {"height": size, "width": size}
    with Image.open(root / config["reference_path"]) as original:
        image = ImageOps.pad(original.convert("RGB"), (size, size), color="white")
    strength = config["reference_strength"]
    if config.get("public_model") != "pixart":
        return {"image": image, "strength": strength}

    # PixArt Sigma exposes noisy latents and custom timesteps rather than an
    # Image2Image class. Encode the actual image, add noise at the selected
    # starting timestep, then run only the remaining denoising trajectory.
    import torch
    from diffusers import DPMSolverMultistepScheduler
    from diffusers.utils.torch_utils import randn_tensor

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = pipe._execution_device
    pipe.scheduler.set_timesteps(config["inference_steps"], device=device)
    remaining = max(1, int(config["inference_steps"] * strength))
    timesteps = pipe.scheduler.timesteps[-remaining:].tolist()
    pipe.scheduler.set_timesteps(timesteps=timesteps, device=device)
    pixels = pipe.image_processor.preprocess(image, height=size, width=size)
    with torch.no_grad():
        latent = pipe.vae.encode(pixels.to(device=device, dtype=pipe.vae.dtype)).latent_dist.mode()
        latent = latent * pipe.vae.config.scaling_factor
        noise = randn_tensor(latent.shape, generator=generator, device=device, dtype=latent.dtype)
        latent = pipe.scheduler.add_noise(latent, noise, pipe.scheduler.timesteps[:1])
    # The pipeline multiplies supplied latents by init_noise_sigma (1 for DPM).
    return {
        "height": size,
        "width": size,
        "use_resolution_binning": False,
        "latents": latent / pipe.scheduler.init_noise_sigma,
        "timesteps": timesteps,
    }
