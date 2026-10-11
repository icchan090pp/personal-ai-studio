"""Bounded-memory image/video exports; high resolutions use Lanczos upscaling."""

from PIL import Image, ImageOps

RESOLUTIONS = {
    "custom": (2100, 2050),
    "native": None,
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}


def resize_output(image, resolution):
    size = RESOLUTIONS[resolution]
    image = image.convert("RGB")
    return ImageOps.pad(image, size, method=Image.Resampling.LANCZOS) if size else image


def save_image(image, destination, resolution="720p"):
    resize_output(image, resolution).save(destination, format="PNG")


def save_video(frames, destination, resolution="720p", fps=8):
    import imageio.v2 as imageio
    import numpy as np

    with imageio.get_writer(
        str(destination), fps=fps, codec="libx264", quality=9, macro_block_size=1
    ) as writer:
        for frame in frames:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(np.asarray(frame, dtype=np.uint8))
            writer.append_data(np.asarray(resize_output(frame, resolution)))
