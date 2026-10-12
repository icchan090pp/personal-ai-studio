"""CPU tiled Real-ESRGAN restoration. RRDB architecture adapted from BasicSR.

Copyright 2018-2022 BasicSR Authors, Apache-2.0; see docs/licenses/BasicSR.txt.
Adapted to x4 inference, strict weights-only loading and bounded CPU tiles.
Real-ESRGAN pretrained weights: Xintao Wang, BSD-3-Clause.
"""

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageFilter

MODELS = {
    "anime": {
        "name": "Real-ESRGAN / アニメ・イラスト",
        "blocks": 6,
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "release": "v0.2.2.4",
        "sha256": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    },
    "photo": {
        "name": "Real-ESRGAN / 写真・汎用",
        "blocks": 23,
        "file": "RealESRGAN_x4plus.pth",
        "release": "v0.1.0",
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    },
}


def network(blocks):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            for index in range(1, 6):
                setattr(
                    self,
                    f"conv{index}",
                    nn.Conv2d(64 + (index - 1) * 32, 32 if index < 5 else 64, 3, 1, 1),
                )
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            values = [x]
            for index in range(1, 5):
                values.append(self.lrelu(getattr(self, f"conv{index}")(torch.cat(values, 1))))
            return self.conv5(torch.cat(values, 1)) * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = Dense(), Dense(), Dense()

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
            self.body = nn.Sequential(*(RRDB() for _ in range(blocks)))
            for name in ("conv_body", "conv_up1", "conv_up2", "conv_hr"):
                setattr(self, name, nn.Conv2d(64, 64, 3, 1, 1))
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            first = self.conv_first(x)
            x = first + self.conv_body(self.body(first))
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(x)))

    return RRDBNet()


def weights(root, model):
    path = Path(root) / "models" / model["file"]
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == model["sha256"]:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/{model['release']}/{model['file']}"
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 80 * 1024 * 1024:
                    raise ValueError("高精細化モデルの容量上限を超えています")
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError("高精細化モデルの検証に失敗しました")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def tiled(model, image, progress=lambda _value: None, tile=96, padding=16):
    import torch

    values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    height, width = values.shape[:2]
    output = np.empty((height * 4, width * 4, 3), dtype=np.uint8)
    count = ((height + tile - 1) // tile) * ((width + tile - 1) // tile)
    done = 0
    with torch.inference_mode():
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                y1, x1 = min(y + tile, height), min(x + tile, width)
                top, left = max(0, y - padding), max(0, x - padding)
                bottom, right = min(height, y1 + padding), min(width, x1 + padding)
                tensor = torch.from_numpy(
                    values[top:bottom, left:right].transpose(2, 0, 1).copy()
                ).unsqueeze(0)
                restored = model(tensor)
                if not torch.isfinite(restored).all():
                    raise ValueError("高精細化の計算結果が不正です")
                pixels = restored[
                    0, :, (y - top) * 4 : (y1 - top) * 4, (x - left) * 4 : (x1 - left) * 4
                ].clamp(0, 1)
                output[y * 4 : y1 * 4, x * 4 : x1 * 4] = (
                    pixels.permute(1, 2, 0).mul(255).round().byte().numpy()
                )
                done += 1
                progress(5 + done * 90 // count)
    return Image.fromarray(output)


def enhance(image, root, model="anime", scale=4, denoise=False, progress=lambda _value: None):
    import torch

    if model not in MODELS or scale not in {2, 4}:
        raise ValueError("高精細化の設定を確認してください")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    spec = MODELS[model]
    original_size = list(image.size)
    working = image.convert("RGB")
    working.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    if denoise:
        working = working.filter(ImageFilter.MedianFilter(3))
    net = network(spec["blocks"])
    checkpoint = torch.load(weights(root, spec), map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint.get("params_ema", checkpoint.get("params")), strict=True)
    net.eval()
    output = tiled(net, working, progress)
    if scale == 2:
        output = output.resize((working.width * 2, working.height * 2), Image.Resampling.LANCZOS)
    return output, {
        "backend": "cpu",
        "model": spec["name"],
        "weights_sha256": spec["sha256"],
        "source_size": original_size,
        "working_size": list(working.size),
        "output_size": list(output.size),
        "scale": scale,
        "denoise": denoise,
    }
