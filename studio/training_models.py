"""Pinned, architecture-compatible bases for image LoRA personalization."""

IMAGE_TRAINING_MODELS = {
    "tiny": {
        "id": "segmind/tiny-sd",
        "name": "Tiny-SD / 軽量・CPU学習",
        "revision": "cad0bd7495fa6c4bcca01b19a723dc91627fe84f",
        "backend": "cpu",
        "ram_gb": 8,
        "resolution": 256,
        "steps": 30,
        "safetensors": False,
        "description": "まずはこちら。AMD・Intel搭載PCでもCPUで追加学習します。RAM 8GB以上が目安。",
    },
    "small": {
        "id": "segmind/small-sd",
        "name": "Small-SD / CPU学習",
        "revision": "f8b087e68071a29bbe85a4b07e6e1e8998d59aa7",
        "backend": "cpu",
        "ram_gb": 12,
        "resolution": 256,
        "steps": 30,
        "safetensors": False,
        "description": "Tiny-SDより大きい画像モデル。CPU学習対応、RAM 12GB以上が目安。処理時間も増えます。",
    },
    "illustration": {
        "id": "Lykon/dreamshaper-8",
        "name": "DreamShaper 8 / アニメ・イラスト / CPU学習",
        "revision": "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917",
        "backend": "cpu",
        "ram_gb": 16,
        "resolution": 256,
        "steps": 30,
        "safetensors": True,
        "description": "アニメ・イラスト・写実の表現を学習済みの公開モデルから育成。CPU・RAM 16GB以上が目安。Tiny-SDより取得容量・処理時間が増えます。",
    },
    "sd15": {
        "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "name": "Stable Diffusion 1.5 / CUDA学習",
        "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "backend": "cuda",
        "ram_gb": 16,
        "resolution": 512,
        "steps": 500,
        "safetensors": True,
        "description": "汎用画像モデル。NVIDIA CUDA・GPUメモリ16GB以上が目安。",
    },
    "dreamshaper": {
        "id": "Lykon/dreamshaper-8",
        "name": "DreamShaper 8 / CUDA学習",
        "revision": "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917",
        "backend": "cuda",
        "ram_gb": 16,
        "resolution": 512,
        "steps": 500,
        "safetensors": True,
        "description": "SD 1.5派生のイラスト・写実モデル。NVIDIA CUDA・GPUメモリ16GB以上が目安。",
    },
}

VIDEO_TRAINING_MODELS = {
    "animatediff": {
        "id": "Lykon/dreamshaper-8",
        "name": "AnimateDiff + DreamShaper 8 / CPU動画学習",
        "revision": "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917",
        "motion_model": "guoyww/animatediff-motion-adapter-v1-5-2",
        "motion_revision": "6167b88ffe39b4441fdf2113e77b99a6f56b7906",
        "backend": "cpu",
        "ram_gb": 16,
        "resolution": 256,
        "steps": 10,
        "license": "DreamShaper: CreativeML OpenRAIL-M / AnimateDiff: 配布元の利用条件を確認",
        "description": "ノートPC向けの短い動画。CPUで映像の時間方向を含むLoRAを学習。RAM 16GB以上・数分〜数十分以上。Kokoroの日本語音声を後から合成できます。",
    },
    "cogvideo": {
        "id": "THUDM/CogVideoX-5b-I2V",
        "name": "CogVideoX 5B I2V / CUDA動画学習",
        "backend": "cuda",
        "ram_gb": 32,
        "steps": 500,
        "license": "CogVideoX License（配布元を確認）",
        "description": "生成画像から約6秒の動画。LoRA学習は専用GPU向け、GPUメモリ48〜80GBが目安です。",
    },
}


def video_catalog():
    return [
        {
            **value,
            "key": key,
            "kind": "video",
            "trainable": True,
            "weights_url": "https://huggingface.co/" + value.get("motion_model", value["id"]),
        }
        for key, value in VIDEO_TRAINING_MODELS.items()
    ]


def catalog():
    return [
        {
            **value,
            "key": key,
            "kind": "image",
            "trainable": True,
            "license": "CreativeML OpenRAIL-M",
            "weights_url": "https://huggingface.co/" + value["id"],
        }
        for key, value in IMAGE_TRAINING_MODELS.items()
    ]
