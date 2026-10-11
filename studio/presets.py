"""Sampling quality presets, shared by image and video generation."""

LEVELS = {
    "normal": {
        "name": "ノーマル",
        "image_steps": 20,
        "video_steps": 20,
        "image_guidance": 6.5,
        "video_guidance": 5.0,
        "training_steps": 500,
        "rank": 8,
        "learning_rate": 0.0001,
    },
    "power": {
        "name": "パワー",
        "image_steps": 35,
        "video_steps": 35,
        "image_guidance": 7.5,
        "video_guidance": 6.0,
        "training_steps": 1000,
        "rank": 16,
        "learning_rate": 0.000075,
    },
    "excellent": {
        "name": "エクセレント（深考）",
        "image_steps": 50,
        "video_steps": 50,
        "image_guidance": 8.5,
        "video_guidance": 7.0,
        "training_steps": 2000,
        "rank": 32,
        "learning_rate": 0.00005,
    },
    "ultra": {
        "name": "ウルトラ（レベル200）",
        "image_steps": 200,
        "video_steps": 200,
        "image_guidance": 8.5,
        "video_guidance": 7.0,
        "training_steps": 5000,
        "rank": 32,
        "learning_rate": 0.00005,
    },
}
