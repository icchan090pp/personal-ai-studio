"""Pretrained sound generation and speech synthesis, separate from sound training."""

AUDIO_MODELS = {
    "audioldm2": {
        "id": "cvssp/audioldm2-large",
        "name": "AudioLDM2 Large / 効果音・環境音",
        "license": "CC-BY-NC-SA-4.0（非商用）",
        "revision": "4b0b875a9e0c5305dfc917da808584e50e1c7ed4",
    },
    "kokoro": {
        "id": "hexgrad/Kokoro-82M",
        "name": "Kokoro 82M / 日本語読み上げ",
        "license": "Apache-2.0",
        "revision": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    },
}


def generate_audio(config, path):
    from studio.sound import write_audio

    if config["audio_model"] == "audioldm2":
        import torch
        from diffusers import AudioLDM2Pipeline

        model = AUDIO_MODELS["audioldm2"]
        pipe = AudioLDM2Pipeline.from_pretrained(
            model["id"], revision=model["revision"], torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        audio = pipe(
            prompt=config["audio_prompt"] or config["prompt"],
            negative_prompt=config["negative_prompt"],
            audio_length_in_s=6.125,
            num_inference_steps=config["inference_steps"],
            guidance_scale=3.5,
            generator=torch.Generator(device="cpu").manual_seed(config["seed"]),
        ).audios[0]
        write_audio(path, audio, rate=16000)
    elif config["audio_model"] == "kokoro":
        import numpy as np
        from huggingface_hub import hf_hub_download
        from kokoro import KModel, KPipeline

        model = AUDIO_MODELS["kokoro"]
        checkpoint = hf_hub_download(model["id"], "kokoro-v1_0.pth", revision=model["revision"])
        definition = hf_hub_download(model["id"], "config.json", revision=model["revision"])
        import torch

        network = (
            KModel(repo_id=model["id"], config=definition, model=checkpoint)
            .to("cuda" if torch.cuda.is_available() else "cpu")
            .eval()
        )
        pipeline = KPipeline(lang_code="j", model=network, repo_id=model["id"])
        voice = hf_hub_download(model["id"], "voices/jf_alpha.pt", revision=model["revision"])
        parts = []
        total = 0
        for _, _, audio in pipeline(
            config["audio_prompt"] or config["prompt"],
            voice=voice,
            speed=config.get("speech_speed", 1.0),
        ):
            data = audio.detach().cpu().numpy()
            parts.append(data[: max(0, 30 * 24000 - total)])
            total += len(data)
            if total >= 30 * 24000:
                break
        if not parts:
            raise ValueError("読み上げる文章を入力してください")
        write_audio(path, np.concatenate(parts), rate=24000)
    else:
        raise ValueError("対応する音声モデルを選択してください")
