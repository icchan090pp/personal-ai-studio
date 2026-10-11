"""Real CPU optimization and serialization, using a tiny random diffusion network.

This checks math and compatibility, not SD 1.5 or CogVideoX generation quality.
"""

import json
import zipfile

import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from transformers.models.clip.tokenization_clip import bytes_to_unicode

from studio.gpu import prepare_training_rows, save_test_sample, train_adapter
from tests.test_studio import png


def tiny_pipeline(tmp_path):
    torch.manual_seed(123)
    chars = list(bytes_to_unicode().values())
    vocab = {
        token: index
        for index, token in enumerate(
            chars + [char + "</w>" for char in chars] + ["<|startoftext|>", "<|endoftext|>"]
        )
    }
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))
    merges = tmp_path / "merges.txt"
    merges.write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(vocab_path), str(merges), model_max_length=77)
    text = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=77,
        )
    )
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        cross_attention_dim=32,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        norm_num_groups=8,
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(32, 64),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        norm_num_groups=8,
        sample_size=64,
    )
    return StableDiffusionPipeline(
        vae=vae,
        text_encoder=text,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDPMScheduler(num_train_timesteps=100, steps_offset=1, clip_sample=False),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )


def test_real_optimizer_export_reload_and_inference(tmp_path):
    torch.set_num_threads(1)
    pipe = tiny_pipeline(tmp_path)
    frozen_before = {key: value.detach().clone() for key, value in pipe.unet.named_parameters()}
    folder = tmp_path / "jobs/test"
    folder.mkdir(parents=True)
    dataset = tmp_path / "projects/model/images"
    dataset.mkdir(parents=True)
    (dataset / "sample.png").write_bytes(png())
    project = {"id": "model", "name": "numerical test", "trigger": "myconcept"}
    config = {
        "steps": 10,
        "rank": 4,
        "learning_rate": 0.001,
        "seed": 42,
        "base_model": "tiny-random-test-network",
        "images": [{"id": "sample", "caption": "a photo of myconcept"}],
    }
    progress = []
    metric_records = []
    train_adapter(
        pipe,
        tmp_path,
        project,
        config,
        folder,
        progress.append,
        device="cpu",
        resolution=64,
        report=metric_records.append,
    )
    state = get_peft_model_state_dict(pipe.unet)
    assert all(torch.isfinite(value).all() for value in state.values())
    assert any(value.abs().sum() > 0 for key, value in state.items() if "lora_B" in key)
    for key, value in pipe.unet.named_parameters():
        original_key = key.replace(".base_layer.", ".")
        if original_key in frozen_before:
            assert torch.equal(value, frozen_before[original_key]), original_key
    assert progress[-1] == 95
    assert [point["step"] for point in metric_records] == [1, 10]
    assert all(
        point["loss"] >= 0 and point["gradient_norm"] >= 0 and point["lora_norm"] > 0
        for point in metric_records
    )
    assert all(point["learning_rate"] == config["learning_rate"] for point in metric_records)
    with zipfile.ZipFile(folder / "adapter.zip") as archive:
        assert set(archive.namelist()) == {
            "pytorch_lora_weights.safetensors",
            "stability-matrix-lora.safetensors",
            "model-card.json",
        }
    kohya = load_file(folder / "adapter/stability-matrix-lora.safetensors")
    assert any(key.endswith(".alpha") and value.item() == 4 for key, value in kohya.items())
    assert all(key.startswith("lora_unet_") for key in kohya)
    restored = tiny_pipeline(tmp_path)
    restored.load_lora_weights(str(folder / "adapter"), adapter_name="default")
    restored_state = get_peft_model_state_dict(restored.unet)
    assert state.keys() == restored_state.keys()
    assert all(torch.equal(state[key], restored_state[key]) for key in state)
    # Diffusers also interprets the Kohya export successfully, with identical adapter weights.
    kohya_restored = tiny_pipeline(tmp_path)
    kohya_restored.load_lora_weights(
        str(folder / "adapter"),
        weight_name="stability-matrix-lora.safetensors",
        adapter_name="default",
    )
    kohya_state = get_peft_model_state_dict(kohya_restored.unet)
    assert state.keys() == kohya_state.keys()
    assert all(torch.equal(state[key], kohya_state[key]) for key in state)
    restored.unet.eval()
    image = restored(
        "a photo of myconcept",
        height=64,
        width=64,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(42),
    ).images[0]
    assert image.size == (64, 64)
    assert len(set(image.getdata())) > 1
    save_test_sample(
        restored, "myconcept", 42, folder / "test.png", device="cpu", resolution=64, steps=2
    )
    assert (folder / "test.png").is_file()
    # Exercise the shared numerical path for every additional training mode, including
    # continuing the already-loaded adapter for reward-weighted fine-tuning.
    for mode in ("semi_supervised", "style", "reinforcement", "generation_test", "imitation"):
        mode_pipe = restored if mode == "reinforcement" else tiny_pipeline(tmp_path)
        before = (
            {k: v.clone() for k, v in get_peft_model_state_dict(restored.unet).items()}
            if mode == "reinforcement"
            else None
        )
        mode_config = {
            **config,
            "mode": mode,
            "steps": 2,
            "images": [
                {"id": "sample", "caption": "a photo of myconcept", "labeled": 1, "rating": 5}
            ],
        }
        prepare_training_rows(tmp_path, project, mode_config)
        mode_folder = tmp_path / "jobs" / mode
        mode_folder.mkdir()
        train_adapter(
            mode_pipe,
            tmp_path,
            project,
            mode_config,
            mode_folder,
            lambda _v: None,
            device="cpu",
            resolution=64,
        )
        mode_state = get_peft_model_state_dict(mode_pipe.unet)
        assert all(torch.isfinite(value).all() for value in mode_state.values())
        assert (mode_folder / "adapter.zip").is_file()
        if before:
            assert any(not torch.equal(before[key], mode_state[key]) for key in before)
