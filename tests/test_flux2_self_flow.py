import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path

import torch
from PIL import Image

os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

from diffsynth.models.flux2_dit import Flux2DiT


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "flux2"
    / "model_training"
    / "train_self_flow.py"
)
SPEC = importlib.util.spec_from_file_location("train_self_flow", SCRIPT_PATH)
TRAIN_SELF_FLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_SELF_FLOW)


def tiny_dit():
    return Flux2DiT(
        in_channels=8,
        out_channels=8,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        timestep_guidance_channels=16,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )


def test_dual_timestep_noising_and_mask():
    clean = torch.zeros(2, 8, 4)
    noise = torch.ones_like(clean)
    mask = TRAIN_SELF_FLOW.build_token_mask(2, 8, 0.25, clean.device)
    mixed, cleaner, sigma_tau = TRAIN_SELF_FLOW.dual_timestep_noising(
        clean,
        noise,
        torch.tensor([0.8, 0.6]),
        torch.tensor([0.2, 0.4]),
        mask,
    )
    assert mask.sum(dim=1).tolist() == [2, 2]
    assert mixed.shape == clean.shape
    assert cleaner.shape == clean.shape
    assert sigma_tau.shape == (2, 8, 1)
    assert torch.allclose(cleaner[0], torch.full_like(cleaner[0], 0.2))


def test_flux2_per_token_timestep_hidden_and_backward():
    model = tiny_dit()
    batch, image_tokens, text_tokens = 2, 4, 3
    output, hidden = model(
        hidden_states=torch.randn(batch, image_tokens, 8),
        encoder_hidden_states=torch.randn(batch, text_tokens, 12),
        timestep=torch.rand(batch, image_tokens),
        img_ids=torch.zeros(1, image_tokens, 4),
        txt_ids=torch.zeros(1, text_tokens, 4),
        return_hidden_state_at=2,
    )
    loss = output.square().mean() + hidden.square().mean()
    loss.backward()
    assert output.shape == (batch, image_tokens, 8)
    assert hidden.shape == (batch, image_tokens, 16)
    assert model.x_embedder.weight.grad is not None


def test_ema_update():
    student = torch.nn.Linear(2, 2, bias=False)
    teacher = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
        teacher.weight.zero_()
    TRAIN_SELF_FLOW.update_ema_teacher(student, teacher, 0.75)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.5))


def test_tar_image_caption_dataset_is_read_only(tmp_path):
    image = Image.new("RGB", (24, 16), color=(10, 20, 30))
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()

    tar_path = tmp_path / "images.tar"
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("sample.jpg")
        info.size = len(image_bytes)
        archive.addfile(info, io.BytesIO(image_bytes))

    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "image": "sample.jpg",
                "tar_file": str(tar_path),
                "caption": "a test image",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_mtime = metadata_path.stat().st_mtime_ns
    tar_mtime = tar_path.stat().st_mtime_ns

    dataset = TRAIN_SELF_FLOW.TarImageCaptionDataset(
        metadata_path=metadata_path,
        image_column="image",
        caption_column="caption",
        tar_column="tar_file",
        height=32,
        width=48,
    )
    sample = dataset[0]

    assert sample["image"].size == (48, 32)
    assert sample["prompt"] == "a test image"
    assert metadata_path.stat().st_mtime_ns == metadata_mtime
    assert tar_path.stat().st_mtime_ns == tar_mtime


def test_flux2_dynamic_resolution_preserves_aspect_ratio(tmp_path):
    image_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (800, 1200), color=(10, 20, 30)).save(image_path)
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps({"image": image_path.name, "caption": "a portrait"}) + "\n",
        encoding="utf-8",
    )

    dataset = TRAIN_SELF_FLOW.ImageCaptionDataset(
        metadata_path=metadata_path,
        image_root=tmp_path,
        caption_column="caption",
        height=None,
        width=None,
        max_pixels=512 * 512,
    )
    sample = dataset[0]
    width, height = sample["image"].size

    assert width * height <= 512 * 512
    assert width % 16 == 0
    assert height % 16 == 0
    assert abs(width / height - 800 / 1200) < 0.03
