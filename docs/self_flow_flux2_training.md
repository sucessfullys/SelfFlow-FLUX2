# FLUX.2 Klein Self-Flow Full-Parameter Training

This implementation adds Self-Flow-style text-to-image training to the existing
DiffSynth-Studio FLUX.2 pipeline. It trains every parameter of `pipe.dit`.
The VAE and text encoder remain frozen. The projection head is also optimized.

## Method

For every sample, the trainer draws two scheduler timesteps `t` and `s`, then
selects 25 percent of image latent tokens. Selected tokens use `s`; the rest use
`t`. The student receives the resulting mixed noisy latent and a per-image-token
timestep tensor. The EMA teacher receives the same noise realization at
`min(t, s)`.

The loss is:

```text
L = L_gen + gamma * L_rep
L_rep = -mean(cosine(project(student_hidden), stopgrad(teacher_hidden)))
```

`L_gen` uses the repository's rectified-flow target `noise - clean_latent` and
the existing FLUX.2 scheduler weights.

FLUX.2 Klein-4B has 5 double-stream blocks followed by 20 single-stream blocks.
They are indexed as one logical sequence `[0, 24]`. Ratio `r` maps to
`round(r * 24)`, so the defaults select student layer 7 and teacher layer 17.
Only image-token hidden states are aligned. Text tokens use the base timestep
`t` for student modulation because they are conditioning tokens, not noised
data tokens.

## Dataset Adapter

The default configuration uses `DummyTextImageDataset`. For real data, set
`dataset_type: metadata` and provide CSV, JSON, or JSONL metadata. Each record
must contain an image path and caption:

```json
{"image": "images/000001.jpg", "prompt": "a red bicycle"}
```

Set `image_root`, `image_column`, and `caption_column` as needed. The current
pipeline preprocessing path intentionally requires micro batch size 1; use
gradient accumulation to increase the effective batch.

The default real-data preprocessing follows DiffSynth-Studio's native FLUX.2
training path: `height` and `width` are unset, aspect ratio is preserved,
images larger than `max_pixels` are downscaled, and both dimensions are aligned
down to multiples of 16. Set both `height` and `width` only when fixed-resolution
center-crop training is explicitly desired.

## Smoke Test

The full-model smoke test uses dummy data, 128x128 images, and two optimizer
steps:

```bash
bash scripts/train_flux2_klein_4b_self_flow.sh smoke
```

The lightweight CPU regression test does not load the 4B checkpoint:

```bash
pytest -q tests/test_flux2_self_flow.py
```

## Full Training

Edit the metadata fields in
`configs/train/flux2_klein_4b_self_flow.yaml`, or pass them on the command line:

```bash
bash scripts/train_flux2_klein_4b_self_flow.sh train \
  --metadata_path /path/to/metadata.jsonl \
  --image_root /path/to/dataset \
  --max_steps 1000 \
  --gradient_accumulation_steps 4
```

The launch configuration uses Accelerate with 8 processes, bf16, DeepSpeed
ZeRO-3, and activation checkpointing. No interactive `accelerate config` step
is required.

## Checkpoints and Resume

Each `checkpoint-N` directory contains:

- Accelerate/DeepSpeed training state for exact resume.
- `student.safetensors`, directly loadable with `pipe.dit.load_state_dict`.
- `self_flow_projector.safetensors`.
- `ema_teacher.safetensors` when `--save_ema_teacher` is enabled.
- `trainer_state.json` with the optimizer-step count.

Resume with:

```bash
bash scripts/train_flux2_klein_4b_self_flow.sh train \
  --metadata_path /path/to/metadata.jsonl \
  --image_root /path/to/dataset \
  --resume_from_checkpoint /path/to/checkpoint-500
```

## Memory Notes

The EMA teacher adds another bf16 denoiser copy and a second forward pass.
The default `max_pixels` is 1024x1024, matching the repository's FLUX.2 full
training example. For OOM failures, reduce `max_pixels` first, keep micro batch
size 1, increase gradient accumulation, shorten captions, or add
parameter/optimizer CPU offload to the DeepSpeed JSON. Exporting gathered
student or EMA weights also needs host RAM.

## Differences From the Paper

The paper's released code targets class-conditioned SiT and does not include its
training loop. This port applies the published Self-Flow objective to FLUX.2's
joint text-image transformer. It aligns image tokens only, uses FLUX.2's native
flow schedule and weighting, and uses `t` for non-noised text-token modulation.
The EMA teacher is distributed through the same Accelerate/DeepSpeed model
wrapper rather than a separate teacher service.
