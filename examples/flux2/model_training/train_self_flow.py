import argparse
import csv
import io
import json
import os
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core import ModelConfig
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DummyTextImageDataset(Dataset):
    def __init__(self, length=8, height=256, width=256, seed=0):
        self.length = length
        self.height = height
        self.width = width
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.randint(
            0,
            256,
            (self.height, self.width, 3),
            generator=generator,
            dtype=torch.uint8,
        ).numpy()
        return {
            "image": Image.fromarray(image, mode="RGB"),
            "prompt": f"a synthetic smoke test image number {index}",
        }


class ImageCaptionDataset(Dataset):
    """Metadata adapter for CSV, JSON, or JSONL image-caption datasets."""

    def __init__(
        self,
        metadata_path,
        image_root="",
        image_column="image",
        caption_column="prompt",
        height=None,
        width=None,
        max_pixels=1024 * 1024,
    ):
        self.metadata_path = Path(metadata_path)
        self.image_root = Path(image_root)
        self.image_column = image_column
        self.caption_column = caption_column
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.image_processor = ImageCropAndResize(
            height=height,
            width=width,
            max_pixels=max_pixels,
            height_division_factor=16,
            width_division_factor=16,
        )
        suffix = self.metadata_path.suffix.lower()
        if suffix == ".csv":
            with self.metadata_path.open("r", encoding="utf-8") as file:
                self.records = list(csv.DictReader(file))
        elif suffix == ".jsonl":
            with self.metadata_path.open("r", encoding="utf-8") as file:
                self.records = [json.loads(line) for line in file if line.strip()]
        elif suffix == ".json":
            with self.metadata_path.open("r", encoding="utf-8") as file:
                self.records = json.load(file)
        else:
            raise ValueError("metadata_path must end in .csv, .json, or .jsonl")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.image_root / record[self.image_column]
        with Image.open(image_path) as image:
            image = self._prepare_image(image)
        return {
            "image": image,
            "prompt": str(record[self.caption_column]),
        }

    def _prepare_image(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        return self.image_processor(image)


class TarImageCaptionDataset(ImageCaptionDataset):
    """Read image-caption samples from tar archives without modifying the source."""

    def __init__(self, *args, tar_column="tar_file", tar_cache_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.tar_column = tar_column
        self.tar_cache_size = tar_cache_size
        self._tar_cache = OrderedDict()

    def _get_tar(self, tar_path):
        tar_path = str(tar_path)
        archive = self._tar_cache.pop(tar_path, None)
        if archive is None:
            archive = tarfile.open(tar_path, mode="r:*")
        self._tar_cache[tar_path] = archive
        while len(self._tar_cache) > self.tar_cache_size:
            _, oldest = self._tar_cache.popitem(last=False)
            oldest.close()
        return archive

    def __getitem__(self, index):
        record = self.records[index]
        archive = self._get_tar(record[self.tar_column])
        member = archive.extractfile(record[self.image_column])
        if member is None:
            raise FileNotFoundError(
                f"{record[self.image_column]} not found in {record[self.tar_column]}"
            )
        image_bytes = member.read()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = self._prepare_image(image)
        return {
            "image": image,
            "prompt": str(record[self.caption_column]),
        }

    def __del__(self):
        for archive in getattr(self, "_tar_cache", {}).values():
            archive.close()


class SelfFlowProjectionHead(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear_2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, hidden_states):
        return self.linear_2(F.silu(self.linear_1(hidden_states)))


def layer_index_from_ratio(dit, ratio):
    num_layers = len(dit.transformer_blocks) + len(dit.single_transformer_blocks)
    if num_layers == 0:
        raise ValueError("FLUX.2 DiT has no transformer blocks.")
    return int(round(float(ratio) * (num_layers - 1)))


def build_token_mask(batch_size, num_tokens, mask_ratio, device, generator=None):
    masked_tokens = max(1, min(num_tokens, int(round(num_tokens * mask_ratio))))
    scores = torch.rand(
        batch_size,
        num_tokens,
        device=device,
        generator=generator,
    )
    indices = scores.topk(masked_tokens, dim=1).indices
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    return mask.scatter_(1, indices, True)


def dual_timestep_noising(clean_latents, noise, sigma_t, sigma_s, mask):
    sigma_t = sigma_t[:, None]
    sigma_s = sigma_s[:, None]
    sigma_tau = torch.where(mask, sigma_s, sigma_t).unsqueeze(-1)
    mixed = (1.0 - sigma_tau) * clean_latents + sigma_tau * noise
    sigma_min = torch.minimum(sigma_t, sigma_s).unsqueeze(-1)
    cleaner = (1.0 - sigma_min) * clean_latents + sigma_min * noise
    return mixed, cleaner, sigma_tau


@torch.no_grad()
def update_ema_teacher(student, teacher, decay):
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher.named_parameters():
        student_param = student_params[name]
        teacher_param.data.mul_(decay).add_(
            student_param.data.to(teacher_param.dtype),
            alpha=1.0 - decay,
        )
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        if name in student_buffers:
            teacher_buffer.copy_(student_buffers[name])


class Flux2SelfFlowTrainingModule(torch.nn.Module):
    def __init__(
        self,
        pipe,
        teacher_dit,
        gamma=0.8,
        ema_decay=0.9999,
        mask_ratio=0.25,
        student_layer_ratio=0.3,
        teacher_layer_ratio=0.7,
        use_gradient_checkpointing=True,
    ):
        super().__init__()
        self.pipe = pipe
        self.pipe.freeze_except(["dit"])
        self.teacher_dit = teacher_dit.eval().requires_grad_(False)
        self.projector = SelfFlowProjectionHead(self.pipe.dit.inner_dim).to(
            dtype=self.pipe.torch_dtype,
            device=self.pipe.device,
        )
        self.gamma = gamma
        self.ema_decay = ema_decay
        self.mask_ratio = mask_ratio
        self.student_layer = layer_index_from_ratio(
            self.pipe.dit,
            student_layer_ratio,
        )
        self.teacher_layer = layer_index_from_ratio(
            self.teacher_dit,
            teacher_layer_ratio,
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def train(self, mode=True):
        super().train(mode)
        self.pipe.eval()
        self.pipe.dit.train(mode)
        self.projector.train(mode)
        self.teacher_dit.eval()
        return self

    def trainable_parameters(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _pipeline_inputs(self, sample):
        inputs_shared = {
            "input_image": sample["image"],
            "height": sample["image"].size[1],
            "width": sample["image"].size[0],
            "embedded_guidance": 1.0,
            "cfg_scale": 1.0,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": False,
        }
        inputs_posi = {"prompt": sample["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit,
                self.pipe,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )
        return {**inputs_shared, **inputs_posi}

    def _dit_forward(
        self,
        dit,
        latents,
        timesteps,
        text_timesteps,
        inputs,
        layer_index,
    ):
        output, hidden = dit(
            hidden_states=latents,
            timestep=timesteps / 1000.0,
            text_timestep=text_timesteps / 1000.0,
            guidance=None,
            encoder_hidden_states=inputs["prompt_embeds"],
            txt_ids=inputs["text_ids"],
            img_ids=inputs["image_ids"],
            use_gradient_checkpointing=(
                self.use_gradient_checkpointing and dit is self.pipe.dit
            ),
            return_hidden_state_at=layer_index,
        )
        return output[:, : latents.shape[1]], hidden

    def forward(self, sample):
        inputs = self._pipeline_inputs(sample)
        clean_latents = inputs["input_latents"]
        batch_size, num_tokens, _ = clean_latents.shape
        device = clean_latents.device

        timestep_count = len(self.pipe.scheduler.timesteps)
        timestep_ids = torch.randint(
            0,
            timestep_count,
            (batch_size, 2),
            device=device,
        )
        scheduler_timesteps = self.pipe.scheduler.timesteps.to(device)
        scheduler_sigmas = self.pipe.scheduler.sigmas.to(
            device=device,
            dtype=clean_latents.dtype,
        )
        t_ids, s_ids = timestep_ids.unbind(dim=1)
        timestep_t = scheduler_timesteps[t_ids]
        timestep_s = scheduler_timesteps[s_ids]
        sigma_t = scheduler_sigmas[t_ids]
        sigma_s = scheduler_sigmas[s_ids]

        noise = torch.randn_like(clean_latents)
        mask = build_token_mask(
            batch_size,
            num_tokens,
            self.mask_ratio,
            device,
        )
        mixed_latents, cleaner_latents, _ = dual_timestep_noising(
            clean_latents,
            noise,
            sigma_t,
            sigma_s,
            mask,
        )
        token_timesteps = torch.where(
            mask,
            timestep_s[:, None],
            timestep_t[:, None],
        )
        cleaner_timesteps = torch.minimum(timestep_t, timestep_s)

        prediction, student_hidden = self._dit_forward(
            self.pipe.dit,
            mixed_latents,
            token_timesteps,
            timestep_t,
            inputs,
            self.student_layer,
        )
        with torch.no_grad():
            _, teacher_hidden = self._dit_forward(
                self.teacher_dit,
                cleaner_latents,
                cleaner_timesteps,
                cleaner_timesteps,
                inputs,
                self.teacher_layer,
            )

        target = noise - clean_latents
        timestep_weights = self.pipe.scheduler.linear_timesteps_weights.to(
            device=device,
            dtype=torch.float32,
        )
        token_weights = torch.where(
            mask,
            timestep_weights[s_ids, None],
            timestep_weights[t_ids, None],
        )
        per_token_mse = (prediction.float() - target.float()).square().mean(dim=-1)
        loss_gen = (per_token_mse * token_weights).mean()

        projected_student = self.projector(student_hidden)
        loss_rep = -F.cosine_similarity(
            projected_student.float(),
            teacher_hidden.detach().float(),
            dim=-1,
        ).mean()
        loss = loss_gen + self.gamma * loss_rep
        return {
            "loss": loss,
            "loss_gen": loss_gen.detach(),
            "loss_rep": loss_rep.detach(),
            "mask_ratio": mask.float().mean().detach(),
            "t_mean": timestep_t.float().mean().detach(),
            "s_mean": timestep_s.float().mean().detach(),
        }


def load_config_defaults(parser):
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with open(preliminary.config, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        parser.set_defaults(**config)
    return parser.parse_args()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Full-parameter Self-Flow training for FLUX.2 Klein."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--base_model", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument(
        "--dataset_type",
        choices=["dummy", "metadata", "metadata_tar"],
        default="dummy",
    )
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="prompt")
    parser.add_argument("--tar_column", type=str, default="tar_file")
    parser.add_argument("--tar_cache_size", type=int, default=8)
    parser.add_argument("--dummy_length", type=int, default=8)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--student_layer_ratio", type=float, default=0.3)
    parser.add_argument("--teacher_layer_ratio", type=float, default=0.7)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--save_ema_teacher", action="store_true")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--log_every", type=int, default=1)
    return parser


def build_dataset(args):
    if args.dataset_type == "dummy":
        return DummyTextImageDataset(
            length=args.dummy_length,
            height=args.height or 256,
            width=args.width or 256,
            seed=args.seed,
        )
    if not args.metadata_path:
        raise ValueError("--metadata_path is required for dataset_type=metadata")
    dataset_class = (
        TarImageCaptionDataset
        if args.dataset_type == "metadata_tar"
        else ImageCaptionDataset
    )
    dataset_kwargs = dict(
        metadata_path=args.metadata_path,
        image_root=args.image_root,
        image_column=args.image_column,
        caption_column=args.caption_column,
        height=args.height,
        width=args.width,
        max_pixels=args.max_pixels,
    )
    if args.dataset_type == "metadata_tar":
        dataset_kwargs.update(
            tar_column=args.tar_column,
            tar_cache_size=args.tar_cache_size,
        )
    return dataset_class(**dataset_kwargs)


def collate_single_sample(samples):
    if len(samples) != 1:
        raise ValueError(
            "The current FLUX.2 PIL preprocessing path supports micro batch size 1. "
            "Use gradient accumulation for a larger effective batch."
        )
    return samples[0]


def build_pipeline_and_teacher(base_model, device):
    base = Path(base_model)
    text_encoder_files = sorted(str(path) for path in (base / "text_encoder").glob("*.safetensors"))
    if not text_encoder_files:
        raise FileNotFoundError(f"No text encoder weights found under {base / 'text_encoder'}")
    model_configs = [
        ModelConfig(path=text_encoder_files),
        ModelConfig(path=str(base / "transformer" / "diffusion_pytorch_model.safetensors")),
        ModelConfig(path=str(base / "vae" / "diffusion_pytorch_model.safetensors")),
    ]
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=str(base / "tokenizer")),
    )
    pipe.scheduler.set_timesteps(1000, training=True)
    teacher_pool = pipe.download_and_load_models(
        [
            ModelConfig(
                path=str(
                    base / "transformer" / "diffusion_pytorch_model.safetensors"
                )
            )
        ]
    )
    teacher_dit = teacher_pool.fetch_model("flux2_dit")
    return pipe, teacher_dit


def state_dict_subset(state_dict, prefix):
    return {
        key[len(prefix):]: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def save_training_checkpoint(
    accelerator,
    model,
    output_dir,
    global_step,
    save_ema_teacher,
):
    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(str(checkpoint_dir))
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        student = state_dict_subset(state_dict, "pipe.dit.")
        projector = state_dict_subset(state_dict, "projector.")
        save_file(student, str(checkpoint_dir / "student.safetensors"))
        save_file(projector, str(checkpoint_dir / "self_flow_projector.safetensors"))
        if save_ema_teacher:
            teacher = state_dict_subset(state_dict, "teacher_dit.")
            save_file(teacher, str(checkpoint_dir / "ema_teacher.safetensors"))
        with (checkpoint_dir / "trainer_state.json").open("w", encoding="utf-8") as file:
            json.dump({"global_step": global_step}, file)
    accelerator.wait_for_everyone()


def main():
    args = load_config_defaults(build_parser())
    if not args.base_model or not args.output_dir:
        raise ValueError("base_model and output_dir must be set by config or CLI.")
    if args.train_batch_size != 1:
        raise ValueError("Set train_batch_size=1 and scale with gradient accumulation.")

    accelerator_kwargs = {
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)
        ],
    }
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() != "true":
        accelerator_kwargs.update(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
        )
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    accelerate.utils.set_seed(args.seed, device_specific=True)

    pipe, teacher_dit = build_pipeline_and_teacher(
        args.base_model,
        accelerator.device,
    )
    model = Flux2SelfFlowTrainingModule(
        pipe=pipe,
        teacher_dit=teacher_dit,
        gamma=args.gamma,
        ema_decay=args.ema_decay,
        mask_ratio=args.mask_ratio,
        student_layer_ratio=args.student_layer_ratio,
        teacher_layer_ratio=args.teacher_layer_ratio,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    dataset = build_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_single_sample,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        lr_scheduler,
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)

    global_step = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        state_path = Path(args.resume_from_checkpoint) / "trainer_state.json"
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as file:
                global_step = int(json.load(file)["global_step"])

    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        print(
            "Self-Flow logical layers: "
            f"student={unwrapped.student_layer}, teacher={unwrapped.teacher_layer}; "
            f"gamma={args.gamma}, mask_ratio={args.mask_ratio}, ema={args.ema_decay}"
        )

    model.train()
    while global_step < args.max_steps:
        for sample in dataloader:
            with accelerator.accumulate(model):
                losses = model(sample)
                accelerator.backward(losses["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unwrapped.trainable_parameters(),
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema_teacher(
                        unwrapped.pipe.dit,
                        unwrapped.teacher_dit,
                        args.ema_decay,
                    )
                    global_step += 1
                    if accelerator.is_main_process and global_step % args.log_every == 0:
                        print(
                            f"step={global_step} "
                            f"loss={losses['loss'].item():.6f} "
                            f"loss_gen={losses['loss_gen'].item():.6f} "
                            f"loss_rep={losses['loss_rep'].item():.6f} "
                            f"mask={losses['mask_ratio'].item():.4f} "
                            f"t={losses['t_mean'].item():.1f} "
                            f"s={losses['s_mean'].item():.1f}"
                        )
                    if (
                        args.checkpointing_steps > 0
                        and global_step % args.checkpointing_steps == 0
                    ):
                        save_training_checkpoint(
                            accelerator,
                            model,
                            args.output_dir,
                            global_step,
                            args.save_ema_teacher,
                        )
            if global_step >= args.max_steps:
                break

    if (
        args.checkpointing_steps <= 0
        or global_step % args.checkpointing_steps != 0
    ):
        save_training_checkpoint(
            accelerator,
            model,
            args.output_dir,
            global_step,
            args.save_ema_teacher,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
