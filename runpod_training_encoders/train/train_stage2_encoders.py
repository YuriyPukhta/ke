#!/usr/bin/env python3
"""Stage 2 Encoder-Only Trainer: joint fine-tune for text_encoder & style_encoder with GAN & SLM/WavLM.

Freezes decoder, bert, bert_encoder, predictor, predictor_encoder, text_aligner, pitch_extractor.
Updates text_encoder, style_encoder, mpd, msd, wd with gradient clipping.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from runpod_training_encoders.data.npz_dataset import build_npz_dataloader
from runpod_training_encoders.paths import ensure_styletts2_on_path, ensure_volume
from runpod_training_encoders.train.checkpointing import (
    apply_retention,
    load_checkpoint,
    resolve_resume,
    save_checkpoint,
    update_latest_pointer,
)
from runpod_training_encoders.train.freeze import apply_phase, trainable_param_count
from runpod_training_encoders.train.logging_utils import RunLogger
from runpod_training_encoders.train.model_factory import build_training_model, load_yaml
from runpod_training_encoders.train.schedulers import build_warmup_cosine_scheduler
from runpod_training_encoders.train.slmadv_nodiff import EchoStyleSampler

logger = logging.getLogger("runpod_training_encoders.stage2_encoders")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def build_optimizers(model: dict, lrs: dict[str, float]) -> dict[str, AdamW]:
    opts = {}
    for key, module in model.items():
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            continue
        lr = lrs.get(key, lrs.get("default", 1e-4))
        betas = (0.9, 0.99) if key in ("bert_encoder", "bert") else (0.0, 0.99)
        wd = 0.01 if key in ("bert_encoder", "bert") else 1e-4
        opts[key] = AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=1e-9)
    return opts


def clip_grad_norm(model: dict, keys: list[str], max_norm: float = 1.0) -> None:
    params = []
    for k in keys:
        if k in model:
            params.extend([p for p in model[k].parameters() if p.requires_grad and p.grad is not None])
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def step_optimizers(opts, keys=None):
    for key in keys or list(opts):
        if key in opts:
            opts[key].step()
            opts[key].zero_grad(set_to_none=True)


def run_stage2_encoders(config: dict, args: argparse.Namespace) -> None:
    ensure_styletts2_on_path()
    from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, WavLMLoss
    from utils import log_norm, maximum_path, mask_from_lens

    layout = ensure_volume(Path(config.get("run_root", "/workspace/kokoro-run-encoders")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(config.get("bf16", False)) and device.type == "cuda"
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
    stage_cfg = config["stage2"]
    data_cfg = config["data_params"]
    loss_params = config["loss_params"]

    train_loader = build_npz_dataloader(
        index_jsonl=data_cfg["train_index"],
        data_root=data_cfg["root_path"],
        targets_root=data_cfg.get("targets_root"),
        batch_size=config.get("batch_size", 12),
        num_workers=data_cfg.get("num_workers", 8),
        validation=False,
        ood_texts=data_cfg.get("OOD_data"),
        use_cached_f0n=bool(data_cfg.get("use_cached_f0n", False)),
        speaker_balanced=True,
        seed=config.get("seed", 42),
    )

    model, model_params = build_training_model(config, device)
    first_stage = Path(config.get("first_stage_path") or (layout["stage1"] / "first_stage.pth"))
    if not first_stage.is_file():
        raise FileNotFoundError(f"Stage 1 checkpoint missing: {first_stage}")

    load_checkpoint(
        first_stage,
        model,
        load_optimizer=False,
        ignore_modules=["predictor_encoder", "msd", "mpd", "wd", "diffusion"],
    )
    if "style_encoder" in model and "predictor_encoder" in model:
        model["predictor_encoder"] = copy.deepcopy(model["style_encoder"]).to(device)

    phase = "stage2_encoders_only"
    apply_phase(model, phase)

    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model["mpd"], model["msd"]).to(device)
    dl = DiscriminatorLoss(model["mpd"], model["msd"]).to(device)
    sr = config["preprocess_params"]["sr"]
    lambda_slm = float(loss_params.get("lambda_slm", 1))
    enable_slm = bool(config.get("slm", True)) and (lambda_slm > 0)
    wl = None
    if enable_slm:
        wl = WavLMLoss(model_params.slm.model, model["wd"], sr, model_params.slm.sr).to(device)

    run_logger = RunLogger(layout["root"], "stage2_encoders", enable_wandb=config.get("wandb", True))
    start_epoch = 0
    iters = 0
    resume_path = resolve_resume(args.resume, layout["state"], "stage2")
    if resume_path:
        start_epoch, iters, _ = load_checkpoint(resume_path, model, load_optimizer=False)
        logger.info("resumed stage2_encoders from %s epoch=%s iters=%s", resume_path, start_epoch, iters)

    epochs = int(stage_cfg.get("epochs", 10))
    max_len = int(config.get("max_len", 300))
    accum = int(config.get("grad_accum", 2))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    n_down = model["text_aligner"].n_down
    total_steps = epochs * max(1, len(train_loader))

    lambda_mel = float(loss_params.get("lambda_mel", 5))
    lambda_gen = float(loss_params.get("lambda_gen", 1))

    lrs = stage_cfg["lrs"]["joint"]
    opts = build_optimizers(model, lrs)
    schedulers = {
        k: build_warmup_cosine_scheduler(opt, total_steps, stage_cfg.get("warmup_ratio", 0.05))
        for k, opt in opts.items()
    }
    logger.info("Stage 2 Encoders-Only starting: epochs=%s trainable=%.2fM modules=%s max_grad_norm=%s", epochs, trainable_param_count(model) / 1e6, list(opts.keys()), max_grad_norm)

    for epoch in range(start_epoch, epochs):
        apply_phase(model, phase)
        for key in model:
            if key in opts:
                model[key].train()
            else:
                model[key].eval()

        running = {"mel": 0.0, "gen": 0.0, "disc": 0.0, "slm": 0.0}
        n_batches = 0

        for i, batch in enumerate(train_loader):
            waves = batch[0]
            texts = batch[1].to(device)
            input_lengths = batch[2].to(device)
            mels = batch[5].to(device)
            mel_input_length = batch[6].to(device)

            with autocast, torch.no_grad():
                mask = length_to_mask(mel_input_length // (2**n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)
                _, _, s2s_attn = model["text_aligner"](mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            if int(mel_input_length.min().item()) < 80:
                continue

            with autocast:
                t_en = model["text_encoder"](texts, input_lengths, text_mask)
                asr = t_en @ s2s_attn_mono

                gs = []
                for bib in range(len(mel_input_length)):
                    mel = mels[bib, :, : int(mel_input_length[bib].item())]
                    gs.append(model["style_encoder"](mel.unsqueeze(0).unsqueeze(1)))
                s_trg_style = torch.stack(gs).squeeze(1)

            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len = mel_len // 2 * 2
            if mel_len < 32 or int(mel_input_length.min().item() / 2 - 1) < mel_len:
                continue

            en, gt, wav_list = [], [], []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, max(1, mel_length - mel_len))
                en.append(asr[bib, :, random_start : random_start + mel_len])
                gt.append(mels[bib, :, (random_start * 2) : ((random_start + mel_len) * 2)])
                y = waves[bib]
                start_wav = (random_start * 2) * 300
                end_wav = ((random_start + mel_len) * 2) * 300
                wav_list.append(y[start_wav:end_wav])

            en = torch.stack(en)
            gt = torch.stack(gt)

            with torch.no_grad():
                F0_real, _, _ = model["pitch_extractor"](gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

            with autocast:
                y_rec = model["decoder"](en, F0_real, real_norm, s_trg_style)
                if y_rec.dim() == 2:
                    y_rec = y_rec.unsqueeze(1)

            wav_gt = []
            for w in wav_list:
                target_len = y_rec.shape[-1]
                ww = np.asarray(w, dtype=np.float32)
                if ww.shape[0] < target_len:
                    ww = np.pad(ww, (0, target_len - ww.shape[0]))
                wav_gt.append(ww[:target_len])
            wav_gt = torch.tensor(np.stack(wav_gt), device=device).float().unsqueeze(1)

            if "mpd" in opts:
                for opt in opts.values():
                    opt.zero_grad(set_to_none=True)
                loss_disc = dl(wav_gt, y_rec.detach())
                (loss_disc / accum).backward()
                clip_grad_norm(model, ["mpd", "msd", "wd"], max_norm=max_grad_norm)
                step_optimizers(opts, [k for k in ("mpd", "msd", "wd") if k in opts])
            else:
                loss_disc = torch.tensor(0.0, device=device)

            with autocast:
                loss_mel = stft_loss(y_rec.squeeze(1), wav_gt.squeeze(1))
                loss_gen = gl(wav_gt, y_rec)
                loss = lambda_mel * loss_mel + lambda_gen * loss_gen

                if enable_slm and wl is not None:
                    loss_slm = wl(wav_gt.squeeze(1), y_rec.squeeze(1))
                    loss = loss + lambda_slm * loss_slm
                else:
                    loss_slm = torch.tensor(0.0, device=device)

            (loss / accum).backward()
            if (i + 1) % accum == 0:
                clip_grad_norm(model, ["text_encoder", "style_encoder"], max_norm=max_grad_norm)
                step_optimizers(opts, [k for k in ("text_encoder", "style_encoder") if k in opts])
                for sch in schedulers.values():
                    sch.step()

            running["mel"] += float(loss_mel.detach().item())
            running["gen"] += float(loss_gen.detach().item())
            running["disc"] += float(loss_disc.detach().item()) if torch.is_tensor(loss_disc) else float(loss_disc)
            running["slm"] += float(loss_slm.detach().item()) if torch.is_tensor(loss_slm) else float(loss_slm)
            n_batches += 1
            iters += 1

            if iters % int(config.get("log_interval", 50)) == 0:
                avg = {k: v / max(n_batches, 1) for k, v in running.items()}
                lr_now = {k: opts[k].param_groups[0]["lr"] for k in opts}
                run_logger.log_train(iters, avg, lr_now)
                logger.info("iter %s phase=%s %s", iters, phase, avg)

        ckpt = layout["stage2"] / f"epoch_2nd_{epoch:05d}.pth"
        save_checkpoint(ckpt, model, opts, schedulers, epoch + 1, iters, phase, running)
        latest = layout["stage2"] / "latest.pth"
        save_checkpoint(latest, model, opts, schedulers, epoch + 1, iters, phase, running)
        update_latest_pointer(
            layout["state"],
            "stage2",
            latest,
            {"epoch": epoch + 1, "iters": iters, "wandb_id": run_logger.wandb_id, "phase": phase},
        )
        apply_retention(layout["stage2"], keep_last=int(config.get("keep_last", 3)))

    final_ckpt = layout["stage2"] / "final.pth"
    save_checkpoint(final_ckpt, model, None, None, epochs, iters, "stage2", running)
    run_logger.log_summary({"stage": "stage2_encoders", "iters": iters})
    run_logger.close()
    logger.info("Stage 2 Encoders Complete → %s", final_ckpt)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", default=None, help="'auto' or checkpoint path")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    run_stage2_encoders(config, args)


if __name__ == "__main__":
    main()
