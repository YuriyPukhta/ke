#!/usr/bin/env python3
"""Stage 1 trainer: ASR aligner warm-up → acoustic fine-tune.

Wraps kikiri StyleTTS2 modules without modifying third_party sources.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from runpod_training.data.npz_dataset import build_npz_dataloader
from runpod_training.paths import ensure_styletts2_on_path, ensure_volume
from runpod_training.train.checkpointing import (
    apply_retention,
    load_checkpoint,
    resolve_resume,
    save_checkpoint,
    update_latest_pointer,
)
from runpod_training.train.freeze import apply_phase, trainable_param_count
from runpod_training.train.logging_utils import RunLogger, configure_run_logging
from runpod_training.train.model_factory import (
    build_training_model,
    load_kokoro_base,
    load_yaml,
    maybe_init_style_encoder,
)
from runpod_training.train.schedulers import build_warmup_cosine_scheduler, set_group_lr

logger = logging.getLogger("runpod_training.stage1")


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
        opts[key] = AdamW(params, lr=lr, weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9)
    return opts


def step_optimizers(opts: dict[str, AdamW], keys=None):
    for key in keys or list(opts):
        if key in opts:
            opts[key].step()
            opts[key].zero_grad(set_to_none=True)


def run_stage1(config: dict, args: argparse.Namespace) -> None:
    ensure_styletts2_on_path()
    from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, WavLMLoss
    from utils import log_norm, maximum_path, mask_from_lens

    layout = ensure_volume(Path(config.get("run_root", "/workspace/kokoro-run")))
    configure_run_logging(layout["root"], "stage1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_cfg = config["stage1"]
    data_cfg = config["data_params"]
    lr_cfg = stage_cfg["lrs"]

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
    val_loader = build_npz_dataloader(
        index_jsonl=data_cfg["val_index"],
        data_root=data_cfg["root_path"],
        targets_root=data_cfg.get("targets_root"),
        batch_size=config.get("batch_size", 12),
        num_workers=max(1, data_cfg.get("num_workers", 8) // 2),
        validation=True,
        ood_texts=data_cfg.get("OOD_data"),
        use_cached_f0n=False,
        speaker_balanced=False,
    )

    model, model_params = build_training_model(config, device)
    load_kokoro_base(model, config["pretrained_model"])
    style_audit = maybe_init_style_encoder(model, config)
    logger.info("style init audit: %s", style_audit)

    # Losses
    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model["mpd"], model["msd"]).to(device)
    dl = DiscriminatorLoss(model["mpd"], model["msd"]).to(device)
    sr = config["preprocess_params"]["sr"]
    wl = WavLMLoss(model_params.slm.model, model["wd"], sr, model_params.slm.sr).to(device)

    run_logger = RunLogger(layout["root"], "stage1", enable_wandb=config.get("wandb", True))
    best_mel = float("inf")
    start_epoch = 0
    iters = 0

    resume_path = resolve_resume(args.resume, layout["state"], "stage1")
    if resume_path:
        start_epoch, iters, _ = load_checkpoint(resume_path, model, load_optimizer=False)
        logger.info("resumed stage1 from %s @ epoch=%s iters=%s", resume_path, start_epoch, iters)

    warmup_epochs = int(stage_cfg.get("warmup_epochs", 1))
    max_warmup = int(stage_cfg.get("warmup_epochs_max", 2))
    acoustic_epochs = int(stage_cfg.get("acoustic_epochs", 9))
    total_epochs = max_warmup + acoustic_epochs
    max_len = int(config.get("max_len", 300))
    accum = int(config.get("grad_accum", 2))
    lambda_mel = float(config["loss_params"].get("lambda_mel", 5.0))
    lambda_mono = float(config["loss_params"].get("lambda_mono", 1.0))
    lambda_s2s = float(config["loss_params"].get("lambda_s2s", 1.0))
    lambda_gen = float(config["loss_params"].get("lambda_gen", 1.0))
    TMA_epoch = int(config["loss_params"].get("TMA_epoch", 0))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = total_epochs * steps_per_epoch

    n_down = model["text_aligner"].n_down
    align_ok = False

    for epoch in range(start_epoch, total_epochs):
        phase = "aligner_warmup" if (epoch < warmup_epochs or (epoch < max_warmup and not align_ok)) else "stage1"
        if phase == "stage1" and epoch < warmup_epochs:
            # force transition after gate
            pass
        apply_phase(model, phase if phase == "aligner_warmup" else "stage1")
        lrs = lr_cfg["warmup"] if phase == "aligner_warmup" else lr_cfg["acoustic"]
        opts = build_optimizers(model, lrs)
        schedulers = {
            k: build_warmup_cosine_scheduler(opt, total_steps, stage_cfg.get("warmup_ratio", 0.05))
            for k, opt in opts.items()
        }
        logger.info(
            "epoch %s phase=%s trainable=%sM",
            epoch,
            phase,
            trainable_param_count(model) / 1e6,
        )

        running = {"mel": 0.0, "mono": 0.0, "s2s": 0.0, "gen": 0.0, "disc": 0.0}
        n_batches = 0
        for model_key in model:
            model[model_key].train()
            if phase == "aligner_warmup" and model_key != "text_aligner":
                model[model_key].eval()
            if model_key == "pitch_extractor":
                model[model_key].eval()

        for i, batch in enumerate(train_loader):
            waves = batch[0]
            texts = batch[1].to(device)
            input_lengths = batch[2].to(device)
            mels = batch[5].to(device)
            mel_input_length = batch[6].to(device)

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2**n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)

            ppgs, s2s_pred, s2s_attn = model["text_aligner"](mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

            with torch.no_grad():
                attn_mask = (
                    (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                )
                attn_mask = attn_mask * (~text_mask).unsqueeze(-1).expand(
                    text_mask.shape[0], text_mask.shape[1], mask.shape[-1]
                ).float()
                attn_mask = attn_mask < 1
            s2s_attn = s2s_attn.masked_fill(attn_mask, 0.0)

            with torch.no_grad():
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            loss_s2s = 0.0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s = loss_s2s + F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s = loss_s2s / texts.size(0)
            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

            loss = lambda_s2s * loss_s2s + lambda_mono * loss_mono
            loss_mel = torch.tensor(0.0, device=device)
            loss_gen = torch.tensor(0.0, device=device)
            loss_disc = torch.tensor(0.0, device=device)

            if phase == "stage1":
                t_en = model["text_encoder"](texts, input_lengths, text_mask)
                asr = t_en @ (s2s_attn if bool(random.getrandbits(1)) else s2s_attn_mono)

                mel_len = min([int(mel_input_length.min().item() / 2 - 1), max_len // 2])
                mel_len = mel_len // 2 * 2
                if mel_len < 32:
                    continue
                en = []
                gt = []
                wav_crops = []
                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item())
                    random_start = np.random.randint(0, max(1, mel_length - mel_len))
                    en.append(asr[bib, :, random_start : random_start + mel_len])
                    gt.append(mels[bib, :, random_start : random_start + mel_len])
                    y = waves[bib]
                    # cache-faithful wave indexing (no 5000 pad): hop 300
                    start_wav = random_start * 300
                    end_wav = (random_start + mel_len) * 300
                    wav_crops.append(y[start_wav:end_wav])
                en = torch.stack(en)
                gt = torch.stack(gt)
                with torch.no_grad():
                    F0_real, _, _ = model["pitch_extractor"](gt.unsqueeze(1))
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                s = model["style_encoder"](gt.unsqueeze(1))
                y_rec = model["decoder"](en, F0_real, real_norm, s)
                if y_rec.dim() == 2:
                    y_rec = y_rec.unsqueeze(1)

                wav_gt = []
                for w in wav_crops:
                    target_len = y_rec.shape[-1]
                    ww = np.asarray(w, dtype=np.float32)
                    if ww.shape[0] < target_len:
                        ww = np.pad(ww, (0, target_len - ww.shape[0]))
                    wav_gt.append(ww[:target_len])
                wav_gt = torch.tensor(np.stack(wav_gt), device=device).float().unsqueeze(1)

                if epoch >= TMA_epoch and "mpd" in opts:
                    for opt in opts.values():
                        opt.zero_grad(set_to_none=True)
                    loss_disc = dl(wav_gt, y_rec.detach())
                    (loss_disc / accum).backward()
                    step_optimizers(opts, [k for k in ("mpd", "msd") if k in opts])

                # Mel reconstruction: STFT multi-res on waveforms (StyleTTS2 uses mel L1 + gen)
                loss_mel = stft_loss(y_rec.squeeze(1), wav_gt.squeeze(1))
                loss_gen = gl(wav_gt, y_rec)
                loss = loss + lambda_mel * loss_mel + lambda_gen * loss_gen

            (loss / accum).backward()
            if (i + 1) % accum == 0:
                if phase == "aligner_warmup":
                    step_optimizers(opts, ["text_aligner"])
                else:
                    step_optimizers(opts, list(opts))
                for sch in schedulers.values():
                    sch.step()

            running["mel"] += float(loss_mel.detach().item()) if torch.is_tensor(loss_mel) else float(loss_mel)
            running["mono"] += float(loss_mono.detach().item())
            running["s2s"] += float(loss_s2s.detach().item())
            running["gen"] += float(loss_gen.detach().item()) if torch.is_tensor(loss_gen) else float(loss_gen)
            running["disc"] += float(loss_disc.detach().item()) if torch.is_tensor(loss_disc) else float(loss_disc)
            n_batches += 1
            iters += 1

            if iters % int(config.get("log_interval", 50)) == 0:
                avg = {k: v / max(n_batches, 1) for k, v in running.items()}
                lr_now = {k: opts[k].param_groups[0]["lr"] for k in opts}
                run_logger.log_train(iters, avg, lr_now)
                logger.info("iter %s phase=%s %s", iters, phase, avg)

        # Validation / alignment health
        model["text_aligner"].eval()
        val_mono = val_s2s = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                texts = batch[1].to(device)
                input_lengths = batch[2].to(device)
                mels = batch[5].to(device)
                mel_input_length = batch[6].to(device)
                mask = length_to_mask(mel_input_length // (2**n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)
                _, s2s_pred, s2s_attn = model["text_aligner"](mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
                loss_s2s = 0.0
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                    loss_s2s = loss_s2s + F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
                loss_s2s = loss_s2s / texts.size(0)
                val_mono += float(loss_mono.item())
                val_s2s += float(loss_s2s.item())
                val_n += 1
                if val_n >= int(stage_cfg.get("val_batches", 20)):
                    break
        val_metrics = {
            "mono": val_mono / max(val_n, 1),
            "s2s": val_s2s / max(val_n, 1),
            "mel": running["mel"] / max(n_batches, 1),
        }
        run_logger.log_val(iters, val_metrics)
        # Alignment gate: mono falling and finite s2s
        if phase == "aligner_warmup" and val_metrics["mono"] < float(stage_cfg.get("align_mono_gate", 2.0)):
            align_ok = True
            logger.info("alignment gate passed: mono=%.4f", val_metrics["mono"])

        ckpt = layout["stage1"] / f"epoch_1st_{epoch:05d}.pth"
        save_checkpoint(ckpt, model, opts, schedulers, epoch + 1, iters, phase, val_metrics)
        latest = layout["stage1"] / "latest.pth"
        save_checkpoint(latest, model, opts, schedulers, epoch + 1, iters, phase, val_metrics)
        if val_metrics["mel"] < best_mel:
            best_mel = val_metrics["mel"]
            save_checkpoint(layout["stage1"] / "best_mel.pth", model, opts, schedulers, epoch + 1, iters, phase, val_metrics)
        update_latest_pointer(
            layout["state"],
            "stage1",
            latest,
            {"epoch": epoch + 1, "iters": iters, "wandb_id": run_logger.wandb_id, "phase": phase},
        )
        apply_retention(layout["stage1"], keep_last=int(config.get("keep_last", 3)))

    # Export first_stage.pth for Stage 2 handoff
    first_stage = layout["stage1"] / "first_stage.pth"
    save_checkpoint(first_stage, model, None, None, total_epochs, iters, "stage1", {"best_mel": best_mel})
    run_logger.log_summary({"stage": "stage1", "best_mel": best_mel, "iters": iters})
    run_logger.close()
    logger.info("Stage 1 complete → %s", first_stage)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", default=None, help="'auto' or checkpoint path")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    run_stage1(config, args)


if __name__ == "__main__":
    main()
