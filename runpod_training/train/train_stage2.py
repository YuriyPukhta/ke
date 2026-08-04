#!/usr/bin/env python3
"""Stage 2 trainer: prosody predictor + joint GAN/SLM (diffusion disabled)."""

from __future__ import annotations

import argparse
import copy
import logging
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
from runpod_training.train.model_factory import build_training_model, load_yaml
from runpod_training.train.schedulers import build_warmup_cosine_scheduler

logger = logging.getLogger("runpod_training.stage2")


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
        betas = (0.9, 0.99) if key == "bert_encoder" else (0.0, 0.99)
        wd = 0.01 if key == "bert_encoder" else 1e-4
        opts[key] = AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=1e-9)
    return opts


def step_optimizers(opts, keys=None):
    for key in keys or list(opts):
        if key in opts:
            opts[key].step()
            opts[key].zero_grad(set_to_none=True)


def run_stage2(config: dict, args: argparse.Namespace) -> None:
    ensure_styletts2_on_path()
    from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, WavLMLoss
    from utils import log_norm, maximum_path, mask_from_lens

    layout = ensure_volume(Path(config.get("run_root", "/workspace/kokoro-run")))
    configure_run_logging(layout["root"], "stage2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model["predictor_encoder"] = copy.deepcopy(model["style_encoder"]).to(device)
    logger.info("loaded %s; predictor_encoder cloned from style_encoder", first_stage)

    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model["mpd"], model["msd"]).to(device)
    dl = DiscriminatorLoss(model["mpd"], model["msd"]).to(device)
    sr = config["preprocess_params"]["sr"]
    wl = WavLMLoss(model_params.slm.model, model["wd"], sr, model_params.slm.sr).to(device)

    run_logger = RunLogger(layout["root"], "stage2", enable_wandb=config.get("wandb", True))
    start_epoch = 0
    iters = 0
    resume_path = resolve_resume(args.resume, layout["state"], "stage2")
    if resume_path:
        start_epoch, iters, _ = load_checkpoint(resume_path, model, load_optimizer=False)
        logger.info("resumed stage2 from %s epoch=%s iters=%s", resume_path, start_epoch, iters)

    epochs = int(stage_cfg.get("epochs", 10))
    joint_epoch = int(loss_params.get("joint_epoch", 3))
    max_len = int(config.get("max_len", 300))
    accum = int(config.get("grad_accum", 2))
    n_down = model["text_aligner"].n_down
    best_val = float("inf")
    total_steps = epochs * max(1, len(train_loader))

    lambda_mel = float(loss_params.get("lambda_mel", 5))
    lambda_dur = float(loss_params.get("lambda_dur", 1))
    lambda_ce = float(loss_params.get("lambda_ce", 20))
    lambda_f0 = float(loss_params.get("lambda_F0", 1))
    lambda_norm = float(loss_params.get("lambda_norm", 1))
    lambda_gen = float(loss_params.get("lambda_gen", 1))
    lambda_slm = float(loss_params.get("lambda_slm", 1))

    for epoch in range(start_epoch, epochs):
        phase = "stage2_joint" if epoch >= joint_epoch else "stage2_pre_joint"
        apply_phase(model, phase)
        lrs = stage_cfg["lrs"]["joint" if phase == "stage2_joint" else "pre_joint"]
        opts = build_optimizers(model, lrs)
        schedulers = {
            k: build_warmup_cosine_scheduler(opt, total_steps, stage_cfg.get("warmup_ratio", 0.05))
            for k, opt in opts.items()
        }
        logger.info("epoch %s phase=%s trainable=%.2fM", epoch, phase, trainable_param_count(model) / 1e6)

        for key in model:
            model[key].train()
        model["bert"].eval()
        model["text_encoder"].eval()
        model["text_aligner"].eval()
        model["pitch_extractor"].eval()
        if phase != "stage2_joint":
            model["style_encoder"].eval()
            model["decoder"].eval()

        running = {"mel": 0.0, "dur": 0.0, "f0": 0.0, "norm": 0.0, "gen": 0.0, "slm": 0.0}
        n_batches = 0

        for i, batch in enumerate(train_loader):
            waves = batch[0]
            texts = batch[1].to(device)
            input_lengths = batch[2].to(device)
            mels = batch[5].to(device)
            mel_input_length = batch[6].to(device)

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2**n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)
                _, _, s2s_attn = model["text_aligner"](mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                t_en = model["text_encoder"](texts, input_lengths, text_mask)
                asr = t_en @ s2s_attn_mono
                d_gt = s2s_attn_mono.sum(-1).detach()
                bert_dur = model["bert"](texts, attention_mask=(~text_mask).int())

            # Global styles per utterance (avgpool cannot be batched with padding)
            ss, gs = [], []
            for bib in range(len(mel_input_length)):
                mel = mels[bib, :, : int(mel_input_length[bib].item())]
                ss.append(model["predictor_encoder"](mel.unsqueeze(0).unsqueeze(1)))
                gs.append(model["style_encoder"](mel.unsqueeze(0).unsqueeze(1)))
            s_trg_dur = torch.stack(ss).squeeze(1)
            # gs used only for optional diffusion (disabled)

            d_en = model["bert_encoder"](bert_dur).transpose(-1, -2)
            d, p = model["predictor"](d_en, s_trg_dur, input_lengths, s2s_attn_mono, text_mask)

            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len = mel_len // 2 * 2
            if mel_len < 32 or int(mel_input_length.min().item() / 2 - 1) < mel_len:
                continue

            en, gt, p_en, wav_list = [], [], [], []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, max(1, mel_length - mel_len))
                en.append(asr[bib, :, random_start : random_start + mel_len])
                p_en.append(p[bib, :, random_start : random_start + mel_len])
                gt.append(mels[bib, :, (random_start * 2) : ((random_start + mel_len) * 2)])
                y = waves[bib][(random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300]
                wav_list.append(torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device))

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            if gt.size(-1) < 80:
                continue

            with torch.no_grad():
                F0_real, _, _ = model["pitch_extractor"](gt.unsqueeze(1))
                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

            s = model["style_encoder"](gt.unsqueeze(1))
            s_dur = model["predictor_encoder"](gt.unsqueeze(1))

            y_rec_gt_pred = model["decoder"](en, F0_real, N_real, s)
            if y_rec_gt_pred.dim() == 2:
                y_rec_gt_pred = y_rec_gt_pred.unsqueeze(1)

            F0_fake, N_fake = model["predictor"].F0Ntrain(p_en, s_dur)
            y_rec = model["decoder"](en, F0_fake, N_fake, s)
            if y_rec.dim() == 2:
                y_rec = y_rec.unsqueeze(1)

            loss_f0 = F.smooth_l1_loss(F0_real, F0_fake) / 10
            loss_n = F.smooth_l1_loss(N_real, N_fake)

            loss_dur = 0.0
            loss_ce = 0.0
            for _s2s_pred, _text_input, _text_length in zip(d, d_gt, input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for pp in range(_s2s_trg.shape[0]):
                    _s2s_trg[pp, : int(_text_input[pp])] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                loss_dur = loss_dur + F.l1_loss(
                    _dur_pred[1 : _text_length - 1], _text_input[1 : _text_length - 1]
                )
                loss_ce = loss_ce + F.binary_cross_entropy_with_logits(
                    _s2s_pred.flatten(), _s2s_trg.flatten()
                )
            loss_dur = loss_dur / texts.size(0)
            loss_ce = loss_ce / texts.size(0)

            max_w = y_rec.size(-1)
            wav_gt = torch.zeros(len(wav_list), 1, max_w, device=device)
            for bib, w in enumerate(wav_list):
                n = min(w.numel(), max_w)
                wav_gt[bib, 0, :n] = w[:n]
            wav_target = wav_gt if phase == "stage2_joint" else y_rec_gt_pred.detach()
            if wav_target.size(-1) != max_w:
                wav_target = F.pad(wav_target, (0, max(0, max_w - wav_target.size(-1))))[..., :max_w]

            loss_mel = stft_loss(y_rec.squeeze(1), wav_target.squeeze(1))
            loss = (
                lambda_mel * loss_mel
                + lambda_dur * loss_dur
                + lambda_ce * loss_ce
                + lambda_f0 * loss_f0
                + lambda_norm * loss_n
            )
            loss_gen = torch.tensor(0.0, device=device)
            loss_slm = torch.tensor(0.0, device=device)

            if phase == "stage2_joint":
                disc_out = dl(wav_target, y_rec.detach())
                loss_disc = disc_out[0] if isinstance(disc_out, tuple) else disc_out
                for opt in opts.values():
                    opt.zero_grad(set_to_none=True)
                (loss_disc.mean() / accum).backward()
                step_optimizers(opts, [k for k in ("mpd", "msd") if k in opts])
                gen_out = gl(wav_target, y_rec)
                loss_gen = gen_out[0] if isinstance(gen_out, tuple) else gen_out
                loss_gen = loss_gen.mean() if torch.is_tensor(loss_gen) else loss_gen
                try:
                    loss_slm = wl(wav_target.detach().squeeze(1), y_rec.squeeze(1)).mean()
                except Exception:
                    loss_slm = torch.tensor(0.0, device=device)
                loss = loss + lambda_gen * loss_gen + lambda_slm * loss_slm

            (loss / accum).backward()
            if (i + 1) % accum == 0:
                step_optimizers(opts, list(opts))
                for sch in schedulers.values():
                    sch.step()

            running["mel"] += float(loss_mel.detach().item())
            running["dur"] += float(loss_dur.detach().item()) if torch.is_tensor(loss_dur) else float(loss_dur)
            running["f0"] += float(loss_f0.detach().item())
            running["norm"] += float(loss_n.detach().item())
            running["gen"] += float(loss_gen.detach().item()) if torch.is_tensor(loss_gen) else float(loss_gen)
            running["slm"] += float(loss_slm.detach().item()) if torch.is_tensor(loss_slm) else float(loss_slm)
            n_batches += 1
            iters += 1
            if iters % int(config.get("log_interval", 50)) == 0:
                avg = {k: v / max(n_batches, 1) for k, v in running.items()}
                lr_now = {k: opts[k].param_groups[0]["lr"] for k in opts}
                run_logger.log_train(iters, avg, lr_now)
                logger.info("iter %s %s", iters, avg)

        val_mel = running["mel"] / max(n_batches, 1)
        val_metrics = {"mel": val_mel, "dur": running["dur"] / max(n_batches, 1)}
        run_logger.log_val(iters, val_metrics)

        ckpt = layout["stage2"] / f"epoch_2nd_{epoch:05d}.pth"
        save_checkpoint(ckpt, model, opts, schedulers, epoch + 1, iters, phase, val_metrics)
        latest = layout["stage2"] / "latest.pth"
        save_checkpoint(latest, model, opts, schedulers, epoch + 1, iters, phase, val_metrics)
        if val_mel < best_val:
            best_val = val_mel
            save_checkpoint(
                layout["stage2"] / "best_val.pth", model, opts, schedulers, epoch + 1, iters, phase, val_metrics
            )
        update_latest_pointer(
            layout["state"],
            "stage2",
            latest,
            {"epoch": epoch + 1, "iters": iters, "wandb_id": run_logger.wandb_id, "phase": phase},
        )
        apply_retention(layout["stage2"], keep_last=int(config.get("keep_last", 3)))

    final = layout["stage2"] / "final.pth"
    save_checkpoint(final, model, None, None, epochs, iters, "stage2", {"best_val": best_val})
    run_logger.log_summary({"stage": "stage2", "best_val": best_val, "iters": iters})
    run_logger.close()
    logger.info("Stage 2 complete → %s", final)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_stage2(load_yaml(args.config), args)


if __name__ == "__main__":
    main()
