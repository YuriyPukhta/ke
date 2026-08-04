#!/usr/bin/env python3
"""Stage 2 trainer: prosody predictor + joint GAN/SLM/SLM-adv (diffusion disabled)."""

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

from runpod_training.data.npz_dataset import build_npz_dataloader
from runpod_training.paths import ensure_styletts2_on_path, ensure_volume
from runpod_training.train.checkpointing import (
    apply_retention,
    load_checkpoint,
    resolve_resume,
    save_checkpoint,
    save_checkpoint_resilient,
    update_latest_pointer,
)
from runpod_training.train.freeze import apply_phase, trainable_param_count
from runpod_training.train.logging_utils import RunLogger
from runpod_training.train.model_factory import build_training_model, load_yaml
from runpod_training.train.schedulers import build_warmup_cosine_scheduler
from runpod_training.train.slmadv_nodiff import EchoStyleSampler

logger = logging.getLogger("runpod_training.stage2")
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


def step_optimizers(opts, keys=None):
    for key in keys or list(opts):
        if key in opts:
            opts[key].step()
            opts[key].zero_grad(set_to_none=True)


def _patch_safe_tprls() -> None:
    """StyleTTS2 TPRLS does mean(empty) → NaN when the relative mask is empty."""
    import losses as st2_losses

    if getattr(st2_losses, "_kokoro_safe_tprls", False):
        return

    def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
        loss = 0.0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            tau = 0.04
            m_DG = torch.median((dr - dg))
            mask = dr < (dg + m_DG)
            if bool(mask.any()):
                L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[mask])
            else:
                L_rel = dr.new_zeros(())
            loss = loss + (tau - F.relu(tau - L_rel))
        return loss

    def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
        # StyleTTS2 names the zip vars (dg, dr) but zips (real, generated).
        loss = 0.0
        for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
            tau = 0.04
            m_DG = torch.median((dr - dg))
            mask = dr < (dg + m_DG)
            if bool(mask.any()):
                L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[mask])
            else:
                L_rel = dr.new_zeros(())
            loss = loss + (tau - F.relu(tau - L_rel))
        return loss

    st2_losses.discriminator_TPRLS_loss = discriminator_TPRLS_loss
    st2_losses.generator_TPRLS_loss = generator_TPRLS_loss
    st2_losses._kokoro_safe_tprls = True  # type: ignore[attr-defined]
    logger.info("patched StyleTTS2 TPRLS for empty-mask NaN safety")


def run_stage2(config: dict, args: argparse.Namespace) -> None:
    ensure_styletts2_on_path()
    from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, WavLMLoss
    from utils import log_norm, maximum_path, mask_from_lens

    _patch_safe_tprls()

    layout = ensure_volume(Path(config.get("run_root", "/workspace/kokoro-run")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(config.get("bf16", False)) and device.type == "cuda"
    # bf16 does not need GradScaler; keeps master weights in fp32 via AdamW.
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
    logger.info("amp bf16=%s", use_bf16)
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
    slmadv_cfg = config.get("slmadv_params") or {}
    enable_slmadv = bool(config.get("slmadv", True)) and bool(slmadv_cfg)
    lambda_slm = float(loss_params.get("lambda_slm", 1))
    enable_slm = bool(config.get("slm", True)) and (lambda_slm > 0 or enable_slmadv)
    wl = None
    slmadv = None
    slm_sampler = EchoStyleSampler()
    if enable_slm:
        wl = WavLMLoss(model_params.slm.model, model["wd"], sr, model_params.slm.sr).to(device)
        if enable_slmadv:
            from Modules.slmadv import SLMAdversarialLoss

            slmadv = SLMAdversarialLoss(
                model,
                wl,
                slm_sampler,
                int(slmadv_cfg.get("min_len", 100)),
                int(slmadv_cfg.get("max_len", 500)),
                batch_percentage=float(slmadv_cfg.get("batch_percentage", 0.5)),
                skip_update=int(slmadv_cfg.get("iter", 10)),
                sig=float(slmadv_cfg.get("sig", 1.5)),
            )
            logger.info(
                "SLM-adv enabled (StyleTTS2 Modules.slmadv, echo-sampler w/o diffusion): "
                "min_len=%s max_len=%s skip_update=%s",
                slmadv.min_len,
                slmadv.max_len,
                slmadv.skip_update,
            )
    else:
        from runpod_training.train.freeze import set_requires_grad

        set_requires_grad(model["wd"], False)
        model["wd"].cpu()
        model["wd"].eval()
        logger.info(
            "SLM fully disabled (slm=false or lambda_slm=0 & slmadv=false) — "
            "no WavLM load/forward, wd on CPU"
        )

    run_logger = RunLogger(layout["root"], "stage2", enable_wandb=config.get("wandb", True))
    start_epoch = 0
    iters = 0
    resume_path = resolve_resume(args.resume, layout["state"], "stage2")
    if resume_path:
        start_epoch, iters, _ = load_checkpoint(resume_path, model, load_optimizer=False)
        logger.info("resumed stage2 from %s epoch=%s iters=%s", resume_path, start_epoch, iters)
        if not enable_slm and "wd" in model:
            from runpod_training.train.freeze import set_requires_grad

            set_requires_grad(model["wd"], False)
            model["wd"].cpu()
            model["wd"].eval()

    epochs = int(stage_cfg.get("epochs", 10))
    joint_epoch = int(loss_params.get("joint_epoch", 3))
    max_len = int(config.get("max_len", 300))
    accum = int(config.get("grad_accum", 2))
    n_down = model["text_aligner"].n_down
    best_val = float("inf")
    total_steps = epochs * max(1, len(train_loader))
    slmadv_thresh = float(slmadv_cfg.get("thresh", 5))
    slmadv_scale = float(slmadv_cfg.get("scale", 0.01))

    lambda_mel = float(loss_params.get("lambda_mel", 5))
    lambda_dur = float(loss_params.get("lambda_dur", 1))
    lambda_ce = float(loss_params.get("lambda_ce", 20))
    lambda_f0 = float(loss_params.get("lambda_F0", 1))
    lambda_norm = float(loss_params.get("lambda_norm", 1))
    lambda_gen = float(loss_params.get("lambda_gen", 1))
    enable_gan = bool(config.get("gan", True)) and lambda_gen > 0
    joint_freeze_acoustic = bool(config.get("joint_freeze_acoustic", False))
    if not enable_gan:
        logger.info("GAN disabled (gan=false or lambda_gen=0) — no MPD/MSD disc/gen steps")
    if joint_freeze_acoustic:
        logger.info("joint_freeze_acoustic=true — decoder/style stay frozen in joint")
    save_interval = int(config.get("save_interval", 500))
    shm_ckpt_dir = Path(config.get("shm_ckpt_dir", "/dev/shm/kokoro-ckpts"))
    try:
        shm_ckpt_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        shm_ckpt_dir = Path("/tmp/kokoro-ckpts")
        shm_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if save_interval > 0:
        logger.info("mid-epoch checkpoints every %s iters → %s + volume latest", save_interval, shm_ckpt_dir)

    for epoch in range(start_epoch, epochs):
        phase = "stage2_joint" if epoch >= joint_epoch else "stage2_pre_joint"
        apply_phase(model, phase)
        if not enable_slm and "wd" in model:
            from runpod_training.train.freeze import set_requires_grad

            set_requires_grad(model["wd"], False)
            model["wd"].cpu()
            model["wd"].eval()
        if phase == "stage2_joint":
            from runpod_training.train.freeze import set_requires_grad

            if joint_freeze_acoustic:
                for key in ("decoder", "style_encoder"):
                    if key in model:
                        set_requires_grad(model[key], False)
            if not enable_gan:
                for key in ("mpd", "msd"):
                    if key in model:
                        set_requires_grad(model[key], False)
                        model[key].cpu()
                        model[key].eval()
                logger.info("GAN nets mpd/msd moved to CPU")
        lrs = stage_cfg["lrs"]["joint" if phase == "stage2_joint" else "pre_joint"]
        opts = build_optimizers(model, lrs)
        schedulers = {
            k: build_warmup_cosine_scheduler(opt, total_steps, stage_cfg.get("warmup_ratio", 0.05))
            for k, opt in opts.items()
        }
        logger.info("epoch %s phase=%s trainable=%.2fM", epoch, phase, trainable_param_count(model) / 1e6)

        for key in model:
            model[key].train()
        model["text_encoder"].eval()
        model["text_aligner"].eval()
        model["pitch_extractor"].eval()
        if not enable_slm and "wd" in model:
            model["wd"].eval()
        if phase != "stage2_joint":
            model["bert"].eval()
            model["style_encoder"].eval()
            model["decoder"].eval()
        elif joint_freeze_acoustic:
            model["style_encoder"].eval()
            model["decoder"].eval()
        if phase == "stage2_joint" and not enable_gan:
            if "mpd" in model:
                model["mpd"].eval()
            if "msd" in model:
                model["msd"].eval()
        # StyleTTS2 stage2: bert stays trainable in joint (used by SLM-adv)

        running = {
            "mel": 0.0,
            "dur": 0.0,
            "ce": 0.0,
            "f0": 0.0,
            "norm": 0.0,
            "gen": 0.0,
            "slm": 0.0,
            "slm_gen": 0.0,
            "slm_disc": 0.0,
        }
        n_batches = 0

        for i, batch in enumerate(train_loader):
            waves = batch[0]
            texts = batch[1].to(device)
            input_lengths = batch[2].to(device)
            ref_texts = batch[3].to(device)
            ref_lengths = batch[4].to(device)
            mels = batch[5].to(device)
            mel_input_length = batch[6].to(device)

            with autocast, torch.no_grad():
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

            # Style/predictor encoders use 5x5 convs — skip short utterances (StyleTTS2 needs ~80 mel frames).
            if int(mel_input_length.min().item()) < 80:
                continue

            with autocast:
                # Global styles per utterance (avgpool cannot be batched with padding)
                ss, gs = [], []
                for bib in range(len(mel_input_length)):
                    mel = mels[bib, :, : int(mel_input_length[bib].item())]
                    ss.append(model["predictor_encoder"](mel.unsqueeze(0).unsqueeze(1)))
                    gs.append(model["style_encoder"](mel.unsqueeze(0).unsqueeze(1)))
                s_trg_dur = torch.stack(ss).squeeze(1)
                s_trg_style = torch.stack(gs).squeeze(1)
                s_trg = torch.cat([s_trg_style, s_trg_dur], dim=-1).detach()

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

            with autocast:
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
                use_soft_mel = bool(config.get("joint_soft_mel", False)) or phase != "stage2_joint"
                wav_mel_target = y_rec_gt_pred.detach() if use_soft_mel else wav_gt
                if wav_mel_target.size(-1) != max_w:
                    wav_mel_target = F.pad(
                        wav_mel_target, (0, max(0, max_w - wav_mel_target.size(-1)))
                    )[..., :max_w]
                wav_gan_target = wav_gt
                if wav_gan_target.size(-1) != max_w:
                    wav_gan_target = F.pad(
                        wav_gan_target, (0, max(0, max_w - wav_gan_target.size(-1)))
                    )[..., :max_w]

                loss_mel = stft_loss(y_rec.squeeze(1), wav_mel_target.squeeze(1))
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
                    y_gen = torch.nan_to_num(y_rec, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
                    y_tgt = torch.nan_to_num(wav_gan_target, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
                    if enable_gan:
                        y_disc = torch.nan_to_num(y_rec.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
                        disc_out = dl(y_tgt, y_disc)
                        loss_disc = disc_out[0] if isinstance(disc_out, tuple) else disc_out
                        loss_disc = loss_disc.mean() if torch.is_tensor(loss_disc) else loss_disc
                        if torch.is_tensor(loss_disc) and torch.isfinite(loss_disc):
                            for opt in opts.values():
                                opt.zero_grad(set_to_none=True)
                            (loss_disc / accum).backward()
                            step_optimizers(opts, [k for k in ("mpd", "msd") if k in opts])
                        else:
                            logger.warning("iter %s skip non-finite disc loss", iters)
                        gen_out = gl(y_tgt, y_gen)
                        loss_gen = gen_out[0] if isinstance(gen_out, tuple) else gen_out
                        loss_gen = loss_gen.mean() if torch.is_tensor(loss_gen) else loss_gen
                        if not (torch.is_tensor(loss_gen) and torch.isfinite(loss_gen)):
                            logger.warning("iter %s skip non-finite gen loss", iters)
                            loss_gen = torch.tensor(0.0, device=device)
                        else:
                            loss = loss + lambda_gen * loss_gen
                    if wl is not None and lambda_slm > 0:
                        try:
                            loss_slm = wl(y_tgt.detach().squeeze(1), y_gen.squeeze(1)).mean()
                        except Exception:
                            loss_slm = torch.tensor(0.0, device=device)
                        if not (torch.is_tensor(loss_slm) and torch.isfinite(loss_slm)):
                            logger.warning("iter %s skip non-finite slm loss", iters)
                            loss_slm = torch.tensor(0.0, device=device)
                        else:
                            loss = loss + lambda_slm * loss_slm

                if torch.is_tensor(loss) and not torch.isfinite(loss):
                    logger.warning("iter %s skip non-finite total loss", iters)
                    for opt in opts.values():
                        opt.zero_grad(set_to_none=True)
                    continue
                (loss / accum).backward()
            if (i + 1) % accum == 0:
                step_optimizers(opts, list(opts))
                for sch in schedulers.values():
                    sch.step()

            loss_slm_gen = torch.tensor(0.0, device=device)
            loss_slm_disc = torch.tensor(0.0, device=device)
            if (
                phase == "stage2_joint"
                and slmadv is not None
                and (i + 1) % accum == 0
            ):
                # StyleTTS2 train_second: 50% in-domain text, else OOD ref_texts
                use_ind = bool(np.random.rand() < 0.5)
                if use_ind:
                    slm_ref_texts, slm_ref_lengths = texts, input_lengths
                else:
                    slm_ref_texts, slm_ref_lengths = ref_texts, ref_lengths
                slm_sampler.style = s_trg.float() if torch.is_tensor(s_trg) else s_trg
                try:
                    with torch.autocast(device_type="cuda", enabled=False):
                        slm_out = slmadv(
                            iters,
                            wav_gt.float(),
                            y_rec_gt_pred.float(),
                            waves,
                            mel_input_length,
                            slm_ref_texts,
                            slm_ref_lengths,
                            use_ind,
                            s_trg.detach().float(),
                            s_trg.detach().float(),
                        )
                except Exception as exc:
                    logger.warning("iter %s SLM-adv failed: %s", iters, exc)
                    slm_out = None
                if slm_out is not None:
                    d_loss_slm, loss_gen_lm, _y_pred = slm_out
                    if torch.is_tensor(loss_gen_lm) and torch.isfinite(loss_gen_lm):
                        for opt in opts.values():
                            opt.zero_grad(set_to_none=True)
                        loss_gen_lm.backward()
                        # StyleTTS2 gradient scaling
                        total_norm = {}
                        for key in model.keys():
                            parameters = [
                                p
                                for p in model[key].parameters()
                                if p.grad is not None and p.requires_grad
                            ]
                            sq = sum(float(p.grad.detach().data.norm(2).item()) ** 2 for p in parameters)
                            total_norm[key] = sq**0.5
                        if total_norm.get("predictor", 0.0) > slmadv_thresh:
                            scale = 1.0 / max(total_norm["predictor"], 1e-6)
                            for key in model.keys():
                                for p in model[key].parameters():
                                    if p.grad is not None:
                                        p.grad.mul_(scale)
                        for p in model["predictor"].duration_proj.parameters():
                            if p.grad is not None:
                                p.grad.mul_(slmadv_scale)
                        for p in model["predictor"].lstm.parameters():
                            if p.grad is not None:
                                p.grad.mul_(slmadv_scale)
                        # StyleTTS2 steps bert_encoder / bert / predictor (/ diffusion skipped)
                        step_optimizers(
                            opts,
                            [k for k in ("bert_encoder", "bert", "predictor") if k in opts],
                        )
                        loss_slm_gen = loss_gen_lm.detach()
                    # StyleTTS2: update wd when discriminator term was computed this iter
                    if d_loss_slm != 0 and torch.is_tensor(d_loss_slm) and torch.isfinite(d_loss_slm):
                        for opt in opts.values():
                            opt.zero_grad(set_to_none=True)
                        d_loss_slm.backward(retain_graph=True)
                        step_optimizers(opts, [k for k in ("wd",) if k in opts])
                        loss_slm_disc = d_loss_slm.detach()

            def _acc(key, val):
                v = float(val.detach().item()) if torch.is_tensor(val) else float(val)
                if math.isfinite(v):
                    running[key] += v

            _acc("mel", loss_mel)
            _acc("dur", loss_dur)
            _acc("ce", loss_ce)
            _acc("f0", loss_f0)
            _acc("norm", loss_n)
            _acc("gen", loss_gen)
            _acc("slm", loss_slm)
            _acc("slm_gen", loss_slm_gen)
            _acc("slm_disc", loss_slm_disc)
            n_batches += 1
            iters += 1
            if iters % int(config.get("log_interval", 50)) == 0:
                avg = {k: v / max(n_batches, 1) for k, v in running.items()}
                lr_now = {k: opts[k].param_groups[0]["lr"] for k in opts}
                run_logger.log_train(iters, avg, lr_now)
                logger.info("iter %s %s", iters, avg)

            if save_interval > 0 and iters % save_interval == 0:
                mid_metrics = {
                    "mel": running["mel"] / max(n_batches, 1),
                    "dur": running["dur"] / max(n_batches, 1),
                    "ce": running["ce"] / max(n_batches, 1),
                }
                # Keep epoch == current loop index so resume continues this same epoch.
                shm_mid = shm_ckpt_dir / "mid.pth"
                shm_latest = shm_ckpt_dir / "latest.pth"
                vol_latest = layout["stage2"] / "latest.pth"
                save_checkpoint_resilient(
                    shm_mid, model, opts, schedulers, epoch, iters, phase, mid_metrics
                )
                save_checkpoint_resilient(
                    shm_latest, model, opts, schedulers, epoch, iters, phase, mid_metrics
                )
                saved = save_checkpoint_resilient(
                    vol_latest,
                    model,
                    opts,
                    schedulers,
                    epoch,
                    iters,
                    phase,
                    mid_metrics,
                    mirror=shm_latest,
                )
                try:
                    update_latest_pointer(
                        layout["state"],
                        "stage2",
                        Path(saved) if saved else shm_latest,
                        {
                            "epoch": epoch,
                            "iters": iters,
                            "wandb_id": run_logger.wandb_id,
                            "phase": phase,
                            "mid": True,
                        },
                    )
                except OSError as exc:
                    logger.warning("state pointer update failed: %s", exc)
                logger.info("mid-epoch checkpoint iters=%s (shm mid+latest, volume latest)", iters)

        val_mel = running["mel"] / max(n_batches, 1)
        val_metrics = {
            "mel": val_mel,
            "dur": running["dur"] / max(n_batches, 1),
            "ce": running["ce"] / max(n_batches, 1),
        }
        run_logger.log_val(iters, val_metrics)

        ckpt = layout["stage2"] / f"epoch_2nd_{epoch:05d}.pth"
        shm_latest = shm_ckpt_dir / "latest.pth"
        save_checkpoint_resilient(ckpt, model, opts, schedulers, epoch + 1, iters, phase, val_metrics, mirror=shm_latest)
        latest = layout["stage2"] / "latest.pth"
        save_checkpoint_resilient(latest, model, opts, schedulers, epoch + 1, iters, phase, val_metrics, mirror=shm_latest)
        save_checkpoint_resilient(shm_latest, model, opts, schedulers, epoch + 1, iters, phase, val_metrics)
        if val_mel < best_val:
            best_val = val_mel
            save_checkpoint_resilient(
                layout["stage2"] / "best_val.pth",
                model,
                opts,
                schedulers,
                epoch + 1,
                iters,
                phase,
                val_metrics,
                mirror=shm_ckpt_dir / "best_val.pth",
            )
        try:
            update_latest_pointer(
                layout["state"],
                "stage2",
                latest,
                {"epoch": epoch + 1, "iters": iters, "wandb_id": run_logger.wandb_id, "phase": phase},
            )
        except OSError as exc:
            logger.warning("state pointer update failed: %s", exc)
        try:
            apply_retention(layout["stage2"], keep_last=int(config.get("keep_last", 3)))
        except OSError as exc:
            logger.warning("retention failed: %s", exc)

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
