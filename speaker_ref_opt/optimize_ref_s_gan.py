"""
Optimize two Kokoro style vectors (timbre[:128] + absolute prosody[128:]) with
StyleTTS2 waveform GAN (MPD+MSD d_loss / loss_gen) + mel + WavLM.

No neural encoder — Adam on the two vectors directly.
Forced d_gt alignment; optional F0/N override (jdc_ft / cache / rmvpe / baseline).

  .venv\\Scripts\\python.exe speaker_ref_opt/optimize_ref_s_gan.py \\
      --speaker 6209 --steps 200 --batch-size 4 --lambda-gen 1.0 \\
      --f0n-source jdc_ft --d-gt-source tma_ft \\
      --enc-device cuda --kokoro-device cuda --slm-device cpu --disc-device cpu
"""
from __future__ import annotations

import argparse
import html as html_lib
import random
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(HERE))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache import DEFAULT_FEATURES_ROOT  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import freeze_module  # noqa: E402
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.slm_loss import WavLMFeatureLoss  # noqa: E402
from timbre_encoder.voice_bank import load_bank, normalize_to_bank  # noqa: E402

from optimize_speaker_ref_s import (  # noqa: E402
    assemble_pack_both,
    build_alignment,
    kokoro_ids,
    load_wav,
    make_ref_s,
    pack_slot,
)
from st2_gan import StyleTTS2GAN  # noqa: E402
from train_encoder import (  # noqa: E402
    d_gt_fields,
    load_rows,
    resolve_f0n_override,
)


@torch.no_grad()
def write_listen(
    out_dir: Path,
    kmodel: KModel,
    baseline_pack: torch.Tensor,
    timbre: torch.Tensor,
    prosody: torch.Tensor,
    holdout: list[dict[str, str]],
    device: torch.device,
    features_root: Path,
    args: argparse.Namespace,
) -> None:
    listen = out_dir / "listen_test"
    listen.mkdir(parents=True, exist_ok=True)
    d_col, _ = d_gt_fields(args.d_gt_source)
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>ref_s GAN</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:1.5rem auto}"
        "audio{width:100%}.card{margin:1rem 0;padding:1rem;border:1px solid #ccc}</style>"
        "</head><body>",
        f"<h1>{html_lib.escape(out_dir.name)}</h1>",
        f"<p>vectors both · d_gt={html_lib.escape(args.d_gt_source)} · "
        f"f0n={html_lib.escape(args.f0n_source)} · lambda_gen={args.lambda_gen}</p>",
        "<p>ref | opt+ov_f0n | timbre+base_p+ov | base_a+prosody+ov | baseline+ov</p>",
    ]
    t_cpu = timbre.detach()
    p_cpu = prosody.detach()
    for i, row in enumerate(holdout[: args.listen_n]):
        ph = (row.get("phonemes_st2") or "").strip()
        d_gt = parse_d_gt(row.get(d_col))
        if not ph or not d_gt:
            continue
        input_ids, ids, dropped = kokoro_ids(kmodel, ph, device)
        if dropped or len(ids) != len(ph):
            continue
        aln = build_alignment(d_gt, device)
        if aln.shape[1] != input_ids.shape[1]:
            continue
        wav = load_wav(row["audio_path"], args.max_wav_sec)
        slot = pack_slot(baseline_pack, len(ph))
        base_a = slot[0, :128]
        base_p = slot[0, 128:]
        ov = resolve_f0n_override(
            kmodel,
            input_ids,
            slot,
            aln,
            row,
            wav,
            features_root,
            args.f0n_source if args.f0n_source != "baseline" else "jdc_ft",
            device,
        )

        def synth(a: torch.Tensor, p: torch.Tensor):
            ref_s = make_ref_s(a.to(device), p.to(device))
            kw = {}
            if ov is not None:
                kw["f0_override"], kw["n_override"] = ov
            audio, _, _, _ = kmodel.forward_with_alignment(input_ids, ref_s, aln, **kw)
            if audio.numel() < 100:
                return None
            return audio.detach().cpu().numpy().reshape(-1)

        stem = f"{i:02d}_{row['speaker_id']}"
        sf.write(str(listen / f"{stem}_0_ref.wav"), wav.numpy(), SAMPLE_RATE)
        block = [
            f"<div class='card'><h3>{html_lib.escape(stem)}</h3>",
            f"<p>{html_lib.escape((row.get('text') or '')[:160])}</p>",
            f"<p>ref</p><audio controls src='{stem}_0_ref.wav'></audio>",
        ]

        def add(label: str, audio, name: str) -> None:
            if audio is None:
                block.append(f"<p>{label}: SKIP</p>")
                return
            sf.write(str(listen / name), audio, SAMPLE_RATE)
            block.append(f"<p>{label}</p><audio controls src='{name}'></audio>")

        add("opt + ov_f0n", synth(t_cpu, p_cpu), f"{stem}_1_opt.wav")
        add("timbre + base_p + ov", synth(t_cpu, base_p), f"{stem}_2_timbre_base_p.wav")
        add("base_a + prosody + ov", synth(base_a, p_cpu), f"{stem}_3_base_a_prosody.wav")
        add("baseline + ov", synth(base_a, base_p), f"{stem}_4_baseline.wav")
        block.append("</div>")
        lines.extend(block)
    lines.append("</body></html>")
    (listen / "index.html").write_text("\n".join(lines), encoding="utf-8")
    print(f"listen -> {listen / 'index.html'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optimize timbre+prosody vectors with StyleTTS2 GAN"
    )
    ap.add_argument("--speaker", type=str, default="6209")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--baseline", type=str, default="mean")
    ap.add_argument(
        "--bank-cache",
        type=str,
        default=str(HERE / "runs/speaker_ref_s_6209_bankmean_lr1e5/voice_bank.pt"),
    )
    ap.add_argument("--kokoro-device", type=str, default="cuda")
    ap.add_argument("--slm-device", type=str, default="cpu")
    ap.add_argument("--disc-device", type=str, default="cpu")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lambda-mel", type=float, default=0.25)
    ap.add_argument("--lambda-slm", type=float, default=1.0)
    ap.add_argument("--slm-cos", action="store_true")
    ap.add_argument("--lambda-res", type=float, default=0.0)
    ap.add_argument("--lambda-gen", type=float, default=1.0)
    ap.add_argument("--disc-lr", type=float, default=1e-4)
    ap.add_argument("--slim-disc", action="store_true")
    ap.add_argument("--slm-layers", type=str, default="4,8,12")
    ap.add_argument(
        "--f0n-source",
        choices=("cache", "rmvpe", "jdc_ft", "baseline"),
        default="jdc_ft",
    )
    ap.add_argument(
        "--d-gt-source",
        choices=("baseline", "tma_ft"),
        default="tma_ft",
    )
    ap.add_argument("--features-root", type=str, default=str(DEFAULT_FEATURES_ROOT))
    ap.add_argument("--max-clips", type=int, default=64)
    ap.add_argument("--max-phonemes", type=int, default=50)
    ap.add_argument("--max-wav-sec", type=float, default=8.0)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--listen-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--norm-to-bank", action="store_true")
    ap.add_argument("--z-clip", type=float, default=2.5)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if not args.slm_cos and args.lambda_slm > 0:
        args.slm_cos = True

    device = torch.device(
        args.kokoro_device
        if args.kokoro_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    slm_device = torch.device(args.slm_device)
    disc_device = torch.device(
        args.disc_device
        if args.disc_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    features_root = Path(args.features_root)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            HERE
            / "runs"
            / (
                f"ref_s_gan_{args.speaker}_dgt{args.d_gt_source}"
                f"_f0{args.f0n_source}_mel{args.lambda_mel}_gen{args.lambda_gen}"
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("VECTOR OPT + StyleTTS2 GAN (timbre + absolute prosody)", flush=True)
    print(f"  Kokoro  -> {device}", flush=True)
    print(f"  WavLM   -> {slm_device}", flush=True)
    print(f"  GAN disc-> {disc_device} (lambda_gen={args.lambda_gen})", flush=True)
    print(f"  d_gt    -> {args.d_gt_source}", flush=True)
    print(f"  F0/N    -> {args.f0n_source}", flush=True)
    print("=" * 60, flush=True)
    print(
        f"speaker={args.speaker} steps={args.steps} batch={args.batch_size} lr={args.lr} "
        f"mel={args.lambda_mel} slm={args.lambda_slm} gen={args.lambda_gen} "
        f"res={args.lambda_res} out={out_dir}",
        flush=True,
    )

    need_cache = args.f0n_source in ("cache", "rmvpe", "jdc_ft")
    rows = load_rows(
        Path(args.manifest),
        args.speaker or None,
        max_clips=args.max_clips + args.holdout,
        max_phonemes=args.max_phonemes,
        require_f0n=need_cache,
        features_root=features_root if need_cache else None,
        f0n_source=args.f0n_source,
        d_gt_source=args.d_gt_source,
    )
    if len(rows) < args.holdout + args.batch_size:
        raise SystemExit(f"not enough clips: {len(rows)}")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    holdout = rows[: args.holdout]
    train_rows = rows[args.holdout :]
    if args.max_clips > 0:
        train_rows = train_rows[: args.max_clips]
    print(f"train={len(train_rows)} holdout={len(holdout)}", flush=True)

    pipeline = KPipeline(lang_code="a", model=False)
    bank = None
    if args.baseline == "mean":
        cache = Path(args.bank_cache)
        if not cache.is_file():
            raise SystemExit(f"bank cache missing: {cache}")
        bank = load_bank(cache, device="cpu")
        baseline_pack = bank.mean_pack.to(device)
        timbre0 = bank.timbre_mean.detach().to(device)
        prosody0 = bank.prosody_mean.detach().to(device)
        print("init = bank mean timbre + prosody", flush=True)
    else:
        baseline_pack = pipeline.load_voice(args.baseline).to(device)
        if baseline_pack.dim() == 2:
            baseline_pack = baseline_pack.unsqueeze(1)
        timbre0 = baseline_pack[0, 0, :128].detach().clone()
        prosody0 = baseline_pack[0, 0, 128:].detach().clone()

    print(f"Loading Kokoro on {device}...", flush=True)
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    freeze_module(kmodel)

    timbre = timbre0.clone().requires_grad_(True)
    prosody = prosody0.clone().requires_grad_(True)
    opt = torch.optim.Adam([timbre, prosody], lr=args.lr)

    slm = None
    if args.lambda_slm > 0:
        layers = [int(x) for x in args.slm_layers.split(",") if x.strip()]
        print(
            f"Loading WavLM on {slm_device} layers={layers} include_cos={args.slm_cos}...",
            flush=True,
        )
        slm = WavLMFeatureLoss(
            layers=layers, device=slm_device, include_cos=args.slm_cos
        )

    gan: StyleTTS2GAN | None = None
    opt_d = None
    if args.lambda_gen > 0:
        print(
            f"Loading StyleTTS2 MPD+MSD on {disc_device} slim={args.slim_disc}...",
            flush=True,
        )
        gan = StyleTTS2GAN(slim=args.slim_disc).to(disc_device)
        gan.train()
        opt_d = torch.optim.Adam(gan.parameters(), lr=args.disc_lr)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(
            f"cuda allocated after load: {torch.cuda.memory_allocated()/1024**2:.0f} MiB",
            flush=True,
        )

    d_col, _ = d_gt_fields(args.d_gt_source)
    history: list[dict[str, float]] = []
    skipped = 0
    bs = max(1, args.batch_size)

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        acc = {
            "loss": 0.0,
            "mel": 0.0,
            "slm": 0.0,
            "res": 0.0,
            "gen": 0.0,
            "d": 0.0,
        }
        n_ok = 0
        attempts = 0
        while n_ok < bs and attempts < bs * 20:
            attempts += 1
            row = train_rows[rng.randrange(len(train_rows))]
            ph = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get(d_col))
            if not ph or not d_gt:
                skipped += 1
                continue
            try:
                input_ids, ids, dropped = kokoro_ids(kmodel, ph, device)
                if dropped or len(ids) != len(ph):
                    skipped += 1
                    continue
                aln = build_alignment(d_gt, device)
                if aln.shape[1] != input_ids.shape[1]:
                    skipped += 1
                    continue
                wav = load_wav(row["audio_path"], args.max_wav_sec)
                if wav.numel() < SAMPLE_RATE // 4:
                    skipped += 1
                    continue

                slot = pack_slot(baseline_pack, len(ph))
                ref_s = make_ref_s(timbre, prosody)
                with torch.no_grad():
                    ov = resolve_f0n_override(
                        kmodel,
                        input_ids,
                        slot,
                        aln,
                        row,
                        wav,
                        features_root,
                        args.f0n_source,
                        device,
                    )
                if ov is None:
                    skipped += 1
                    continue
                f0_ov, n_ov = ov
                audio, _, _, _ = kmodel.forward_with_alignment(
                    input_ids, ref_s, aln, f0_override=f0_ov, n_override=n_ov
                )
                if audio.numel() < 100:
                    skipped += 1
                    continue
                pred_w = audio.view(1, -1) if audio.dim() == 1 else audio
                real_w = wav.view(1, -1).to(pred_w.device)

                zero = timbre.new_zeros(())
                loss_mel = zero
                loss_slm = zero
                loss_res = zero
                if args.lambda_mel > 0:
                    pred_mel = waveform_to_mel(pred_w, device=device)
                    tgt_mel = waveform_to_mel(real_w, device=device)
                    t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
                    if t < 4:
                        skipped += 1
                        continue
                    loss_mel = F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t].detach())
                if slm is not None and args.lambda_slm > 0:
                    tw = min(pred_w.shape[-1], real_w.shape[-1])
                    loss_slm = slm(pred_w[..., :tw], real_w[..., :tw].detach())
                if args.lambda_res > 0:
                    loss_res = (timbre - timbre0).pow(2).mean() + (
                        prosody - prosody0
                    ).pow(2).mean()

                loss_g = (
                    args.lambda_mel * loss_mel
                    + args.lambda_slm * loss_slm
                    + args.lambda_res * loss_res
                )
                loss_gen_v = 0.0
                loss_d_v = 0.0
                if gan is not None and opt_d is not None:
                    opt_d.zero_grad(set_to_none=True)
                    d_loss = gan.d_loss(
                        real_w.detach().to(disc_device),
                        pred_w.detach().to(disc_device),
                    )
                    d_loss.backward()
                    opt_d.step()
                    loss_d_v = float(d_loss.detach().item())
                    opt_d.zero_grad(set_to_none=True)
                    g_loss = gan.g_loss(
                        real_w.detach().to(disc_device),
                        pred_w.to(disc_device),
                    )
                    loss_g = loss_g + args.lambda_gen * g_loss.to(device)
                    loss_gen_v = float(g_loss.detach().item())

                (loss_g / bs).backward()
                acc["mel"] += float(loss_mel.detach().item())
                acc["slm"] += float(
                    loss_slm.detach().item() if torch.is_tensor(loss_slm) else 0.0
                )
                acc["res"] += float(
                    loss_res.detach().item() if torch.is_tensor(loss_res) else 0.0
                )
                acc["gen"] += loss_gen_v
                acc["d"] += loss_d_v
                acc["loss"] += float(loss_g.detach().item())
                n_ok += 1
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                skipped += 1
                print(f"step={step} OOM microbatch, skip", flush=True)
            except Exception as e:
                skipped += 1
                if step == 1 and n_ok == 0:
                    print(f"step={step} skip: {e}", flush=True)

        if n_ok == 0:
            print(f"step={step} empty batch", flush=True)
            continue
        if n_ok < bs:
            scale = bs / n_ok
            for p in (timbre, prosody):
                if p.grad is not None:
                    p.grad.mul_(scale)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([timbre, prosody], args.grad_clip)
        opt.step()

        if args.norm_to_bank and bank is not None:
            with torch.no_grad():
                timbre.copy_(
                    normalize_to_bank(
                        timbre, bank.timbre_mean.to(device), bank.timbre_std.to(device), args.z_clip
                    )
                )
                prosody.copy_(
                    normalize_to_bank(
                        prosody,
                        bank.prosody_mean.to(device),
                        bank.prosody_std.to(device),
                        args.z_clip,
                    )
                )

        hist = {k: acc[k] / n_ok for k in acc}
        hist["step"] = float(step)
        history.append(hist)
        if step == 1 or step % 10 == 0 or step == args.steps:
            mem = ""
            if device.type == "cuda":
                mem = f" alloc={torch.cuda.memory_allocated()/1024**2:.0f}MiB"
            print(
                f"step={step}/{args.steps} batch={n_ok}/{bs} "
                f"loss={hist['loss']:.4f} mel={hist['mel']:.4f} "
                f"slm={hist['slm']:.4f} gen={hist['gen']:.4f} d={hist['d']:.4f} "
                f"skipped={skipped}{mem}",
                flush=True,
            )
        if device.type == "cuda" and step % 20 == 0:
            torch.cuda.empty_cache()

    t_cpu = timbre.detach().cpu()
    p_cpu = prosody.detach().cpu()
    ckpt = {
        "timbre": t_cpu,
        "prosody": p_cpu,
        "prosody_mode": "absolute",
        "speaker": args.speaker,
        "baseline": args.baseline,
        "steps": args.steps,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    if gan is not None:
        ckpt["gan"] = gan.state_dict()
    torch.save(ckpt, out_dir / "ref_s.pt")
    torch.save({"timbre": t_cpu, "speaker": args.speaker}, out_dir / "timbre.pt")
    torch.save({"prosody": p_cpu, "speaker": args.speaker}, out_dir / "prosody.pt")
    pack = assemble_pack_both(t_cpu, p_cpu, baseline_pack.cpu(), mode="replace")
    torch.save(pack, out_dir / "voice_pack.pt")

    def avg(key: str, xs: list[dict[str, float]]) -> float:
        if not xs:
            return float("nan")
        return sum(h[key] for h in xs) / len(xs)

    first = history[:50]
    last = history[-50:] if len(history) >= 50 else history
    summary = {
        "speaker": args.speaker,
        "steps": args.steps,
        "batch_size": bs,
        "lr": args.lr,
        "lambda_mel": args.lambda_mel,
        "lambda_slm": args.lambda_slm,
        "lambda_gen": args.lambda_gen,
        "f0n_source": args.f0n_source,
        "d_gt_source": args.d_gt_source,
        "train_clips": len(train_rows),
        "skipped": skipped,
        "loss_first50": avg("loss", first),
        "loss_last50": avg("loss", last),
        "mel_last50": avg("mel", last),
        "slm_last50": avg("slm", last),
        "gen_last50": avg("gen", last),
        "d_last50": avg("d", last),
        "delta_timbre_l2": float((t_cpu - timbre0.cpu()).pow(2).mean()),
        "delta_prosody_l2": float((p_cpu - prosody0.cpu()).pow(2).mean()),
    }
    print("summary", summary, flush=True)
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in summary.items()) + "\n", encoding="utf-8"
    )

    write_listen(
        out_dir,
        kmodel,
        baseline_pack,
        timbre,
        prosody,
        holdout,
        device,
        features_root,
        args,
    )
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
