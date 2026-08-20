"""TensorBoard + optional W&B + JSONL metrics logging."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _append_jsonl(path: Path, row: dict[str, Any], retries: int = 3) -> None:
    """Append one JSON line; tolerate transient network-volume I/O errors."""
    line = json.dumps(row) + "\n"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            return
        except OSError as exc:
            last_exc = exc
            logger.warning(
                "jsonl write failed (%s) attempt %s/%s: %s",
                path.name,
                attempt + 1,
                retries,
                exc,
            )
            time.sleep(0.5 * (attempt + 1))
    # Last resort: write beside the original so training can continue.
    fallback = path.with_suffix(path.suffix + ".fallback")
    try:
        with fallback.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
        logger.error(
            "jsonl write to %s failed after retries; appended to %s (%s)",
            path,
            fallback.name,
            last_exc,
        )
    except OSError as exc:
        logger.error("jsonl write completely failed for %s: %s / fallback: %s", path, last_exc, exc)


class RunLogger:
    def __init__(
        self,
        run_root: Path,
        stage: str,
        enable_wandb: bool = True,
        project: str = "kokoro-libritts",
        run_name: str | None = None,
        resume_id: str | None = None,
    ):
        self.stage = stage
        self.metrics_dir = Path(run_root) / "metrics"
        self.tb_dir = Path(run_root) / "logs" / "tensorboard" / stage
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.train_jsonl = self.metrics_dir / f"{stage}_train.jsonl"
        self.val_jsonl = self.metrics_dir / f"{stage}_val.jsonl"
        self.summary_path = self.metrics_dir / "summary.json"
        self._writer = None
        self._wandb = None
        self._t0 = time.time()

        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.tb_dir))
        except Exception as exc:
            logger.warning("TensorBoard unavailable: %s", exc)

        if enable_wandb and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                self._wandb = wandb.init(
                    project=project,
                    name=run_name or f"{stage}-{int(time.time())}",
                    id=resume_id,
                    resume="allow" if resume_id else None,
                    dir=str(Path(run_root) / "logs"),
                    config={"stage": stage},
                )
            except Exception as exc:
                logger.warning("W&B init failed, continuing with local logs: %s", exc)
                self._wandb = None
        elif enable_wandb:
            logger.info("WANDB_API_KEY not set; using TensorBoard/JSONL only")

    def log_train(self, step: int, metrics: dict[str, float], lr: dict[str, float] | None = None) -> None:
        row = {"step": step, "time": time.time(), **metrics}
        if lr:
            row.update({f"lr/{k}": v for k, v in lr.items()})
        _append_jsonl(self.train_jsonl, row)
        if self._writer is not None:
            try:
                for k, v in metrics.items():
                    self._writer.add_scalar(f"train/{k}", v, step)
                if lr:
                    for k, v in lr.items():
                        self._writer.add_scalar(f"lr/{k}", v, step)
            except Exception as exc:
                logger.warning("TensorBoard train log failed: %s", exc)
        if self._wandb is not None:
            try:
                payload = {f"train/{k}": v for k, v in metrics.items()}
                if lr:
                    payload.update({f"lr/{k}": v for k, v in lr.items()})
                payload["step"] = step
                self._wandb.log(payload, step=step)
            except Exception as exc:
                logger.warning("W&B train log failed: %s", exc)

    def log_val(self, step: int, metrics: dict[str, float]) -> None:
        row = {"step": step, "time": time.time(), **metrics}
        _append_jsonl(self.val_jsonl, row)
        if self._writer is not None:
            try:
                for k, v in metrics.items():
                    self._writer.add_scalar(f"val/{k}", v, step)
            except Exception as exc:
                logger.warning("TensorBoard val log failed: %s", exc)
        if self._wandb is not None:
            try:
                self._wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
            except Exception as exc:
                logger.warning("W&B val log failed: %s", exc)

    def log_audio(self, tag: str, audio, step: int, sample_rate: int = 24000) -> None:
        if self._writer is not None:
            try:
                self._writer.add_audio(tag, audio, step, sample_rate=sample_rate)
            except Exception:
                pass
        if self._wandb is not None:
            try:
                import wandb
                import numpy as np

                arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio
                self._wandb.log({tag: wandb.Audio(np.asarray(arr).squeeze(), sample_rate=sample_rate)}, step=step)
            except Exception:
                pass

    def log_summary(self, payload: dict[str, Any]) -> None:
        payload = {**payload, "elapsed_sec": time.time() - self._t0}
        try:
            self.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.error("summary write failed: %s", exc)

    @property
    def wandb_id(self) -> str | None:
        if self._wandb is None:
            return None
        return getattr(self._wandb, "id", None)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
