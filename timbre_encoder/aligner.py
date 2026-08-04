"""
Frozen StyleTTS2 Text Aligner (TMA) + pure-PyTorch monotonic alignment search (MAS).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_ASR_DIR = _ROOT / "third_party" / "styletts2_asr"
_THIRD = _ROOT / "third_party"
if str(_THIRD) not in sys.path:
    sys.path.insert(0, str(_THIRD))


def _load_asr_cnn():
    """Load ASRCNN from vendored StyleTTS2 ASR without polluting imports."""
    layers_path = _ASR_DIR / "layers.py"
    models_path = _ASR_DIR / "models.py"

    spec_layers = importlib.util.spec_from_file_location("styletts2_asr_layers", layers_path)
    layers_mod = importlib.util.module_from_spec(spec_layers)
    sys.modules["styletts2_asr_layers"] = layers_mod
    # models.py does `from .layers import ...` — register as package
    pkg_name = "styletts2_asr_pkg"
    if pkg_name not in sys.modules:
        import types

        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_ASR_DIR)]
        sys.modules[pkg_name] = pkg
    sys.modules[f"{pkg_name}.layers"] = layers_mod
    spec_layers.loader.exec_module(layers_mod)

    spec_models = importlib.util.spec_from_file_location(
        f"{pkg_name}.models", models_path, submodule_search_locations=[str(_ASR_DIR)]
    )
    models_mod = importlib.util.module_from_spec(spec_models)
    sys.modules[f"{pkg_name}.models"] = models_mod
    # Patch relative import
    models_path_src = models_path.read_text(encoding="utf-8")
    models_path_src = models_path_src.replace(
        "from .layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock",
        "from styletts2_asr_pkg.layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock",
    )
    exec(compile(models_path_src, str(models_path), "exec"), models_mod.__dict__)
    return models_mod.ASRCNN


ASRCNN = _load_asr_cnn()

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
ST2_SYMBOL_TO_ID = {s: i for i, s in enumerate(_symbols)}


class TextCleaner:
    def __call__(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            idx = ST2_SYMBOL_TO_ID.get(ch)
            if idx is not None:
                ids.append(idx)
        return ids


def maximum_path_torch(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MAS without Cython. attn/mask: [B, T_text, T_mel]."""
    device = attn.device
    dtype = attn.dtype
    b, t_t, t_s = attn.shape
    paths = torch.zeros_like(attn)
    for bi in range(b):
        tt = int(mask[bi, :, 0].sum().item())
        ts = int(mask[bi, 0, :].sum().item())
        if tt < 1 or ts < 1:
            continue
        value = attn[bi, :tt, :ts].detach().float().cpu().clone()
        direction = torch.zeros(tt, ts, dtype=torch.int8)
        for y in range(tt):
            for x in range(ts):
                v_cur = value[y, x].item()
                if y == 0 and x == 0:
                    continue
                if y == 0:
                    value[y, x] = v_cur + value[y, x - 1]
                    direction[y, x] = 1
                elif x == 0:
                    value[y, x] = v_cur + value[y - 1, x]
                    direction[y, x] = 2
                else:
                    left = value[y, x - 1].item()
                    up = value[y - 1, x].item()
                    if left >= up:
                        value[y, x] = v_cur + left
                        direction[y, x] = 1
                    else:
                        value[y, x] = v_cur + up
                        direction[y, x] = 2
        path = torch.zeros(tt, ts)
        y, x = tt - 1, ts - 1
        while True:
            path[y, x] = 1.0
            if y == 0 and x == 0:
                break
            if y == 0:
                x -= 1
            elif x == 0:
                y -= 1
            else:
                if int(direction[y, x].item()) == 1:
                    x -= 1
                else:
                    y -= 1
        paths[bi, :tt, :ts] = path.to(device=device, dtype=dtype)
    return paths


def mask_from_lens(attn: torch.Tensor, text_lengths: torch.Tensor, mel_lengths: torch.Tensor) -> torch.Tensor:
    b, tt, tm = attn.shape
    device = attn.device
    t_mask = torch.arange(tt, device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
    m_mask = torch.arange(tm, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)
    return t_mask.unsqueeze(-1) & m_mask.unsqueeze(1)


class FrozenTextAligner(torch.nn.Module):
    def __init__(
        self,
        checkpoint: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        config_path = Path(config_path or (_ASR_DIR / "config.yml"))
        checkpoint = Path(
            checkpoint
            or (Path(__file__).resolve().parent / "checkpoints" / "asr_epoch_00080.pth")
        )
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mp = cfg["model_params"]
        self.model = ASRCNN(
            input_dim=mp["input_dim"],
            hidden_dim=mp["hidden_dim"],
            n_token=mp["n_token"],
            token_embedding_dim=mp["token_embedding_dim"],
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.n_down = int(self.model.n_down)
        self.text_cleaner = TextCleaner()
        self._device = torch.device(device)
        self.to(self._device)

    def phonemes_to_ids(self, phonemes: str) -> torch.Tensor:
        ids = self.text_cleaner(phonemes)
        if not ids:
            ids = [0]
        return torch.tensor(ids, dtype=torch.long, device=self._device)

    @torch.no_grad()
    def align(self, mel: torch.Tensor, phonemes: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (a_gt, d_gt, soft_attn).

        a_gt: hard MAS path [1, N_phonemes, T_mel_down] — one 1 per mel frame
        d_gt: frames per phoneme [N_phonemes] = a_gt.sum(-1)
        soft_attn: aligner attention used for MAS [1, N, T_mel_down]

        Matches StyleTTS2 / st2_runtime: attention already softmaxed by ASR;
        MAS via monotonic_align.maximum_path (no second softmax).
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.shape[-1] % 2 == 1:
            mel = mel[..., :-1]
        mel = mel.to(self._device)
        text_ids = self.phonemes_to_ids(phonemes).unsqueeze(0)
        text_len = torch.tensor([text_ids.shape[1]], device=self._device)
        mel_len = torch.tensor([mel.shape[-1]], device=self._device)
        mel_len_down = torch.clamp(mel_len // (2**self.n_down), min=1)

        mask = self.model.length_to_mask(mel_len_down)
        _, _, alignments = self.model(mel, mask, text_ids)
        # [B, T_text+1, T_mel_down] — drop BOS column like StyleTTS2
        attn = alignments[:, 1 : 1 + text_ids.shape[1], : mel_len_down.item()].contiguous()

        # Prefer NumPy MAS (same as train); fall back to pure-torch
        try:
            from monotonic_align import mask_from_lens as mas_mask
            from monotonic_align import maximum_path as mas_path

            mask_st = mas_mask(attn, text_len, mel_len_down)
            a_gt = mas_path(attn, mask_st)
        except Exception:
            soft = attn
            mask_st = mask_from_lens(soft, text_len, mel_len_down)
            soft = soft * mask_st.float()
            a_gt = maximum_path_torch(soft, mask_st.float())

        d_gt = a_gt.sum(dim=-1).squeeze(0)
        return a_gt, d_gt, attn

    @staticmethod
    def d_gt_to_a_gt(d_gt: torch.Tensor | list[int], device=None) -> torch.Tensor:
        """Rebuild hard a_gt [1, N, T] from durations (what we cache in CSV)."""
        if isinstance(d_gt, torch.Tensor):
            d_list = [max(0, int(round(float(x)))) for x in d_gt.tolist()]
            device = device or d_gt.device
        else:
            d_list = [max(0, int(x)) for x in d_gt]
            device = device or torch.device("cpu")
        total = sum(d_list)
        if total < 1:
            return torch.zeros(1, max(1, len(d_list)), 1, device=device)
        a_gt = torch.zeros(1, len(d_list), total, device=device)
        cursor = 0
        for i, dur in enumerate(d_list):
            if dur > 0:
                a_gt[0, i, cursor : cursor + dur] = 1.0
            cursor += dur
        return a_gt

    @torch.no_grad()
    def alignment_for_kokoro(
        self, mel: torch.Tensor, phonemes: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Kokoro alignment [1, N+2, T] with BOS/EOS duration 1."""
        _, d_gt, _ = self.align(mel, phonemes)
        d_list = [1] + [max(1, int(round(float(x)))) for x in d_gt.tolist()] + [1]
        total = sum(d_list)
        aln = torch.zeros(1, len(d_list), total, device=self._device)
        cursor = 0
        for i, dur in enumerate(d_list):
            aln[0, i, cursor : cursor + dur] = 1.0
            cursor += dur
        d_full = torch.tensor(d_list, device=self._device, dtype=torch.long)
        return aln, d_full
