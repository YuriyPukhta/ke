"""
Normalize Kokoro/Misaki phoneme strings to StyleTTS2 TextCleaner alphabet.

Length can change:
  - 1:1 replace (ᵊ→ə): same N tokens; acoustic identity changes, MAS may
    assign different frame counts than if ᵊ were kept.
  - drop (parentheses, combining ̃): N decreases; those mel frames are
    redistributed to neighbors by MAS (not “lost samples”, but different
    phone boundaries).
  - split (ʦ→ts): N increases; one affricate becomes two duration slots.

Always re-run the aligner on the *normalized* string so d_gt matches it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from timbre_encoder.aligner import ST2_SYMBOL_TO_ID

# 1:1 — preserve token count
CHAR_MAP_1TO1: dict[str, str] = {
    "ᵊ": "ə",  # superscript schwa → schwa (main Kokoro/Misaki mismatch)
    "ᵝ": "β",  # labialization mark → beta (closest ST2 IPA)
    "ᵻ": "ᵻ",  # already in ST2; listed for clarity
}

# 1:N expansions (affricates absent from ST2 single-symbol set)
CHAR_EXPAND: dict[str, str] = {
    "ʦ": "ts",
    "ʨ": "tʃ",
    "ʣ": "dz",
    "ʥ": "dʒ",
}

# Drop entirely (no ST2 symbol; not worth inventing a phone)
CHAR_DROP: set[str] = set("()[]<>{}̃")  # includes combining nasal tilde U+0303


@dataclass
class NormResult:
    raw: str
    normalized: str
    changed: bool
    ops: list[str] = field(default_factory=list)
    len_raw: int = 0
    len_norm: int = 0
    n_replace: int = 0
    n_expand: int = 0
    n_drop: int = 0
    unknown_dropped: list[str] = field(default_factory=list)

    @property
    def delta_len(self) -> int:
        """len_norm - len_raw (negative => fewer tokens after norm)."""
        return self.len_norm - self.len_raw

    def ops_str(self) -> str:
        return "|".join(self.ops) if self.ops else ""


def normalize_kokoro_phonemes_for_st2(phonemes: str) -> NormResult:
    """
    Map Kokoro G2P output onto symbols the ST2 aligner accepts.

    Unknown chars still missing from ST2 after map/expand are dropped and
    listed in unknown_dropped (same silent drop TextCleaner would do).
    """
    out_chars: list[str] = []
    ops: list[str] = []
    n_replace = n_expand = n_drop = 0
    unknown: list[str] = []

    for ch in phonemes:
        if ch in CHAR_DROP:
            n_drop += 1
            ops.append(f"drop:{ch}")
            continue
        if ch in CHAR_EXPAND:
            exp = CHAR_EXPAND[ch]
            out_chars.extend(exp)
            n_expand += 1
            ops.append(f"expand:{ch}->{exp}")
            continue
        mapped = CHAR_MAP_1TO1.get(ch, ch)
        if mapped != ch:
            n_replace += 1
            ops.append(f"map:{ch}->{mapped}")
            ch = mapped
        # final gate: must be in ST2
        if ch not in ST2_SYMBOL_TO_ID:
            # multi-char from expand already split; single unknown → drop
            n_drop += 1
            unknown.append(ch)
            ops.append(f"drop_unknown:{ch}")
            continue
        out_chars.append(ch)

    normalized = "".join(out_chars)
    # collapse duplicate op labels for compact CSV (keep order, unique consecutive counts in ops_str via full list)
    # For CSV we store a compact summary separately; keep raw ops list capped
    if len(ops) > 40:
        ops = ops[:40] + [f"...+{len(ops) - 40}more"]

    return NormResult(
        raw=phonemes,
        normalized=normalized,
        changed=normalized != phonemes,
        ops=ops,
        len_raw=len(phonemes),
        len_norm=len(normalized),
        n_replace=n_replace,
        n_expand=n_expand,
        n_drop=n_drop,
        unknown_dropped=unknown,
    )


def norm_summary(nr: NormResult) -> str:
    """Short flag for CSV: e.g. repl=2|drop=1|dlen=-1"""
    if not nr.changed:
        return ""
    parts = []
    if nr.n_replace:
        parts.append(f"repl={nr.n_replace}")
    if nr.n_expand:
        parts.append(f"exp={nr.n_expand}")
    if nr.n_drop:
        parts.append(f"drop={nr.n_drop}")
    parts.append(f"dlen={nr.delta_len}")
    return "|".join(parts)
