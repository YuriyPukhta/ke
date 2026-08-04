"""GT caches for LibriTTS: F0 (JDC / JDC-ft / RMVPE), N (log_norm), d_gt (alignment).

Features live under ``gt_cache/features/{f0,f0_jdc_ft,f0_rmvpe,n}/``.
``d_gt`` is stored in filtered manifest CSVs (``timbre_encoder/manifests/``).
"""
from .utils import DEFAULT_FEATURES_ROOT, feature_path, load_curve

__all__ = ["DEFAULT_FEATURES_ROOT", "feature_path", "load_curve"]
