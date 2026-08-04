"""Legacy shim — prefer ``python -m gt_cache.cache_d_gt`` / ``gt_cache``."""
from gt_cache.cache_d_gt import *  # noqa: F403
from gt_cache.cache_d_gt import main

if __name__ == "__main__":
    main()
