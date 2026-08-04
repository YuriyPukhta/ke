#!/usr/bin/env python3
"""Backward-compatible entrypoint → data.prepare_index """

from runpod_training.data.prepare_index import main

if __name__ == "__main__":
    raise SystemExit(main())
