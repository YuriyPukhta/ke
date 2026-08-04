#!/usr/bin/env python3
"""Upload a file to RunPod ssh.runpod.io proxy (PTY required; no SFTP/SCP)."""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
import time
from pathlib import Path


def read_until(proc: subprocess.Popen, marker: str, timeout: float = 120.0) -> str:
    assert proc.stdout
    buf = ""
    t0 = time.time()
    while time.time() - t0 < timeout:
        ch = proc.stdout.read(1)
        if not ch:
            time.sleep(0.05)
            if time.time() - t0 > timeout:
                break
            continue
        buf += ch
        if marker in buf:
            return buf
    raise TimeoutError(f"timeout waiting for {marker!r}; tail={buf[-800:]!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--remote", required=True)
    p.add_argument("--host", default="ovffqhywr46v3c-6441120d@ssh.runpod.io")
    p.add_argument("--key", type=Path, default=Path.home() / ".ssh" / "id_ed25519_runpod")
    p.add_argument("--chunk-bytes", type=int, default=120_000)
    args = p.parse_args()

    raw = args.local.read_bytes()
    chunks = [
        base64.b64encode(raw[i : i + args.chunk_bytes]).decode("ascii")
        for i in range(0, len(raw), args.chunk_bytes)
    ]
    print(f"bytes={len(raw)} chunks={len(chunks)} -> {args.host}:{args.remote}", flush=True)

    proc = subprocess.Popen(
        [
            "ssh",
            "-tt",
            "-i",
            str(args.key),
            "-o",
            "StrictHostKeyChecking=accept-new",
            args.host,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
    )
    assert proc.stdin and proc.stdout

    def send(s: str) -> None:
        proc.stdin.write(s)
        if not s.endswith("\n"):
            proc.stdin.write("\n")
        proc.stdin.flush()

    time.sleep(3.0)
    send(
        "mkdir -p \"$(dirname '"
        + args.remote
        + "')\"; rm -f '"
        + args.remote
        + "' '"
        + args.remote
        + ".b64'; : > '"
        + args.remote
        + ".b64'; echo UPLOAD_READY"
    )
    read_until(proc, "UPLOAD_READY", 90)
    print("remote ready", flush=True)

    for i, chunk in enumerate(chunks, 1):
        send(f"printf '%s\\n' '{chunk}' >> '{args.remote}.b64'")
        if i % 25 == 0 or i == len(chunks):
            send(f"echo PROGRESS_{i}")
            read_until(proc, f"PROGRESS_{i}", 90)
            print(f"progress {i}/{len(chunks)}", flush=True)

    send(
        f"base64 -d '{args.remote}.b64' > '{args.remote}' && "
        f"rm -f '{args.remote}.b64' && ls -lh '{args.remote}' && "
        f"sha256sum '{args.remote}' | cut -c1-16 && echo UPLOAD_DONE"
    )
    out = read_until(proc, "UPLOAD_DONE", 300)
    print(out[-1200:], flush=True)
    send("exit")
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
