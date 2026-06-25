"""
Prepare a tokenized data shard from a text source.

Phase 0 (CPU smoke tests):
    python -m data.prepare --source synthetic --out data/shards/ --shard-tokens 50000 --total-tokens 200000

Phase 0.5 (H100 real training):
    python -m data.prepare --source fineweb-edu --out data/shards/ \\
        --shard-tokens 10000000 --total-tokens 1000000000 \\
        --eval-tokens 5000000 --eval-out eval/private/

This will:
  1. Stream and tokenize ~1B tokens from FineWeb-Edu (sample-10BT subset).
  2. Write training shards to data/shards/.
  3. Write a held-out eval shard to eval/private/active_tokens.bin for val_bpb.
  4. Build content-addressed manifest at data/data_manifest.json.

Requires `datasets` package: pip install 'ralph-subnet[data]'
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.tokenizer import EOT_TOKEN, get_tokenizer
from data.manifest import DataManifest, build_manifest


def synthetic_stream(seed: int = 1337):
    """A small deterministic text corpus. Not real English — just stable bytes
    so the model has something to fit. Use only for CPU smoke tests."""
    rng = random.Random(seed)
    words = [
        "the", "cat", "sat", "on", "the", "mat", "and", "looked", "around",
        "quietly", "while", "rain", "tapped", "the", "tin", "roof",
        "Ralph", "validates", "training", "recipes", "openly",
        "every", "epoch", "the", "container", "attests", "what", "it", "ran",
        "miners", "search", "patches", "validators", "score", "checkpoints",
    ]
    while True:
        sent_len = rng.randint(6, 18)
        yield " ".join(rng.choice(words) for _ in range(sent_len)) + "."


def fineweb_edu_stream():  # pragma: no cover - exercised only with `datasets` installed
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for row in ds:
        yield row["text"]


def fineweb_raw_stream():  # pragma: no cover - exercised only with `datasets` installed
    """Raw (unfiltered) FineWeb — the data-quality NEGATIVE lever for B6.
    Same web corpus as FineWeb-Edu without the edu-classifier filtering."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for row in ds:
        yield row["text"]


def _source_stream(source: str, seed: int):
    if source == "synthetic":
        return synthetic_stream(seed)
    if source == "fineweb-raw":
        return fineweb_raw_stream()
    return fineweb_edu_stream()  # "fineweb-edu" (default)


def tokenize_into_shards(
    out_dir: Path,
    shard_tokens: int,
    total_tokens: int,
    source: str = "synthetic",
    seed: int = 1337,
    eval_tokens: int = 0,
    eval_out: Path | None = None,
) -> tuple[list[Path], Path | None]:
    """Returns (train_shard_paths, eval_shard_path_or_None)."""
    import time as _time

    tok = get_tokenizer()
    stream = _source_stream(source, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buf: list[int] = []
    written = 0
    shard_idx = 0
    docs = 0
    t0 = _time.time()
    target = total_tokens + eval_tokens

    for text in stream:
        if not text:
            continue
        ids = tok.encode_ordinary(text)
        buf.extend(ids)
        buf.append(EOT_TOKEN)
        docs += 1
        while len(buf) >= shard_tokens:
            shard_path = out_dir / f"shard_{shard_idx:04d}.bin"
            arr = np.array(buf[:shard_tokens], dtype=np.uint16)
            arr.tofile(shard_path)
            paths.append(shard_path)
            shard_idx += 1
            written += shard_tokens
            buf = buf[shard_tokens:]
            elapsed = _time.time() - t0
            rate = written / max(elapsed, 0.01)
            pct = 100 * written / target
            print(
                f"\r  [{pct:5.1f}%] {written / 1e6:.1f}M / {target / 1e6:.0f}M tokens | "
                f"{shard_idx} shards | {docs:,} docs | {rate / 1e6:.2f}M tok/s",
                end="", flush=True,
            )
            if written >= target:
                print()
                break
        if written >= target:
            break
    if buf and written < target:
        shard_path = out_dir / f"shard_{shard_idx:04d}.bin"
        np.array(buf, dtype=np.uint16).tofile(shard_path)
        paths.append(shard_path)
        written += len(buf)
    print()

    eval_path = None
    if eval_tokens > 0 and eval_out is not None:
        eval_out.mkdir(parents=True, exist_ok=True)
        eval_path = eval_out / "active_tokens.bin"
        last_shard = paths[-1]
        shard_data = np.memmap(last_shard, dtype=np.uint16, mode="r")
        if len(shard_data) >= eval_tokens:
            eval_data = shard_data[-eval_tokens:]
            train_data = shard_data[:-eval_tokens]
            np.array(eval_data).tofile(eval_path)
            np.array(train_data).tofile(last_shard)
            print(f"  split {eval_tokens:,} eval tokens from last shard -> {eval_path}")
        else:
            np.array(shard_data).tofile(eval_path)
            paths.pop()
            last_shard.unlink()
            print(f"  used entire last shard ({len(shard_data):,} tokens) as eval -> {eval_path}")

    return paths, eval_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "fineweb-edu", "fineweb-raw"], default="synthetic")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-tokens", type=int, default=100_000)
    p.add_argument("--total-tokens", type=int, default=500_000)
    p.add_argument("--eval-tokens", type=int, default=0,
                   help="Hold out this many tokens from the end for the hidden eval set")
    p.add_argument("--eval-out", type=Path, default=None,
                   help="Directory for held-out eval tokens (default: eval/private/)")
    p.add_argument("--track", default="llm-pretraining-launch")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    eval_out = args.eval_out or (Path(__file__).resolve().parent.parent / "eval" / "private")
    paths, eval_path = tokenize_into_shards(
        args.out,
        shard_tokens=args.shard_tokens,
        total_tokens=args.total_tokens,
        source=args.source,
        seed=args.seed,
        eval_tokens=args.eval_tokens,
        eval_out=eval_out if args.eval_tokens > 0 else None,
    )
    base_dir = args.out.parent
    manifest = build_manifest(
        track=args.track,
        tokenizer="gpt2",
        vocab_size=50257,
        dtype="uint16",
        shards=paths,
        base_dir=base_dir,
    )
    manifest_path = args.manifest if args.manifest else base_dir / "data_manifest.json"
    manifest.write(manifest_path)
    print(f"wrote {len(paths)} shards, {manifest.total_tokens():,} tokens total")
    if eval_path:
        print(f"eval shard: {eval_path}")
    print(f"manifest: {manifest_path}")
    print(f"manifest hash: {manifest.manifest_hash()[:16]}…")
    # HF `datasets`/tokenizers background threads can fault during interpreter
    # finalization (PyGILState_Release). The work is done + files are written, so
    # exit hard before that cleanup runs.
    import os as _os
    sys.stdout.flush(); sys.stderr.flush()
    _os._exit(0)


if __name__ == "__main__":
    main()
