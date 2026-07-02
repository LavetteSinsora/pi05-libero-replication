#!/usr/bin/env python3
"""
Extract LoRA adapter weights from a full openpi checkpoint into a small .npz.

The full training checkpoint is ~4.8 GB (frozen base + LoRA adapters + optimizer
state). Only the LoRA adapters (~84 MB) actually changed during fine-tuning; the
frozen base weights are identical to the pretrained pi05_base checkpoint on GCS.

This script restores the `params` item of a checkpoint, keeps only the leaves
whose flattened path contains "lora", and saves them to a single .npz. Those
weights can later be overlaid onto the base model to reconstruct a full
checkpoint for evaluation or deployment (see reconstruct_lora.py logic in the
Colab notebook).

Usage
-----
  python scripts/extract_lora.py \
    --checkpoint_dir /content/checkpoints/pi05_libero/pi05_libero_object_lora/masked_loss_summed_subsampling/5000 \
    --out /content/drive/MyDrive/pi05_libero_replication/lora/step_5000.npz

Run from the repo root (adds third_party/openpi/src to path automatically).
"""

import dataclasses
import pathlib
import sys

import numpy as np
import tyro

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "third_party" / "openpi" / "src"))

import openpi.models.model as _model  # noqa: E402
from flax.traverse_util import flatten_dict  # noqa: E402


@dataclasses.dataclass
class Args:
    checkpoint_dir: str  # path to the checkpoint step dir (containing params/)
    out: str             # destination .npz path


def run(args: Args) -> None:
    ckpt = pathlib.Path(args.checkpoint_dir)
    params_path = ckpt / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"No params/ under {ckpt}")

    # Restore as numpy so we don't allocate GPU memory.
    params = _model.restore_params(params_path, restore_type=np.ndarray)

    # Flatten to path-keyed leaves; "/"-join the tuple path for a readable key.
    flat = flatten_dict(params, sep="/")
    lora = {k: np.asarray(v) for k, v in flat.items() if "lora" in k}

    if not lora:
        raise RuntimeError(
            f"No LoRA params found in {params_path}. "
            "Was this checkpoint trained with a *_lora model variant?"
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # "/" is illegal in npz member names → encode as "__", decode on reload.
    np.savez(out, **{k.replace("/", "__"): v for k, v in lora.items()})

    total_mb = sum(v.nbytes for v in lora.values()) / 1e6
    print(f"Saved {len(lora)} LoRA tensors ({total_mb:.1f} MB) → {out}")


if __name__ == "__main__":
    run(tyro.cli(Args))
