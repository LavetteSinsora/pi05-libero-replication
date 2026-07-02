#!/usr/bin/env python3
"""
Rebuild a full openpi training checkpoint from a per-step resume bundle.

A "resume bundle" (created by the Colab notebook, §2 `bundle_step`) contains
everything a LoRA fine-tune actually changed:

    step_<n>/
      lora_params.npz        LoRA adapter weights (extract_lora.py format)
      train_state/           verbatim copy of the checkpoint's train_state item:
                             optimizer state (Adam mu/nu — LoRA params only,
                             since opt_state is init'd on the trainable filter)
                             + the step counter
      _CHECKPOINT_METADATA   orbax step-level metadata (copied if present)
      wandb_id.txt           run id for wandb resume (copied if present)

Everything else in a full ~5 GB checkpoint is the frozen pi05_base weights,
which the config's weight_loader re-downloads from GCS (cached under
~/.cache/openpi), exactly as train.py does on a fresh start.

Output: a native orbax checkpoint at
    <checkpoint_base_dir>/<config_name>/<exp_name>/<step>/
indistinguishable from one written by train.py — both `train.py --resume`
and `benchmark.py` accept it.

Run with third_party/openpi as the working directory (the config's
assets_base_dir is relative to it), on a machine where JAX has enough
memory for a full model instantiation (GPU, or CPU with ~20 GB RAM):

  .venv/bin/python ../../scripts/build_resume_checkpoint.py \
    --config-name pi05_libero_object_lora \
    --bundle-dir  /content/drive/.../lora/<exp>/step_15000 \
    --checkpoint-base-dir /content/checkpoints/pi05_libero \
    --exp-name    masked_loss_summed_subsampling
"""

import dataclasses
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import tyro

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "third_party" / "openpi" / "src"))

import flax.nnx as nnx  # noqa: E402
import flax.traverse_util as traverse_util  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402

import openpi.shared.array_typing as at  # noqa: E402
import openpi.shared.nnx_utils as nnx_utils  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.optimizer as _optimizer  # noqa: E402
import openpi.training.utils as training_utils  # noqa: E402


@dataclasses.dataclass
class Args:
    config_name: str          # TrainConfig name, e.g. pi05_libero_object_lora
    bundle_dir: str           # bundle dir, e.g. .../lora/<exp>/step_15000
    checkpoint_base_dir: str  # e.g. /content/checkpoints/pi05_libero
    exp_name: str


class _AssetsShim:
    """Quacks like a DataLoader for save_state's assets callback (norm stats)."""

    def __init__(self, config: _config.TrainConfig):
        self._config = config

    def data_config(self):
        return self._config.data.create(self._config.assets_dirs, self._config.model)


def run(args: Args) -> None:
    config = _config.get_config(args.config_name)
    if config.ema_decay is not None:
        raise NotImplementedError(
            "Resume bundles do not carry EMA params; this config uses EMA. "
            "Bundle-based resume only works for ema_decay=None configs."
        )
    config = dataclasses.replace(
        config, exp_name=args.exp_name, checkpoint_base_dir=args.checkpoint_base_dir
    )

    bundle = pathlib.Path(args.bundle_dir).resolve()
    step = int(bundle.name.rsplit("_", 1)[-1])

    ckpt_dir = config.checkpoint_dir
    if (ckpt_dir / str(step)).exists():
        print(f"Checkpoint {ckpt_dir / str(step)} already exists — nothing to do.")
        return

    # Mirror train.py's init exactly (minus jit/sharding — single host here) so
    # the rebuilt TrainState has the same pytree structure orbax saved.
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params=None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None,
        )

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    train_state_shape = jax.eval_shape(init, init_rng)

    # 1 · Restore step + optimizer moments from the bundle, through the same
    #     CheckpointManager path train.py uses on --resume: stage the bundle's
    #     train_state item into a temp dir shaped like a checkpoint root.
    with tempfile.TemporaryDirectory() as td:
        step_dir = pathlib.Path(td) / str(step)
        step_dir.mkdir()
        shutil.copytree(bundle / "train_state", step_dir / "train_state")
        if (bundle / "_CHECKPOINT_METADATA").exists():
            shutil.copy(bundle / "_CHECKPOINT_METADATA", step_dir / "_CHECKPOINT_METADATA")
        mngr = ocp.CheckpointManager(td, item_handlers={"train_state": ocp.PyTreeCheckpointHandler()})
        target, _ = _checkpoints._split_params(train_state_shape)  # train_state item has params={}
        with at.disable_typechecking():
            restored = mngr.restore(step, items={"train_state": target})["train_state"]
    print(f"Restored optimizer state + step {int(restored.step)} from {bundle}")

    # 2 · Rebuild full params: frozen base weights from the weight loader
    #     (GCS download, cached) + randomly initialised LoRA adapters …
    loaded = config.weight_loader.load(train_state_shape.params.to_pure_dict())
    at.check_pytree_equality(
        expected=train_state_shape.params.to_pure_dict(), got=loaded,
        check_shapes=True, check_dtypes=True,
    )
    partial_params = traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded).items()
         if not isinstance(v, jax.ShapeDtypeStruct)}
    )
    state = init(init_rng, partial_params)

    # 3 · … then overwrite the LoRA leaves with the trained adapters.
    lora = {tuple(k.split("__")): v for k, v in np.load(bundle / "lora_params.npz").items()}
    flat = traverse_util.flatten_dict(state.params.to_pure_dict())
    for k, v in lora.items():
        if k not in flat:
            raise KeyError(f"bundle key {'/'.join(k)} not found in the model's params")
        if flat[k].shape != v.shape:
            raise ValueError(f"shape mismatch at {'/'.join(k)}: model {flat[k].shape}, bundle {v.shape}")
        flat[k] = jnp.asarray(v).astype(flat[k].dtype)
    state.params.replace_by_pure_dict(traverse_util.unflatten_dict(flat))
    print(f"Overlaid {len(lora)} LoRA tensors onto the frozen base weights")

    # 4 · Graft the resumed step + optimizer moments and save natively.
    state = dataclasses.replace(state, step=restored.step, opt_state=restored.opt_state)

    mngr_out, _ = _checkpoints.initialize_checkpoint_dir(
        ckpt_dir, keep_period=config.keep_period, overwrite=False, resume=True
    )
    _checkpoints.save_state(mngr_out, state, _AssetsShim(config), step)
    mngr_out.wait_until_finished()

    if (bundle / "wandb_id.txt").exists() and not (ckpt_dir / "wandb_id.txt").exists():
        shutil.copy(bundle / "wandb_id.txt", ckpt_dir / "wandb_id.txt")

    print(f"Rebuilt checkpoint at {ckpt_dir / str(step)}")


if __name__ == "__main__":
    run(tyro.cli(Args))
