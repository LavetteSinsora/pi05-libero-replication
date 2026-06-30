#!/usr/bin/env python3
"""
Convert LIBERO-OBJECT HDF5 demos to LeRobot v2.1 dataset format.

Re-renders agentview and wrist camera at 256x256 using the exact saved
MuJoCo states. State and actions are copied directly from the HDF5 files.

Run from third_party/openpi/ (uses openpi's venv, which writes v2.1 format):

  Step 1 — render once (slow: ~2h, creates the canonical video files):
    uv run ../../scripts/convert_hdf5_to_lerobot.py --mode render

  Step 2 — create variants (fast: no rendering, just rewrites scalar labels):
    uv run ../../scripts/convert_hdf5_to_lerobot.py --mode variant --variant naive_subsampling
    uv run ../../scripts/convert_hdf5_to_lerobot.py --mode variant --variant summed_subsampling

── Directory layout ────────────────────────────────────────────────────────────

  data/lerobot/pi05-libero/
    libero_object_canonical/        ← rendered once; never used for training
      videos/chunk-000/image/...    ← the actual MP4 files live here
      videos/chunk-000/wrist_image/...
      data/...   meta/...

    libero_object_naive_subsampling/
      videos/  →  symlink to ../libero_object_canonical/videos/
      data/chunk-000/episode_*.parquet   ← naive actions
      meta/

    libero_object_summed_subsampling/
      videos/  →  symlink to ../libero_object_canonical/videos/
      data/chunk-000/episode_*.parquet   ← summed actions
      meta/

  A symlink is a filesystem pointer: a directory entry that transparently
  redirects to another path. When the training data loader opens
  libero_object_naive_subsampling/videos/, the OS silently serves files from
  libero_object_canonical/videos/. The MP4s exist in exactly one place;
  both variant directories share them without any copying.

── Variants ────────────────────────────────────────────────────────────────────

  naive_subsampling
    The action stored for each 10Hz step is the raw 20Hz action at that
    timestep — a delta designed to move the EEF in 50ms, not 100ms.

    At deployment (10Hz), each action is applied for a 100ms window. The
    robot reaches its target in ~50ms then sits idle for the remaining 50ms.
    Effective speed is halved. Worse, the state the robot is in at the next
    observation (having only moved halfway) is a state the model never saw
    during training — training always showed the state after two full 20Hz
    steps, never the halfway state. This distribution shift can cause
    erratic behaviour mid-task.

  summed_subsampling  (recommended)
    The action for each 10Hz step is the sum of the two covered 20Hz actions:

        action_10hz[i] = action_20hz[2i] + action_20hz[2i+1]

    Position and orientation deltas are additive, so the sum gives exactly
    the displacement needed to cover the full 100ms window. The robot moves
    continuously throughout the window and arrives at the state the model
    expects to see at the next observation.

    Gripper is a target command, not a delta, so we take the more recent
    value (action_20hz[2i+1]) rather than summing.

────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# ── paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
HDF5_DIR       = PROJECT_ROOT / "data" / "libero" / "libero_object"
OUTPUT_ROOT    = PROJECT_ROOT / "data" / "lerobot" / "pi05-libero"
CANONICAL_NAME = "libero_object_canonical"

RENDER_H  = 256
RENDER_W  = 256
RECORD_HZ = 20
TARGET_HZ = 10
SUBSAMPLE = RECORD_HZ // TARGET_HZ
FPS       = TARGET_HZ

VARIANTS = ("naive_subsampling", "summed_subsampling")


# ── action computation ───────────────────────────────────────────────────────

def compute_action(actions: np.ndarray, t: int, T: int, variant: str) -> np.ndarray:
    if variant == "naive_subsampling":
        return actions[t].astype(np.float32)

    # summed_subsampling
    t1 = min(t + 1, T - 1)
    return np.concatenate([
        actions[t, :6] + actions[t1, :6],  # additive position + orientation deltas
        actions[t1, 6:],                    # gripper target — take more recent value
    ]).astype(np.float32)


# ── render mode ─────────────────────────────────────────────────────────────

def render_canonical():
    """Full rendering pass. Creates the canonical dataset with all MP4 files."""
    canonical_path = OUTPUT_ROOT / CANONICAL_NAME
    if canonical_path.exists():
        print(f"Removing existing canonical at {canonical_path}")
        shutil.rmtree(canonical_path)

    # repo_id must include the org prefix for lerobot; OUTPUT_ROOT is the root
    dataset = LeRobotDataset.create(
        repo_id=f"pi05-libero/{CANONICAL_NAME}",
        root=OUTPUT_ROOT / CANONICAL_NAME,
        robot_type="panda",
        fps=FPS,
        features={
            "image": {
                "dtype": "video",
                "shape": (RENDER_H, RENDER_W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "video",
                "shape": (RENDER_H, RENDER_W, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                # EEF position (3) + orientation as axis-angle (3) + gripper finger positions (2)
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                # OSC delta command: dx,dy,dz (3) + dax,day,daz (3) + gripper (1)
                # Stored as naive_subsampling here; canonical is never used for training.
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
    )

    bm = benchmark.get_benchmark_dict()["libero_object"]()

    for task_idx in range(bm.get_num_tasks()):
        task      = bm.get_task(task_idx)
        bddl_file = bm.get_task_bddl_file_path(task_idx)
        hdf5_path = HDF5_DIR / f"{task.name}_demo.hdf5"

        if not hdf5_path.exists():
            print(f"  WARNING: {hdf5_path} not found, skipping")
            continue

        print(f"\n[{task_idx+1}/10] {task.name}")
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=RENDER_H,
            camera_widths=RENDER_W,
        )
        env.reset()
        language = env.language_instruction
        print(f"  instruction: {language}")

        with h5py.File(hdf5_path, "r") as f:
            num_demos = len(f["data"])
            for demo_idx in range(num_demos):
                demo    = f[f"data/demo_{demo_idx}"]
                states  = demo["states"][()]
                actions = demo["actions"][()]
                ee_pos  = demo["obs/ee_pos"][()]
                ee_ori  = demo["obs/ee_ori"][()]
                gripper = demo["obs/gripper_states"][()]
                T = len(actions)

                for t in range(0, T, SUBSAMPLE):
                    obs       = env.regenerate_obs_from_state(states[t])
                    agentview = obs["agentview_image"][::-1]
                    wrist     = obs["robot0_eye_in_hand_image"][::-1]
                    state     = np.concatenate([ee_pos[t], ee_ori[t], gripper[t]]).astype(np.float32)
                    dataset.add_frame({
                        "image":       agentview,
                        "wrist_image": wrist,
                        "state":       state,
                        "actions":     actions[t].astype(np.float32),  # naive; canonical is not trained on
                        "task":        language,
                    })

                dataset.save_episode()
                kept = len(range(0, T, SUBSAMPLE))
                print(f"  demo {demo_idx:02d}/{num_demos-1}: {T} frames → {kept} kept")

        env.close()

    print(f"\nCanonical rendered at: {canonical_path}")


# ── variant mode ─────────────────────────────────────────────────────────────

def _episode_action_stats(actions_array: np.ndarray) -> dict:
    """Compute per-episode stats dict for the actions feature."""
    return {
        "min":   actions_array.min(0).tolist(),
        "max":   actions_array.max(0).tolist(),
        "mean":  actions_array.mean(0).tolist(),
        "std":   actions_array.std(0).tolist(),
        "count": [len(actions_array)],
    }


def create_variant(variant: str):
    """
    Fast variant creation — no rendering.

    Reads action values from HDF5, rewrites the actions column of each
    episode parquet, recomputes action statistics, and symlinks the videos/
    directory to the canonical dataset.
    """
    canonical_path = OUTPUT_ROOT / CANONICAL_NAME
    if not canonical_path.exists():
        raise RuntimeError(
            f"Canonical not found at {canonical_path}. Run --mode render first."
        )

    variant_path = OUTPUT_ROOT / f"libero_object_{variant}"
    if variant_path.exists():
        print(f"Removing existing variant at {variant_path}")
        shutil.rmtree(variant_path)

    # Create directory skeleton
    (variant_path / "data" / "chunk-000").mkdir(parents=True)
    (variant_path / "meta").mkdir(parents=True)

    # Symlink videos → ../libero_object_canonical/videos  (relative, portable)
    (variant_path / "videos").symlink_to(
        Path("..") / CANONICAL_NAME / "videos"
    )

    # Copy unchanged meta files
    shutil.copy(canonical_path / "meta" / "info.json",    variant_path / "meta" / "info.json")
    shutil.copy(canonical_path / "meta" / "tasks.jsonl",  variant_path / "meta" / "tasks.jsonl")
    shutil.copy(canonical_path / "meta" / "episodes.jsonl", variant_path / "meta" / "episodes.jsonl")

    # Rewrite parquets and recompute episodes_stats for this variant's actions
    bm = benchmark.get_benchmark_dict()["libero_object"]()
    episode_idx   = 0
    episodes_stats = []

    for task_idx in range(bm.get_num_tasks()):
        task      = bm.get_task(task_idx)
        hdf5_path = HDF5_DIR / f"{task.name}_demo.hdf5"
        if not hdf5_path.exists():
            continue

        with h5py.File(hdf5_path, "r") as f:
            num_demos = len(f["data"])
            for demo_idx in range(num_demos):
                actions_hdf5 = f[f"data/demo_{demo_idx}/actions"][()]
                T = len(actions_hdf5)

                # Read the canonical parquet for this episode
                src = canonical_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
                df  = pd.read_parquet(src)

                # Replace actions column with variant-specific values
                new_actions = [
                    compute_action(actions_hdf5, t, T, variant)
                    for t in range(0, T, SUBSAMPLE)
                ]
                df["actions"] = new_actions

                # Write new parquet
                dst = variant_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
                df.to_parquet(dst, index=False)

                # Recompute action stats for this episode
                canonical_stats_path = canonical_path / "meta" / "episodes_stats.jsonl"
                with open(canonical_stats_path) as fh:
                    all_lines = [json.loads(l) for l in fh if l.strip()]
                canon_ep_stats = next(s for s in all_lines if s["episode_index"] == episode_idx)

                ep_stats = canon_ep_stats.copy()
                ep_stats["stats"] = dict(canon_ep_stats["stats"])
                ep_stats["stats"]["actions"] = _episode_action_stats(np.array(new_actions))
                episodes_stats.append(ep_stats)

                episode_idx += 1

        print(f"[{task_idx+1}/10] {task.name}: {num_demos} demos rewritten")

    # Write updated episodes_stats.jsonl
    with open(variant_path / "meta" / "episodes_stats.jsonl", "w") as fh:
        for ep in episodes_stats:
            fh.write(json.dumps(ep) + "\n")

    print(f"\nVariant '{variant}' written to: {variant_path}")
    print(f"  videos/  →  ../{CANONICAL_NAME}/videos  (symlink)")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["render", "variant"], required=True)
    parser.add_argument("--variant", choices=VARIANTS,
                        help="Required when --mode variant")
    args = parser.parse_args()

    if args.mode == "render":
        render_canonical()
    else:
        if not args.variant:
            parser.error("--variant is required when --mode is 'variant'")
        create_variant(args.variant)


if __name__ == "__main__":
    main()
