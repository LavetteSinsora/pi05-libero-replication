"""
Inspect a single LIBERO demo HDF5 file and print examples of every data modality.

Usage:
    uv run scripts/inspect_demo.py
    uv run scripts/inspect_demo.py --file data/libero/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5
    uv run scripts/inspect_demo.py --demo 3   # inspect demo_3 instead of demo_0
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = (
    REPO_ROOT
    / "data/libero/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5"
)


def sep(title: str = "") -> None:
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'=' * pad} {title} {'=' * (width - pad - len(title) - 2)}")
    else:
        print("=" * width)


def main(file: Path, demo_idx: int) -> None:
    with h5py.File(file, "r") as f:
        data_grp = f["data"]

        # ── File-level metadata ───────────────────────────────────────────────
        sep("FILE METADATA")
        print(f"  File          : {file.name}")
        print(f"  Tag           : {data_grp.attrs.get('tag', 'n/a')}")
        print(f"  Num demos     : {data_grp.attrs['num_demos']}")
        print(f"  Total steps   : {data_grp.attrs['total']}")

        problem_info = json.loads(data_grp.attrs["problem_info"])
        print(f"  Env name      : {data_grp.attrs['env_name']}")

        # ── Language instruction ──────────────────────────────────────────────
        sep("LANGUAGE INSTRUCTION")
        lang = problem_info["language_instruction"]
        print(f"  \"{lang}\"")
        print(f"  (shared across all {data_grp.attrs['num_demos']} demonstrations in this file)")

        # ── Episode-length distribution ───────────────────────────────────────
        sep("EPISODE LENGTH DISTRIBUTION (all 50 demos)")
        lengths = np.array(
            [data_grp[f"demo_{i}/actions"].shape[0] for i in range(data_grp.attrs["num_demos"])]
        )
        print(f"  min={lengths.min()}  max={lengths.max()}  mean={lengths.mean():.1f}  "
              f"median={int(np.median(lengths))}")

        # ── Single-demo inspection ────────────────────────────────────────────
        demo_key = f"demo_{demo_idx}"
        demo = data_grp[demo_key]
        T = demo["actions"].shape[0]

        sep(f"DEMO {demo_idx} OVERVIEW")
        print(f"  Episode length (T): {T} steps")
        print(f"  Datasets inside demo_{demo_idx}:")
        def _print_ds(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"    {name:40s} shape={str(obj.shape):25s} dtype={obj.dtype}")
        demo.visititems(_print_ds)

        # ── Actions ──────────────────────────────────────────────────────────
        sep(f"ACTIONS  (demo_{demo_idx}, shape={demo['actions'].shape}, dtype={demo['actions'].dtype})")
        actions = demo["actions"][:]
        print("  Columns: [dx, dy, dz, dRx, dRy, dRz, gripper]  (OSC_POSE delta + gripper)")
        print("           controller output_max = ±0.05 (pos), ±0.5 (rot), gripper ∈ [-1, 1]")
        print()
        print(f"  First  step : {np.round(actions[0], 4)}")
        print(f"  Middle step : {np.round(actions[T // 2], 4)}")
        print(f"  Last   step : {np.round(actions[-1], 4)}")
        print()
        print("  Per-dim stats (min / max / mean):")
        labels = ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"]
        for i, lbl in enumerate(labels):
            col = actions[:, i]
            print(f"    {lbl:8s}: min={col.min():+.4f}  max={col.max():+.4f}  mean={col.mean():+.4f}")

        # ── Proprioception: EE state ──────────────────────────────────────────
        sep(f"PROPRIOCEPTION — end-effector state  (shape={demo['obs/ee_states'].shape})")
        ee = demo["obs/ee_states"][:]
        print("  Columns: [x, y, z,  roll, pitch, yaw]  (EE position + euler orientation)")
        print(f"  Step 0  : {np.round(ee[0], 4)}")
        print(f"  Step {T//2:3d}: {np.round(ee[T // 2], 4)}")
        print(f"  Step {T-1:3d}: {np.round(ee[-1], 4)}")

        sep(f"PROPRIOCEPTION — EE position  (shape={demo['obs/ee_pos'].shape})")
        ee_pos = demo["obs/ee_pos"][:]
        print("  Columns: [x, y, z]")
        print(f"  Step 0  : {np.round(ee_pos[0], 4)}")

        sep(f"PROPRIOCEPTION — EE orientation  (shape={demo['obs/ee_ori'].shape})")
        ee_ori = demo["obs/ee_ori"][:]
        print("  Columns: [roll, pitch, yaw]  (euler angles in radians)")
        print(f"  Step 0  : {np.round(ee_ori[0], 4)}")

        sep(f"PROPRIOCEPTION — joint states  (shape={demo['obs/joint_states'].shape})")
        joints = demo["obs/joint_states"][:]
        print("  7 Panda joint angles (radians)")
        print(f"  Step 0  : {np.round(joints[0], 4)}")
        print(f"  Step {T-1:3d}: {np.round(joints[-1], 4)}")

        sep(f"PROPRIOCEPTION — gripper state  (shape={demo['obs/gripper_states'].shape})")
        gripper = demo["obs/gripper_states"][:]
        print("  2 values: [left_finger_pos, right_finger_pos]  (positive = open)")
        print(f"  Step 0  : {np.round(gripper[0], 4)}")
        print(f"  Step {T-1:3d}: {np.round(gripper[-1], 4)}")

        sep(f"PROPRIOCEPTION — robot_states  (shape={demo['robot_states'].shape})")
        rs = demo["robot_states"][:]
        print("  9 values = [joint_vel(7), gripper_pos(1), gripper_vel(1)]  (approximate)")
        print(f"  Step 0  : {np.round(rs[0], 4)}")

        # ── Visual observations ───────────────────────────────────────────────
        sep(f"VISION — agentview_rgb (third-person)  shape={demo['obs/agentview_rgb'].shape}")
        agentview = demo["obs/agentview_rgb"][:]
        print(f"  dtype={agentview.dtype}, value range [{agentview.min()}, {agentview.max()}]")
        print(f"  Frame 0   — mean pixel: {agentview[0].mean():.1f}  "
              f"std: {agentview[0].std():.1f}")
        print(f"  Frame {T//2:3d} — mean pixel: {agentview[T // 2].mean():.1f}  "
              f"std: {agentview[T // 2].std():.1f}")
        print(f"  Frame {T-1:3d} — mean pixel: {agentview[-1].mean():.1f}  "
              f"std: {agentview[-1].std():.1f}")

        sep(f"VISION — eye_in_hand_rgb (wrist camera)  shape={demo['obs/eye_in_hand_rgb'].shape}")
        wrist = demo["obs/eye_in_hand_rgb"][:]
        print(f"  dtype={wrist.dtype}, value range [{wrist.min()}, {wrist.max()}]")
        print(f"  Frame 0   — mean pixel: {wrist[0].mean():.1f}  "
              f"std: {wrist[0].std():.1f}")
        print(f"  Frame {T//2:3d} — mean pixel: {wrist[T // 2].mean():.1f}  "
              f"std: {wrist[T // 2].std():.1f}")
        print(f"  Frame {T-1:3d} — mean pixel: {wrist[-1].mean():.1f}  "
              f"std: {wrist[-1].std():.1f}")

        # ── Rewards / dones ───────────────────────────────────────────────────
        sep(f"REWARDS & DONES  (shape={demo['rewards'].shape})")
        rewards = demo["rewards"][:]
        dones = demo["dones"][:]
        print(f"  rewards — unique values: {np.unique(rewards)}  "
              f"(sum={rewards.sum()}, i.e. {int(rewards.sum())} successful steps)")
        print(f"  dones   — unique values: {np.unique(dones)}  "
              f"(final step done={dones[-1]})")

        # ── Summary ───────────────────────────────────────────────────────────
        sep("SUMMARY")
        print(f"""
  Task               : "{lang}"
  Demonstrations     : {data_grp.attrs['num_demos']} (one file per task)
  Episode length     : variable, {lengths.min()}–{lengths.max()} steps @ 20 Hz control

  Per-step data (T steps per demo):
    obs/agentview_rgb    ({T}, 128, 128, 3) uint8  — third-person RGB camera
    obs/eye_in_hand_rgb  ({T}, 128, 128, 3) uint8  — wrist RGB camera
    obs/ee_states        ({T}, 6)           f64    — EE position (x,y,z) + euler (r,p,y)
    obs/ee_pos           ({T}, 3)           f64    — EE position only
    obs/ee_ori           ({T}, 3)           f64    — EE euler orientation only
    obs/joint_states     ({T}, 7)           f64    — Panda joint angles (rad)
    obs/gripper_states   ({T}, 2)           f64    — finger positions
    robot_states         ({T}, 9)           f64    — joints vel + gripper pos/vel
    states               ({T}, 110)         f64    — full MuJoCo simulator state
    actions              ({T}, 7)           f64    — delta EE pose (6) + gripper (1)
    rewards              ({T},)             uint8  — sparse {0, 1}
    dones                ({T},)             uint8  — episode termination flag
""")
        sep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a LIBERO demo HDF5 file.")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help="Path to the .hdf5 demo file",
    )
    parser.add_argument(
        "--demo",
        type=int,
        default=0,
        help="Which demonstration index to inspect (default: 0)",
    )
    args = parser.parse_args()
    main(args.file, args.demo)
