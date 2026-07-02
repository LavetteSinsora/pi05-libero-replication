#!/usr/bin/env python3
"""Smoke test: convert task 0, demo 0 only. Outputs to data/lerobot_smoke/."""
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HDF5_DIR     = PROJECT_ROOT / "data" / "libero" / "libero_object"
OUTPUT_ROOT  = PROJECT_ROOT / "data" / "lerobot_smoke"
REPO_ID      = "pi05_libero/libero_object"

RENDER_H  = 256
RENDER_W  = 256
RECORD_HZ = 20
TARGET_HZ = 10
SUBSAMPLE = RECORD_HZ // TARGET_HZ

output_path = OUTPUT_ROOT / REPO_ID
if output_path.exists():
    shutil.rmtree(output_path)

dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    root=OUTPUT_ROOT,
    robot_type="panda",
    fps=TARGET_HZ,
    features={
        "image":       {"dtype": "video", "shape": (RENDER_H, RENDER_W, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "video", "shape": (RENDER_H, RENDER_W, 3), "names": ["height", "width", "channel"]},
        "state":       {"dtype": "float32", "shape": (8,),  "names": ["state"]},
        "actions":     {"dtype": "float32", "shape": (7,),  "names": ["actions"]},
    },
    image_writer_threads=4,
)

bm = benchmark.get_benchmark_dict()["libero_object"]()
task      = bm.get_task(0)
bddl_file = bm.get_task_bddl_file_path(0)
hdf5_path = HDF5_DIR / f"{task.name}_demo.hdf5"

print(f"Task: {task.name}")
env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=RENDER_H, camera_widths=RENDER_W)
env.reset()
print(f"Instruction: {env.language_instruction}")

with h5py.File(hdf5_path, "r") as f:
    demo    = f["data/demo_0"]
    states  = demo["states"][()]
    actions = demo["actions"][()]
    ee_pos  = demo["obs/ee_pos"][()]
    ee_ori  = demo["obs/ee_ori"][()]
    gripper = demo["obs/gripper_states"][()]
    T = len(actions)
    print(f"Demo 0: {T} raw frames → {len(range(0, T, SUBSAMPLE))} kept at {TARGET_HZ}Hz")

    for t in range(0, T, SUBSAMPLE):
        obs       = env.regenerate_obs_from_state(states[t])
        agentview = obs["agentview_image"][::-1]
        wrist     = obs["robot0_eye_in_hand_image"][::-1]
        state     = np.concatenate([ee_pos[t], ee_ori[t], gripper[t]]).astype(np.float32)
        t1 = min(t + 1, T - 1)
        action = np.concatenate([
            actions[t, :6] + actions[t1, :6],
            actions[t1, 6:],
        ]).astype(np.float32)
        dataset.add_frame({
            "image":       agentview,
            "wrist_image": wrist,
            "state":       state,
            "actions":     action,
            "task":        env.language_instruction,
        })
        if t % 20 == 0:
            print(f"  frame {t}/{T-1}")

dataset.save_episode()
env.close()

print(f"\nVideos written to:")
for p in sorted(output_path.rglob("*.mp4")):
    print(f"  {p}")
