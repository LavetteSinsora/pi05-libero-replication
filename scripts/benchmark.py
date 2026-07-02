#!/usr/bin/env python3
"""
Evaluate a π0.5 policy on LIBERO-OBJECT (10 tasks × 50 trials).

Usage
-----
# Baseline (pretrained π0.5, no fine-tuning):
  python scripts/benchmark.py \
    --config_name pi05_libero \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
    --exp_dir experiments/pi05_base_benchmark

# Fine-tuned checkpoint at step 5000:
  python scripts/benchmark.py \
    --config_name pi05_libero_object_lora \
    --checkpoint_dir checkpoints/pi05_libero_object_lora/masked_loss_summed_subsampling/5000 \
    --exp_dir experiments/pi05_libero_object_lora/masked_loss_summed_subsampling/step_5000

Run from the repo root. Requires the openpi venv or `pip install -e third_party/openpi`.
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
import subprocess
import sys

import imageio
import numpy as np
import tyro
from libero.libero import benchmark as libero_benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

# Add third_party/openpi to path when running outside the venv
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "third_party" / "openpi" / "src"))

import openpi.training.config as _config
import openpi.policies.policy_config as _policy_config

LIBERO_ENV_RESOLUTION = 256
TASK_SUITE = "libero_object"
MAX_STEPS = 280          # longest libero_object demo has 254 steps
NUM_STEPS_WAIT = 10      # let objects settle before acting
REPLAN_STEPS = 5         # execute this many actions from each predicted chunk
RESIZE = 224
MAX_FAILURE_VIDEOS = 3
MAX_SUCCESS_VIDEOS = 1
DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    exp_dir: str
    num_trials_per_task: int = 50
    seed: int = 7
    wandb_project: str = "pi05_libero_replication"
    wandb_enabled: bool = True
    train_step: int | None = None  # for wandb x-axis when mid-training eval


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_env(task, seed: int):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def _write_config_snapshot(exp_dir: pathlib.Path, train_config, checkpoint_dir: str) -> None:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_ROOT / "third_party" / "openpi"),
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    snapshot = {
        "config": dataclasses.asdict(train_config),
        "checkpoint_dir": checkpoint_dir,
        "git_commit": git_commit,
    }
    (exp_dir / "config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, default=str)
    )


def run(args: Args) -> None:
    np.random.seed(args.seed)
    exp_dir = pathlib.Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── load policy ──────────────────────────────────────────────────────────
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    _write_config_snapshot(exp_dir, train_config, args.checkpoint_dir)

    # ── wandb ────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_enabled:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=exp_dir.name,
            config={
                "config_name": args.config_name,
                "checkpoint_dir": args.checkpoint_dir,
                "num_trials_per_task": args.num_trials_per_task,
            },
        )

    # ── task suite ───────────────────────────────────────────────────────────
    suite = libero_benchmark.get_benchmark_dict()[TASK_SUITE]()
    num_tasks = suite.n_tasks
    logging.info(f"Evaluating {num_tasks} tasks × {args.num_trials_per_task} trials")

    all_results = {}
    total_ep, total_succ = 0, 0

    for task_id in range(num_tasks):
        task = suite.get_task(task_id)
        task_name = task.language.replace(" ", "_")
        task_key = f"task_{task_id:02d}_{task_name}"
        initial_states = suite.get_task_init_states(task_id)

        env = _get_env(task, args.seed)

        video_dir = exp_dir / "videos" / task_key
        (video_dir / "success").mkdir(parents=True, exist_ok=True)
        (video_dir / "failure").mkdir(parents=True, exist_ok=True)

        task_succ = 0
        saved_success = 0
        saved_failure = 0

        for ep_idx in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])
            action_plan: collections.deque = collections.deque()
            frames = []
            done = False

            for t in range(MAX_STEPS + NUM_STEPS_WAIT):
                if t < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(DUMMY_ACTION)
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
                wrist_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
                frames.append(img_r)

                if not action_plan:
                    element = {
                        "observation/image": img_r,
                        "observation/wrist_image": wrist_r,
                        "observation/state": np.concatenate([
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]),
                        "prompt": task.language,
                    }
                    chunk = policy.infer(element)["actions"]
                    action_plan.extend(chunk[:REPLAN_STEPS])

                obs, _, done, _ = env.step(action_plan.popleft().tolist())
                if done:
                    task_succ += 1
                    break

            suffix = "success" if done else "failure"
            save_video = (done and saved_success < MAX_SUCCESS_VIDEOS) or (
                not done and saved_failure < MAX_FAILURE_VIDEOS
            )
            if save_video and frames:
                ep_label = f"ep_{ep_idx:02d}.mp4"
                imageio.mimwrite(
                    video_dir / suffix / ep_label,
                    [np.asarray(f) for f in frames],
                    fps=10,
                )
                if done:
                    saved_success += 1
                else:
                    saved_failure += 1

        env.close()
        success_rate = task_succ / args.num_trials_per_task
        all_results[task_key] = {
            "success_rate": success_rate,
            "successes": task_succ,
            "trials": args.num_trials_per_task,
        }
        total_ep += args.num_trials_per_task
        total_succ += task_succ
        logging.info(f"[{task_id+1}/{num_tasks}] {task_key}: {success_rate:.1%}")

    # ── aggregate + write results ─────────────────────────────────────────────
    aggregate_rate = total_succ / total_ep
    results = {
        "aggregate_success_rate": aggregate_rate,
        "per_task": all_results,
    }
    (exp_dir / "results.json").write_text(json.dumps(results, indent=2))
    logging.info(f"Aggregate success rate: {aggregate_rate:.1%}")

    if wandb_run is not None:
        log_dict = {"aggregate_success_rate": aggregate_rate}
        for k, v in all_results.items():
            log_dict[f"{k}/success_rate"] = v["success_rate"]
        if args.train_step is not None:
            wandb.log(log_dict, step=args.train_step)
        else:
            wandb.log(log_dict)
        wandb_run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(run)
