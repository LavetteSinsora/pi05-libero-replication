# π0.5 LIBERO Replication — Colab Experiment Documentation

This document describes what `colab_training.ipynb` is intended to do end-to-end,
the design decisions behind it, and — first — an objective record of the problems
encountered while bringing it up on Google Colab (A100).

---

## Part 0 — Current state: observed problems (description only)

This section records what we are currently seeing and what we have already run
into, objectively and without proposed fixes.

### 0.1 Current blocking error

When the baseline evaluation is launched, `scripts/benchmark.py` is executed by
the openpi virtual-environment interpreter
(`third_party/openpi/.venv/bin/python`, referred to as `PY`). It fails
immediately at import time:

```
Traceback (most recent call last):
  File "/content/pi05-libero-replication/scripts/benchmark.py", line 34, in <module>
    from libero.libero import benchmark as libero_benchmark
ModuleNotFoundError: No module named 'libero'
EXIT: 1
```

Relevant observations around this error:

- The `.venv` was created by `uv sync` using **CPython 3.11.15**. The Colab
  notebook kernel itself is **CPython 3.12.13**. These are two different Python
  installations.
- The first attempt to install LIBERO used
  `uv pip install -e third_party/libero` (no explicit interpreter). Its output
  ended with `Using Python 3.12.13 environment at: /usr`, i.e. it installed into
  the **system** interpreter (`/usr`, 3.12), not the `.venv` (3.11) that `PY`
  points to.
- A second attempt used `uv pip install --python {PY} -e third_party/libero`.
  uv reported `Resolved 1 package`, `Installed 1 package`, and
  `+ libero==0.1.0`. Despite this, `PY -c "import libero"` still raises
  `ModuleNotFoundError: No module named 'libero'`.
- `uv sync` (which set up everything else) reported `Resolved 279 packages` and
  installed the full locked dependency set into `.venv` successfully, including
  `jax==0.5.3` (+ CUDA12 plugins), `torch==2.7.1`, `lerobot==0.1.0` (from git),
  `openpi`, and `openpi-client`.

Observed facts about the LIBERO package layout that are relevant to the import:

- The repository top-level directory `third_party/libero/libero/` does **not**
  contain an `__init__.py`. Only the nested `third_party/libero/libero/libero/`
  … i.e. `libero/libero/__init__.py` exists.
- `setup.py` declares `install_requires=[]` (empty). LIBERO's actual runtime
  dependencies (`robosuite==1.4.0`, `bddl==1.0.1`, `robomimic==0.2.0`,
  `hydra-core`, `easydict`, etc.) are listed only in `requirements.txt`, which
  is not consumed by a `pip`/`uv` install of the package. This is consistent
  with uv reporting only `1 package` resolved.

### 0.2 Earlier problems encountered this session (chronological)

These were observed earlier and either worked around or bypassed; recorded here
for completeness.

1. **`pip install -e third_party/openpi` never completes.** Plain pip's
   dependency resolver entered prolonged backtracking (observed hanging >40–50
   minutes with no completion). The traceback on interrupt showed it deep in
   `pip._internal.resolution.resolvelib` parsing installed-package metadata.
   openpi pins hard versions (`jax[cuda12]==0.5.3`, `torch==2.7.1`,
   `numpy<2.0.0`, `flax==0.10.2`, `orbax-checkpoint==0.11.13`,
   `transformers==4.53.2`) and sources `lerobot`/`dlimp` from git via
   `[tool.uv.sources]`, which pip does not understand.

2. **`uv sync` succeeds.** Replacing the pip install with `uv sync --no-dev`
   built the `.venv` and resolved all 279 packages in seconds.

3. **HuggingFace dataset download stalls.** `snapshot_download` for
   `pi05-libero/libero_object_summed_subsampling` repeatedly froze at ~99%
   (e.g. `181M/184M`), with the progress timer frozen. This occurred with the
   `hf_xet` backend (the "Fetching files" xet fetcher) and again with the
   classic HTTP path. An interrupt traceback showed it inside
   `tqdm.contrib.concurrent._executor_map` (a thread pool). The observed
   transfer rate at the tail was very low (hundreds of B/s to a few hundred
   kB/s).

4. **The dataset repository itself is complete.** A metadata listing showed the
   repo has 1505 files totalling 327.2 MB, including all 1000 video files; no
   single file is larger than ~0.6 MB. The stall is associated with fetching
   many small files, not with a single large or corrupt file.

5. **Partial-download deletion failed.** `rm -rf` of the partially-downloaded
   dataset directory reported `Directory not empty` for a
   `.cache/huggingface/download/...` subpath. This coincided with a
   `snapshot_download` cell still executing in the background (still writing
   files into that cache directory).

6. **`google.colab.auth.authenticate_user()` fails.** It raised
   `MessageError: Error: credential propagation was unsuccessful`, even after
   the browser consent was granted.

7. **`gsutil` reads the public checkpoint bucket without authentication.**
   `gsutil ls gs://openpi-assets/checkpoints/pi05_base/` returned
   `assets/` and `params/` (with a non-fatal notice recommending the
   `gcloud storage` CLI over `gsutil`). openpi routes `gs://openpi-assets`
   specifically through `gsutil` (see `openpi/shared/download.py`).

8. **Subprocess errors are not surfaced.** The `sh()` helper runs commands via
   `subprocess.run(..., check=True)` without capturing output. On failure, the
   notebook shows only the parent `CalledProcessError`; the child process's
   stderr/traceback is not visible inline. Diagnosing failures required re-running
   the command with the child's stdout/stderr captured or streamed.

9. **`huggingface-cli` is deprecated.** Invoking `huggingface-cli download` on
   Colab printed `` `huggingface-cli` is deprecated and no longer works. Use `hf`
   instead. `` and exited non-zero. The replacement is the `hf` CLI.

---

## Part 1 — Purpose of the notebook

`colab_training.ipynb` runs the full π0.5 → LIBERO-OBJECT LoRA fine-tuning
replication as a single, largely unattended Colab session on an A100 GPU. In one
run it:

1. **Evaluates the pretrained π0.5 base model** on LIBERO-OBJECT to establish a
   baseline success rate — using *our* preprocessing and normalization stats.
2. **Fine-tunes** the base model on LIBERO-OBJECT with LoRA adapters for 30,000
   steps, in one continuous training run.
3. **Evaluates every preserved checkpoint** (every 5,000 steps) on LIBERO-OBJECT.
4. **Persists the useful artifacts**: LoRA adapter weights to Google Drive, and
   all evaluation metrics + rollout videos to Weights & Biases (WandB).

The scientific setup (summed-subsampling dataset, masked flow-matching loss on
the 7 real action dims, LoRA on both the PaliGemma 2B backbone and the 300M
action expert, discrete state input) is defined in the training config
`pi05_libero_object_lora` and documented in
`third_party/openpi/src/openpi/training/misc/libero_object_configs.py`.

---

## Part 2 — Execution environment and why subprocesses are used

### 2.1 Two separate Python interpreters

- **Colab kernel Python** (`/usr`, CPython 3.12): runs the notebook cells
  themselves. Used only for lightweight orchestration — mounting Drive, cloning
  the repo, setting environment variables, extracting the dataset, and
  launching subprocesses. It has the packages Colab preinstalls (e.g.
  `huggingface_hub`) but **not** the openpi stack.

- **openpi venv Python** (`third_party/openpi/.venv/bin/python`, CPython 3.11,
  referred to throughout as `PY`): created by `uv sync`. This is the only
  interpreter that has `openpi`, `jax`, `torch`, `lerobot`, `openpi-client`
  (and is intended to have `libero`) installed at the exact locked versions.

### 2.2 Why every ML step runs as a subprocess

The training and evaluation scripts (`scripts/train.py`, `scripts/benchmark.py`,
`scripts/extract_lora.py`) import the openpi stack. Those packages exist **only**
in the `.venv` (3.11), not in the notebook kernel (3.12). The kernel therefore
cannot `import` them directly. Each ML step is invoked as a subprocess using
`PY` so it runs inside the venv where the dependencies live. This is the reason
the notebook uses `subprocess` rather than in-kernel imports for anything that
touches openpi/jax/libero.

The `sh()` helper in §2 is the wrapper used to launch these subprocesses with
the correct working directory.

---

## Part 3 — Storage plan

| Artifact | Location | Rationale |
|---|---|---|
| Full training checkpoints (6 × ~4.8 GB) | Colab **local disk** `/content/checkpoints/...` | Ephemeral. Needed only to resume across the single run and to be read by the evaluator. Not persisted. |
| LoRA adapter weights (6 × ~84 MB) | **Google Drive** `.../pi05_libero_replication/lora/...` | The real, permanent output. Small. Extracted from each full checkpoint. |
| Evaluation metrics + rollout videos | **WandB** (project `pi05_libero_replication`) | Permanent, comparable across runs; videos viewable in the run page. |
| Normalization stats (`norm_stats.json`) | **Git** (committed under `assets/pi05_libero/...`) | Cloned with the repo; small; identical for baseline and fine-tuned evals. |
| Training dataset (~327 MB, LeRobot v2.1) | **Google Drive** tarball → extracted to Colab local disk | See Part 4. |

Full checkpoints stay on local SSD (fast) and are discarded with the session.
Only the ~84 MB LoRA deltas are kept, because the frozen base weights are
identical to the public `pi05_base` checkpoint and need not be re-stored.

---

## Part 4 — Dataset acquisition (updated: Google Drive instead of HuggingFace)

### 4.1 Change of approach

The dataset was originally intended to be pulled from the HuggingFace Hub repo
`pi05-libero/libero_object_summed_subsampling` via `snapshot_download` on each
session. Because that download repeatedly stalled on Colab (see §0.2, items
3–5), the approach was changed to **store the dataset as a single tar archive on
Google Drive** and extract it locally each session.

### 4.2 Why a single tarball, extracted to local disk

- **One file, not 1,505.** A single sequential file transfer over the Drive
  FUSE mount avoids the per-file network round-trips and the parallel-fetch
  stalls seen with the Hub download.
- **Extract to local SSD, not read from Drive directly.** The Drive mount is a
  network FUSE filesystem; random-access video reads during training would
  bottleneck throughput. The archive is therefore extracted to
  `/content/lerobot/...` (local SSD) and training reads from there.

### 4.3 The symlink consideration

On the source machine, the dataset directory's `videos/` entry is a **symlink**
(`videos -> ../libero_object_canonical/videos`). A naive folder upload or a plain
`tar` would capture the symlink rather than the real video bytes. The archive
must therefore be created with symlink dereferencing (`tar -chf`), so the real
`.mp4` files are embedded. The archive built this way contains no symlinks and
materializes all 1,000 videos as regular files.

### 4.4 Resulting layout and path alignment

- Archive location on Drive:
  `MyDrive/pi05_libero_replication/dataset/libero_object_summed_subsampling.tar`
- Extraction target: `/content/lerobot/`, producing
  `/content/lerobot/libero_object_summed_subsampling/`
- Environment: `HF_LEROBOT_HOME=/content/lerobot`, and the config's
  `repo_id="libero_object_summed_subsampling"`.
- LeRobot resolves the dataset path as `HF_LEROBOT_HOME / repo_id`, i.e.
  `/content/lerobot/libero_object_summed_subsampling/`, which matches the
  extraction target. Nothing downstream changes.

### 4.5 Completeness vs. presence

The dataset cell guards work with an *existence* check
(`path.exists() and any(path.iterdir())`). This detects whether a directory is
present and non-empty, but **not** whether it is complete. A directory left
behind by a partial download would satisfy the check and cause extraction to be
skipped. A complete extraction of the tarball contains **1,504 files**
(500 parquet + 1,000 videos + 4 meta files: `info.json`, `episodes.jsonl`,
`episodes_stats.jsonl`, `tasks.jsonl`). The HuggingFace repo shows 1,505 because
the Hub adds a `.gitattributes` file, which LeRobot does not use.

---

## Part 5 — Section-by-section walkthrough of the notebook

### §0 · Config
Defines tokens (`WANDB_API_KEY`, `HF_TOKEN`), sources (`GITHUB_REPO`,
`HF_DATASET`, `BASE_CKPT=gs://openpi-assets/checkpoints/pi05_base`), the fixed
names that must match the training config (`CONFIG_NAME=pi05_libero_object_lora`,
`EXP_NAME=masked_loss_summed_subsampling`), and all paths, including:
- `REPO`, `OPENPI = {REPO}/third_party/openpi`, `PY = {OPENPI}/.venv/bin/python`
- `DATASET_DIR = /content/lerobot/libero_object_summed_subsampling`
- `LOCAL_CKPT = /content/checkpoints/pi05_libero` (ephemeral full checkpoints)
- `DRIVE_ROOT`, `DRIVE_LORA` (permanent LoRA output)
- Knobs: `RUN_BASELINE=True`, `TRIALS_PER_TASK=50`.

### §1a · Mount Drive + clone repo
Mounts Google Drive at `/content/drive`, and clones the parent repo with
`--recurse-submodules` (submodule URLs are HTTPS so no credentials are needed).

### §1b · System libs + Python env
Installs system libraries (`ffmpeg`, `libgl1-mesa-glx`, `libegl1-mesa`), then
uses `uv`:
- `uv sync --no-dev` in `OPENPI` → builds `.venv` with the exact locked deps.
- installs LIBERO into the environment so evaluation rollouts can import it.
- asserts `PY` exists and reports the venv interpreter path.

(Note: the correct target interpreter for the LIBERO install is the subject of
the current issue in §0.1; per instruction, no dependency changes are made in
this document.)

### §1c · Environment variables
Sets `HF_LEROBOT_HOME=/content/lerobot`, `MUJOCO_GL=egl` (GPU off-screen
rendering for LIBERO), `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, and exports
`HF_TOKEN` / `WANDB_API_KEY` if provided. Creates the local checkpoint and Drive
LoRA directories. These variables are inherited by all subprocesses.

### §1d · Dataset (Drive tarball → local disk)
Updated to restore the dataset from the Drive tarball rather than download from
the Hub: if the local dataset directory is absent, it extracts
`.../dataset/libero_object_summed_subsampling.tar` into `/content/lerobot/`.
(See Part 4 for the completeness caveat.)

### §1e · Norm-stats check (and GCS access)
Confirms the committed `norm_stats.json` is present in the cloned repo at
`assets/pi05_libero/pi05_libero_object_lora/libero_object_summed_subsampling/`.
Reading the `pi05_base` checkpoint from `gs://openpi-assets` is done by openpi via
`gsutil`, which was observed to work on Colab without interactive
authentication (the bucket is public).

### §2 · Full experiment (unattended)
The single orchestration cell. It defines `sh()` (subprocess launcher) and
`benchmark(...)` (wraps `scripts/benchmark.py`), then:

1. **Baseline eval** — runs `benchmark.py` against `BASE_CKPT` with
   `--config_name pi05_libero_object_lora`. The evaluator loads *our* norm stats
   from the config's assets dir and passes them explicitly, so the base model is
   evaluated under the same normalization as the fine-tuned model. Results →
   WandB run `pi05_base_benchmark`.

2. **Train 30k** — runs `scripts/train.py pi05_libero_object_lora
   --exp_name masked_loss_summed_subsampling --checkpoint_base_dir {LOCAL_CKPT}`.
   One continuous run. The config sets `save_interval=5000` and
   `keep_period=5000`, so checkpoints at 5000/10000/15000/20000/25000 are
   preserved (and the final step retained as the latest), all on local disk.
   Loss curve → WandB.

3. **Evaluate every preserved checkpoint** — enumerates the numeric checkpoint
   step directories under
   `{LOCAL_CKPT}/{CONFIG_NAME}/{EXP_NAME}/`, and for each: runs `benchmark.py`
   (metrics + rollout videos → WandB run `step_<n>`), then runs
   `extract_lora.py` to pull the ~84 MB LoRA adapters out of the full checkpoint
   into `{DRIVE_LORA}/step_<n>.npz`.

All three stages run as subprocesses through `PY` with `cwd=OPENPI` (so the
config's relative `../../assets/...` path resolves to the repo's `assets/`).

### §3 · Results summary (optional)
Reads the saved `results.json` files and prints a local table of aggregate
success rates (baseline vs. each checkpoint step). All numbers are also in WandB.

---

## Part 6 — Key parameters and where they are defined

- **Config** `pi05_libero_object_lora` (in
  `third_party/openpi/src/openpi/training/misc/libero_object_configs.py`):
  `pi05=True`, `action_horizon=10`, `discrete_state_input=True`,
  `action_dim_actual=7` (masked loss), `paligemma_variant=gemma_2b_lora`,
  `action_expert_variant=gemma_300m_lora`, `ema_decay=None`, `batch_size=32`,
  `num_train_steps=30_000`, `save_interval=5_000`, `keep_period=5_000`,
  `assets_base_dir=../../assets/pi05_libero`,
  `checkpoint_base_dir=../../checkpoints/pi05_libero`.
- **Evaluator** `scripts/benchmark.py`: `TASK_SUITE=libero_object`,
  `MAX_STEPS=280`, `NUM_STEPS_WAIT=10`, `REPLAN_STEPS=5` (receding-horizon
  control), 50 trials/task × 10 tasks = 500 rollouts per checkpoint, saves up to
  5 failure + 3 success videos per task, logs metrics + videos to WandB, loads
  our norm stats explicitly.
- **LoRA extractor** `scripts/extract_lora.py`: restores checkpoint params as
  numpy, keeps leaves whose path contains `lora`, saves to a `.npz`
  (path separators encoded as `__`).
