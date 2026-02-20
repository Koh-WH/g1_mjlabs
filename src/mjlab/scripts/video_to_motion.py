"""
video_to_motion.py — Video → trained MJLab tracking policy in one command.

Runs these exact steps in the correct envs:

    [gvhmr env]  python tools/demo/demo.py --video=<video> -s
    [gmr env]    python scripts/gvhmr_to_robot.py ... --robot unitree_g1
    [gmr env]    python scripts/pkl_to_csv.py --input <pkl>
    [mjlab env]  uv run python src/mjlab/scripts/csv_to_npz.py ...
    [mjlab env]  uv run train Mjlab-Tracking-Flat-Unitree-G1 ...

Usage
-----
    uv run python src/mjlab/scripts/video_to_motion.py \\
        --video      /any/path/to/video.mp4 \\
        --name       my_motion \\
        --gvhmr-dir  /path/to/GVHMR \\
        --gmr-dir    /path/to/GMR \\
        --wandb-entity my-wandb-username

    # Shorter — set dirs via environment variables instead of flags:
    export GVHMR_DIR=/path/to/GVHMR
    export GMR_DIR=/path/to/GMR
    export WANDB_ENTITY=my-wandb-username
    uv run python src/mjlab/scripts/video_to_motion.py \\
        --video /any/path/to/video.mp4 --name my_motion

    # Skip GVHMR if hmr4d_results.pt already exists:
    uv run python src/mjlab/scripts/video_to_motion.py \\
        --hmr4d /path/to/hmr4d_results.pt \\
        --name  my_motion ...

    # Upload artifact only, skip training:
    uv run python src/mjlab/scripts/video_to_motion.py \\
        --video ... --name my_motion ... --no-train

    # Static/tripod camera footage (better GVHMR results):
    uv run python src/mjlab/scripts/video_to_motion.py \\
        --video ... --name my_motion ... --static-cam

Environment variables (alternative to flags)
--------------------------------------------
    GVHMR_DIR       path to GVHMR repo
    GMR_DIR         path to GMR repo
    WANDB_ENTITY    W&B username/entity
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_conda_python(env_name: str) -> Path:
    """Get the python executable path for a conda environment."""
    # Try to get conda env path via conda info
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[video_to_motion] ERROR: conda env list failed")
        sys.exit(1)
    
    # Parse output to find env path
    for line in result.stdout.split('\n'):
        if env_name in line and not line.startswith('#'):
            parts = line.split()
            # Format: "env_name * /path/to/env" or "env_name /path/to/env"
            env_path = parts[-1]
            python_path = Path(env_path) / "bin" / "python"
            if python_path.exists():
                return python_path
    
    # Fallback: try default conda env location
    conda_base = Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python"
    if conda_base.exists():
        return conda_base
    conda_base = Path.home() / "anaconda3" / "envs" / env_name / "bin" / "python"
    if conda_base.exists():
        return conda_base
    
    print(f"[video_to_motion] ERROR: Could not find python for conda env '{env_name}'")
    sys.exit(1)


def run(cmd: list[str], cwd: str | None = None, env_name: str | None = None):
    """Run a command, optionally using a conda env's python."""
    if env_name:
        # Replace 'python' with the full path to the conda env's python
        if cmd[0] == "python":
            python_path = get_conda_python(env_name)
            cmd = [str(python_path)] + cmd[1:]
        else:
            print(f"[video_to_motion] WARNING: env_name specified but command doesn't start with 'python'")
    
    print(f"\n[video_to_motion] $ {' '.join(str(c) for c in cmd)}")
    if cwd:
        print(f"[video_to_motion]   cwd: {cwd}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[video_to_motion] FAILED (exit {result.returncode})")
        sys.exit(result.returncode)


def require_path(p: str | Path | None, label: str) -> Path:
    if p is None:
        print(f"[video_to_motion] ERROR: {label} not set. "
              f"Pass it as a flag or set the corresponding environment variable.")
        sys.exit(1)
    path = Path(p).expanduser().resolve()
    if not path.exists():
        print(f"[video_to_motion] ERROR: {label} not found: {path}")
        sys.exit(1)
    return path


def resolve(flag_val: str | None, env_var: str) -> str | None:
    """Return flag value if given, else fall back to env var."""
    return flag_val or os.environ.get(env_var)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step1_gvhmr(video: Path, gvhmr_dir: Path, static_cam: bool) -> Path:
    print("\n" + "="*60)
    print("STEP 1/4 — GVHMR: extract human pose from video")
    print("="*60)

    cmd = ["python", "tools/demo/demo.py", f"--video={video}", "-s"]
    if static_cam:
        cmd.append("--static_cam")
    run(cmd, cwd=str(gvhmr_dir), env_name="gvhmr")

    output = gvhmr_dir / "outputs" / "demo" / video.stem / "hmr4d_results.pt"
    if not output.exists():
        candidates = list((gvhmr_dir / "outputs").rglob("hmr4d_results.pt"))
        if not candidates:
            print(f"[video_to_motion] ERROR: hmr4d_results.pt not found under {gvhmr_dir}/outputs/")
            sys.exit(1)
        output = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    print(f"[video_to_motion] ✓ GVHMR output: {output}")
    return output


def step2_gmr_retarget(hmr4d_pt: Path, gmr_dir: Path, pkl_path: Path) -> Path:
    print("\n" + "="*60)
    print("STEP 2/4 — GMR: retarget human pose to Unitree G1")
    print("="*60)

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "python", "scripts/gvhmr_to_robot.py",
        "--gvhmr_pred_file", str(hmr4d_pt),
        "--robot", "unitree_g1",
        "--save_path", str(pkl_path),
    ], cwd=str(gmr_dir), env_name="gmr")

    if not pkl_path.exists():
        print(f"[video_to_motion] ERROR: pkl not written: {pkl_path}")
        sys.exit(1)

    print(f"[video_to_motion] ✓ pkl: {pkl_path}")
    return pkl_path


def step3_pkl_to_csv(pkl_path: Path, gmr_dir: Path) -> Path:
    print("\n" + "="*60)
    print("STEP 3/4 — GMR: convert pkl → CSV")
    print("="*60)

    run([
        "python", "scripts/pkl_to_csv.py",
        "--input", str(pkl_path),
    ], cwd=str(gmr_dir), env_name="gmr")

    csv_path = pkl_path.with_suffix(".csv")
    if not csv_path.exists():
        print(f"[video_to_motion] ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    print(f"[video_to_motion] ✓ CSV: {csv_path}")
    return csv_path


def step4a_csv_to_npz(
    csv_path: Path,
    name: str,
    mjlab_dir: Path,
    input_fps: float,
    output_fps: float,
    wandb_entity: str,
) -> str:
    print("\n" + "="*60)
    print("STEP 4a/4 — MJLab: csv_to_npz → W&B artifact")
    print("="*60)

    run([
        "uv", "run", "python", "src/mjlab/scripts/csv_to_npz.py",
        "--input-file", str(csv_path),
        "--output-name", name,
        f"--input-fps={input_fps}",
        f"--output-fps={output_fps}",
    ], cwd=str(mjlab_dir))

    # Registry path format: <entity>-org/wandb-registry-Motions/<name>
    registry = f"{wandb_entity}-org/wandb-registry-Motions/{name}"
    print(f"[video_to_motion] ✓ W&B artifact: {registry}")
    return registry


def step4b_train(
    registry: str,
    name: str,
    mjlab_dir: Path,
    num_envs: int,
    max_iter: int,
    save_interval: int,
    wandb_entity: str,
    early_stop_patience: int = 500,
    early_stop_enabled: bool = True,
):
    print("\n" + "="*60)
    print("STEP 4b/4 — MJLab: train tracking policy")
    if early_stop_enabled:
        print(f"  Early stopping enabled: {early_stop_patience} iters without improvement")
    print("="*60)

    cmd = [
        "uv", "run", "train",
        "Mjlab-Tracking-Flat-Unitree-G1",
        "--registry-name", registry,
        "--env.scene.num-envs", str(num_envs),
        "--agent.max-iterations", str(max_iter),
        "--agent.save-interval", str(save_interval),
        "--agent.run-name", name,
    ]

    if not early_stop_enabled:
        # Standard blocking call
        run(cmd, cwd=str(mjlab_dir))
        return

    # Start training as non-blocking subprocess
    print(f"\n[video_to_motion] $ {' '.join(str(c) for c in cmd)}")
    print(f"[video_to_motion]   cwd: {mjlab_dir}")
    process = subprocess.Popen(cmd, cwd=str(mjlab_dir))

    try:
        # Wait a bit for W&B run to initialize
        import time
        time.sleep(30)

        # Monitor training progress
        monitor_training_with_early_stop(
            process=process,
            patience=early_stop_patience,
            check_interval=30,  # check every 30 seconds
            wandb_entity=wandb_entity,
        )

        # Wait for graceful shutdown
        process.wait(timeout=60)
        if process.returncode != 0 and process.returncode != -15:  # -15 is SIGTERM
            print(f"[video_to_motion] Training failed (exit {process.returncode})")
            sys.exit(process.returncode)

    except KeyboardInterrupt:
        print("\n[video_to_motion] Training interrupted by user")
        process.terminate()
        process.wait(timeout=10)
        sys.exit(1)


def monitor_training_with_early_stop(
    process: subprocess.Popen,
    patience: int,
    check_interval: int = 30,
    wandb_entity: str = None,
):
    """Monitor training and terminate if reward plateaus."""
    import time

    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        print("[video_to_motion] WARNING: wandb not installed, early stopping disabled")
        process.wait()
        return

    # Find the run - it should be the most recent in the project
    time.sleep(5)  # give wandb a moment to register the run

    print("[video_to_motion] Monitoring training progress for early stopping...")

    # Try to get entity from wandb config if not provided
    if wandb_entity is None:
        try:
            wandb_entity = api.default_entity
        except:
            wandb_entity = "mjlab"  # fallback

    best_reward = float('-inf')
    iters_since_improvement = 0
    last_iter = 0
    last_improvement_iter = 0

    while process.poll() is None:  # while training is running
        time.sleep(check_interval)

        try:
            # Get the most recent run (this training run)
            project_path = f"{wandb_entity}/mjlab"
            runs = api.runs(path=project_path, order="-created_at", per_page=1)
            if not runs:
                continue

            run = runs[0]
            
            # Try to get history with pandas, fall back to non-pandas
            try:
                history = run.history(keys=["rollout/ep_rew_mean", "_step"], samples=1000, pandas=True)
            except:
                history = run.history(keys=["rollout/ep_rew_mean", "_step"], samples=1000, pandas=False)

            # Handle both pandas DataFrame and list formats
            if isinstance(history, list):
                if not history:
                    continue
                # Convert list of dicts to simple format
                latest = history[-1]
                if "rollout/ep_rew_mean" not in latest or "_step" not in latest:
                    continue
                current_iter = int(latest["_step"])
                current_reward = float(latest["rollout/ep_rew_mean"])
            else:
                # pandas DataFrame
                if history.empty or "rollout/ep_rew_mean" not in history.columns:
                    continue
                latest = history.iloc[-1]
                current_iter = int(latest["_step"])
                current_reward = float(latest["rollout/ep_rew_mean"])

            # Skip if no new data
            if current_iter <= last_iter:
                continue

            last_iter = current_iter

            # Check for improvement
            if current_reward > best_reward:
                best_reward = current_reward
                last_improvement_iter = current_iter
                iters_since_improvement = 0
                print(f"[early_stop] iter={current_iter:5d}  reward={current_reward:.2f}  ✓ new best")
            else:
                iters_since_improvement = current_iter - last_improvement_iter
                print(f"[early_stop] iter={current_iter:5d}  reward={current_reward:.2f}  "
                      f"({iters_since_improvement}/{patience} no improvement)")

            # Trigger early stop
            if iters_since_improvement >= patience:
                print(f"\n[early_stop] No improvement for {patience} iters - stopping training")
                print(f"[early_stop] Best reward: {best_reward:.2f}")
                process.terminate()
                return

        except Exception as e:
            # Don't crash on transient API errors
            print(f"[early_stop] Warning: {e}")
            continue

    print("[early_stop] Training completed normally")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Video → MJLab tracking policy (GVHMR + GMR + csv_to_npz + train)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source — mutually exclusive
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", metavar="PATH",
                     help="Path to any video file (.mp4, .avi, .mov, etc.)")
    src.add_argument("--hmr4d", metavar="PATH",
                     help="Path to existing hmr4d_results.pt — skips GVHMR step")

    # Required
    p.add_argument("--name", required=True, metavar="NAME",
                   help="Motion name used for W&B artifact and training run")

    # Tool directories — flag or env var
    p.add_argument("--gvhmr-dir", metavar="DIR", default=None,
                   help="Path to GVHMR repo root  [env: GVHMR_DIR]")
    p.add_argument("--gmr-dir",   metavar="DIR", default=None,
                   help="Path to GMR repo root    [env: GMR_DIR]")
    p.add_argument("--mjlab-dir", metavar="DIR", default=None,
                   help="Path to mjlab repo root  [env: MJLAB_DIR]  "
                        "(auto-detected from script location if not set)")

    # W&B entity
    p.add_argument("--wandb-entity", metavar="ENTITY", default=None,
                   help="W&B username or entity   [env: WANDB_ENTITY]  "
                        "Run 'wandb status' to see yours")

    # GVHMR options
    p.add_argument("--static-cam", action="store_true",
                   help="Tell GVHMR the camera is static (tripod footage — better results)")

    # FPS
    p.add_argument("--input-fps",  type=float, default=30.0,
                   help="FPS of the input video (default: 30)")
    p.add_argument("--output-fps", type=float, default=50.0,
                   help="Target FPS for MJLab training (default: 50)")

    # Training hyperparams
    p.add_argument("--num-envs",      type=int, default=1024,
                   help="Number of parallel environments (default: 1024)")
    p.add_argument("--max-iter",      type=int, default=5000,
                   help="Maximum training iterations (default: 5000)")
    p.add_argument("--save-interval", type=int, default=500,
                   help="Save checkpoint every N iterations (default: 500)")

    # Early stopping
    p.add_argument("--early-stop-patience", type=int, default=1000,
                   help="Stop if no reward improvement for N iters (default: 1000)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping (train to max-iter regardless)")

    # Control flow
    p.add_argument("--no-train", action="store_true",
                   help="Stop after uploading W&B artifact, skip training")

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve dirs — flag > env var > auto-detect (mjlab only)
    gvhmr_dir_raw  = resolve(args.gvhmr_dir,    "GVHMR_DIR")
    gmr_dir_raw    = resolve(args.gmr_dir,       "GMR_DIR")
    mjlab_dir_raw  = resolve(args.mjlab_dir,     "MJLAB_DIR")
    wandb_entity   = resolve(args.wandb_entity,  "WANDB_ENTITY")

    # Auto-detect mjlab root from this script's location:
    # src/mjlab/scripts/video_to_motion.py → ../../.. = mjlab root
    if mjlab_dir_raw is None:
        mjlab_dir_raw = str(Path(__file__).resolve().parents[3])
        print(f"[video_to_motion] Auto-detected mjlab root: {mjlab_dir_raw}")

    # Validate required dirs
    gmr_dir   = require_path(gmr_dir_raw,   "--gmr-dir / GMR_DIR")
    mjlab_dir = require_path(mjlab_dir_raw, "--mjlab-dir / MJLAB_DIR")

    if wandb_entity is None:
        print("[video_to_motion] ERROR: W&B entity not set.")
        print("  Pass --wandb-entity <name>  or  export WANDB_ENTITY=<name>")
        print("  Run 'wandb status' to see your current entity.")
        sys.exit(1)

    # PKL and CSV live in GMR's output folder
    gmr_output = gmr_dir / "output"
    gmr_output.mkdir(parents=True, exist_ok=True)
    pkl_path = gmr_output / f"{args.name}.pkl"

    print(f"\n{'='*60}")
    print(f"  video_to_motion  →  {args.name}")
    print(f"{'='*60}")
    if args.video:
        print(f"  video      : {args.video}")
    else:
        print(f"  hmr4d      : {args.hmr4d}")
    print(f"  gmr-dir    : {gmr_dir}")
    print(f"  mjlab-dir  : {mjlab_dir}")
    print(f"  entity     : {wandb_entity}")
    print(f"  train      : {'no' if args.no_train else 'yes'}")

    # ── Step 1: Video → hmr4d_results.pt ────────────────────────────────
    if args.hmr4d:
        hmr4d_pt = require_path(args.hmr4d, "--hmr4d")
        print(f"\n[video_to_motion] Skipping GVHMR, using: {hmr4d_pt}")
    else:
        gvhmr_dir = require_path(gvhmr_dir_raw, "--gvhmr-dir / GVHMR_DIR")
        video = require_path(args.video, "--video")
        hmr4d_pt = step1_gvhmr(video, gvhmr_dir, args.static_cam)

    # ── Step 2: hmr4d_results.pt → robot_motion.pkl ─────────────────────
    step2_gmr_retarget(hmr4d_pt, gmr_dir, pkl_path)

    # ── Step 3: robot_motion.pkl → motion.csv ───────────────────────────
    csv_path = step3_pkl_to_csv(pkl_path, gmr_dir)

    # ── Step 4a: CSV → NPZ → W&B artifact ───────────────────────────────
    registry = step4a_csv_to_npz(
        csv_path     = csv_path,
        name         = args.name,
        mjlab_dir    = mjlab_dir,
        input_fps    = args.input_fps,
        output_fps   = args.output_fps,
        wandb_entity = wandb_entity,
    )

    if args.no_train:
        print(f"\n{'='*60}")
        print(f"✓ Done (--no-train).  Artifact: {registry}")
        print(f"{'='*60}")
        print(f"\n  Train manually:")
        print(f"    cd {mjlab_dir}")
        print(f"    uv run train Mjlab-Tracking-Flat-Unitree-G1 \\")
        print(f"        --registry-name {registry} \\")
        print(f"        --env.scene.num-envs {args.num_envs} \\")
        print(f"        --agent.max-iterations {args.max_iter} \\")
        print(f"        --agent.save-interval {args.save_interval} \\")
        print(f"        --agent.run-name {args.name}")
        return

    # ── Step 4b: Train ───────────────────────────────────────────────────
    step4b_train(
        registry      = registry,
        name          = args.name,
        mjlab_dir     = mjlab_dir,
        num_envs      = args.num_envs,
        max_iter      = args.max_iter,
        save_interval = args.save_interval,
        wandb_entity  = wandb_entity,
        early_stop_patience = args.early_stop_patience,
        early_stop_enabled  = not args.no_early_stop,
    )

    print(f"\n{'='*60}")
    print(f"✓ All done!  Motion: {args.name}")
    print(f"{'='*60}")
    print(f"\n  Visualise:")
    print(f"    cd {mjlab_dir}")
    print(f"    uv run play Mjlab-Tracking-Flat-Unitree-G1 \\")
    print(f"        --wandb-run-path {wandb_entity}/mjlab/<run-id>")


if __name__ == "__main__":
    main()