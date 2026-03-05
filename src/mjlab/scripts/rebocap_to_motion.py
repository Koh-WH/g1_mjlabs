"""
Usage
-----
    uv run python src/mjlab/scripts/rebocap_to_motion.py \
        --bvh        /home/koh-wh/Downloads/GMR/bvh/starjumps.bvh \
        --name       starjumps_rebocap \
        --gmr-dir    /home/koh-wh/Downloads/GMR \
        --wandb-entity kohwh-nanyang-technological-university-singapore

Environment variables (alternative to flags)
--------------------------------------------
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
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[rebocap_to_motion] ERROR: conda env list failed")
        sys.exit(1)
    
    for line in result.stdout.split('\n'):
        if env_name in line and not line.startswith('#'):
            parts = line.split()
            env_path = parts[-1]
            python_path = Path(env_path) / "bin" / "python"
            if python_path.exists():
                return python_path
    
    conda_base = Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python"
    if conda_base.exists():
        return conda_base
    conda_base = Path.home() / "anaconda3" / "envs" / env_name / "bin" / "python"
    if conda_base.exists():
        return conda_base
    
    print(f"[rebocap_to_motion] ERROR: Could not find python for conda env '{env_name}'")
    sys.exit(1)


def run(cmd: list[str], cwd: str | None = None, env_name: str | None = None):
    """Run a command, optionally using a conda env's python."""
    if env_name:
        if cmd[0] == "python":
            python_path = get_conda_python(env_name)
            cmd = [str(python_path)] + cmd[1:]
        else:
            print(f"[rebocap_to_motion] WARNING: env_name specified but command doesn't start with 'python'")
    
    print(f"\n[rebocap_to_motion] $ {' '.join(str(c) for c in cmd)}")
    if cwd:
        print(f"[rebocap_to_motion]   cwd: {cwd}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[rebocap_to_motion] FAILED (exit {result.returncode})")
        sys.exit(result.returncode)


def require_path(p: str | Path | None, label: str) -> Path:
    if p is None:
        print(f"[rebocap_to_motion] ERROR: {label} not set. "
              f"Pass it as a flag or set the corresponding environment variable.")
        sys.exit(1)
    path = Path(p).expanduser().resolve()
    if not path.exists():
        print(f"[rebocap_to_motion] ERROR: {label} not found: {path}")
        sys.exit(1)
    return path


def resolve(flag_val: str | None, env_var: str) -> str | None:
    """Return flag value if given, else fall back to env var."""
    return flag_val or os.environ.get(env_var)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step1_convert_bvh(bvh_in: Path, gmr_dir: Path) -> Path:
    print("\n" + "="*60)
    print("STEP 1/4 — GMR: convert mixamo to lafan1")
    print("="*60)

    # Create 'converted' subdirectory and set output path
    converted_dir = bvh_in.parent / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    bvh_out = converted_dir / f"{bvh_in.stem}_converted.bvh"
    
    run([
        "python", "scripts/convert_mixamo_to_lafan1.py",
        str(bvh_in), str(bvh_out) # Removed --in_place
    ], cwd=str(gmr_dir), env_name="gmr")

    if not bvh_out.exists():
        print(f"[rebocap_to_motion] ERROR: Converted BVH not found: {bvh_out}")
        sys.exit(1)

    print(f"[rebocap_to_motion] ✓ Converted BVH: {bvh_out}")
    return bvh_out


def step2_bvh_to_robot(bvh_in: Path, gmr_dir: Path, pkl_path: Path) -> Path:
    print("\n" + "="*60)
    print("STEP 2/4 — GMR: retarget BVH to Unitree G1")
    print("="*60)

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "python", "scripts/bvh_to_robot.py",
        "--bvh_file", str(bvh_in),
        "--robot", "unitree_g1",
        "--format", "mixamo",
        "--rate_limit",
        "--save_path", str(pkl_path),
    ], cwd=str(gmr_dir), env_name="gmr")

    if not pkl_path.exists():
        print(f"[rebocap_to_motion] ERROR: pkl not written: {pkl_path}")
        sys.exit(1)

    print(f"[rebocap_to_motion] ✓ pkl: {pkl_path}")
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
        print(f"[rebocap_to_motion] ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    print(f"[rebocap_to_motion] ✓ CSV: {csv_path}")
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

    registry = f"{wandb_entity}-org/wandb-registry-Motions/{name}"
    print(f"[rebocap_to_motion] ✓ W&B artifact: {registry}")
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
        run(cmd, cwd=str(mjlab_dir))
        return

    print(f"\n[rebocap_to_motion] $ {' '.join(str(c) for c in cmd)}")
    print(f"[rebocap_to_motion]   cwd: {mjlab_dir}")
    process = subprocess.Popen(cmd, cwd=str(mjlab_dir))

    try:
        import time
        time.sleep(30)

        monitor_training_with_early_stop(
            process=process,
            patience=early_stop_patience,
            check_interval=30,
            wandb_entity=wandb_entity,
        )

        process.wait(timeout=60)
        if process.returncode != 0 and process.returncode != -15:
            print(f"[rebocap_to_motion] Training failed (exit {process.returncode})")
            sys.exit(process.returncode)

    except KeyboardInterrupt:
        print("\n[rebocap_to_motion] Training interrupted by user")
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
        print("[rebocap_to_motion] WARNING: wandb not installed, early stopping disabled")
        process.wait()
        return

    time.sleep(5) 

    print("[rebocap_to_motion] Monitoring training progress for early stopping...")

    if wandb_entity is None:
        try:
            wandb_entity = api.default_entity
        except:
            wandb_entity = "mjlab" 

    best_reward = float('-inf')
    iters_since_improvement = 0
    last_iter = 0
    last_improvement_iter = 0

    while process.poll() is None:
        time.sleep(check_interval)

        try:
            project_path = f"{wandb_entity}/mjlab"
            runs = api.runs(path=project_path, order="-created_at", per_page=1)
            if not runs:
                continue

            run = runs[0]
            
            try:
                history = run.history(keys=["rollout/ep_rew_mean", "_step"], samples=1000, pandas=True)
            except:
                history = run.history(keys=["rollout/ep_rew_mean", "_step"], samples=1000, pandas=False)

            if isinstance(history, list):
                if not history:
                    continue
                latest = history[-1]
                if "rollout/ep_rew_mean" not in latest or "_step" not in latest:
                    continue
                current_iter = int(latest["_step"])
                current_reward = float(latest["rollout/ep_rew_mean"])
            else:
                if history.empty or "rollout/ep_rew_mean" not in history.columns:
                    continue
                latest = history.iloc[-1]
                current_iter = int(latest["_step"])
                current_reward = float(latest["rollout/ep_rew_mean"])

            if current_iter <= last_iter:
                continue

            last_iter = current_iter

            if current_reward > best_reward:
                best_reward = current_reward
                last_improvement_iter = current_iter
                iters_since_improvement = 0
                print(f"[early_stop] iter={current_iter:5d}  reward={current_reward:.2f}  ✓ new best")
            else:
                iters_since_improvement = current_iter - last_improvement_iter
                print(f"[early_stop] iter={current_iter:5d}  reward={current_reward:.2f}  "
                      f"({iters_since_improvement}/{patience} no improvement)")

            if iters_since_improvement >= patience:
                print(f"\n[early_stop] No improvement for {patience} iters - stopping training")
                print(f"[early_stop] Best reward: {best_reward:.2f}")
                process.terminate()
                return

        except Exception as e:
            print(f"[early_stop] Warning: {e}")
            continue

    print("[early_stop] Training completed normally")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Rebocap BVH → MJLab tracking policy (convert + GMR + csv_to_npz + train)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bvh", metavar="PATH",
                     help="Path to input .bvh file")
    src.add_argument("--converted-bvh", metavar="PATH",
                     help="Path to already converted .bvh file — skips Step 1")

    # Required
    p.add_argument("--name", required=True, metavar="NAME",
                   help="Motion name used for output files, W&B artifact, and training run")

    # Tool directories
    p.add_argument("--gmr-dir",   metavar="DIR", default=None,
                   help="Path to GMR repo root    [env: GMR_DIR]")
    p.add_argument("--mjlab-dir", metavar="DIR", default=None,
                   help="Path to mjlab repo root  [env: MJLAB_DIR]  "
                        "(auto-detected from script location if not set)")

    # W&B entity
    p.add_argument("--wandb-entity", metavar="ENTITY", default=None,
                   help="W&B username or entity   [env: WANDB_ENTITY]")

    # FPS
    p.add_argument("--input-fps",  type=float, default=30.0,
                   help="FPS of the input BVH (default: 30)")
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
    p.add_argument("--early-stop-patience", type=int, default=200,
                   help="Stop if no reward improvement for N iters (default: 200)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping (train to max-iter regardless)")

    # Control flow
    p.add_argument("--no-train", action="store_true",
                   help="Stop after uploading W&B artifact, skip training")

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve dirs
    gmr_dir_raw    = resolve(args.gmr_dir,       "GMR_DIR")
    mjlab_dir_raw  = resolve(args.mjlab_dir,     "MJLAB_DIR")
    wandb_entity   = resolve(args.wandb_entity,  "WANDB_ENTITY")

    if mjlab_dir_raw is None:
        mjlab_dir_raw = str(Path(__file__).resolve().parents[3])
        print(f"[rebocap_to_motion] Auto-detected mjlab root: {mjlab_dir_raw}")

    gmr_dir   = require_path(gmr_dir_raw,   "--gmr-dir / GMR_DIR")
    mjlab_dir = require_path(mjlab_dir_raw, "--mjlab-dir / MJLAB_DIR")

    if wandb_entity is None:
        print("[rebocap_to_motion] ERROR: W&B entity not set.")
        print("  Pass --wandb-entity <name>  or  export WANDB_ENTITY=<name>")
        sys.exit(1)

    # Output paths
    gmr_output = gmr_dir / "output"
    gmr_output.mkdir(parents=True, exist_ok=True)
    pkl_path = gmr_output / f"{args.name}.pkl"

    print(f"\n{'='*60}")
    print(f"  rebocap_to_motion  →  {args.name}")
    print(f"{'='*60}")
    if args.bvh:
        print(f"  bvh        : {args.bvh}")
    else:
        print(f"  converted  : {args.converted_bvh}")
    print(f"  gmr-dir    : {gmr_dir}")
    print(f"  mjlab-dir  : {mjlab_dir}")
    print(f"  entity     : {wandb_entity}")
    print(f"  train      : {'no' if args.no_train else 'yes'}")

    # ── Step 1: convert_mixamo_to_lafan1 ────────────────────────────────
    if args.converted_bvh:
        converted_bvh = require_path(args.converted_bvh, "--converted-bvh")
        print(f"\n[rebocap_to_motion] Skipping conversion, using: {converted_bvh}")
    else:
        bvh_in = require_path(args.bvh, "--bvh")
        converted_bvh = step1_convert_bvh(bvh_in, gmr_dir)

    # ── Step 2: bvh_to_robot.py ─────────────────────────────────────────
    step2_bvh_to_robot(converted_bvh, gmr_dir, pkl_path)

    # ── Step 3: pkl_to_csv.py ───────────────────────────────────────────
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


if __name__ == "__main__":
    main()