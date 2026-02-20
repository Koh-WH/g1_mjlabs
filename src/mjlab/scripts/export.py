"""Export a trained MJLab velocity policy to TorchScript (.pt) and ONNX.

Drop into src/mjlab/scripts/ alongside play.py and train.py.

Default output layout (relative to the mjlab project root):
    src/mjlab/export/pt/policy.pt      ← Isaac Gym  (torch.jit.load)
    src/mjlab/export/onnx/policy.onnx  ← Isaac Lab  (future)

Usage
-----
    # Default output paths:
    python export.py Mjlab-Velocity-Flat-Unitree-G1 \\
        --wandb-run-path your-org/mjlab/run-id

    # Override output root:
    python export.py Mjlab-Velocity-Flat-Unitree-G1 \\
        --checkpoint-file logs/rsl_rl/g1_velocity/2026-xx-xx/model_4998.pt \\
        --export-root /some/other/path

Register as `uv run export` by adding to pyproject.toml [project.scripts]:
    export = "mjlab.scripts.export:main"
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.rl.exporter import (
    attach_onnx_metadata,
    export_velocity_policy_as_onnx,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExportConfig:
    wandb_run_path:  Optional[str] = None
    checkpoint_file: Optional[str] = None
    # Root export directory. Defaults to src/mjlab/export/ relative to cwd.
    # Files are placed in <export_root>/pt/ and <export_root>/onnx/ automatically.
    export_root:     Optional[str] = None
    device:          str            = "cpu"  # cpu recommended for export


def run_export(task_id: str, cfg: ExportConfig) -> None:
    configure_torch_backends()

    env_cfg   = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1

    # Resolve checkpoint ------------------------------------------------
    if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"[export] Local checkpoint: {resume_path}")
        wandb_run_path = None
    elif cfg.wandb_run_path is not None:
        log_root = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        resume_path, was_cached = get_wandb_checkpoint_path(
            log_root, Path(cfg.wandb_run_path)
        )
        print(f"[export] Checkpoint: {resume_path.name} ({'cached' if was_cached else 'downloaded'})")
        wandb_run_path = cfg.wandb_run_path
    else:
        raise ValueError("Provide --wandb-run-path or --checkpoint-file")

    # Resolve motion file for tracking tasks ----------------------------
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    is_tracking = "motion" in env_cfg.commands and isinstance(
        env_cfg.commands["motion"], MotionCommandCfg
    )
    if is_tracking:
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        if wandb_run_path is not None:
            import wandb
            api        = wandb.Api()
            wandb_run  = api.run(wandb_run_path)
            art        = next(
                (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
            )
            if art is None:
                raise RuntimeError("No motion artifact found in the W&B run.")
            motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")
            print(f"[export] Motion file: {motion_cmd.motion_file}")
        else:
            raise ValueError(
                "Tracking tasks require --wandb-run-path so the motion artifact "
                "can be resolved. Use --wandb-run-path instead of --checkpoint-file, "
                "or manually set motion_file."
            )

    # Resolve output directories ----------------------------------------
    # Use a task-specific slug as subdirectory so different tasks never
    # overwrite each other.  e.g.:
    #   src/mjlab/export/pt/velocity_flat_g1/policy.pt
    #   src/mjlab/export/onnx/tracking_flat_g1/policy.onnx
    task_slug = task_id.lower().replace("mjlab-", "").replace("-", "_")
    export_root = Path(cfg.export_root) if cfg.export_root else Path("src/mjlab/export")
    pt_dir   = export_root / "pt"   / task_slug
    onnx_dir = export_root / "onnx" / task_slug
    pt_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] pt   → {pt_dir}")
    print(f"[export] onnx → {onnx_dir}")

    # Build env ---------------------------------------------------------
    print("[export] Building environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy -------------------------------------------------------
    print("[export] Loading policy...")
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner     = runner_cls(env, asdict(agent_cfg), device=cfg.device)
    runner.load(str(resume_path), map_location=cfg.device)

    # Export 1: TorchScript (.pt) for Isaac Gym -------------------------
    # Use _TorchPolicyExporter — the canonical JIT exporter that handles
    # actor, normalizer, and optional RNN correctly (torch.jit.script).
    print("[export] Exporting TorchScript (.pt) for Isaac Gym...")

    from mjlab.utils.lab_api.rl.exporter import _TorchPolicyExporter

    policy = runner.alg.policy
    obs_dim = policy.actor[0].in_features
    print(f"[export] Obs dim: {obs_dim}  →  Actions: {policy.actor[-1].out_features}")

    # Check for empirical obs normalizer and verify its stats match obs_dim.
    # actor_obs_normalizer exists as an attribute but may have been initialized
    # with mismatched stats (e.g., from a previous task loaded in the same
    # process). Validate shape before using it.
    _raw_norm = getattr(policy, "actor_obs_normalizer", None)
    if _raw_norm is not None:
        try:
            norm_dim = _raw_norm._mean.shape[0]
            if norm_dim == obs_dim:
                normalizer = _raw_norm
                print(f"[export] Normalizer: actor_obs_normalizer ({norm_dim}-dim) — included")
            else:
                normalizer = None
                print(f"[export] Normalizer: actor_obs_normalizer dim mismatch "
                      f"({norm_dim} vs obs {obs_dim}) — skipping, using identity")
        except AttributeError:
            normalizer = None
            print("[export] Normalizer: not found — using identity")
    else:
        normalizer = None
        print("[export] Normalizer: not found — using identity")

    pt_exporter = _TorchPolicyExporter(policy, normalizer)
    pt_exporter.eval()

    pt_path = pt_dir / "policy.pt"
    pt_exporter.export(str(pt_dir), "policy.pt")
    pt_size = pt_path.stat().st_size / 1024
    print(f"[export] {pt_path}  ({pt_size:.1f} KB)")

    # Verify
    dummy_obs = torch.zeros(1, obs_dim, device=cfg.device)
    verify    = torch.jit.load(str(pt_path), map_location=cfg.device)
    out       = verify(dummy_obs)
    print(f"[export] Verified: input (1, {obs_dim}) → output {tuple(out.shape)}")

    # Export 2: ONNX (use tracking exporter if needed) -----------------
    print("[export] Exporting ONNX...")
    if is_tracking:
        from mjlab.tasks.tracking.rl.exporter import (
            export_motion_policy_as_onnx,
            attach_onnx_metadata as tracking_attach_metadata,
        )
        export_motion_policy_as_onnx(
            env=env.unwrapped,
            actor_critic=runner.alg.policy,
            normalizer=normalizer,
            path=str(onnx_dir),
            filename="policy.onnx",
            verbose=False,
        )
        tracking_attach_metadata(
            env=env.unwrapped,
            run_path=wandb_run_path or str(resume_path),
            path=str(onnx_dir),
            filename="policy.onnx",
        )
    else:
        export_velocity_policy_as_onnx(
            actor_critic=runner.alg.policy,
            normalizer=normalizer,
            path=str(onnx_dir),
            filename="policy.onnx",
            verbose=False,
        )
        attach_onnx_metadata(
            env=env.unwrapped,
            run_path=wandb_run_path or str(resume_path),
            path=str(onnx_dir),
            filename="policy.onnx",
        )
    onnx_path = onnx_dir / "policy.onnx"
    onnx_size = onnx_path.stat().st_size / 1024
    print(f"[export] {onnx_path}  ({onnx_size:.1f} KB)")

    print(f"\n[export] Done:")
    print(f"[export]   {pt_path}   → Isaac Gym  (deploy_isaacgym_g1.py)")
    print(f"[export]   {onnx_path} → Isaac Lab  (future)")

    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    cfg = tyro.cli(
        ExportConfig,
        args=remaining_args,
        default=ExportConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )

    run_export(chosen_task, cfg)


if __name__ == "__main__":
    main()