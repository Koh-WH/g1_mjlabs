"""
play_keyboard.py — keyboard-controlled play script for MJLabs G1 velocity task.

Wraps the existing play.py infrastructure and injects arrow-key velocity commands
directly into the environment's CommandManager twist term.

Controls
--------
    UP    / DOWN  : forward / backward  (vx)
    LEFT  / RIGHT : yaw left / right    (omega)
    Q     / E     : lateral left/right  (vy)
    SPACE         : stop (zero all velocities)
    ENTER         : reset environment
    ESC           : quit

Usage
-----
    # With a W&B checkpoint:
    python play_keyboard.py Mjlab-Velocity-Flat-Unitree-G1 \
        --wandb-run-path your-org/mjlab/run-id

    # With a local checkpoint:
    python play_keyboard.py Mjlab-Velocity-Flat-Unitree-G1 \
        --checkpoint-file logs/rsl_rl/g1_velocity/2026-xx-xx/model_xx.pt
"""

import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner


@dataclass(frozen=True)
class KBPlayConfig:
    wandb_run_path: Optional[str] = None
    checkpoint_file: Optional[str] = None
    num_envs: int = 1
    device: Optional[str] = None

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import (
    KEY_DOWN,
    KEY_E,
    KEY_ENTER,
    KEY_EQUAL,
    KEY_LEFT,
    KEY_MINUS,
    KEY_P,
    KEY_Q,
    KEY_R,
    KEY_RIGHT,
    KEY_SPACE,
    KEY_UP,
)


# ---------------------------------------------------------------------------
# Velocity settings — tune these to taste
# ---------------------------------------------------------------------------

VX_STEP   = 0.2   # m/s per arrow-key press  (forward / backward)
VY_STEP   = 0.2   # m/s per Q/E press        (lateral)
ANG_STEP  = 0.3   # rad/s per arrow-key press (yaw)
VX_MAX    = 1.0
VY_MAX    = 0.5
ANG_MAX   = 1.0


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


# ---------------------------------------------------------------------------
# Velocity command injector
# ---------------------------------------------------------------------------

class VelocityCommandInjector:

    def __init__(self, env: ManagerBasedRlEnv):
        self.env = env
        self._lock = threading.Lock()
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._omega: float = 0.0

        # Locate the twist command term once up front.
        # MJLab / IsaacLab style: env.command_manager._terms["twist"]
        try:
            self._term = env.command_manager._terms["twist"]
        except (AttributeError, KeyError):
            # Fall back — iterate terms to find one with a .command tensor
            self._term = None
            for name, term in vars(env.command_manager).get("_terms", {}).items():
                if hasattr(term, "command"):
                    self._term = term
                    print(f"[keyboard] Using command term: '{name}'")
                    break

        if self._term is None:
            raise RuntimeError(
                "Could not find a 'twist' command term in the environment's "
                "CommandManager.  Check that the task uses UniformVelocityCommand."
            )
        
        print("[keyboard] VelocityCommandInjector ready.")
        print("[keyboard] Controls: UP/DOWN=fwd/back  LEFT/RIGHT=yaw  Q/E=lateral  SPACE=stop")

    # ------------------------------------------------------------------ #
    #  Called from the MuJoCo *viewer thread* via key_callback            #
    # ------------------------------------------------------------------ #

    def on_key(self, key: int) -> None:
        with self._lock:
            if key == KEY_UP:
                self._vx = _clamp(self._vx + VX_STEP, VX_MAX)
            elif key == KEY_DOWN:
                self._vx = _clamp(self._vx - VX_STEP, VX_MAX)
            elif key == KEY_LEFT:
                self._omega = _clamp(self._omega + ANG_STEP, ANG_MAX)
            elif key == KEY_RIGHT:
                self._omega = _clamp(self._omega - ANG_STEP, ANG_MAX)
            elif key == KEY_Q:
                self._vy = _clamp(self._vy + VY_STEP, VY_MAX)
            elif key == KEY_E:
                self._vy = _clamp(self._vy - VY_STEP, VY_MAX)
            elif key == KEY_SPACE:
                self._vx = self._vy = self._omega = 0.0

            vx, vy, omega = self._vx, self._vy, self._omega

        # Print current target so you can see what's happening
        print(f"\r[keyboard] vx={vx:+.2f}  vy={vy:+.2f}  omega={omega:+.2f}   ", end="", flush=True)

    # ------------------------------------------------------------------ #
    #  Called from the *main loop thread* just before env.step()         #
    # ------------------------------------------------------------------ #

    def apply(self) -> None:
        """Write current velocity target into the command tensor."""
        with self._lock:
            vx, vy, omega = self._vx, self._vy, self._omega

        # command shape: (num_envs, 3)  — [vx, vy, omega]
        cmd = self._term.command  # torch.Tensor on the env device
        cmd[:, 0] = vx
        cmd[:, 1] = vy
        cmd[:, 2] = omega


# ---------------------------------------------------------------------------
# Patched viewer that calls injector.apply() on every sim step
# ---------------------------------------------------------------------------

class KeyboardNativeMujocoViewer(NativeMujocoViewer):
    """Thin subclass that applies the keyboard velocity every step."""

    def __init__(self, env, policy, injector: VelocityCommandInjector, **kwargs):
        self._injector = injector
        super().__init__(env, policy, key_callback=injector.on_key, **kwargs)

    def step_simulation(self) -> None:
        # Inject velocity *before* the policy reads observations
        self._injector.apply()
        super().step_simulation()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_torch_backends()

    import mjlab.tasks  # noqa: F401  — populates the task registry

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    cfg = tyro.cli(KBPlayConfig, args=remaining_args)

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(chosen_task, play=True)
    agent_cfg = load_rl_cfg(chosen_task)

    # Force single env for keyboard play — easier to control
    env_cfg.scene.num_envs = cfg.num_envs



    # Resolve checkpoint
    if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"[INFO] Loading checkpoint: {resume_path.name}")
    elif cfg.wandb_run_path is not None:
        log_root = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        resume_path, was_cached = get_wandb_checkpoint_path(log_root, Path(cfg.wandb_run_path))
        print(f"[INFO] Checkpoint: {resume_path.name} ({'cached' if was_cached else 'downloaded'})")
    else:
        raise ValueError("Provide either --wandb-run-path or --checkpoint-file")

    # Build env
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy
    runner_cls = load_runner_cls(chosen_task) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Create the velocity injector
    injector = VelocityCommandInjector(env.unwrapped)

    # Launch viewer with keyboard support
    viewer = KeyboardNativeMujocoViewer(env, policy, injector)
    viewer.run()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[keyboard] Interrupted.")