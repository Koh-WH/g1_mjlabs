"""
play_keyboard.py — keyboard-controlled play script for MJLabs G1 velocity task.

Engine-style control: holding a key ramps velocity up to max, releasing ramps
it back down to zero — like a throttle pedal, not a toggle switch.

Controls
--------
    UP    / DOWN  : forward / backward  (vx)
    LEFT  / RIGHT : yaw left / right    (omega)
    Q     / E     : lateral left/right  (vy)
    SPACE         : emergency stop (instant zero)
    ENTER         : reset environment
    -     / =     : slow down / speed up simulation

Usage
-----
    python play_keyboard.py Mjlab-Velocity-Flat-Unitree-G1 \
        --wandb-run-path your-org/mjlab/run-id

    python play_keyboard.py Mjlab-Velocity-Flat-Unitree-G1 \
        --checkpoint-file logs/rsl_rl/g1_velocity/2026-xx-xx/model_xx.pt
"""

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import tyro
from pynput.keyboard import Key, KeyCode, Listener
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer


# ---------------------------------------------------------------------------
# Tuning — adjust to taste
# ---------------------------------------------------------------------------

VX_MAX   = 1.0    # m/s    max forward / backward speed
VY_MAX   = 0.5    # m/s    max lateral speed
ANG_MAX  = 1.0    # rad/s  max yaw rate

# How fast velocity ramps up when key held, and ramps down when released.
# Units: fraction of max_speed per second.
# e.g. ACCEL=1.2 means 0 → VX_MAX in ~0.83 s
ACCEL    = 1.2    # ramp-up rate   (hold key)
DECEL    = 2.0    # ramp-down rate (release key)  — brakes faster than throttle


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KBPlayConfig:
    wandb_run_path:  Optional[str] = None
    checkpoint_file: Optional[str] = None
    num_envs:        int            = 1
    device:          Optional[str]  = None


# ---------------------------------------------------------------------------
# Velocity command injector  (engine model)
# ---------------------------------------------------------------------------

class VelocityCommandInjector:

    def __init__(self, env: ManagerBasedRlEnv):
        # Current actual velocities
        self._vx:    float = 0.0
        self._vy:    float = 0.0
        self._omega: float = 0.0

        self._lock   = threading.Lock()
        self._held: set = set()
        self._last_t: float = time.perf_counter()

        # Locate the command tensor ----------------------------------------
        try:
            self._term = env.command_manager._terms["twist"]
        except (AttributeError, KeyError):
            self._term = None
            for name, term in vars(env.command_manager).get("_terms", {}).items():
                if hasattr(term, "command"):
                    self._term = term
                    print(f"[keyboard] Using command term: '{name}'")
                    break
        if self._term is None:
            raise RuntimeError(
                "Could not find a 'twist' command term. "
                "Is this a velocity-tracking task?"
            )

        # Start pynput listener --------------------------------------------
        self._listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

        print("\n[keyboard] Engine-mode controls active:")
        print("  Hold  UP / DOWN      -> accelerate forward / backward")
        print("  Hold  LEFT / RIGHT   -> accelerate yaw left / right")
        print("  Hold  Q / E          -> accelerate lateral left / right")
        print("  SPACE                -> emergency stop (instant)")
        print("  ENTER (in viewer)    -> reset environment")
        print()

    # ------------------------------------------------------------------
    # pynput callbacks  (background thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _token(key):
        if isinstance(key, KeyCode) and key.char is not None:
            return key.char.lower()
        return key

    def _on_press(self, key):
        with self._lock:
            self._held.add(self._token(key))

    def _on_release(self, key):
        with self._lock:
            self._held.discard(self._token(key))

    # ------------------------------------------------------------------
    # apply() — called every sim step from main loop thread
    # ------------------------------------------------------------------

    def apply(self) -> None:
        now = time.perf_counter()
        with self._lock:
            dt   = min(now - self._last_t, 0.05)   # cap to avoid big jumps
            self._last_t = now
            held = set(self._held)                  # snapshot

        # Emergency stop
        if Key.space in held:
            self._vx = self._vy = self._omega = 0.0
        else:
            # Desired velocity from currently held keys (opposing keys cancel)
            fwd  = Key.up    in held
            back = Key.down  in held
            left = Key.left  in held
            rgt  = Key.right in held
            q_k  = 'q' in held
            e_k  = 'e' in held

            want_vx    = VX_MAX  if (fwd  and not back)  else (-VX_MAX  if (back and not fwd)   else 0.0)
            want_vy    = VY_MAX  if (q_k  and not e_k)   else (-VY_MAX  if (e_k  and not q_k)   else 0.0)
            want_omega = ANG_MAX if (left and not rgt)    else (-ANG_MAX if (rgt  and not left)  else 0.0)

            self._vx    = self._ramp(self._vx,    want_vx,    VX_MAX,  dt)
            self._vy    = self._ramp(self._vy,    want_vy,    VY_MAX,  dt)
            self._omega = self._ramp(self._omega, want_omega, ANG_MAX, dt)

        # Write into command tensor — shape (num_envs, 3)
        cmd = self._term.command
        cmd[:, 0] = self._vx
        cmd[:, 1] = self._vy
        cmd[:, 2] = self._omega

        print(
            f"\r[cmd] vx={self._vx:+.2f}  vy={self._vy:+.2f}  omega={self._omega:+.2f}   ",
            end="", flush=True,
        )

    @staticmethod
    def _ramp(current: float, target: float, limit: float, dt: float) -> float:
        """Ramp current toward target using ACCEL/DECEL rates."""
        if target == 0.0:
            # Decelerate to zero
            step = DECEL * limit * dt
            if current > 0:
                return max(0.0, current - step)
            elif current < 0:
                return min(0.0, current + step)
            return 0.0
        else:
            # Accelerate toward target
            step = ACCEL * limit * dt
            if current < target:
                return min(target, current + step)
            else:
                return max(target, current - step)

    def stop(self):
        self._listener.stop()


# ---------------------------------------------------------------------------
# Patched viewer
# ---------------------------------------------------------------------------

class KeyboardNativeMujocoViewer(NativeMujocoViewer):
    def __init__(self, env, policy, injector: VelocityCommandInjector, **kwargs):
        self._injector = injector
        super().__init__(env, policy, **kwargs)

    def step_simulation(self) -> None:
        self._injector.apply()
        super().step_simulation()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_torch_backends()

    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    cfg = tyro.cli(KBPlayConfig, args=remaining_args)

    device    = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg   = load_env_cfg(chosen_task, play=True)
    agent_cfg = load_rl_cfg(chosen_task)
    env_cfg.scene.num_envs = cfg.num_envs

    # Resolve checkpoint
    if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"[INFO] Loading checkpoint: {resume_path.name}")
    elif cfg.wandb_run_path is not None:
        log_root = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        resume_path, was_cached = get_wandb_checkpoint_path(
            log_root, Path(cfg.wandb_run_path)
        )
        print(f"[INFO] Checkpoint: {resume_path.name} ({'cached' if was_cached else 'downloaded'})")
    else:
        raise ValueError("Provide --wandb-run-path or --checkpoint-file")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(chosen_task) or OnPolicyRunner
    runner     = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    policy     = runner.get_inference_policy(device=device)

    injector = VelocityCommandInjector(env.unwrapped)

    try:
        viewer = KeyboardNativeMujocoViewer(env, policy, injector)
        viewer.run()
    finally:
        injector.stop()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[keyboard] Interrupted.")