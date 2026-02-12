Original mjlabs [Readme](docs/README.md)  
  
Reinforcement learning with isaacgym --> [IsaacGym](https://github.com/Koh-WH/g1_isaacgym)  
Teleoperation with Rebocap --> [ReboCap](https://github.com/Koh-WH/Rebocap_mujoco)  
  
# Table of Contents
- [Setup](#Setup)
- [Train examples](#Train-examples)
- [Keyboard controls](#Keyboard-controls)
- [Environments available](#Available-Environments-in-mjlab)
  
# Folder Structure
```
mjlab/
├── actuator/
├── asset_zoo/
│   └── robots/
│       ├── unitree_go1/
│       ├── unitree_g1/
│       └── i2rt_yam/
├── entity/
├── envs/
│   └── mdp/
│       └── actions/
├── managers/
├── rl/
├── scene/
├── scripts/
├── sensor/
├── sim/
├── tasks/
│   ├── manipulation/
│   │   ├── mdp/
│   │   └── config/
│   │       └── yam/
│   ├── tracking/
│   │   ├── mdp/
│   │   ├── config/
│   │   │   └── g1/
│   │   ├── rl/
│   │   └── scripts/
│   └── velocity/
│       ├── mdp/
│       ├── config/
│       │   ├── g1/
│       │   └── go1/
│       └── rl/
├── terrains/
├── utils/
│   ├── lab_api/
│   │   ├── tasks/
│   │   └── rl/
│   ├── wrappers/
│   ├── noise/
│   └── buffers/
└── viewer/
    ├── viser/
    └── native/
```
  
# Setup
- Install in exisiting env using uv or can use other installation method. --> [Installation link](#https://mujocolab.github.io/mjlab/source/installation.html)  
Local installation.  
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```bash
git clone https://github.com/mujocolab/mjlab.git
```
cd to within mjlabs  
```
bash uv add --editable /path/to/cloned/mjlab
```
Test Demo.  
```bash
uv run demo
```
  
Create WandB account ---> [Create WandB account](#https://wandb.ai/login)  
  
# Train examples
Following the original readme.  
```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```
Run path take from Wandb.   
```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```
  
## Adding more arguments --> [Full Argument list](docs/source/train_arg.rst)
Example:  
2000 iterations can only stand, robot does not know movement.  
```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 1024 --agent.max-iterations 2000 --agent.run-name "g1_demo" --agent.save-interval 200
```
```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path kohwh-nanyang-technological-university-singapore/mjlab/99bej7pb
```
Resume and train to 5000 iterations.  
```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 1024 --agent.resume True --agent.load-run "g1_demo_2000" --agent.max-iterations 3000 --agent.save-interval 200
```
```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path kohwh-nanyang-technological-university-singapore/mjlab/ft0swacc
```
  
# Keyboard controls
Velocity keyboard control.  
  
Keyboard control that reads with every key press.  
```bash
uv run python src/mjlab/scripts/play_keyboard_1.py Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path kohwh-nanyang-technological-university-singapore/mjlab/ft0swacc
```
Keyboard control that reads key hold.  
```bash
uv add pynput
```
```bash
uv run python src/mjlab/scripts/play_keyboard_2.py Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path kohwh-nanyang-technological-university-singapore/mjlab/ft0swacc
```
```bash
uv run python src/mjlab/scripts/play_keyboard_2.py Mjlab-Velocity-Rough-Unitree-G1 --wandb-run-path kohwh-nanyang-technological-university-singapore/mjlab/ft0swacc
```
  
# Available Environments in mjlab
```bash
uv run python src/mjlab/scripts/list_envs.py
```
```
| # | Task ID                                            |  
|---|----------------------------------------------------|  
| 1 | Mjlab-Lift-Cube-Yam                                |  
| 2 | Mjlab-Tracking-Flat-Unitree-G1                     |  
| 3 | Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation |  
| 4 | Mjlab-Velocity-Flat-Unitree-G1                     |  
| 5 | Mjlab-Velocity-Flat-Unitree-Go1                    |  
| 6 | Mjlab-Velocity-Rough-Unitree-G1                    |  
| 7 | Mjlab-Velocity-Rough-Unitree-Go1                   |  
```
  
