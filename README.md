Original mjlabs [Readme](docs/README.md)  
  
Reinforcement learning with isaacgym --> [IsaacGym](https://github.com/Koh-WH/g1_isaacgym)  
Teleoperation with Rebocap --> [ReboCap](https://github.com/Koh-WH/Rebocap_mujoco)  
  
# Table of Contents
- [Setup](#Setup)
- [Train examples](#Train-examples)
- [Keyboard controls](#Keyboard-controls)
- [Environments available](#Available-Environments-in-mjlab)
- [Motiom imitation](#Motiom-imitation)
- [Video To Motion](#Full-pipeline-for-Video_To_Motion)
    
# Folder Structure
```
src/mjlab/
├── actuator/
├── asset_zoo/
│   └── robots/
│       ├── unitree_go1/
│       ├── unitree_g1/
│       └── i2rt_yam/
├── deploy/
├── entity/
├── envs/
│   └── mdp/
│       └── actions/
├── export/
├── managers/
├── pkl/
│   ├── csv/
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
Work in base directory ~/Downloads/mjlab$    

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
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path {Path}
```
Resume and train to 5000 iterations.  
```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 1024 --agent.resume True --agent.load-run "g1_demo_2000" --agent.max-iterations 3000 --agent.save-interval 200
```
```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path {Path}
```
  
# Keyboard controls
Velocity keyboard control.  
  
Keyboard control that reads with every key press.  
```bash
uv run python src/mjlab/scripts/play_keyboard_1.py Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path {Path}
```
Keyboard control that reads key hold. Accerlate and deccelerate.    
```bash
uv add pynput
```
Flat terrain.  
```bash
uv run python src/mjlab/scripts/play_keyboard_2.py Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path {Path}
```
Rough terrain.  
```bash
uv run python src/mjlab/scripts/play_keyboard_2.py Mjlab-Velocity-Rough-Unitree-G1 --wandb-run-path {Path}
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
  
# Motiom imitation
```bash
uv add joblib
```
If pkl file, convert to csv.  
```bash
uv run python src/mjlab/scripts/pkl_to_csv.py --file=src/mjlab/pkl/Roundhouse_kick.pkl
```
Log in to your WandB account; access Registry from teams; under Core on the left. Create a new registry collection with the name " Motions" and artifact type "All Types".  
  
## Action 1 (Roundhouse_kick):
Calculate velocities and convert to npz; uploads to wandb registry.  
```bash
uv run python src/mjlab/scripts/csv_to_npz.py   --input-file src/mjlab/pkl/csv/Roundhouse_kick.csv   --output-name roundhouse_kick   --input-fps 30 --output-fps 50
```
Train imitation. {Name} can be found in details of registry.  
```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name {Name} --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.save-interval 200 --agent.run-name "roundhouse_kick"
```
Play imitation.  
```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path {Path}
```
If need to train more.  
```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name {Name} --env.scene.num-envs 1024 --agent.max-iterations 5000 --agent.save-interval 200 --agent.run-name "roundhouse_kick" --agent.resume True --agent.load-run "roundhouse_kick"
```
  
## Action 2 (dance):
```bash
uv run python src/mjlab/scripts/csv_to_npz.py --input-file src/mjlab/pkl/csv/dance1_subject2.csv --output-name dance1_subject2.npz --input-fps 30 --output-fps 50
```
```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name {Name} --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.save-interval 200 --agent.run-name "dance1"
```
```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path {Path}
```
If need to train more.  
```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name {Name} --env.scene.num-envs 1024 --agent.max-iterations 5000 --agent.save-interval 500 --agent.run-name "dance1.1" --agent.resume True --agent.load-run "dance1"
```
  
# Full pipeline for Video_To_Motion  
## Setup
Have to setup 2 coonda envs first and download the gits data.  
[GVHMR](https://github.com/zju3dv/GVHMR/tree/main) -> Motion Recovery.  
[GMR](https://github.com/YanjieZe/GMR/tree/master) -> Motion remapping.  
  
1) Setup the 2 conda envs:  
```bash
git clone https://github.com/YanjieZe/GMR.git
cd GMR/
conda create -n gmr python=3.10 -y
conda activate gmr
mkdir outputs
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
conda deactivate

git clone https://github.com/zju3dv/GVHMR.git
cd GVHMR/
conda create -n gvhmr python=3.10 -y
conda activate gvhmr
pip install chumpy --no-build-isolation
pip install -r requirements.txt
pip install -e .
mkdir outputs
mkdir -p inputs/checkpoints
```
2) Setup the [SMPLX](https://smpl-x.is.tue.mpg.de/login.php) and [SMPL](https://smpl.is.tue.mpg.de/login.php) data accordingly.  
For GVHMR follow the "Weights" section at the [install.md](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md).  
For GMR follow the "Data Preparation" section at the [readme](https://github.com/YanjieZe/GMR/tree/master).  
  
## Running video_to_motion.py
Cut video to allow better detection of human and learning.   
```bash
ffmpeg -ss {when to start cutting} -i {mp4_path} -t {how long to cut} -c copy {new_mp4_path}
```
Activate one conda env first.  
```bash
conda activate {env}
```
Go to mjlab directory. Add pandas for early stop detection.  
```bash
cd ~/Downloads/mjlab
uv add pandas
```  
```bash
uv run python src/mjlab/scripts/video_to_motion.py \
    --video {mp4_path} \
    --name {name_of_training} \
    --gvhmr-dir {GVHMR_directory} \
    --gmr-dir {GMR_directory} \
    --wandb-entity {wandb_name} \
    --max-iter 50000 \
    --early-stop-patience 200
```
```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path {Path}
```
  
## Breakdown of video_to_motion.py
outputs of mesh -> (../GVHMR/outputs/demo).  
outputs of mapping to g1 -> (../GMR/output).  
convert to npz using mjlabs -> file in wandb registry.  
train and play in mjlabs -> (.../mjlab/logs).  
  
Example of what is done when video_to_motion.py is called:  
```bash
(gvhmr) ~/Downloads/GVHMR$ python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
```
```bash
(gmr) ~/Downloads/GMR$ python scripts/gvhmr_to_robot.py --gvhmr_pred_file /home/koh-wh/Downloads/GVHMR/outputs/demo/tennis/hmr4d_results.pt --robot unitree_g1 --save_path /home/koh-wh/Downloads/GMR/outputs/tennis.pkl
```
```bash
(gmr) ~/Downloads/GMR$ python scripts/pkl_to_csv.py --input /home/koh-wh/Downloads/GMR/outputs/tennis.pkl
```
```bash
(gmr) ~/Downloads/mjlab$ uv run python src/mjlab/scripts/csv_to_npz.py --input-file /home/koh-wh/Downloads/GMR/outputs/tennis.csv --output-name tennis --input-fps 30 --output-fps 50
```
```bash
(gmr) ~/Downloads/mjlab$ uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name {wandb_name}/wandb-registry-Motions/tennis --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.save-interval 500 --agent.run-name "tennis"
```
```bash
(gmr) ~/Downloads/mjlab$ uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path {wandb_name}/mjlab/bxtf8h1t
```