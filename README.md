Original mjlabs [Readme](docs/README.md)  
  
Reinforcement learning with isaacgym --> [IsaacGym](https://github.com/Koh-WH/g1_isaacgym)  
Teleoperation with Rebocap --> [ReboCap](https://github.com/Koh-WH/Rebocap_mujoco)  
  
# Table of Contents
- [Setup](#Setup)
- [Train examples](#Train-examples)
- [Keyboard controls](#Keyboard-controls)
- [Environments available](#Available-Environments-in-mjlab)
- [Motion imitation](#Motion-imitation)
- [Video To Motion](#Full-pipeline-for-Video_To_Motion)
- [Rebocap To Motion](#Full-pipeline-for-Rebocap_To_Motion)
    
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
- Install in exisiting env using uv or can use other installation method. --> [Installation link](https://mujocolab.github.io/mjlab/source/installation.html)  
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
  
Create WandB account ---> [Create WandB account](https://wandb.ai/login)  
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
## 🎥 Demo Videos
Velocity keyboard control  
<table>
  <tr>
    <td align="center" width="20%">
      <b>Demo 1</b><br>
      Velocity Tracking Flat<br><br>
      <video src="https://github.com/user-attachments/assets/8cb1a638-f14c-42ed-a85c-df32da32af16" width="200" autoplay loop muted playsinline></video>
    </td>
    <td align="center" width="20%">
      <b>Demo 2</b><br>
      Velocity Tracking Rough<br><br>
      <video src="https://github.com/user-attachments/assets/917cd305-9298-4a3a-a9e6-eb8ed4c3351d" width="200" autoplay loop muted playsinline></video>
    </td>
  </tr>
</table>
  
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
  
# Motion imitation
## 🎥 Demo Videos
<table>
  <tr>
    <td align="center" width="20%">
      <b>Demo 1</b><br>
      Roundhouse Kick<br><br>
      <video src="https://github.com/user-attachments/assets/a453dc0f-2b35-460f-9818-436e90d99cec" width="200" autoplay loop muted playsinline></video>
    </td>
    <td align="center" width="20%">
      <b>Demo 2</b><br>
      Dance Motion<br><br>
      <video src="https://github.com/user-attachments/assets/afcd21eb-b268-483b-8c3f-01660170509d" width="200" autoplay loop muted playsinline></video>
    </td>
  </tr>
</table>
  
```bash
uv add joblib
```
If pkl file, convert to csv.  
```bash
uv run python src/mjlab/scripts/pkl_to_csv.py --file=src/mjlab/pkl/Roundhouse_kick.pkl
```
Log in to your WandB account; access Registry from teams; under Core on the left. Create a new registry collection with the name " Motions" and artifact type "All Types".  
  
## Action 1 (Roundhouse_kick)
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
  
## Action 2 (dance)
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
  
### Download npz to local
Download and rename.  
```bash
uv run wandb artifact get {path_of_npz_file_in_registry}:latest --root ./motion_data
```
Play local checkpoint and local motion file.  
```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1   --checkpoint-file {path_to_policy.pt}   --motion-file {path_to_motion.npz}
```
  
# Full pipeline for Video_To_Motion  
## 🎥 Demo Videos
<table>
  <tr>
    <td align="center" width="20%">
      <b>Demo 1</b><br>
      Shadowboxing<br><br>
      <video src="https://github.com/user-attachments/assets/35ee547e-09c7-46ab-a6fa-5fefe25e9cf2" width="200" autoplay loop muted playsinline></video>
    </td>
    <td align="center" width="20%">
      <b>Demo 2</b><br>
      Tennis Motion<br><br>
      <video src="https://github.com/user-attachments/assets/588415c6-7fef-4762-89c5-792b306881e8" width="200" autoplay loop muted playsinline></video>
    </td>
  </tr>
  <tr>
    <td align="center" width="20%">
      <b>Human mesh example</b><br><br>
      <video src="https://github.com/user-attachments/assets/298691fb-7705-4324-a8c4-6433c1e8fe93" width="200" autoplay loop muted playsinline></video>
    </td>
    <td align="center" width="20%">
      <b>Robot Mapping</b><br><br>
      <video src="https://github.com/user-attachments/assets/528a54a2-315c-48eb-9ea4-ef660e10542c" width="200" autoplay loop muted playsinline></video>
    </td>
  </tr>
</table>
  
## Setup
Have to setup 2 coonda envs first and download the gits data.  
[GVHMR I used](https://github.com/Koh-WH/g1_gvhmr.git), [Original GVHMR](https://github.com/zju3dv/GVHMR/tree/main) -> Motion Recovery.  
[GMR I edited](https://github.com/Koh-WH/g1_gmr.git), [Original GMR](https://github.com/YanjieZe/GMR/tree/master) -> Motion remapping.  
  
1) Setup the 2 conda envs:  
```bash
git clone https://github.com/Koh-WH/g1_gmr.git
cd GMR/
conda create -n gmr python=3.10 -y
conda activate gmr
mkdir outputs
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
conda deactivate

git clone https://github.com/Koh-WH/g1_gvhmr.git
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
```
GVHMR/
└── inputs/
    └── checkpoints/
        ├── body_models/
        │   ├── smpl/
        │   │   ├── SMPL_FEMALE.pkl
        │   │   ├── SMPL_MALE.pkl
        │   │   └── SMPL_NEUTRAL.pkl
        │   └── smplx/
        │       ├── SMPLX_FEMALE.npz
        │       ├── SMPLX_MALE.npz
        │       └── SMPLX_NEUTRAL.npz
        ├── dpvo/
        ├── gvhmr/
        ├── hmr2/
        ├── vitpose/
        └── yolo/
```
  
For GMR follow the "Data Preparation" section at the [readme](https://github.com/YanjieZe/GMR/tree/master).  
```
GMR/
├── assets/
│   ├── body_models/
│   │   └── smplx/
│   │       ├── SMPLX_FEMALE.npz
│   │       ├── SMPLX_MALE.npz
│   │       └── SMPLX_NEUTRAL.npz
```  
  
## Running video_to_motion.py
Cut video to allow better detection of human and learning.   
```bash
ffmpeg -ss {when to start cutting} -i {mp4_path} -t {how long to cut} -c copy {new_mp4_path}
```
Activate one conda env first.  
```bash
conda activate gvhmr
```
Go to mjlab directory. Add pandas for early stop detection.  
```bash
cd ~/Downloads/mjlab
uv add pandas
```  
```bash
uv run python src/mjlab/scripts/video_to_motion.py \
    --video docs/videos_images/shadowboxing.mp4 \
    --name shadowboxing \
    --gvhmr-dir ../GVHMR \
    --gmr-dir ../GMR \
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
Already in mjlabs directory with one conda env activated.  
```bash
conda activate gvhmr
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
```
```bash
conda activate gmr
cd ~/Downloads/GMR
python scripts/gvhmr_to_robot.py --gvhmr_pred_file /home/koh-wh/Downloads/GVHMR/outputs/demo/tennis/hmr4d_results.pt --robot unitree_g1 --save_path /home/koh-wh/Downloads/GMR/output/tennis.pkl
```
```bash
python scripts/pkl_to_csv.py --input /home/koh-wh/Downloads/GMR/output/tennis.pkl
```
```bash
cd ~/Downloads/mjlab
uv run python src/mjlab/scripts/csv_to_npz.py --input-file /home/koh-wh/Downloads/GMR/output/tennis.csv --output-name tennis --input-fps 30 --output-fps 50
```
```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name {wandb_name}/wandb-registry-Motions/tennis --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.save-interval 500 --agent.run-name "tennis"
```
  
# Full pipeline for Rebocap_To_Motion 
## 🎥 Demo Videos 
<table>
  <tr>
    <td align="center" width="33%">
      <b>1. Input BVH (Squats)</b><br>
      <i>Rebocap MoCap Data</i><br><br>
      <div style="border: 2px dashed #d0d7de; border-radius: 6px; padding: 40px 0; width: 70%; margin: 0 auto; background-color: #f6f8fa; font-family: monospace; font-size: 14px; color: #57606a;">
        📄 <a href="https://github.com/Koh-WH/g1_gmr/blob/main/bvh/squats.bvh" style="text-decoration: none; color: #0969da;">squats.bvh</a>
        <br><br>
        🎥 <a href="https://github.com/Koh-WH/g1_mjlabs/blob/main/docs/videos_images/squats_.MP4" style="text-decoration: none; color: #0969da;">Input Video</a>
      </div>
    </td>
    <td align="center" width="20%">
      <b>2. Mapped Motion</b><br>
      <i>GMR Retargeting</i><br><br>
      <video src="https://github.com/user-attachments/assets/da537d8e-b210-474b-b06f-8e4816e9cc77" width="200%" autoplay loop muted playsinline></video>
    </td>
    <td align="center" width="20%">
      <b>3. Trained Output</b><br>
      <i>MJLab Tracking Policy</i><br><br>
      <video src="https://github.com/user-attachments/assets/5a788dd7-b444-4c64-ad27-3a7b2ee3f852" width="200%" autoplay loop muted playsinline></video>
    </td>
  </tr>

  <tr>
    <td align="center" width="33%">
      <b>1. Input BVH (Starjumps)</b><br>
      <i>Rebocap MoCap Data</i><br><br>
      <div style="border: 2px dashed #d0d7de; border-radius: 6px; padding: 40px 0; width: 70%; margin: 0 auto; background-color: #f6f8fa; font-family: monospace; font-size: 14px; color: #57606a;">
        📄 <a href="https://github.com/Koh-WH/g1_gmr/blob/main/bvh/starjumps.bvh" style="text-decoration: none; color: #0969da;">starjumps.bvh</a>
        <br><br>
        🎥 <a href="https://github.com/Koh-WH/g1_mjlabs/blob/main/docs/videos_images/starjumps_.MP4" style="text-decoration: none; color: #0969da;">Input Video</a>
      </div>
    </td>
    <td align="center" width="20%">
      <b>2. Mapped Motion</b><br>
      <i>GMR Retargeting</i><br><br>
      <video src="https://github.com/user-attachments/assets/b16f2ace-10ce-4bb3-83f7-926845111746" width="200%" autoplay loop muted playsinline></video>
    </td>
    <td align="center" width="20%">
      <b>3. Trained Output</b><br>
      <i>MJLab Tracking Policy</i><br><br>
      <video src="https://github.com/user-attachments/assets/f59b38e6-78a9-4772-a037-5bd8e5d034cc" width="200%" autoplay loop muted playsinline></video>
    </td>
  </tr>
</table>
  
## Setup
Need to adjust the axis in GMR using `NeutralPose.bvh` to match [G1 sdk.](https://support.unitree.com/home/en/G1_developer)  
This is done [here.](https://github.com/Koh-WH/g1_gmr/blob/main/docs/readme.md)  
<table>
  <tr>
    <td align="center" width="20%">
      <b>Neutral Pose</b><br>
      <video src="https://github.com/user-attachments/assets/b9ca4453-347f-407b-a08e-e7b1765a5413" width="200" autoplay loop muted playsinline></video>
    </td>
  </tr>
</table>
  
## Running rebocap_to_motion.py
```bash
conda activate gmr
cd ~/Downloads/mjlab
uv run python src/mjlab/scripts/rebocap_to_motion.py \
  --bvh ../GMR/bvh/squats.bvh \
  --name squats_rebocap \
  --gmr-dir ../GMR \
  --wandb-entity {wandb_name} \
  --num-envs 1024 \
  --max-iter 5000 \
  --save-interval 500
```
  
## Breakdown of rebocap_to_motion.py
outputs of mapping to g1 -> (../GMR/output).  
convert to npz using mjlabs -> file in wandb registry.  
train and play in mjlabs -> (.../mjlab/logs).  
  
Example of what is done when rebocap_to_motion.py is called:  
Already in mjlabs directory with one conda env activated.  
```bash
cd ~/Downloads/GMR
python scripts/convert_mixamo_to_lafan1.py /home/koh-wh/Downloads/GMR/bvh/NeutralPose.bvh /home/koh-wh/Downloads/GMR/bvh/converted/NeutralPose_converted.bvh 
python scripts/convert_mixamo_to_lafan1.py /home/koh-wh/Downloads/GMR/bvh/starjumps.bvh /home/koh-wh/Downloads/GMR/bvh/converted/starjumps_converted.bvh 
```
```bash
python scripts/bvh_to_robot.py     --bvh_file /home/koh-wh/Downloads/GMR/bvh/converted/NeutralPose_converted.bvh     --robot unitree_g1     --format mixamo     --rate_limit     --save_path output/neutralpose_rebocap.pkl

python scripts/bvh_to_robot.py     --bvh_file /home/koh-wh/Downloads/GMR/bvh/converted/starjumps_converted.bvh     --robot unitree_g1     --format mixamo     --rate_limit     --save_path output/starjumps_rebocap.pkl
```
```bash
python scripts/pkl_to_csv.py --input /home/koh-wh/Downloads/GMR/output/starjumps_rebocap.pkl
```
```bash
cd ..
cd mjlabs/
uv run python src/mjlab/scripts/csv_to_npz.py --input-file /home/koh-wh/Downloads/GMR/output/starjumps_rebocap.csv --output-name starjumps_rebocap --input-fps 30 --output-fps 50

uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name kohwh-nanyang-technological-university-singapore-org/wandb-registry-Motions/starjumps_rebocap --env.scene.num-envs 1024 --agent.max-iterations 5000 --agent.save-interval 500 --agent.run-name "starjumps_rebocap"
```
  