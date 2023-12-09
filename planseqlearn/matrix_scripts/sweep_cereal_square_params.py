import os
import random
import time

env_names = [
    "robosuite_PickPlaceCereal",
    "robosuite_NutAssemblySquare",
]

discounts = [0.99, 0.96]

hardcoded_orientations = [True, False]

camera_names = [
    "agentview",
    "robot0_eye_in_hand",
]
num_seeds = 3
num_exps = 0
for discount in discounts:
    for hardcoded_orientation in hardcoded_orientations:
        for camera_name in camera_names:
            for env_name in env_names:
                for _ in range(num_seeds):
                    seed = random.randint(0, 100000000)
                    run_name = f"mprl_drqv2_d4rl_latest_{env_name}_sweep_params_v1"
                    params = f"discount={discount} hardcoded_orientations={hardcoded_orientation} camera_name={camera_name}"
                    cmd = f"python planseqlearn/train.py agent=drqv2 use_proprio=False use_wandb=True save_video=True num_train_frames=1000000 wandb.project_name=mprl wandb.run_name={run_name} task={env_name} mprl=True path_length=50 experiment_id={run_name} matrix=True seed={seed} {params}"
                    os.system(cmd)
                    num_exps += 1
                    time.sleep(10)

print(f"Ran {num_exps} experiments")
