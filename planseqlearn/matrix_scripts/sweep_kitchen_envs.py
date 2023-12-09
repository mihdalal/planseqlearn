import os
import time

env_names = [
    "kitchen_kitchen-microwave-v0",
    "kitchen_kitchen-hinge-v0",
    "kitchen_kitchen-kettle-v0",
    "kitchen_kitchen-light-v0",
    "kitchen_kitchen-tlb-v0",
]

for env_name in env_names:
    run_name = f"mprl_drqv2_d4rl_latest_{env_name}"
    cmd = f"python planseqlearn/train.py agent=drqv2 use_wandb=True save_video=True num_train_frames=10000000 wandb.project_name=mprl wandb.run_name={run_name} task={env_name} mprl=True path_length=25 experiment_id={run_name} matrix=True"
    os.system(cmd)
    time.sleep(10)
