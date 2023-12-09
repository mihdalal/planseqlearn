import os
import random
import time

env_names = [
    "metaworld_assembly-v2",
    "metaworld_disassemble-v2",
]
num_seeds = 1
for idx, env_name in enumerate(env_names):
    for _ in range(num_seeds):
        seed = random.randint(0, 100000000)
        run_name = f"metaworld_e2e_baseline_final_{env_name}"
        cmd = f"python planseqlearn/train.py agent=drqv2 use_wandb=True seed={seed} debug=True save_video=True num_train_frames=5000000 eval_every_frames=50000 num_eval_episodes=10 wandb.project_name=mprl wandb.run_name={run_name} task={env_name} mprl=False path_length=500 action_repeat=2 experiment_id={run_name} experiment_subdir=metaworld_e2e_iclr_rerun replay_buffer_size=750000 camera_name=corner matrix=True"
        os.system(cmd)
