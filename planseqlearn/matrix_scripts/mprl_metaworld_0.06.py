import os
import random
import time

env_names = [
    "metaworld_hammer-v2",
    "metaworld_bin-picking-v2",
]
num_seeds = 3
for _ in range(num_seeds):
    for idx, env_name in enumerate(env_names):
        seed = random.randint(0, 100000000)
        run_name = f"metaworld_mprl_baseline_final_redo_006_{env_name}"
        cmd = f"python planseqlearn/train.py experiment_subdir=metaworld_mprl_iclr_rerun agent=drqv2 use_wandb=True seed={seed} debug=True save_video=True num_train_frames=2000000 eval_every_frames=20000 num_eval_episodes=10 wandb.project_name=mprl wandb.run_name={run_name} task={env_name} mprl=True path_length=200 camera_name=gripperPOVpos action_repeat=2 experiment_id={run_name} matrix=True replay_buffer_size=750000"
        os.system(cmd)
    break
