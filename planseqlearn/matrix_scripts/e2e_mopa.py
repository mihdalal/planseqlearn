import os
import random
import time

env_names = [
    "mopa_SawyerLiftObstacle-v0",
    "mopa_SawyerAssemblyObstacle-v0",
]
num_seeds = 3
for idx, env_name in enumerate(env_names):
    for _ in range(num_seeds):
        seed = random.randint(0, 100000000)
        run_name = f"mopa_e2e_baseline_final_{env_name}"
        cmd = f"python planseqlearn/train.py experiment_subdir=e2e_baseline_final agent=drqv2 use_wandb=True seed={seed} debug=True save_video=True num_train_frames=5000000 eval_every_frames=50000 num_eval_episodes=10 wandb.project_name=mprl wandb.run_name={run_name} task={env_name} mprl=False path_length=500 action_repeat=2 experiment_id={run_name} matrix=True"
        os.system(cmd)
        break
