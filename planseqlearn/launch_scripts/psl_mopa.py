import os
import random
import time

env_names = [
    "mopa_SawyerLiftObstacle-v0",
    "mopa_SawyerAssemblyObstacle-v0",
    "mopa_SawyerPushObstacle-v0",
]
abbrev_env_names = ["assembly", "push"]
num_seeds = 3
for idx, env_name in enumerate(env_names):
    for _ in range(num_seeds):
        seed = random.randint(0, 100000000)
        run_name = f"mopa_psl_final_{env_name}_wrist"
        cmd = f"python planseqlearn/train.py experiment_subdir=mopa_psl_iclr_rerun agent=drqv2 use_wandb=True seed={seed} debug=True save_video=True num_train_frames=2000000 eval_every_frames=20000 num_eval_episodes=10 wandb.project_name=psl wandb.run_name={run_name} task={env_name} psl=True path_length=200 action_repeat=2 experiment_id={run_name}  replay_buffer_size=750000"
        os.system(cmd)
        break
