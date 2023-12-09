import os
import random
import time

env_names = [
    "kitchen_kitchen-microwave-v0",
    "kitchen_kitchen-kettle-v0",
    "kitchen_kitchen-light-v0",
    "kitchen_kitchen-tlb-v0",
    "kitchen_kitchen-slider-v0",
    "kitchen_kitchen-kettle-burner-v0",
    "kitchen_kitchen-kettle-light-burner-v0"
    "kitchen_kitchen-kettle-light-burner-slider-v0",
    "kitchen_kitchen-microwave-kettle-light-burner-slider-v0",
    # 'kitchen_kitchen-hinge-microwave-kettle-light-burner-slider-v0',
]
num_seeds = 3
for idx, env_name in enumerate(env_names):
    for _ in range(num_seeds):
        seed = random.randint(0, 100000000)
        run_name = f"drqv2_e2e_baseline_final_{env_name}"
        cmd = f"python planseqlearn/train.py experiment_subdir=e2e_baseline_final_iclr_v3 agent=drqv2 use_wandb=True seed={seed} debug=True save_video=True num_train_frames=2800000 eval_every_frames=50000 replay_buffer_size=750000 num_eval_episodes=10 wandb.project_name=mprl_paper_results wandb.run_name={run_name} task={env_name} mprl=False path_length=280 action_repeat=2 experiment_id={run_name} matrix=True"
        os.system(cmd)
        time.sleep(5)
