import os
import random
import time

env_names = [
    "kitchen_kitchen-microwave-v0",
    "kitchen_kitchen-kettle-v0",
    "kitchen_kitchen-light-v0",
    "kitchen_kitchen-tlb-v0",
    "kitchen_kitchen-trb-v0",
    "kitchen_kitchen-blb-v0",
    "kitchen_kitchen-brb-v0",
    "kitchen_kitchen-slider-v0",
    "kitchen_kitchen-hinge-v0",
    "kitchen_kitchen-left-hinge-v0",
    "kitchen_kitchen-kettle-burner-v0",
    "kitchen_kitchen-kettle-light-burner-v0"
    "kitchen_kitchen-kettle-light-burner-slider-v0",
    "kitchen_kitchen-microwave-kettle-light-burner-slider-v0",
    "kitchen_kitchen-hinge-microwave-kettle-light-burner-slider-v0",
]
num_seeds = 1
for env_name in env_names:
    for _ in range(num_seeds):
        if env_name in [
            "kitchen_kitchen-microwave-v0",
            "kitchen_kitchen-kettle-v0",
            "kitchen_kitchen-light-v0",
            "kitchen_kitchen-tlb-v0",
            "kitchen_kitchen-trb-v0",
            "kitchen_kitchen-blb-v0",
            "kitchen_kitchen-brb-v0",
            "kitchen_kitchen-slider-v0",
            "kitchen_kitchen-hinge-v0",
            "kitchen_kitchen-left-hinge-v0",
        ]:
            path_length = 25
        elif env_name in ["kitchen_kitchen-kettle-burner-v0"]:
            path_length = 50
        elif env_name in ["kitchen_kitchen-kettle-light-burner-v0"]:
            path_length = 75
        elif env_name in ["kitchen_kitchen-kettle-light-burner-slider-v0"]:
            path_length = 100
        elif env_name in ["kitchen_kitchen-microwave-kettle-light-burner-slider-v0"]:
            path_length = 125
        elif env_name in [
            "kitchen_kitchen-hinge-microwave-kettle-light-burner-slider-v0"
        ]:
            path_length = 150
        num_train_frames = 10000 * path_length
        seed = random.randint(0, 100000)
        run_name = f"drqv2_psl_kitchen_eef_{env_name}"
        cmd = f"python planseqlearn/train.py experiment_subdir=psl_kitchen_eef num_train_frames={num_train_frames} eval_every_frames=10000 num_eval_episodes=10 camera_name='wrist' debug=True replay_buffer_size=750000 agent=drqv2 use_wandb=True save_video=True seed={seed} wandb.project_name=psl wandb.run_name={run_name} action_repeat=1 task={env_name} psl=True path_length={path_length} experiment_id={run_name}"
        os.system(cmd)
        time.sleep(5)
