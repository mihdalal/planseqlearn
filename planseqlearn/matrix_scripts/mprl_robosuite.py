import os
import random
import time

env_names = [
    "robosuite_Lift",
    "robosuite_Door",
    "robosuite_PickPlaceBread",
    "robosuite_PickPlaceCan",
    "robosuite_PickPlaceCereal",
    "robosuite_PickPlaceMilk",
    "robosuite_NutAssemblyRound",
    "robosuite_NutAssemblySquare",
    "robosuite_NutAssembly",
    "robosuite_PickPlaceCerealMilk",
    "robosuite_PickPlaceCanBread",
]
num_seeds = 3
for idx, env_name in enumerate(env_names):
    valid_obj_names = ""
    if env_name == "robosuite_PickPlaceCerealMilk":
        valid_obj_names = '\\\'[\\\"Cereal\\\", \\\"Milk\\\"]\\\''
    if env_name == "robosuite_PickPlaceCanBread":
        valid_obj_names = '\\\'[\\\"Can\\\", \\\"Bread\\\"]\\\''
    if env_name in ["robosuite_Lift", "robosuite_Door"]:
        path_length = 25
    elif env_name in [
        "robosuite_PickPlaceBread",
        "robosuite_PickPlaceCan",
        "robosuite_PickPlaceCereal",
        "robosuite_PickPlaceMilk",
        "robosuite_NutAssemblyRound",
        "robosuite_NutAssemblySquare",
    ]:
        path_length = 50
    else:
        path_length = 100
    num_train_frames = 10000 * path_length
    for _ in range(num_seeds):
        seed = random.randint(0, 100000000)
        run_name = f"drqv2_mprl_replicate_v3_{env_name}"
        cmd = f"python planseqlearn/train.py agent=drqv2 use_wandb=True seed={seed} experiment_subdir=mprl_replicate_v3 debug=True save_video=True num_train_frames={num_train_frames} camera_name='robot0_eye_in_hand' eval_every_frames=10000 num_eval_episodes=10 wandb.project_name=mprl_replicate_v3 wandb.run_name={run_name} task={env_name} valid_obj_names={valid_obj_names} mprl=True path_length={path_length} action_repeat=1 experiment_id={run_name} matrix=True"
        os.system(cmd)
        print(cmd)
        time.sleep(5)
