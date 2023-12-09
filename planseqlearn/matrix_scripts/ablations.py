import random
import os

num_seeds = 3
env_name = "robosuite_PickPlaceCan"
hardcoded_high_level_plans = [True, False]
camera_names = ["robot0_eye_in_hand", "agentview"]
noisy_pose_estimates = True
pose_sigmas = [0.01, 0.025, 0.1, 0.5]

# for pose_sigma in pose_sigmas:
#     for _ in range(num_seeds):
#         seed = random.randint(0, 100000000)
#         run_name = f"drqv2_mprl_ablations_{env_name}"
#         extra_args = f"noisy_pose_estimates=True pose_sigma={pose_sigma}"
#         cmd = f"python planseqlearn/train.py {extra_args} experiment_subdir=mprl_final_paper_ablations_noisy_pose replay_buffer_size=750000 agent=drqv2 use_wandb=True seed={seed} camera_name=robot0_eye_in_hand debug=True save_video=True num_train_frames=500000 eval_every_frames=10000 num_eval_episodes=10 wandb.project_name=mprl_paper_results wandb.run_name={run_name} task={env_name} mprl=True path_length=50 action_repeat=1 experiment_id={run_name} matrix=True"
#         os.system(cmd)
# print(cmd)

# for hardcoded_high_level_plan in hardcoded_high_level_plans:
#     for _ in range(num_seeds):
#         seed = random.randint(0, 100000000)
#         run_name = f"drqv2_mprl_ablations_{env_name}"
#         extra_args = f"hardcoded_high_level_plan={hardcoded_high_level_plan}"
#         cmd = f"python planseqlearn/train.py {extra_args} experiment_subdir=mprl_final_paper_ablations_hardcoded_high_level_plan replay_buffer_size=750000 agent=drqv2 use_wandb=True seed={seed} camera_name=robot0_eye_in_hand debug=True save_video=True num_train_frames=500000 eval_every_frames=10000 num_eval_episodes=10 wandb.project_name=mprl_paper_results wandb.run_name={run_name} task={env_name} mprl=True path_length=50 action_repeat=1 experiment_id={run_name} matrix=True"
#         os.system(cmd)
# print(cmd)

for camera_name in camera_names:
    for _ in range(num_seeds):
        seed = random.randint(0, 100000000)
        run_name = f"drqv2_mprl_ablations_{env_name}"
        cmd = f"python planseqlearn/train.py experiment_subdir=mprl_final_paper_ablations_camera replay_buffer_size=750000 agent=drqv2 use_wandb=True seed={seed} camera_name={camera_name} debug=True save_video=True num_train_frames=500000 eval_every_frames=10000 num_eval_episodes=10 wandb.project_name=mprl_paper_results wandb.run_name={run_name} task={env_name} mprl=True path_length=50 action_repeat=1 experiment_id={run_name} matrix=True"
        os.system(cmd)
        # print(cmd)

for _ in range(num_seeds):
    seed = random.randint(0, 100000000)
    run_name = f"drqv2_mprl_ablations_{env_name}"
    cmd = f"python planseqlearn/train.py experiment_subdir=mprl_final_paper_ablations_camera use_fixed_plus_wrist_view=True replay_buffer_size=750000 agent=drqv2 use_wandb=True seed={seed} camera_name={camera_name} debug=True save_video=True num_train_frames=500000 eval_every_frames=10000 num_eval_episodes=10 wandb.project_name=mprl_paper_results wandb.run_name={run_name} task={env_name} mprl=True path_length=50 action_repeat=1 experiment_id={run_name} matrix=True"
    os.system(cmd)
    # print(cmd)
