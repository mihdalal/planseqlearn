import numpy as np
import robosuite as suite
import torch
from matplotlib import pyplot as plt
from robosuite.utils.transform_utils import *
from robosuite.wrappers.gym_wrapper import GymWrapper
from rlkit.torch.model_based.dreamer.visualization import make_video
from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu
#from rlkit.mprl.mp_env import MPEnv
from planseqlearn.mnm.robosuite_mp_env import RobosuiteMPEnv
import cv2

def save_img(env, camera_name, filename, flip=False):
    # frame = env.sim.render(camera_name=camera_name, width=500, height=500)
    frame = env.sim.render(width=500, camera_name=camera_name, height=500)
    if flip:
        frame = np.flipud(frame)
    plt.imshow(frame)
    plt.savefig(filename)


if __name__ == "__main__":
    mp_env_kwargs = dict(
        vertical_displacement=0.04,
        teleport_instead_of_mp=True,
        use_joint_space_mp=False,
        randomize_init_target_pos=False,
        mp_bounds_low=(-1.45, -1.25, 0.45),
        mp_bounds_high=(0.45, 0.85, 2.25),
        backtrack_movement_fraction=0.001,
        clamp_actions=True,
        update_with_true_state=True,
        grip_ctrl_scale=0.0025,
        planning_time=20,
        hardcoded_high_level_plan=True,
        terminate_on_success=False,
        plan_to_learned_goals=False,
        reset_at_grasped_state=False,
        verify_stable_grasp=True,
        hardcoded_orientations=True,
        use_pcd_collision_check=False,
        use_vision_pose_estimation=False,
        use_vision_placement_check=False,
        use_vision_grasp_check=False,
        burn_in=True,
        use_llm_plan=True, 
        text_plan=[("silver round nut", "grasp"), ("silver peg", "place")] + [("gold square nut", "grasp"), ("gold peg", "place")] 
    )
    robosuite_args = dict(
        robots="Panda",
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        use_object_obs=True,
        env_name="NutAssembly",
        reward_scale=2.0,
        horizon=500,
    )
    # OSC controller spec
    controller_args = dict(
        type="OSC_POSE",
        input_max=1,
        input_min=-1,
        output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        kp=150,
        damping=1,
        impedance_mode="fixed",
        kp_limits=[0, 300],
        damping_limits=[0, 10],
        position_limits=None,
        orientation_limits=None,
        uncouple_pos_ori=True,
        control_delta=True,
        interpolation=None,
        ramp_ratio=0.2,
    )
    robosuite_args["controller_configs"] = controller_args
    mp_env_kwargs["controller_configs"] = controller_args
    env = suite.make(
        **robosuite_args,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names="frontview",
        camera_heights=480,
        camera_widths=640,
    )
    #env = MPEnv(GymWrapper(env), **mp_env_kwargs)
    env = RobosuiteMPEnv(
        env,
        "NutAssembly",
        **mp_env_kwargs,
    )
    num_episodes = 2
    total = 0
    ptu.device = torch.device("cuda")
    frames = []
    all_success_rates = []
    for seed in [0]:
        np.random.seed(seed)
        env.intermediate_frames = []
        success_rate = 0
        for s in tqdm(range(num_episodes)):
            curr_frames = []
            # skip_mp= not (seed == 4 and s == 18)
            o = env.reset(get_intermediate_frames=True)
            # if not ((seed == 4 and s == 18) or (seed == 1 and s == 11)):
            #     continue                
            start_img = env.get_image().copy()
            if len(env.intermediate_frames) > 0:
                for frame in env.intermediate_frames:
                    curr_frames.append(frame)
                env.intermediate_frames = []
            rs = []
            for i in range(25):
                a = np.concatenate(([0, 0, -0.3], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                curr_frames.append(env.get_image())
                # env.render()
            for i in range(15):
                a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                curr_frames.append(env.get_image())
            for i in range(20):
                a = np.concatenate(([0, 0, 0.1], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                if len(env.intermediate_frames) > 0:
                    curr_frames.extend(env.intermediate_frames)
                    env.intermediate_frames = []
                curr_frames.append(env.get_image())
            for i in range(40):
                a = np.concatenate(([0, 0, -0.3], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                curr_frames.append(env.get_image())
            for i in range(10):
                a = np.concatenate(([0, 0, 0], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                curr_frames.append(env.get_image())

            for i in range(25):
                a = np.concatenate(([0, 0, -0.2], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                curr_frames.append(env.get_image())
                # env.render()
            for i in range(15):
                a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                curr_frames.append(env.get_image())
            for i in range(20):
                a = np.concatenate(([0, 0, 0.1], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                if len(env.intermediate_frames) > 0:
                    curr_frames.extend(env.intermediate_frames)
                    env.intermediate_frames = []
                curr_frames.append(env.get_image())
            for i in range(40):
                a = np.concatenate(([0, 0, -0.3], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                curr_frames.append(env.get_image())
            for i in range(10):
                a = np.concatenate(([0, 0, 0], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                curr_frames.append(env.get_image())
            cv2.imwrite("img_4.png", env.get_image(camera_name='agentview', width=960, height=540))
            print(env._check_success())
            plt.plot(rs)
            success_rate += env._check_success()
            print("Running success rate: ", success_rate / (s + 1))
            if env._check_success():
                frames.extend(curr_frames)
                print("success: seed {}, episode {}".format(seed, s))
                cv2.imwrite('success_{}_{}.png'.format(seed, s), start_img)
            if seed == 4 and s == 18:
                break
        print(f"Success Rate: {success_rate/num_episodes}")
        all_success_rates.append(success_rate / num_episodes)
    print("Mean success", np.mean(all_success_rates))
    print("Std success", np.std(all_success_rates))
    make_video(frames, "videos", 0, use_wandb=False)

