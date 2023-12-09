import mujoco_py
import numpy as np
import robosuite as suite
import torch
from matplotlib import pyplot as plt
from robosuite.utils.transform_utils import *
from robosuite.wrappers.gym_wrapper import GymWrapper
from tqdm import tqdm
from planseqlearn.mnm.robosuite_mp_env import RobosuiteMPEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.mprl.mp_env import *
from rlkit.torch.model_based.dreamer.visualization import make_video

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=bool)
    args = parser.parse_args()
    mp_env_kwargs = dict(
        vertical_displacement=0.08,
        #vertical_displacement=0.06,
        teleport_instead_of_mp=True,
        use_joint_space_mp=False,
        randomize_init_target_pos=False,
        mp_bounds_low=(-1.45, -1.25, 0.45),
        mp_bounds_high=(0.45, 0.85, 2.25),
        backtrack_movement_fraction=0.001,
        clamp_actions=True,
        update_with_true_state=True,
        grip_ctrl_scale=0.0025,
        planning_time=0.5 * 100,
        hardcoded_high_level_plan=True,
        terminate_on_success=False,
        plan_to_learned_goals=False,
        reset_at_grasped_state=False,
        verify_stable_grasp=True,
        hardcoded_orientations=False,
        use_pcd_collision_check=False,
        use_vision_pose_estimation=True,
        use_vision_placement_check=True,
        use_vision_grasp_check=True,
        # noisy_pose_estimates=False,
        # pose_sigma=0,
    )
    robosuite_args = dict(
        robots="Panda",
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        use_object_obs=True,
        env_name="PickPlace",
        # valid_obj_names=["Cereal", "Milk", "Can", "Bread"],
        valid_obj_names=["Cereal", "Milk"],
        reward_scale=4.0,
    )
    controller_configs = dict(
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
    robosuite_args["controller_configs"] = controller_configs
    mp_env_kwargs["controller_configs"] = controller_configs
    np.random.seed(2)
    env = suite.make(
        **robosuite_args,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names="frontview",
        camera_heights=480,
        camera_widths=640,
        hard_reset=False,
    )
    if args.a:
        print(f"A")
        env = MPEnv(GymWrapper(env), **mp_env_kwargs)
    else:
        print(f"B")
        env = RobosuiteMPEnv(env, "PickPlace", **mp_env_kwargs); env._wrapped_env.reset()
    all_success_rates = []
    for seed in range(1):
        num_episodes = 1
        total = 0
        ptu.device = torch.device("cuda")
        success_rate = 0
        frames = []
        np.set_printoptions(suppress=True)
        env.intermediate_frames = []
        np.random.seed(1)
        for s in tqdm(range(num_episodes)):
            o = env.reset(get_intermediate_frames=True)
            if len(env.intermediate_frames) > 0:
                for frame in env.intermediate_frames:
                    frames.append(frame)
                env.intermediate_frames = []
            rs = []
            frames.append(env.get_image())
            for i in range(20):
                a = np.concatenate(([0, 0, -0.3], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            for i in range(5):
                a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            for i in range(20):
                a = np.concatenate(([0, 0, 0.1], [0, 0, 0, 1]))
                o, r, d, info = env.step(a, get_intermediate_frames=True)
                if len(env.intermediate_frames) > 0:
                    for frame in env.intermediate_frames:
                        frames.append(frame)
                    env.intermediate_frames = []
                rs.append(r)
                frames.append(env.get_image())
            for i in range(10):
                print(f"i: {i}")
                a = np.concatenate(([0, 0, 0.0], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            #breakpoint()
            for i in range(10):
                a = np.concatenate(([0, 0, 0.1], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            for i in range(20):
                a = np.concatenate(([0, 0, -0.3], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            np.set_printoptions(suppress=True)
            for i in range(5):
                a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            for i in range(20):
                a = np.concatenate(([0, 0, 0.1], [0, 0, 0, 1]))
                o, r, d, info = env.step(a, get_intermediate_frames=True)
                if len(env.intermediate_frames) > 0:
                    for frame in env.intermediate_frames:
                        frames.append(frame)
                    env.intermediate_frames = []
                rs.append(r)
                frames.append(env.get_image())
            for i in range(10): # usd to be 10
                a = np.concatenate(([0, 0, 0.0], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            for i in range(10):
                a = np.concatenate(([0, 0, 0.1], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            for i in range(15):
                a = np.concatenate(([0, 0, -0.3], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            for i in range(10):
                a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            for i in range(30):
                a = np.concatenate(([0, 0, 0.1], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            for i in range(10):
                a = np.concatenate(([0, 0, 0.0], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())
            for i in range(10):
                a = np.concatenate(([0, 0, 0.1], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                frames.append(env.get_image())

            # for i in range(25):
            #     a = np.concatenate(([0, 0, -0.3], [0, 0, 0, -1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     frames.append(env.get_image())
            # for i in range(10):
            #     a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     frames.append(env.get_image())
            # for i in range(30):
            #     a = np.concatenate(([0, 0, 0.1], [0, 0, 0, 1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     frames.append(env.get_image())
            # for i in range(10):
            #     a = np.concatenate(([0, 0, 0.0], [0, 0, 0, -1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     frames.append(env.get_image())
            # for i in range(10):
            #     a = np.concatenate(([0, 0, 0.1], [0, 0, 0, -1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     frames.append(env.get_image())

            # for i in range(25):
            #     a = np.concatenate(([0, 0, -0.3], [0, 0, 0, -1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     frames.append(env.get_image())
            # for i in range(10):
            #     a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     frames.append(env.get_image())
            # for i in range(30):
            #     a = np.concatenate(([0, 0, 0.1], [0, 0, 0, 1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     frames.append(env.get_image())
            # for i in range(10):
            #     a = np.concatenate(([0, 0, 0.0], [0, 0, 0, -1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     frames.append(env.get_image())
            # for i in range(10):
            #     a = np.concatenate(([0, 0, 0.1], [0, 0, 0, -1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     frames.append(env.get_image())

            # plt.plot(rs)
            # plt.savefig(f"test_{s}.png")
            # plt.clf()
            # cumsum = np.cumsum(rs)
            # plt.plot(cumsum)
            # plt.savefig(f"test_{s}_cumsum.png")
            # plt.clf()
            # plt.show()
            success_rate += env._check_success()
            print(env._check_success())
            print(f"Final qpos")
            print(env.sim.data.qpos)
            print("Running Success Rate: ", success_rate / (s + 1))
        print(f"Success Rate: {success_rate/num_episodes}")
        all_success_rates.append(success_rate / num_episodes)
    print("Mean success", np.mean(all_success_rates))
    print("Std success", np.std(all_success_rates))
    if args.a:
        make_video(frames, "videos", 2, use_wandb=False)
    else:
        make_video(frames, "videos", 3, use_wandb=False)