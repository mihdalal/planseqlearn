import numpy as np
import robosuite as suite
import torch
from matplotlib import pyplot as plt
from robosuite.utils.transform_utils import *
from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu

from planseqlearn.mnm.robosuite_mp_env import RobosuiteMPEnv


def rob_save_img(env, camera_name, filename):
    img = env.sim.render(camera_name=camera_name, width=500, height=500)
    plt.imshow(np.flipud(img))
    plt.savefig(filename)


if __name__ == "__main__":
    robosuite_args = dict(
        robots="Panda",
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        use_object_obs=True,
        env_name="Lift",
        horizon=50,
        use_distance_reduced_to_object_reward=True,
        use_min_prev_distance=False,
        dist_reduced_reward_scale=1,
        first_grasp_reward=True,
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
    env = suite.make(
        **robosuite_args,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names="frontview",
        camera_heights=1024,
        camera_widths=1024,
    )
    env = RobosuiteMPEnv(
        env,
        "Lift",
        vertical_displacement=0.05,
        hardcoded_orientations=False,
        teleport_instead_of_mp=False,
        use_joint_space_mp=False,
        use_vision_pose_estimation=False,
        burn_in=True,
    )
    num_episodes = 10
    total = 0
    ptu.device = torch.device("cuda")
    success_rate = 0
    cumsum = 0
    cumsums = []
    for s in tqdm(range(num_episodes)):
        o = env.reset()
        rob_save_img(env, "sideview", "reset.png")
        rs = []
        for i in range(150):
            a = np.concatenate(
                (
                    env.sim.data.body_xpos[env.cube_body_id] - env._eef_xpos,
                    [0, 0, 0, -1],
                )
            )
            o, r, d, info = env.step(a)
            rs.append(r)
            cumsum += r
            cumsums.append(cumsum)
            # env.render()
            cube_pos = env.sim.data.body_xpos[env.cube_body_id]
            gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)
        for i in range(50):
            a = np.concatenate(([0, 0, -0.1], [0, 0, 0, -1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            cumsum += r
            cumsums.append(cumsum)
            # env.render()
        for i in range(50):
            a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            cumsum += r
            cumsums.append(cumsum)
            # env.render()
        rob_save_img(env, "sideview", "start.png")
        for i in range(50):
            a = np.concatenate(([0, 0, 0.1], [0, 0, 0, 1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            cumsum += r
            cumsums.append(cumsum)
            if d:
                break
            # env.render()
        print(env._check_success())
        plt.plot(rs)
        plt.savefig("rs.png")
        plt.clf()
        plt.plot(cumsums)
        plt.savefig("cumsums.png")
        plt.clf()
        success_rate += env._check_success()
    print(f"Success Rate: {success_rate/num_episodes}")
