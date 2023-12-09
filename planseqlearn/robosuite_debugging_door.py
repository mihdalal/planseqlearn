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

def get_object_pose(env, obj_idx=0):
    name = env.name.split("_")[1]
    if name.endswith("Lift"):
        object_pos = env.sim.data.qpos[9:12].copy()
        object_quat = T.convert_quat(env.sim.data.qpos[12:16].copy(), to="xyzw")
    elif name.startswith("PickPlaceMilk"):
        object_pos = env.sim.data.qpos[9:12].copy()
        object_quat = T.convert_quat(env.sim.data.qpos[12:16].copy(), to="xyzw")
    elif name.startswith("PickPlaceBread"):
        object_pos = env.sim.data.qpos[16:19].copy()
        object_quat = T.convert_quat(env.sim.data.qpos[19:23].copy(), to="xyzw")
    elif name.startswith("PickPlaceCereal"):
        object_pos = env.sim.data.qpos[23:26].copy()
        object_quat = T.convert_quat(env.sim.data.qpos[26:30].copy(), to="xyzw")
    elif name.startswith("PickPlaceCan"):
        object_pos = env.sim.data.qpos[30:33].copy()
        object_quat = T.convert_quat(env.sim.data.qpos[33:37].copy(), to="xyzw")
    elif name.endswith("PickPlace"):
        new_obj_idx = compute_correct_obj_idx(env, obj_idx=obj_idx)
        object_pos = env.sim.data.qpos[
            9 + 7 * new_obj_idx : 12 + 7 * new_obj_idx
        ].copy()
        object_quat = T.convert_quat(
            env.sim.data.qpos[12 + 7 * new_obj_idx : 16 + 7 * new_obj_idx].copy(),
            to="xyzw",
        )
    elif name.startswith("Door"):
        object_pos = np.array(
            [env.sim.data.qpos[env.hinge_qpos_addr]]
        )  # this is not what they are, but they will be decoded properly
        object_quat = np.array(
            [env.sim.data.qpos[env.handle_qpos_addr]]
        )  # this is not what they are, but they will be decoded properly
    elif name.startswith("Wipe"):
        object_pos = np.zeros(3)
        object_quat = np.zeros(4)
    elif "NutAssembly" in name:
        if name.endswith("Square"):
            nut = env.nuts[0]
        elif name.endswith("Round"):
            nut = env.nuts[1]
        elif name.endswith("NutAssembly"):
            nut = env.nuts[1 - obj_idx]  # first nut is round, second nut is square
        nut_name = nut.name
        object_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[nut_name]])
        object_quat = T.convert_quat(
            env.sim.data.body_xquat[env.obj_body_id[nut_name]], to="xyzw"
        )
    else:
        raise NotImplementedError()
    return object_pos, object_quat


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=bool)
    args = parser.parse_args()
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
        #use_vision_placement_check=True,
        #use_vision_grasp_check=True,
    )
    robosuite_args = dict(
        robots="Panda",
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        use_object_obs=True,
        env_name="Door",
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
    if args.a:
        print(f"A")
        env = MPEnv(GymWrapper(env), **mp_env_kwargs)
    else:
        print(f"B")
        env = RobosuiteMPEnv(env, "Door", **mp_env_kwargs); env._wrapped_env.reset()
    num_episodes = 1
    total = 0
    ptu.device = torch.device("cuda")
    np.random.seed(1)
    frames = []
    all_success_rates = []
    for seed in range(1):
        np.random.seed(seed)
        env.intermediate_frames = []
        success_rate = 0
        for s in tqdm(range(num_episodes)):
            o = env.reset(get_intermediate_frames=True)
            print(env.sim.data.qpos)
            # load in policy 
            print(env._check_success())
            # plt.savefig(f"plots/{s}.png")
            success_rate += env._check_success()
            np.set_printoptions(suppress=True)
            print("Running success rate: ", success_rate / (s + 1))
        print(f"Success Rate: {success_rate/num_episodes}")
        all_success_rates.append(success_rate / num_episodes)
    print("Mean success", np.mean(all_success_rates))
    print("Std success", np.std(all_success_rates))
    make_video(frames, "videos", 2, use_wandb=False)