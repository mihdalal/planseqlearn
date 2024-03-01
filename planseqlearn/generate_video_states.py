import mujoco_py
import argparse 
from planseqlearn.environments.robosuite_dm_env import make_robosuite
from planseqlearn.environments.metaworld_dm_env import make_metaworld
from planseqlearn.environments.mopa_dm_env import make_mopa
from planseqlearn.environments.kitchen_dm_env import make_kitchen
import torch 
import numpy as np
from planseqlearn.psl.env_text_plans import *
import pickle

from planseqlearn.utils import make_video 

def robosuite_gen_video(env_name, camera_name, suite):
    # create environment 
    agent = torch.load(f"psl_policies/{suite}/{env_name}.pt")["agent"]
    if env_name == "PickPlaceCanBread" or env_name == "PickPlaceCerealMilk":
        env_name = "PickPlace"
    if suite == "robosuite":
        env = make_robosuite(
            name=env_name,
            frame_stack=3,
            action_repeat=2,
            discount=1.0,
            camera_name=camera_name,
            psl=True,
            use_mp=True,
            use_sam_segmentation=False,
            use_vision_pose_estimation=False,
            text_plan=ROBOSUITE_PLANS[env_name],
            vertical_displacement=0.08,
            estimate_orientation=False,
        )
        inner_env = env._env._env._env._env._env
        mp_env = inner_env._env
    elif suite == "metaworld":
        env = make_metaworld(
                name=env_name,
                frame_stack=3,
                action_repeat=2,
                discount=1.0,
                camera_name=camera_name,
                psl=True,
                text_plan=METAWORLD_PLANS[env_name],
                use_mp=True,
                use_sam_segmentation=False,
                use_vision_pose_estimation=False,
                seed=0
            )
        inner_env = env._env._env._env._env._env._env
        mp_env = inner_env._env
    elif suite == "mopa":
        env = make_mopa(
            name=env_name,
            frame_stack=3,
            action_repeat=2,
            psl=True,
            text_plan=MOPA_PLANS[env_name],
            use_mp=True,
            use_sam_segmentation=False,
            use_vision_pose_estimation=False,
            seed=0
        )
        inner_env = env._env._env._env._env._env
        mp_env = inner_env._env
    elif suite == "kitchen":
        env = make_kitchen(
            name=env_name,
            frame_stack=3,
            action_repeat=2,
            discount=1.0,
            camera_name=camera_name,
            psl=True,
            text_plan=KITCHEN_PLANS[env_name],
            use_mp=True,
            use_sam_segmentation=False,
            seed=0
        )
        inner_env = env._env._env._env._env._env._env
        mp_env = inner_env._env
    frames = []
    clean_frames = []
    np.random.seed(0)
    o = env.reset()
    states = dict(
        qpos=mp_env.intermediate_qposes,
        qvel=mp_env.intermediate_qvels,
    )
    mp_env.intermediate_qposes = []
    mp_env.intermediate_qvels = []
    frames.extend(mp_env.intermediate_frames)
    with torch.no_grad():
        for _ in range(100):
            act = agent.act(o.observation, step=_, eval_mode=True)
            o = env.step(act)
            if len(mp_env.intermediate_qposes) > 0:
                states['qpos'].extend(mp_env.intermediate_qposes)
                states['qvel'].extend(mp_env.intermediate_qvels)
                mp_env.intermediate_qposes = []
                mp_env.intermediate_qvels = []
                frames.extend(mp_env.intermediate_frames)
                mp_env.intermediate_frames = []
            if suite == 'mopa':
                frames.append(env.get_vid_image())
            else:
                frames.append(env.get_image())
            states["qpos"].append(inner_env.sim.data.qpos.copy())
            states["qvel"].append(inner_env.sim.data.qvel.copy())
            if o.last() or o.reward['success']:
                break
    print(o.reward)
    states["qpos"] = np.array(states["qpos"])
    states["qvel"] = np.array(states["qvel"])
    np.savez(f"states/{env_name}_{camera_name}_states.npz", **states)
    video_filename = f"{env_name}_{camera_name}.mp4"
    make_video(frames, "videos", video_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help='Name of the environment')
    parser.add_argument('--camera_name', type=str, help='Name of the environment')   
    parser.add_argument('--suite', type=str, default='robosuite', help='Type of environment')
    args = parser.parse_args()
    robosuite_gen_video(args.env_name, args.camera_name, args.suite)  