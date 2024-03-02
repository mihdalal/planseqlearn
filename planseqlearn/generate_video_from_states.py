import os
import mujoco_py
import argparse

from tqdm import tqdm
from planseqlearn.environments.robosuite_dm_env import make_robosuite
from planseqlearn.environments.metaworld_dm_env import make_metaworld
from planseqlearn.environments.mopa_dm_env import make_mopa
from planseqlearn.environments.kitchen_dm_env import make_kitchen
import torch
import numpy as np
from planseqlearn.psl.env_text_plans import *
import pickle
import robosuite as suite
import imageio
import cv2

from planseqlearn.utils import make_video


def robosuite_gen_video(env_name, camera_name):
    # create environment
    env = suite.make(
        robots="Panda",
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        env_name=env_name,
        has_renderer=True,  # no on-screen renderer
        has_offscreen_renderer=False,  # no off-screen renderer
        use_camera_obs=False,  # no camera observations
        renderer="nvisii",
    )
    env.reset()
    states = np.load(f"states/{env_name}_{camera_name}_states.npz")
    for step in tqdm(range(states["qpos"].shape[0])):
        qpos = states["qpos"][step]
        qvel = states["qvel"][step]
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
        env.step(np.zeros(8))
        env.render()
    # load png files from images folder
    frames = []
    for _ in range(1, len(os.listdir("images"))):
        im_path = os.path.join("images", "image_{}.png".format(_))
        frames.append(cv2.imread(im_path))
    video_filename = f"rendered_videos/{env_name}_{camera_name}.mp4"
    make_video(frames, "rendered_videos", video_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, help="Name of the environment")
    parser.add_argument("--camera_name", type=str, help="Name of the environment")
    args = parser.parse_args()
    robosuite_gen_video(args.env_name, args.camera_name)
