import numpy as np
import mujoco_py
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from planseqlearn.environments.kitchen_dm_env import make_kitchen
from planseqlearn.environments.metaworld_dm_env import make_metaworld, ENV_CAMERA_DICT
from planseqlearn.environments.mopa_dm_env import make_mopa
from planseqlearn.environments.robosuite_dm_env import make_robosuite
import torch
from rlkit.torch.model_based.dreamer.visualization import make_video
from PIL import Image, ImageDraw, ImageFont
import copy

FONT = ImageFont.truetype("arial.ttf", 25)
BOLD_FONT = ImageFont.truetype("Arial Bold.ttf", 25)
LARGE_FONT = ImageFont.truetype("Arial Bold.ttf", 30)


def make_env(env_name):
    suite, environment = env_name.split("_")
    if suite == "kitchen":
        env = make_kitchen(
            environment,
            frame_stack=3,
            action_repeat=1,
            discount=1.0,
            seed=0,
            camera_name="wrist_cam",
            add_segmentation_to_obs=False,
            noisy_mask_drop_prob=0.0,
            mprl=True,
            use_mp=True,
        )
    if suite == "robosuite":
        env = make_robosuite(
            environment,
            frame_stack=3,
            action_repeat=1,
            discount=1.0,
            seed=0,
            camera_name="robot0_eye_in_hand",
            add_segmentation_to_obs=False,
            noisy_mask_drop_prob=0.0,
            video_mode=True,
            mprl=True,
        )
    if suite == "metaworld":
        env = make_metaworld(
            environment,
            frame_stack=3,
            action_repeat=1,
            discount=1.0,
            seed=0,
            camera_name=ENV_CAMERA_DICT[environment],
            add_segmentation_to_obs=False,
            noisy_mask_drop_prob=0.0,
            mprl=True,
            use_mp=True,
        )
    if suite == "mopa":
        env = make_mopa(
            environment,
            frame_stack=3,
            action_repeat=1,
            discount=1.0,
            seed=0,
        )
    return env


def main(args):
    env = make_env(args.env_name)
    # load in policy
    policy_file = torch.load(args.policy_path)
    agent = policy_file["agent"]
    frames = []
    mp_idxs = []
    o = env.reset()
    frames.extend(env.intermediate_frames)
    env.intermediate_frames = []
    mp_idxs.extend([i for i in range(len(frames))])
    for step in range(env.max_path_length):
        with torch.no_grad():
            action = agent.act(o.observation, step=step, eval_mode=True)
        o = env.step(action)
        if len(env.intermediate_frames) > 0:
            frames.extend(env.intermediate_frames)
            mp_idxs.extend([i + len(frames) for i in range(len(intermediate_frames))])
            env.intermediate_frames = []
        if args.env_name.startswith("metaworld"):
            frames.append(env.get_image()[:, :, ::-1])
        else:
            frames.append(env.get_image())
        if o.reward["success"] or o.last():
            break

    if args.gen_clean_video:
        clean_frames = copy.deepcopy(frames)
        make_video(clean_frames, "videos", 1, use_wandb=False)
    for i in range(len(frames)):
        pil_img = Image.fromarray(frames[i].astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)
        draw.text((20, 10), f"{env.env_name}", font=LARGE_FONT)
        if i in mp_idxs:
            draw.text((20, 55), "Motion Planner", font=BOLD_FONT, fill="#FFFFFF")
            draw.text((800, 15), "2x speed", font=BOLD_FONT, fill="#FFFFFF")
            draw.rectangle([(13, 53), (205, 85)], outline="white", width=3)
            draw.text((20, 90), "Local Policy", font=FONT, fill="#D3D3D3")
        else:
            draw.text((20, 55), "Motion Planner", font=FONT, fill="#D3D3D3")
            draw.text((800, 15), "0.5x speed", font=BOLD_FONT, fill="#FFFFFF")
            draw.rectangle([(13, 88), (170, 120)], outline="white", width=3)
            draw.text((20, 90), "Local Policy", font=BOLD_FONT, fill="#FFFFFF")
        frames[i] = np.array(pil_img).astype(np.uint8)
    make_video(frames, "videos", 0, use_wandb=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_path", type=str, required=False, help="Path to the policy file"
    )
    parser.add_argument(
        "--env_name", type=str, required=True, help="Name of environment"
    )
    parser.add_argument(
        "--gen_clean_video",
        type=bool,
        required=False,
        default=False,
        help="Generate a separate video with no text.",
    )
    args = parser.parse_args()
    main(args)
