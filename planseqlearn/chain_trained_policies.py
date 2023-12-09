import mujoco_py
import os
from pathlib import Path

from planseqlearn.environments import dmc
from planseqlearn.environments.adroit_dm_env import make_adroit
from planseqlearn.environments.distracting_dmc import make_distracting_dmc
from planseqlearn.environments.kitchen_dm_env import make_kitchen
from planseqlearn.environments.metaworld_dm_env import make_metaworld
from planseqlearn.environments.robosuite_dm_env import make_robosuite
from rlkit.torch.model_based.dreamer.visualization import make_video
from PIL import Image, ImageDraw, ImageFont
import copy

FONT = ImageFont.truetype("arial.ttf", 25)
BOLD_FONT = ImageFont.truetype("Arial Bold.ttf", 25)
LARGE_FONT = ImageFont.truetype("Arial Bold.ttf", 30)


os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import hydra
import numpy as np
import torch
from planseqlearn import utils
from planseqlearn.video import VideoRecorder
import cv2

torch.backends.cudnn.benchmark = True


def make_env(cfg, is_eval):
    if cfg.task_name.split("_", 1)[0] == "metaworld":
        env = make_metaworld(
            cfg.task_name.split("_", 1)[1],
            cfg.frame_stack,
            cfg.action_repeat,
            cfg.discount,
            cfg.seed,
            cfg.camera_name,
            cfg.add_segmentation_to_obs,
            cfg.noisy_mask_drop_prob,
            cfg.use_rgbm,
            cfg.slim_mask_cfg,
            cfg.mprl,
        )
    elif cfg.task_name.split("_", 1)[0] == "robosuite":
        env = make_robosuite(
            cfg.task_name.split("_", 1)[1],
            cfg.frame_stack,
            cfg.action_repeat,
            cfg.discount,
            cfg.seed,
            cfg.camera_name,
            cfg.add_segmentation_to_obs,
            cfg.noisy_mask_drop_prob,
            cfg.use_rgbm,
            cfg.slim_mask_cfg,
            cfg.mprl,
            cfg.path_length,
            cfg.vertical_displacement,
            cfg.hardcoded_orientations,
            cfg.valid_obj_names,
            cfg.steps_of_high_level_plan_to_complete,
            cfg.use_proprio,
            cfg.use_fixed_plus_wrist_view,
            cfg.pose_sigma,
            cfg.noisy_pose_estimates,
            cfg.hardcoded_high_level_plan,
        )
    elif cfg.task_name.split("_", 1)[0] == "adroit":
        env = make_adroit(
            cfg.task_name.split("_", 1)[1],
            cfg.frame_stack,
            cfg.action_repeat,
            cfg.discount,
            cfg.seed,
            cfg.camera_name,
            cfg.add_segmentation_to_obs,
            cfg.noisy_mask_drop_prob,
            cfg.use_rgbm,
            cfg.slim_mask_cfg,
        )
    elif cfg.task_name.split("_", 1)[0] == "kitchen":
        env = make_kitchen(
            cfg.task_name.split("_", 1)[1],
            cfg.frame_stack,
            cfg.action_repeat,
            cfg.discount,
            cfg.seed,
            cfg.camera_name,
            cfg.add_segmentation_to_obs,
            cfg.noisy_mask_drop_prob,
            cfg.use_rgbm,
            cfg.slim_mask_cfg,
            cfg.path_length,
            cfg.mprl,
        )
    elif cfg.task_name.split("_", 1)[0] == "distracting":
        background_dataset_videos = "val" if is_eval else "train"
        env = make_distracting_dmc(
            cfg.task_name.split("_", 1)[1],
            cfg.frame_stack,
            cfg.action_repeat,
            cfg.seed,
            cfg.add_segmentation_to_obs,
            cfg.distraction.difficulty,
            cfg.distraction.types,
            cfg.distraction.dataset_path,
            background_dataset_videos,
            cfg.noisy_mask_drop_prob,
            cfg.use_rgbm,
            cfg.slim_mask_cfg,
        )
    elif cfg.task_name.split("_", 1)[0] == "mopa":
        env = make_mopa(
            name=cfg.task_name.split("_", 1)[1],
            frame_stack=cfg.frame_stack,
            action_repeat=cfg.action_repeat,
            discount=cfg.discount,
            seed=cfg.seed,
            horizon=cfg.path_length,
            mprl=cfg.mprl,
            is_eval=is_eval,
        )
    else:
        env = dmc.make(cfg.task_name, cfg.frame_stack, cfg.action_repeat, cfg.seed)
    return env


class ChainedWorkspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,
            metaworld_camera_name=self.cfg.camera_name
            if cfg.task_name.split("_", 1)[0] == "metaworld"
            else None,
            use_wandb=self.cfg.use_wandb,
        )
        self.eval_env = make_env(self.cfg, is_eval=True)

        self.agents = []
        for idx, snapshot in enumerate(self.cfg.agent_snapshots):
            with open(snapshot, "rb") as f:
                payload = torch.load(f)
            self.agents.append(payload["agent"])

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        if self.cfg.has_success_metric:
            mean_max_success, mean_mean_success, mean_last_success = 0, 0, 0
        all_frames = []
        all_clean_frames = []
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        self.video_recorder.init(self.eval_env, enabled=True)
        success_ep = 0
        while eval_until_episode(episode):
            if self.cfg.has_success_metric:
                current_episode_max_success = 0
                current_episode_mean_success = 0
                current_episode_last_success = 0
            current_episode_step = 0
            time_step = self.eval_env.reset()
            self.video_recorder.record(self.eval_env)
            agent_idx = 0
            global_step = 0
            prev_agent_idx = 0
            obs_frames = []
            ep_frames = []
            ep_frames.extend(self.eval_env.intermediate_frames[::2])
            self.eval_env._env._env._env._env._env._env.intermediate_frames = []
            ep_clean_frames = []
            ep_clean_frames.extend(self.eval_env.clean_frames)
            self.eval_env._env._env._env._env._env._env.clean_frames = []
            mp_idxs = [i for i in range(len(ep_frames))]
            while not time_step.last():
                if prev_agent_idx != agent_idx:
                    if self.cfg.task_name.startswith("robosuite"):
                        final_obs = (
                            self.eval_env._env._env._env._env._env._env.cam_sensor[0](
                                None
                            ).transpose(2, 0, 1)
                        )
                    elif self.cfg.task_name.startswith("kitchen"):
                        final_obs = self.eval_env.physics.render(
                            **self.eval_env._render_kwargs
                        ).transpose(2, 0, 1)
                    self.eval_env._frames[0].append(final_obs)
                    self.eval_env._frames[0].append(final_obs)
                    self.eval_env._frames[0].append(final_obs)
                    time_step.observation["pixels"] = np.concatenate(
                        (final_obs, final_obs, final_obs), axis=0
                    )
                    time_step.observation["state"] = np.concatenate(
                        (
                            self.eval_env._env._env._env._env._env._env._eef_xpos,
                            self.eval_env._env._env._env._env._env._env._eef_xquat,
                        ),
                        axis=0,
                    )
                current_episode_step += 1
                with torch.no_grad(), utils.eval_mode(self.agents[agent_idx]):
                    action = self.agents[agent_idx].act(
                        time_step.observation, global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                if len(self.eval_env._env._env._env._env._env._env.intermediate_frames) > 0:
                    mp_idxs.extend([i + len(mp_idxs) for i in range(len(self.eval_env._env._env._env._env._env._env.intermediate_frames) // 2)])
                    ep_frames.extend(self.eval_env._env._env._env._env._env._env.intermediate_frames[::2])
                    ep_clean_frames.extend(self.eval_env._env._env._env._env._env._env.clean_frames[::2])
                    self.eval_env._env._env._env._env._env._env.intermediate_frames = []
                    self.eval_env._env._env._env._env._env._env.clean_frames = []
                self.video_recorder.record(self.eval_env)
                #all_frames.append(self.eval_env.get_image())
                ep_frames.append(self.eval_env.get_image())
                ep_clean_frames.append(self.eval_env.get_image())
                prev_agent_idx = agent_idx
                agent_idx = (self.eval_env.num_high_level_steps - 1) // 2  # note this will only work for mprl

                if self.cfg.has_success_metric:
                    total_reward += time_step.reward["reward"]
                    success = int(time_step.reward["success"])
                    current_episode_max_success = max(
                        current_episode_max_success, success
                    )
                    current_episode_last_success = success
                    current_episode_mean_success += success
                else:
                    total_reward += time_step.reward
                step += 1
                # if agent_idx > 0:
                #     current_episode_max_success = 1
                #     break
            if self.cfg.has_success_metric:
                mean_max_success += current_episode_max_success
                current_episode_max_success = 1
                if current_episode_max_success > 0:
                    # add text to images
                    for i in range(len(ep_frames)):
                        pil_img = Image.fromarray(ep_frames[i].astype(np.uint8))
                        draw = ImageDraw.Draw(pil_img)
                        draw.text((20, 10), f"{self.eval_env.name.split('_')[1]}", font=LARGE_FONT)
                        if i in mp_idxs:
                            draw.text((20, 55), "Motion Planner", font=BOLD_FONT, fill="#FFFFFF")
                            draw.text((800, 15), "2x speed", font=BOLD_FONT, fill="#FFFFFF")
                            draw.rectangle([(13, 53), (205, 85)], outline="white", width=3)
                            draw.text((20, 90), "Local Policy", font=FONT, fill="#D3D3D3")
                        else:
                            draw.text((20, 55), "Motion Planner", font=FONT, fill="#D3D3D3")
                            draw.text((800, 15), "1x speed", font=BOLD_FONT, fill="#FFFFFF")
                            draw.rectangle([(13, 88), (170, 120)], outline="white", width=3)
                            draw.text((20, 90), "Local Policy", font=BOLD_FONT, fill="#FFFFFF")
                        ep_frames[i] = np.array(pil_img).astype(np.uint8)
                    success_ep += 1
                    all_frames.extend(ep_frames)
                    all_clean_frames.extend(ep_clean_frames)
                mean_last_success += current_episode_last_success
                mean_mean_success += current_episode_mean_success / current_episode_step
            episode += 1
            print(f"Curr num successes: {success_ep}")
        self.video_recorder.save(f"{0}", step=0)
        make_video(all_frames, "videos", 0, use_wandb=False)
        make_video(all_clean_frames, "videos", 1, use_wandb=False)
        print("episode_reward", total_reward / episode)
        print("episode_length", step * self.cfg.action_repeat / episode)
        if self.cfg.has_success_metric:
            print("max_success", mean_max_success / episode)
            print("last_success", mean_last_success / episode)
            print("mean_success", mean_mean_success / episode)


@hydra.main(config_path="cfgs", config_name="eval_cfg")
def main(cfg):
    workspace = ChainedWorkspace(cfg)
    workspace.eval()


if __name__ == "__main__":
    np.random.seed(1)
    main()