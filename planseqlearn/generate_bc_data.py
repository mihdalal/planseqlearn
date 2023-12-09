import os
from collections import OrderedDict
from pathlib import Path

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import hydra
import numpy as np
import torch
from dm_env import specs
from termcolor import colored

from planseqlearn import utils
from planseqlearn.environments import dmc
from planseqlearn.environments.adroit_dm_env import make_adroit
from planseqlearn.environments.metaworld_dm_env import make_metaworld
from planseqlearn.logger import _format
from planseqlearn.replay_buffer import ReplayBufferStorage

torch.backends.cudnn.benchmark = True

PRINT_FORMAT = [
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("success", "S", "int"),
]


def print_episode_stats(data):
    prefix = "Data Generation"
    prefix = colored(prefix, "blue")
    pieces = [f"| {prefix: <14}"]
    for key, disp_key, ty in PRINT_FORMAT:
        value = data.get(key, 0)
        pieces.append(_format(disp_key, value, ty))
    print(" | ".join(pieces))


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        payload = torch.load(self.cfg.snapshot)
        self.has_success_metric = payload["has_success_metric"]
        self.action_repeat = payload["action_repeat"]

        # create envs
        if payload["task_name"].split("_")[0] == "metaworld":
            self.eval_env = make_metaworld(
                payload["task_name"].split("_")[1],
                payload["frame_stack"],
                payload["action_repeat"],
                payload["discount"],
                self.cfg.seed,
                payload["camera_name"],
                self.cfg.add_segmentation_to_obs,
            )
            reward_spec = OrderedDict(
                [
                    ("reward", specs.Array((1,), np.float32, "reward")),
                    ("success", specs.Array((1,), np.int16, "reward")),
                ]
            )
        elif payload["task_name"].split("_")[0] == "adroit":
            self.eval_env = make_adroit(
                payload["task_name"].split("_")[1],
                payload["frame_stack"],
                payload["action_repeat"],
                payload["discount"],
                self.cfg.seed,
                payload["camera_name"],
                self.cfg.add_segmentation_to_obs,
            )
            reward_spec = OrderedDict(
                [
                    ("reward", specs.Array((1,), np.float32, "reward")),
                    ("success", specs.Array((1,), np.int16, "reward")),
                ]
            )
        else:
            self.eval_env = dmc.make(
                payload["task_name"],
                payload["frame_stack"],
                payload["action_repeat"],
                self.cfg.seed,
            )
            reward_spec = specs.Array((1,), np.float32, "reward")

        discount_spec = specs.Array((1,), np.float32, "discount")
        data_specs = {
            "observation": self.eval_env.observation_spec(),
            "action": self.eval_env.action_spec(),
            "reward": reward_spec,
            "discount": discount_spec,
        }
        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / "data")

        self.agent = payload["agent"]

    def generate_bc_data(self):
        episode = 0
        eval_until_episode = utils.Until(self.cfg.num_episodes)
        while eval_until_episode(episode):
            step, total_reward = 0, 0
            if self.has_success_metric:
                max_success = 0

            time_step = self.eval_env.reset()
            time_steps = [time_step]

            while not time_step.last():
                step += 1
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, step, eval_mode=True)
                time_step = self.eval_env.step(action)
                time_steps.append(time_step)
                if self.has_success_metric:
                    total_reward += time_step.reward["reward"]
                    success = int(time_step.reward["success"])
                    max_success = max(max_success, success)
                else:
                    total_reward += time_step.reward

            if max_success > 0:
                for time_step in time_steps:
                    self.replay_storage.add(time_step)
                episode += 1

            episode_stats = {
                "episode": episode - 1,
                "episode_length": step * self.action_repeat,
                "episode_reward": total_reward,
                "success": max_success,
            }
            print_episode_stats(episode_stats)


@hydra.main(config_path="cfgs", config_name="generate_bc_data_config")
def main(cfg):
    from generate_bc_data import Workspace as W

    workspace = W(cfg)
    workspace.generate_bc_data()


if __name__ == "__main__":
    main()
