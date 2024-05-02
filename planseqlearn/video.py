# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import wandb


class VideoRecorder:
    def __init__(
        self,
        root_dir,
        metaworld_camera_name=None,
        render_size=512,
        fps=20,
        use_wandb=False,
    ):
        if root_dir is not None:
            self.save_dir = root_dir / "eval_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.metaworld_camera_name = metaworld_camera_name
        self.use_wandb = use_wandb

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if not self.enabled:
            return

        if self.metaworld_camera_name is not None:
            frame = env.physics.render(
                height=540,
                width=960,
                mode="offscreen",
                camera_name=self.metaworld_camera_name,
            )
        elif hasattr(env, "fixed_view_physics"):
            frame = env.fixed_view_physics.render(
                height=540,
                width=960,
            )
        elif hasattr(env, "physics"):
            frame = env.physics.render(height=540, width=960)
        else:
            frame = env.render()
        self.frames.append(frame)

    def save(self, file_prefix, step):
        if not self.enabled:
            return

        wandb_videos = {}
        file_name = f"{file_prefix}.mp4"
        path = str(self.save_dir / file_name)
        imageio.mimsave(path, self.frames, fps=self.fps)

        if self.use_wandb:
            wandb_videos["rgb"] = wandb.Video(path, fps=self.fps, format="mp4")

        if self.use_wandb:
            wandb.log(wandb_videos, step=step)


class FrameRecorder:
    def __init__(self, root_dir, fps=20, output_size=(96, 96), use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir / "eval_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.fps = fps
        self.named_frames = {}
        self.output_size = output_size
        self.use_wandb = use_wandb

    def init(self, agent, obs, enabled=True):
        self.named_frames = {}
        self.enabled = self.save_dir is not None and enabled
        self.record(agent, obs)

    def record(self, agent, obs):
        if not self.enabled:
            return

        new_frames = agent.get_frames_to_record(obs)

        for name, frame in new_frames.items():
            if self.output_size is not None:
                frame = cv2.resize(
                    frame, dsize=self.output_size, interpolation=cv2.INTER_CUBIC
                )

            if name not in self.named_frames:
                self.named_frames[name] = [frame]
            else:
                self.named_frames[name].append(frame)

    def save(self, file_prefix, step):
        if not self.enabled:
            return

        wandb_videos = {}
        for name, frames in self.named_frames.items():
            file_name = f"{file_prefix}_{name}.mp4"
            path = str(self.save_dir / file_name)
            imageio.mimsave(path, frames, fps=self.fps)

            if self.use_wandb:
                wandb_videos[name] = wandb.Video(path, fps=self.fps, format="mp4")

        if self.use_wandb:
            wandb.log(wandb_videos, step=step)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / "train_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if not self.enabled:
            return

        frame = cv2.resize(
            obs[-3:].transpose(1, 2, 0),
            dsize=(self.render_size, self.render_size),
            interpolation=cv2.INTER_CUBIC,
        )
        self.frames.append(frame)

    def save(self, file_name):
        if not self.enabled:
            return

        path = self.save_dir / file_name
        imageio.mimsave(str(path), self.frames, fps=self.fps)
