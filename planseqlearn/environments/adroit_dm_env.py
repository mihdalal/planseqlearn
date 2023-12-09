import d4rl
import dm_env
import gym
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import specs

from planseqlearn.environments.wrappers import (
    ActionDTypeWrapper,
    ActionRepeatWrapper,
    ExtendedTimeStepWrapper,
    FrameStackWrapper,
    NoisyMaskWrapper,
    Render_Wrapper,
    SegmentationFilter,
    SegmentationToRobotMaskWrapper,
    SlimMaskWrapper,
    StackRGBAndMaskWrapper,
    get_env_action_spec,
    get_env_observation_spec,
)


class Adroit_Wrapper(dm_env.Environment):
    def __init__(self, env_name: str, discount=1.0, seed=None):
        self.discount = discount
        self._env = gym.make(env_name)
        self._env.seed(seed)
        self.env_name = env_name
        self.physics = Render_Wrapper(self._env.sim.render)
        self._reset_next_step = True
        self.current_step = 0
        self._observation_spec = None
        self._action_spec = None
        self.robot_segmentation_ids = list(range(7, 52))

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        self.current_step = 0
        self.physics = Render_Wrapper(self._env.sim.render)
        observation = self._env.reset()
        observation = observation.astype(self._env.observation_space.dtype)
        self.current_step += 1
        return dm_env.restart(observation)

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        observation, reward, done, info = self._env.step(action)
        self.current_step += 1

        observation = observation.astype(self._env.observation_space.dtype)

        reward_dict = {"reward": reward, "success": info["goal_achieved"]}

        if done:
            self._reset_next_step = True
            return dm_env.truncation(reward_dict, observation, self.discount)

        return dm_env.transition(reward_dict, observation, self.discount)

    def observation_spec(self) -> specs.BoundedArray:
        if self._observation_spec is not None:
            return self._observation_spec

        spec = get_env_observation_spec(self._env)

        self._observation_spec = spec
        return self._observation_spec

    def action_spec(self) -> specs.BoundedArray:
        if self._action_spec is not None:
            return self._action_spec

        spec = get_env_action_spec(self._env)

        self._action_spec = spec
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_adroit(
    name,
    frame_stack,
    action_repeat,
    discount,
    seed,
    camera_name,
    add_segmentation_to_obs,
    noisy_mask_drop_prob,
    use_rgbm=False,
    slim_mask_cfg=None,
):
    env = Adroit_Wrapper(env_name=name, discount=discount, seed=seed)

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat, use_metaworld_reward_dict=True)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    frame_keys = []

    rgb_key = "pixels"
    frame_keys.append(rgb_key)
    render_kwargs = dict(height=84, width=84, mode="offscreen", camera_name=camera_name)
    env = pixels.Wrapper(
        env, pixels_only=False, render_kwargs=render_kwargs, observation_key=rgb_key
    )

    if add_segmentation_to_obs:
        segmentation_key = "segmentation"
        frame_keys.append(segmentation_key)
        segmentation_kwargs = dict(
            height=84 * 3,
            width=84 * 3,
            mode="offscreen",
            camera_name=camera_name,
            segmentation=True,
        )
        env = pixels.Wrapper(
            env,
            pixels_only=False,
            render_kwargs=segmentation_kwargs,
            observation_key=segmentation_key,
        )
        env = SegmentationToRobotMaskWrapper(env, segmentation_key)

        env = SegmentationFilter(env, segmentation_key)

        if noisy_mask_drop_prob > 0:
            env = NoisyMaskWrapper(
                env, segmentation_key, prob_drop=noisy_mask_drop_prob
            )

        if slim_mask_cfg and slim_mask_cfg.use_slim_mask:
            env = SlimMaskWrapper(
                env,
                segmentation_key,
                slim_mask_cfg.scale,
                slim_mask_cfg.threshold,
                slim_mask_cfg.sigma,
            )

        if use_rgbm:
            env = StackRGBAndMaskWrapper(
                env, rgb_key, segmentation_key, new_key="pixels"
            )
            frame_keys = ["pixels"]

    env = FrameStackWrapper(env, frame_stack, frame_keys)
    env = ExtendedTimeStepWrapper(env, has_success_metric=True)
    return env
