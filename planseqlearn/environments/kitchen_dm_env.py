import d4rl  # Import to register d4rl environments to Gym
import dm_env
import gym
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import specs

import planseqlearn.environments.kitchen_custom_envs  # Import to register kitchen custom environments to Gym
from planseqlearn.environments.wrappers import (
    ActionDTypeWrapper,
    ActionRepeatWrapper,
    Camera_Render_Wrapper,
    ExtendedTimeStepWrapper,
    FrameStackWrapper,
    Wrist_Camera_Render_Wrapper,
    get_env_action_spec,
    get_env_observation_spec,
)
from planseqlearn.psl.kitchen_mp_env import KitchenPSLEnv

class Kitchen_Wrapper(dm_env.Environment):
    def __init__(
        self,
        env_name: str,
        text_plan,
        discount=1.0,
        seed=None,
        path_length=280,
        psl=False,
        camera_name="fixed",
        use_sam_segmentation=False,
        use_mp=False,
    ):
        self.discount = discount
        env_kwargs = dict(
            dense=False,
            image_obs=True,
            action_scale=1,
            control_mode="end_effector",
            frame_skip=40,
            max_path_length=path_length,
        )
        # do preprocessing
        from d4rl.kitchen.env_dict import ALL_KITCHEN_ENVIRONMENTS
        self._env = ALL_KITCHEN_ENVIRONMENTS[env_name](**env_kwargs)
        if psl:
            self._env = KitchenPSLEnv(
                self._env,
                env_name,
                use_vision_pose_estimation=False,
                teleport_instead_of_mp=not use_mp,
                use_joint_space_mp=True,
                use_sam_segmentation=use_sam_segmentation,
                text_plan=text_plan,
            )
        # self._env.seed(seed)
        self.env_name = env_name
        self.fixed_view_physics = Camera_Render_Wrapper(
            self._env.sim,
            lookat=[-0.3, 0.5, 2.0],
            distance=1.86,
            azimuth=90,
            elevation=-60,
        )
        self.camera_name = camera_name
        if camera_name == "wrist":
            self.physics = Wrist_Camera_Render_Wrapper(
                self._env.sim,
                lookat=[-0.3, 0.5, 2.0],
                distance=1.86,
                azimuth=90,
                elevation=-60,
            )
        elif camera_name == "fixed":
            self.physics = self.fixed_view_physics
        self._reset_next_step = True
        self.current_step = 0
        self.episode_total_score = 0
        self._observation_spec = None
        self._action_spec = None
        self.path_length = path_length
        self.robot_segmentation_ids = list(range(2, 49))

    def reset(self, **kwargs) -> dm_env.TimeStep:
        self._reset_next_step = False
        self.current_step = 0
        self.episode_total_score = 0
        observation = self._env.reset(**kwargs)
        observation = observation.astype(self._env.observation_space.dtype)
        self.current_step += 1
        return dm_env.restart(observation)

    def step(self, action: int, **kwargs) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        observation, reward, done, info = self._env.step(action)
        self.current_step += 1

        observation = observation.astype(self._env.observation_space.dtype)

        self.episode_total_score += info["score"]
        num_tasks = len(self._env.TASK_ELEMENTS)
        success = self.episode_total_score == num_tasks

        reward_dict = {"reward": reward, "success": success, **info}
        if done or self.current_step == self.path_length:
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


def make_kitchen(
    name,
    frame_stack,
    action_repeat,
    discount,
    seed,
    camera_name,
    text_plan,
    path_length=280,
    psl=False,
    use_sam_segmentation=False,
    use_mp=False,
    control_mode="end_effector"
):
    assert camera_name in ["wrist", "fixed"]

    env = Kitchen_Wrapper(
        env_name=name,
        discount=discount,
        seed=seed,
        path_length=path_length,
        psl=psl,
        camera_name=camera_name,
        use_sam_segmentation=use_sam_segmentation,
        text_plan=text_plan,
        use_mp=use_mp,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat, use_metaworld_reward_dict=True)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    frame_keys = []

    rgb_key = "pixels"
    frame_keys.append(rgb_key)
    render_kwargs = dict(height=84, width=84, segmentation=False)
    env = pixels.Wrapper(
        env, pixels_only=False, render_kwargs=render_kwargs, observation_key=rgb_key
    )

    env = FrameStackWrapper(env, frame_stack, frame_keys)
    env = ExtendedTimeStepWrapper(env, has_success_metric=True)
    return env
