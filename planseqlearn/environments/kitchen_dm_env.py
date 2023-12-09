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
    NoisyMaskWrapper,
    RandomCameraWrapper,
    SegmentationFilter,
    SegmentationToRobotMaskWrapper,
    SlimMaskWrapper,
    StackRGBAndMaskWrapper,
    Wrist_Camera_Render_Wrapper,
    get_env_action_spec,
    get_env_observation_spec,
)
from planseqlearn.mnm.kitchen_mp_env import KitchenMPEnv
from d4rl.kitchen.env_dict import ALL_KITCHEN_ENVIRONMENTS


class CombinedPhysicsWrapper:
    def __init__(self, physics1, physics2):
        self.physics1 = physics1
        self.physics2 = physics2

    def render(self, **render_kwargs):
        """
        Render from both physics, and concatenate across channel dimension
        """
        render1 = self.physics1.render(**render_kwargs)
        render2 = self.physics2.render(**render_kwargs)
        return np.concatenate([render1, render2], axis=-1)


class Kitchen_Wrapper(dm_env.Environment):
    def __init__(
        self,
        env_name: str,
        discount=1.0,
        seed=None,
        path_length=280,
        psl=False,
        camera_name="fixed",
        use_eef=True,
    ):
        self.discount = discount
        if use_eef:
            env_kwargs=dict(
                dense=False, # test also with dense = False
                image_obs=True,
                action_scale=1,
                control_mode="end_effector",
                frame_skip=40,
                max_path_length=path_length,
            )
            # do preprocessing 
            self._env = ALL_KITCHEN_ENVIRONMENTS[env_name](**env_kwargs)
        else:
            self._env = gym.make(env_name)
        if psl:
            self._env = KitchenMPEnv(
                self._env,
                env_name,
                use_vision_pose_estimation=False,
                teleport_instead_of_mp=True,
                use_joint_space_mp=False,
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
        elif camera_name == "wrist+fixed":
            wrist_physics = Wrist_Camera_Render_Wrapper(
                self._env.sim,
                lookat=[-0.3, 0.5, 2.0],
                distance=1.86,
                azimuth=90,
                elevation=-60,
            )
            self.physics = CombinedPhysicsWrapper(
                wrist_physics, self.fixed_view_physics
            )
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
    path_length=280,
    psl=False,
    use_eef=True,
):
    assert camera_name in ["wrist", "fixed"]

    env = Kitchen_Wrapper(
        env_name=name,
        discount=discount,
        seed=seed,
        path_length=path_length,
        psl=psl,
        camera_name=camera_name,
        use_eef=use_eef,
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
