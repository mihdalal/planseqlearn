import dm_env
import gym
import numpy as np
import collections
from dm_control.suite.wrappers import action_scale
from dm_env import specs

from planseqlearn.environments.wrappers import (
    ActionDTypeWrapper,
    ActionRepeatWrapper,
    ExtendedTimeStepWrapper,
    FrameStackWrapper,
)

from planseqlearn.psl.mopa_mp_env import MoPAPSLEnv
from mopa_rl.config.default_configs import (
    LIFT_OBSTACLE_CONFIG,
    LIFT_CONFIG,
    ASSEMBLY_OBSTACLE_CONFIG,
    PUSHER_OBSTACLE_CONFIG,
)


class MoPA_Env_Wrapper(dm_env.Environment):
    def __init__(
        self,
        env_name,
        text_plan,
        discount=1.0,
        horizon=100,
        psl=False,
        use_sam_segmentation=False,
        use_mp=False,
        use_vision_pose_estimation=False,
    ):
        if env_name == "SawyerLift-v0":
            config = LIFT_CONFIG
            if psl:
                config["camera_name"] = "eye_in_hand"
        elif env_name == "SawyerLiftObstacle-v0":
            config = LIFT_OBSTACLE_CONFIG
            if psl:
                config["camera_name"] = "eye_in_hand"
        elif env_name == "SawyerAssemblyObstacle-v0":
            config = ASSEMBLY_OBSTACLE_CONFIG
            if psl:
                config["camera_name"] = "eye_in_hand"
        elif env_name == "SawyerPushObstacle-v0":
            config = PUSHER_OBSTACLE_CONFIG
            if psl:
                config["camera_name"] = "eye_in_hand"
        config['max_episode_steps'] = horizon
        env = gym.make(**config)
        ik_env = gym.make(**config)
        self.name = env_name
        self.horizon = horizon
        self._env = MoPAPSLEnv(
            env,
            env_name,
            ik_env=ik_env,
            use_vision_pose_estimation=use_vision_pose_estimation,
            teleport_instead_of_mp=not use_mp,
            config=config,
            text_plan=text_plan,
            use_sam_segmentation=use_sam_segmentation,
        )
        self.discount = discount
        self.dummy_img = self._env.get_image()

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        self.current_step = 0
        observation = self._env.reset()
        new_observation = collections.OrderedDict()
        new_observation["state"] = observation.copy()
        new_observation["pixels"] = self._env.get_image()
        return dm_env.restart(new_observation)

    def step(self, action) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()
        observation, reward, _, info = self._env.step(action)
        self.current_step += 1
        reward_dict = {"reward": reward, "success": self._env._check_success()}
        new_observation = collections.OrderedDict()
        new_observation["state"] = observation.copy()
        new_observation["pixels"] = self._env.get_image()
        if self.current_step == self.horizon:
            self._reset_next_step = True
            return dm_env.truncation(reward_dict, new_observation, self.discount)
        return dm_env.transition(reward_dict, new_observation, self.discount)

    def observation_spec(self) -> specs.BoundedArray:
        proprio_spec = specs.BoundedArray(
            shape=self._env._wrapped_env.observation_space.low.shape,
            dtype=np.float64,
            minimum=self._env._wrapped_env.observation_space.low,
            maximum=self._env._wrapped_env.observation_space.high,
        )
        spec = collections.OrderedDict()
        spec["state"] = proprio_spec
        pixels_spec = specs.Array(
            shape=self.dummy_img.shape, dtype=np.uint8, name="pixels"
        )
        spec["pixels"] = pixels_spec

        self._observation_spec = spec
        return self._observation_spec

    def action_spec(self) -> specs.BoundedArray:
        spec = specs.BoundedArray(
            shape=self._env._wrapped_env.action_space.low.shape,
            dtype=np.float64,
            minimum=-1 * np.ones_like(self._env._wrapped_env.action_space.low),
            maximum=1 * np.ones_like(self._env._wrapped_env.action_space.high),
            name="action",
        )

        self._action_spec = spec
        return self._action_spec

    def render(self):
        return self._env.get_image()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_mopa(
    name,
    frame_stack,
    action_repeat,
    seed,
    text_plan,
    horizon=100,
    psl=True,
    use_sam_segmentation=False,
    use_mp=False,
    use_vision_pose_estimation=False,
):
    np.random.seed(seed)
    env = MoPA_Env_Wrapper(
        name, 
        horizon=horizon, 
        psl=psl,
        use_sam_segmentation=use_sam_segmentation,
        text_plan=text_plan,
        use_mp=use_mp,
        use_vision_pose_estimation=use_vision_pose_estimation,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(
        env,
        action_repeat,
        use_metaworld_reward_dict=True,
    )
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = FrameStackWrapper(env, frame_stack, ["pixels"])
    env = ExtendedTimeStepWrapper(env, has_success_metric=True)
    return env
