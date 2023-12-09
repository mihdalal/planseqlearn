import random
import mujoco_py
import dm_env
import metaworld
import gym
import numpy as np
import collections
import matplotlib.pyplot as plt
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

# rlkit imports
# from rlkit.mopa.mopa_env import MoPAMPEnv, get_site_pose
from planseqlearn.mnm.mopa_mp_env import MoPAMPEnv
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
        discount=1.0,
        seed=None,
        proprioceptive_state=True,
        horizon=100,
        psl=False,
        is_eval=False,
    ):
        if env_name == "SawyerLift-v0":
            config = LIFT_CONFIG
            if psl:
                config["camera_name"] = "eye_in_hand"
        elif env_name == "SawyerLiftObstacle-v0":
            config = LIFT_OBSTACLE_CONFIG
            if psl:
                config["camera_name"] = "eye_in_hand"
            # if is_eval:
            #     config["camera_name"] = "visview"
        elif env_name == "SawyerAssemblyObstacle-v0":
            config = ASSEMBLY_OBSTACLE_CONFIG
            if psl:
                config["camera_name"] = "eye_in_hand"
        elif env_name == "SawyerPushObstacle-v0":
            config = PUSHER_OBSTACLE_CONFIG
            if psl:
                config["camera_name"] = "eye_in_hand"
        env = gym.make(**config)
        ik_env = gym.make(**config)
        self.name = env_name
        self.horizon = horizon
        self._env = MoPAMPEnv(
            env,
            env_name,
            ik_env=ik_env,
            teleport_on_grasp=True,
            use_vision_pose_estimation=True,
            teleport_instead_of_mp=True,
            config=config,
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
    discount,
    seed,
    horizon=100,
    psl=True,
    is_eval=False,
):
    np.random.seed(seed)
    env = MoPA_Env_Wrapper(name, horizon=horizon, psl=psl, is_eval=is_eval)
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
