import collections
import cv2

import dm_env
import numpy as np
from dm_control.suite.wrappers import action_scale
from dm_env import specs

from planseqlearn.environments.wrappers import (
    ActionDTypeWrapper,
    ActionRepeatWrapper,
    ExtendedTimeStepWrapper,
    FrameStackWrapper,
)
import robosuite as suite
from planseqlearn.psl.robosuite_mp_env import RobosuitePSLEnv


def get_proprioceptive_spec(spec, num_proprioceptive_features):
    return specs.BoundedArray(
        shape=(num_proprioceptive_features,),
        dtype=spec.dtype,
        minimum=spec.minimum[0:num_proprioceptive_features],
        maximum=spec.maximum[0:num_proprioceptive_features],
        name="observation",
    )


class Robosuite_Wrapper(dm_env.Environment):
    def __init__(
        self,
        env_name: str,
        discount=1.0,
        camera_names=None,
        psl=False,
        path_length=500,
        vertical_displacement=0.08,
        estimate_orientation=True,
        valid_obj_names=None,
        use_proprio=True,
        use_sam_segmentation=False,
    ):
        self.discount = discount
        self.env_name = env_name
        self.use_proprio = use_proprio
        controller_configs = dict(
            type="OSC_POSE",
            input_max=1,
            input_min=-1,
            output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            kp=150,
            damping=1,
            impedance_mode="fixed",
            kp_limits=[0, 300],
            damping_limits=[0, 10],
            position_limits=None,
            orientation_limits=None,
            uncouple_pos_ori=True,
            control_delta=True,
            interpolation=None,
            ramp_ratio=0.2,
        )
        if env_name == "NutAssembly":
            reward_scale = 2.0
            extra_kwargs = dict()
        elif env_name == "PickPlace":
            reward_scale = 4.0
            extra_kwargs = dict(
                valid_obj_names=valid_obj_names,
            )
        else:
            reward_scale = 1.0
            extra_kwargs = dict()
        self._env = suite.make(
            robots="Panda",
            reward_shaping=True,
            control_freq=20,
            ignore_done=True,
            env_name=env_name,
            controller_configs=controller_configs,
            horizon=path_length,
            camera_widths=84,
            camera_heights=84,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=camera_names,
            hard_reset=False,
            reward_scale=reward_scale,
            **extra_kwargs,
        )
        # Base dictionary for common parameters
        mp_env_kwargs = dict(
            teleport_instead_of_mp=True,
            mp_bounds_low=(-1.45, -1.25, 0.45),
            mp_bounds_high=(0.45, 0.85, 2.25),
            backtrack_movement_fraction=0.001,
            grip_ctrl_scale=0.0025,
            planning_time=20,
            controller_configs=controller_configs,
            use_vision_pose_estimation=False,
            use_vision_placement_check=False,
            use_vision_grasp_check=False,
            estimate_orientation=False,
            use_sam_segmentation=use_sam_segmentation,
        )

        if env_name == "Lift":
            mp_env_kwargs.update(
                dict(
                    vertical_displacement=0.04,
                    mp_bounds_low=(-1.45, -1.25, 0.8),
                )
            )
        elif env_name.startswith("PickPlace"):
            mp_env_kwargs.update(
                dict(
                    vertical_displacement=vertical_displacement,
                    estimate_orientation=estimate_orientation,
                )
            )
        elif env_name.startswith("NutAssembly"):
            mp_env_kwargs.update(
                dict(
                    vertical_displacement=0.02,
                    estimate_orientation=True,
                )
            )
        elif env_name.startswith("Door"):
            mp_env_kwargs.update(
                dict(
                    vertical_displacement=0.06,
                )
            )

        if psl:
            self._env = RobosuitePSLEnv(self._env, env_name, **mp_env_kwargs)
        self._reset_next_step = True
        self.psl = psl
        self.current_step = 0
        self._observation_spec = None
        self._action_spec = None
        self.NUM_PROPRIOCEPTIVE_STATES = 7
        self.camera_names = camera_names
        self.camera_widths = [84]
        self.camera_heights = [84]
        self.add_cameras()

    def add_cameras(self):
        for cam_name in ["frontview"]:
            # Add cameras associated to our arrays
            cam_sensors, _ = self._create_camera_sensors(
                cam_name,
                cam_w=960,
                cam_h=540,
                cam_d=False,
                cam_segs=None,
                modality="image",
            )
            self.cam_sensor = cam_sensors

    def render(self):
        im = self.cam_sensor[0](None)
        im = cv2.flip(im, 0)
        return im

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        self.current_step = 0
        observation = self._env.reset()
        new_observation = collections.OrderedDict()
        if self.use_proprio:
            new_observation["state"] = np.concatenate(
                (observation["robot0_eef_pos"], observation["robot0_eef_quat"])
            )
        new_observation["pixels"] = observation[f"{self.camera_names[0]}_image"]
        return dm_env.restart(new_observation)

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        observation, reward, d, info = self._env.step(action)
        self.current_step += 1

        reward_dict = {"reward": reward, "success": self._env._check_success()}
        new_observation = collections.OrderedDict()
        if self.use_proprio:
            new_observation["state"] = np.concatenate(
                (observation["robot0_eef_pos"], observation["robot0_eef_quat"])
            )
        new_observation["pixels"] = observation[f"{self.camera_names[0]}_image"]

        if self.current_step == self._env.horizon or d:
            self._reset_next_step = True
            return dm_env.truncation(reward_dict, new_observation, self.discount)
        return dm_env.transition(reward_dict, new_observation, self.discount)

    def observation_spec(self) -> specs.BoundedArray:
        if self._observation_spec is not None:
            return self._observation_spec
        spec = collections.OrderedDict()
        if self.use_proprio:
            proprio_spec = specs.BoundedArray(
                shape=(self.NUM_PROPRIOCEPTIVE_STATES,),
                dtype=self._env.observation_spec()["robot0_eef_pos"].dtype,
                minimum=-1 * np.ones(self.NUM_PROPRIOCEPTIVE_STATES),
                maximum=1 * np.ones(self.NUM_PROPRIOCEPTIVE_STATES),
                name="observation",
            )
            spec["state"] = proprio_spec

        pixels_obs_space = self._env.observation_spec()[f"{self.camera_names[0]}_image"]
        pixels_spec = specs.Array(
            shape=pixels_obs_space.shape, dtype=np.uint8, name="pixels"
        )
        spec["pixels"] = pixels_spec

        self._observation_spec = spec
        return self._observation_spec

    def action_spec(self) -> specs.BoundedArray:
        if self._action_spec is not None:
            return self._action_spec
        spec = specs.BoundedArray(
            shape=self._env.action_spec[0].shape,
            dtype=self._env.action_spec[0].dtype,
            minimum=self._env.action_spec[0],
            maximum=self._env.action_spec[1],
            name="action",
        )

        self._action_spec = spec
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_robosuite(
    name,
    frame_stack,
    action_repeat,
    discount,
    camera_name,
    psl=True,
    path_length=500,
    vertical_displacement=0.08,
    estimate_orientation=True,
    valid_obj_names=None,
    use_proprio=True,
    use_sam_segmentation=False,
):
    env = Robosuite_Wrapper(
        env_name=name,
        discount=discount,
        camera_names=[camera_name],
        psl=psl,
        path_length=path_length,
        vertical_displacement=vertical_displacement,
        estimate_orientation=estimate_orientation,
        valid_obj_names=valid_obj_names,
        use_proprio=use_proprio,
        use_sam_segmentation=use_sam_segmentation,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat, use_metaworld_reward_dict=True)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    frame_keys = []

    rgb_key = "pixels"
    frame_keys.append(rgb_key)
    env = FrameStackWrapper(env, frame_stack, frame_keys)
    env = ExtendedTimeStepWrapper(env, has_success_metric=True)
    return env
