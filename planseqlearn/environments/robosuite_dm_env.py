import collections
import random
import cv2

import dm_env
import metaworld
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import specs

from planseqlearn.environments.metaworld_custom_envs import (
    MT3_Customized,
    MT5_Customized,
    MT10_Customized,
    MT_Door,
)
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
import robosuite as suite
from planseqlearn.mnm.robosuite_mp_env import RobosuiteMPEnv


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
        seed=None,
        camera_names=None,
        psl=False,
        path_length=500,
        vertical_displacement=0.08,
        hardcoded_orientations=True,
        valid_obj_names=None,
        steps_of_high_level_plan_to_complete=-1,
        use_proprio=True,
        use_fixed_plus_wrist_view=False,
        pose_sigma=0.0,
        noisy_pose_estimates=False,
        hardcoded_high_level_plan=True,
    ):
        self.discount = discount
        self.env_name = env_name
        self.use_proprio = use_proprio
        self.use_fixed_plus_wrist_view = use_fixed_plus_wrist_view
        if self.use_fixed_plus_wrist_view:
            camera_names = ["robot0_eye_in_hand", "agentview"]
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
        if env_name == "Lift":
            mp_env_kwargs = dict(
                vertical_displacement=0.04,
                teleport_instead_of_mp=True,
                randomize_init_target_pos=False,
                mp_bounds_low=(-1.45, -1.25, 0.8),
                mp_bounds_high=(0.45, 0.85, 2.25),
                backtrack_movement_fraction=0.001,
                clamp_actions=True,
                update_with_true_state=True,
                grip_ctrl_scale=0.0025,
                planning_time=20,
                verify_stable_grasp=True,
                hardcoded_high_level_plan=True,
                num_ll_actions_per_hl_action=50,
                controller_configs=controller_configs,
                use_vision_pose_estimation=False,
                use_vision_placement_check=False,
                use_vision_grasp_check=False,
                hardcoded_orientations=False,
            )
        elif env_name.startswith("PickPlace"):
            mp_env_kwargs = dict(
                vertical_displacement=vertical_displacement,
                teleport_instead_of_mp=True,
                randomize_init_target_pos=False,
                mp_bounds_low=(-1.45, -1.25, 0.45),
                mp_bounds_high=(0.45, 0.85, 2.25),
                backtrack_movement_fraction=0.001,
                clamp_actions=True,
                update_with_true_state=True,
                grip_ctrl_scale=0.0025,
                planning_time=20,
                hardcoded_high_level_plan=hardcoded_high_level_plan,
                terminate_on_success=False,
                plan_to_learned_goals=False,
                reset_at_grasped_state=False,
                verify_stable_grasp=True,
                hardcoded_orientations=hardcoded_orientations,
                num_ll_actions_per_hl_action=50,
                controller_configs=controller_configs,
                steps_of_high_level_plan_to_complete=steps_of_high_level_plan_to_complete,
                use_vision_pose_estimation=False,
                use_vision_placement_check=False,
                use_vision_grasp_check=False,
                pose_sigma=pose_sigma,
                noisy_pose_estimates=noisy_pose_estimates,
            )
        elif env_name.startswith("NutAssembly"):
            mp_env_kwargs = dict(
                vertical_displacement=0.04,
                teleport_instead_of_mp=True,
                randomize_init_target_pos=False,
                mp_bounds_low=(-1.45, -1.25, 0.45),
                mp_bounds_high=(0.45, 0.85, 2.25),
                backtrack_movement_fraction=0.001,
                clamp_actions=True,
                update_with_true_state=True,
                grip_ctrl_scale=0.0025,
                planning_time=20,
                hardcoded_high_level_plan=True,
                terminate_on_success=False,
                plan_to_learned_goals=False,
                reset_at_grasped_state=False,
                verify_stable_grasp=True,
                hardcoded_orientations=True,
                num_ll_actions_per_hl_action=50,
                controller_configs=controller_configs,
                use_vision_pose_estimation=False,
                use_vision_placement_check=False,
                use_vision_grasp_check=False,
            )
        elif env_name.startswith("Door"):
            mp_env_kwargs = dict(
                vertical_displacement=0.06,
                teleport_instead_of_mp=True,
                randomize_init_target_pos=False,
                mp_bounds_low=(-1.45, -1.25, 0.45),
                mp_bounds_high=(0.45, 0.85, 2.25),
                backtrack_movement_fraction=0.001,
                clamp_actions=True,
                update_with_true_state=True,
                grip_ctrl_scale=0.0025,
                planning_time=20,
                verify_stable_grasp=True,
                hardcoded_high_level_plan=True,
                use_teleports_in_step=False,
                num_ll_actions_per_hl_action=50,
                controller_configs=controller_configs,
                use_vision_pose_estimation=False,
                use_vision_placement_check=False,
                use_vision_grasp_check=False,
                hardcoded_orientations=False,
            )
        if psl:
            self._env = RobosuiteMPEnv(self._env, env_name, **mp_env_kwargs)
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
        if self.use_fixed_plus_wrist_view:
            new_observation["pixels"] = np.concatenate(
                (
                    observation[f"{self.camera_names[0]}_image"],
                    observation[f"{self.camera_names[0]}_image"],
                ),
                axis=2,
            )
        else:
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
        # new_observation['pixels'] = cv2.flip(observation[f'{self.camera_names[0]}_image'], 0)[:, :, ::-1]
        if self.use_fixed_plus_wrist_view:
            new_observation["pixels"] = np.concatenate(
                (
                    observation[f"{self.camera_names[0]}_image"],
                    observation[f"{self.camera_names[0]}_image"],
                ),
                axis=2,
            )
        else:
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
        if self.use_fixed_plus_wrist_view:
            pixels_obs_space = np.concatenate(
                (pixels_obs_space, pixels_obs_space), axis=2
            )
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
    seed,
    camera_name,
    add_segmentation_to_obs,
    noisy_mask_drop_prob,
    use_rgbm=None,
    slim_mask_cfg=None,
    psl=True,
    path_length=500,
    vertical_displacement=0.08,
    hardcoded_orientations=True,
    valid_obj_names=None,
    steps_of_high_level_plan_to_complete=-1,
    use_proprio=True,
    use_fixed_plus_wrist_view=False,
    pose_sigma=0,
    noisy_pose_estimates=False,
    hardcoded_high_level_plan=True,
):
    env = Robosuite_Wrapper(
        env_name=name,
        discount=discount,
        seed=seed,
        camera_names=[camera_name],
        psl=psl,
        path_length=path_length,
        vertical_displacement=vertical_displacement,
        hardcoded_orientations=hardcoded_orientations,
        valid_obj_names=valid_obj_names,
        steps_of_high_level_plan_to_complete=steps_of_high_level_plan_to_complete,
        use_proprio=use_proprio,
        use_fixed_plus_wrist_view=use_fixed_plus_wrist_view,
        pose_sigma=pose_sigma,
        noisy_pose_estimates=noisy_pose_estimates,
        hardcoded_high_level_plan=hardcoded_high_level_plan,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat, use_metaworld_reward_dict=True)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    frame_keys = []

    rgb_key = "pixels"
    frame_keys.append(rgb_key)
    # render_kwargs = dict(height=84,
    #                      width=84,
    #                      mode="offscreen",
    #                      camera_name=camera_name)
    # env = pixels.Wrapper(env,
    #                      pixels_only=False,
    #                      render_kwargs=render_kwargs,
    #                      observation_key=rgb_key)

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
