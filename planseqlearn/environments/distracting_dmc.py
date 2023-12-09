import numpy as np
from distracting_control import suite
from dm_control.suite.wrappers import action_scale, pixels

from planseqlearn.environments.wrappers import (
    ActionDTypeWrapper,
    ActionRepeatWrapper,
    ExtendedTimeStepWrapper,
    FrameStackWrapper,
    NoisyMaskWrapper,
    SegmentationToRobotMaskWrapper,
    SlimMaskWrapper,
    StackRGBAndMaskWrapper,
)

name2robot_seg_ids = {
    "cup_catch": list(range(1, 7)),
    "cartpole_swingup": list(range(3, 5)),
    "cheetah_run": list(range(1, 9)),
    "finger_spin": list(range(1, 5)),
    "reacher_easy": [5, 7, 8, 9],
    "walker_walk": list(range(1, 8)),
}


def make_distracting_dmc(
    name,
    frame_stack,
    action_repeat,
    seed,
    add_segmentation_to_obs,
    difficulty,
    distraction_types,
    background_dataset_path,
    background_dataset_videos,
    noisy_mask_drop_prob,
    use_rgbm=False,
    slim_mask_cfg=None,
):
    domain, task = name.split("_", 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup="ball_in_cup").get(domain, domain)
    env = suite.load(
        domain,
        task,
        difficulty=difficulty,
        distraction_types=distraction_types,
        background_dataset_path=background_dataset_path,
        background_dataset_videos=background_dataset_videos,
        task_kwargs={"random": seed},
        visualize_reward=False,
        distraction_seed=seed,
        from_pixels=False,
    )

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat, use_metaworld_reward_dict=False)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    camera_id = dict(quadruped=2).get(domain, 0)

    frame_keys = []

    rgb_key = "pixels"
    frame_keys.append(rgb_key)
    render_kwargs = dict(height=84, width=84, camera_id=camera_id)
    env = pixels.Wrapper(
        env, pixels_only=False, render_kwargs=render_kwargs, observation_key=rgb_key
    )

    if add_segmentation_to_obs:
        segmentation_key = "segmentation"
        frame_keys.append(segmentation_key)
        segmentation_kwargs = dict(
            height=84, width=84, camera_id=camera_id, segmentation=True
        )
        env = pixels.Wrapper(
            env,
            pixels_only=False,
            render_kwargs=segmentation_kwargs,
            observation_key=segmentation_key,
        )
        env.robot_segmentation_ids = name2robot_seg_ids[name]
        env = SegmentationToRobotMaskWrapper(
            env,
            segmentation_key,
            types_channel=1,
        )

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
    env = ExtendedTimeStepWrapper(env, has_success_metric=False)

    return env
