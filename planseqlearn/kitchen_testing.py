"""
Todo list
- start testing local script on kitchen
- implement motion planning for kitchen
- fix nut assembly square + pick place full
"""
import numpy as np 
from d4rl.kitchen.env_dict import ALL_KITCHEN_ENVIRONMENTS
import matplotlib.pyplot as plt 
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
def save_img(env, filename):
    fixed_view_physics = Camera_Render_Wrapper(
        env.sim,
        lookat=[-0.3, 0.5, 2.0],
        distance=1.86,
        azimuth=90,
        elevation=-60,
    )
    frame = fixed_view_physics.render(height=500, width=500)
    plt.imshow(frame)
    plt.savefig(filename)

def test_hinge(env_kwargs):
    env = ALL_KITCHEN_ENVIRONMENTS['kitchen-hinge-v0'](**env_kwargs)
    
    o = env.reset()
    save_img(env, "0.png")
    env.lift(1)
    save_img(env, "1.png")
    env.angled_x_y_grasp(np.array([-np.pi / 6, -0.3, 1.4, 0]))
    save_img(env, "2.png")
    env.move_delta_ee_pose(np.array([0.5, -1, 0]))
    save_img(env, "3.png")
    env.rotate_about_x_axis(np.array([1]))
    save_img(env, "4.png")
    env.rotate_about_x_axis(np.array([0]))
    save_img(env, "5.png")
    o, r, d, i = env.step(np.zeros_like(env.action_space.low))
    print(f"Success: {i['hinge cabinet success']}")

def test_kettle(env_kwargs):
    env = ALL_KITCHEN_ENVIRONMENTS['kitchen-kettle-v0'](**env_kwargs)
    o = env.reset()
    save_img(env, "0.png")
    env.drop(0.5)
    save_img(env, "1.png")
    env.angled_x_y_grasp(np.array([0, 0.15, 0.7, 1]))
    save_img(env, "2.png")
    env.move_delta_ee_pose(np.array([0.25, 0.7, 0.25]))
    save_img(env, "3.png")
    env.drop(0.25)
    save_img(env, "4.png")
    env.open_gripper(1)
    save_img(env, "5.png")
    o, r, d, i = env.step(np.zeros_like(env.action_space.low))
    print(f"Success: {i['kettle success']}")

def test_light_switch(env_kwargs):
    env = ALL_KITCHEN_ENVIRONMENTS['kitchen-light-v0'](**env_kwargs)
    o = env.reset()
    save_img(env, "0.png")
    env.close_gripper(1)
    save_img(env, "1.png")
    env.lift(0.6)
    save_img(env, "2.png")
    env.move_right(0.45)
    save_img(env, "3.png")
    env.move_forward(1.25)
    save_img(env, "4.png")
    env.move_left(0.45)
    save_img(env, "5.png")
    o, r, d, i = env.step(np.zeros_like(env.action_space.low))
    print(f"Success: {i['light switch success']}")

def test_microwave(env_kwargs):
    env = ALL_KITCHEN_ENVIRONMENTS['kitchen-microwave-v0'](**env_kwargs)
    o = env.reset()
    save_img(env, "0.png")
    env.drop(0.55)
    save_img(env, "1.png")
    env.angled_x_y_grasp(np.array([-np.pi / 6, -0.3, 0.95, 1]))
    save_img(env, "2.png")
    env.move_backward(0.7)
    save_img(env, "3.png")
    o, r, d, i = env.step(np.zeros_like(env.action_space.low))
    print(f"Success: {i['microwave success']}")

def test_top_burner(env_kwargs):
    env = ALL_KITCHEN_ENVIRONMENTS['kitchen-tlb-v0'](**env_kwargs)
    o = env.reset()
    save_img(env, "0.png")
    env.lift(0.6)
    save_img(env, "1.png")
    env.angled_x_y_grasp(np.array([0, 0.5, 1, 1]))
    save_img(env, "2.png")
    env.rotate_about_y_axis(-np.pi / 4)
    save_img(env, "3.png")
    o, r, d, i = env.step(np.zeros_like(env.action_space.low))
    print(f"Success: {i['top burner success']}")

def test_top_right_burner(env_kwargs):
    env = ALL_KITCHEN_ENVIRONMENTS['kitchen-trb-v0'](**env_kwargs)
    o = env.reset()
    save_img(env, "0.png")
    env.lift(0.6)
    save_img(env, "1.png")
    env.angled_x_y_grasp(np.array([0, 1, 1, 1]))
    save_img(env, "2.png")
    env.rotate_about_y_axis(-np.pi / 4)
    save_img(env, "3.png")
    o, r, d, i = env.step(np.zeros_like(env.action_space.low))
    print(f"Success: {i['top right burner success']}")
    print(f"Reward: {r}")
    print(f"Qpos: {env.sim.data.qpos[11:17]}")

def test_bottom_left_burner(env_kwargs):
    env = ALL_KITCHEN_ENVIRONMENTS['kitchen-blb-v0'](**env_kwargs)
    o = env.reset()
    save_img(env, "0.png")
    env.goto_pose(env.get_site_xpos("brbhandle")+ np.array([0, -0.05, -.125]))
    save_img(env, "1.png")
    env.angled_x_y_grasp(np.array([0.0, .0, 0.0, 1]))
    save_img(env, "2.png")
    env.rotate_about_y_axis(-np.pi / 4)
    save_img(env, "3.png")
    o, r, d, i = env.step(np.zeros_like(env.action_space.low))
    print(f"Reward: {r}")
    print(f"Qpos: {env.sim.data.qpos[11:17]}")

def test_bottom_right_burner(env_kwargs):
    env = ALL_KITCHEN_ENVIRONMENTS['kitchen-brb-v0'](**env_kwargs)
    o = env.reset()
    save_img(env, "0.png")
    env.goto_pose(env.get_site_xpos("brbhandle")+ np.array([.025, -0.05, -.125]))
    save_img(env, "1.png")
    env.angled_x_y_grasp(np.array([0.0, .0, 0.0, 1]))
    save_img(env, "2.png")
    prev_qpos = env.sim.data.qpos.copy()
    env.rotate_about_y_axis(-np.pi / 3)
    save_img(env, "3.png")
    o, r, d, i = env.step(np.zeros_like(env.action_space.low))
    print(f"Reward: {r}")
    print(f"Qpos: {env.sim.data.qpos}")
    print(f"qpos_diff: {env.sim.data.qpos[7:] - prev_qpos[7:]}")

def test_make_kitchen_dm_env():
    from planseqlearn.environments.kitchen_dm_env import make_kitchen
    from tqdm import tqdm
    env = make_kitchen(
        name="tlb-v0",
        frame_stack=1,
        action_repeat=1,
        discount=1.0,
        seed=0,
        camera_name="wrist",
        add_segmentation_to_obs=False,
        noisy_mask_drop_prob=0.0,
        use_rgbm=False,
        slim_mask_cfg=None,
        path_length=280,
        mprl=True,
        use_eef=True,
    )
    o = env.reset()
    # test ability to move to generally correct position
    print(f"Initial ee pos: {env.get_ee_pose()}")
    for _ in tqdm(range(10)):
        o = env.step(np.array([1.0, 0., -1.0, 0., 0., 0., 1.0]))
    print(f"End ee pos: {env.get_ee_pose()}")
    frame = np.transpose(o.observation['pixels'], (1, 2, 0))
    plt.imshow(frame)
    plt.savefig("reset.png")

if __name__ == "__main__":
    env_kwargs = dict(
        dense=False,
        image_obs=True,
        action_scale=1.0,
        control_mode="primitives",
        frame_skip=40,
        use_workspace_limits=True,
        usage_kwargs=dict(
            use_dm_backend=True,
            use_raw_action_wrappers=False,
            use_image_obs=True,
            max_path_length=280,
            unflatten_images=False,
        ),
        image_kwargs=dict(),
    )
    usage_kwargs = env_kwargs["usage_kwargs"]
    image_kwargs = env_kwargs["image_kwargs"]
    max_path_length = usage_kwargs["max_path_length"]
    use_dm_backend = usage_kwargs.get("use_dm_backend", True)
    use_raw_action_wrappers = usage_kwargs.get("use_raw_action_wrappers", False)
    use_image_obs = usage_kwargs.get("use_image_obs", True)
    unflatten_images = usage_kwargs.get("unflatten_images", False)
    env_kwargs_new = env_kwargs.copy()
    if "usage_kwargs" in env_kwargs_new:
        del env_kwargs_new["usage_kwargs"]
    if "image_kwargs" in env_kwargs_new:
        del env_kwargs_new["image_kwargs"]
    # uncomment which one you want to test 
    # test_hinge(env_kwargs_new)
    # test_microwave(env_kwargs_new)
    # test_light_switch(env_kwargs_new)
    # test_top_burner(env_kwargs_new)
    # test_kettle(env_kwargs_new)
    test_bottom_right_burner(env_kwargs_new)
    
    