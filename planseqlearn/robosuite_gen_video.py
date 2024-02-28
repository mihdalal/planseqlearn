import mujoco_py
import argparse 
from planseqlearn.environments.robosuite_dm_env import make_robosuite
from planseqlearn.environments.metaworld_dm_env import make_metaworld
from planseqlearn.environments.mopa_dm_env import make_mopa
from planseqlearn.environments.kitchen_dm_env import make_kitchen
import torch 
import numpy as np
from rlkit.torch.model_based.dreamer.visualization import make_video
from planseqlearn.psl.env_text_plans import *
import pickle 

def robosuite_gen_video(env_name, camera_name):
    # create environment 
    agent = torch.load(f"psl_policies/robosuite/{env_name}.pt")["agent"]
    rs_name = env_name 
    if env_name == "PickPlaceCanBread" or env_name == "PickPlaceCerealMilk":
        rs_name = "PickPlace"
    env = make_robosuite(
        name=rs_name,
        frame_stack=3,
        action_repeat=1,
        discount=1.0,
        camera_name=camera_name,
        psl=True,
        use_mp=False,
        use_sam_segmentation=True,
        use_vision_pose_estimation=False,
        text_plan=ROBOSUITE_PLANS[env_name],
        vertical_displacement=0.04,
        estimate_orientation=True,
    )
    frames = []
    clean_frames = []
    np.random.seed(0)
    o = env.reset()
    with torch.no_grad():
        for _ in range(100):
            act = agent.act(o.observation, step=_, eval_mode=True)
            o = env.step(act)
            frames.append(env.get_image())
            if o.last(): # or o.reward["success"]:
                break 
    frames = list(map(lambda x: x[:, :, ::-1], frames))
    make_video(frames, "videos", 0, use_wandb=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help='Name of the environment')
    parser.add_argument('--camera_name', type=str, help='Name of the environment')   
    robosuite_gen_video(args.env_name, args.camera_name)  