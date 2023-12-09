import mujoco_py
import numpy as np
import robosuite as suite 
import torch 
from matplotlib import pyplot as plt 
from robosuite.utils.transform_utils import * 
from robosuite.wrappers.gym_wrapper import GymWrapper 
from tqdm import tqdm 
from planseqlearn.mnm.robosuite_mp_env import RobosuiteMPEnv 
import rlkit.torch.pytorch_util as ptu 
from rlkit.mprl.mp_env import * 
from rlkit.torch.model_based.dreamer.visualization import make_video 
from planseqlearn.environments.robosuite_dm_env import make_robosuite 
def get_object_pose(env, obj_idx=0): 
    name = env.name.split("_")[1] 
    if name.endswith("Lift"): 
        object_pos = 
        env.sim.data.qpos[9:12].copy() 
        object_quat = T.convert_quat(env.sim.data.qpos[12:16].copy(), to="xyzw") 
    elif name.startswith("PickPlaceMilk"): 
        object_pos = env.sim.data.qpos[9:12].copy() 
        object_quat = T.convert_quat(env.sim.data.qpos[12:16].copy(), to="xyzw") 
    elif name.startswith("PickPlaceBread"): 
        object_pos = env.sim.data.qpos[16:19].copy() 
        object_quat = T.convert_quat(env.sim.data.qpos[19:23].copy(), to="xyzw") 
    elif name.startswith("PickPlaceCereal"): 
        object_pos = env.sim.data.qpos[23:26].copy() 
        object_quat = T.convert_quat(env.sim.data.qpos[26:30].copy(), to="xyzw") 
    elif name.startswith("PickPlaceCan"): 
        object_pos = env.sim.data.qpos[30:33].copy() 
        object_quat = T.convert_quat(env.sim.data.qpos[33:37].copy(), to="xyzw") 
    elif name.endswith("PickPlace"): 
        new_obj_idx = compute_correct_obj_idx(env, obj_idx=obj_idx) 
        object_pos = env.sim.data.qpos[ 9 + 7 * new_obj_idx : 12 + 7 * new_obj_idx ].copy() 
        object_quat = T.convert_quat(env.sim.data.qpos[12 + 7 * new_obj_idx : 16 + 7 * new_obj_idx].copy(), to="xyzw",) 
    elif name.startswith("Door"): 
        object_pos = np.array( [env.sim.data.qpos[env.hinge_qpos_addr]] )  
        # this is not what they are, but they will be decoded properly 
        object_quat = np.array( [env.sim.data.qpos[env.handle_qpos_addr]] )  
        # this is not what they are, but they will be decoded properly 
    elif name.startswith("Wipe"): 
        object_pos = np.zeros(3) 
        object_quat = np.zeros(4) 
    elif "NutAssembly" in name: 
        if name.endswith("Square"): 
            nut = env.nuts[0] 
        elif name.endswith("Round"): 
            nut = env.nuts[1] 
        elif name.endswith("NutAssembly"): 
            nut = env.nuts[1 - obj_idx]  
        # first nut is round, second nut is square 
        nut_name = nut.name object_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[nut_name]]) 
        object_quat = T.convert_quat( env.sim.data.body_xquat[env.obj_body_id[nut_name]], to="xyzw" ) 
    else: 
        raise NotImplementedError() 
    return object_pos, object_quat 
if __name__ == "__main__": 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--a', type=bool) 
    args = parser.parse_args() 
    np.random.seed(1) 
    if args.a: 
        print(f"A") 
        env = make_robosuite(
            name="Door", 
            frame_stack=3, 
            action_repeat=1, 
            discount=1.0, 
            seed=0, 
            camera_name="agentview", 
            add_segmentation_to_obs=False, 
            noisy_mask_drop_prob=0.0, 
            use_old=True, 
        ) 
    else: 
        print(f"B") 
        env = make_robosuite( 
            name="Door", 
            frame_stack=3, 
            action_repeat=1, 
            discount=1.0, 
            seed=0, 
            camera_name="agentview", 
            add_segmentation_to_obs=False, 
            noisy_mask_drop_prob=0.0, 
            use_old=False, 
        ) 
    num_episodes = 3 
    total = 0 
    ptu.device = torch.device("cuda") 
    frames = [] 
    all_success_rates = [] 
    for seed in range(3): 
        env.intermediate_frames = [] 
        success_rate = 0 
        import torch 
        policy = torch.load('policies/door.pt') 
        agent = policy['agent'] 
        for s in tqdm(range(num_episodes)): 
            o = env.reset() 
            for i in range(40): 
                with torch.no_grad(): 
                    action = agent.act(o.observation, step=i, eval_mode=True) 
                    o = env.step(action) 
                    if o.last() or o.reward['success']: 
                        break 
                    frame = env.get_image() 
                    frames.append(frame) 
            print(env.sim.data.qpos) # load in policy 
            print(env._check_success()) # 
            plt.savefig(f"plots/{s}.png") 
            success_rate += env._check_success() 
            np.set_printoptions(suppress=True) 
            print("Running success rate: ", success_rate / (s + 1)) 
        print(f"Success Rate: {success_rate/num_episodes}") 
        all_success_rates.append(success_rate / num_episodes) 
    print("Mean success", np.mean(all_success_rates)) 
    print("Std success", np.std(all_success_rates)) 
    if args.a: 
        make_video(frames, "videos", 2, use_wandb=False) 
    else: 
        make_video(frames, "videos", 3, use_wandb=False)