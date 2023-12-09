import numpy as np
import mujoco_py 
from planseqlearn.environments.metaworld_dm_env import make_metaworld, ENV_CAMERA_DICT
from tqdm import tqdm
import argparse 
import matplotlib.pyplot as plt

def save_img(env, camera_name, filename, flip=False):
    frame = env.sim.render(width=500, camera_name=camera_name, height=500)
    if flip:
        frame = np.flipud(frame)
    plt.imshow(frame)
    plt.savefig(filename) 

def assembly(args, env):
    successes = 0
    for trial in range(args.num_trials):
        t = env.reset()
        for _ in range(5):
            t = env.step(np.array([0., 0., -1.0, 0.0]))
        for _ in range(10):
            t = env.step(np.array([0., 0., 0., 1.0]))
        for _ in range(8):
            t = env.step(np.array([0., 0., 1.0, 1.0]))
        for _ in range(25):
            t = env.step(np.array([0.2, 0., -1.0, 1.0]))
        for _ in range(15):
            t = env.step(np.array([0., 0., 0., -1.0]))
        successes += t.reward['success']
    print(f"Success rate: {successes/args.num_trials}")

def disassemble(args, env):
    successes = 0
    for trial in range(args.num_trials):
        t = env.reset()
        for _ in range(5):
            t = env.step(np.array([0., 0., -1.0, 0.0]))
        for _ in range(10):
            t = env.step(np.array([0., 0., 0., 1.0]))
        for _ in range(20):
            t = env.step(np.array([0., 0., 1.0, 1.0]))
        successes += t.reward['success']
    print(f"Success rate: {successes/args.num_trials}")

def bin_picking(args, env):
    successes = 0
    for trial in range(args.num_trials):
        t = env.reset()
        for _ in range(15):
            t = env.step(np.array([0., 0., 0., 1.0])) 
        for _ in range(10):
            t = env.step(np.array([0., 0., 1.0, 1.0]))
        for _ in range(15):
            t = env.step(np.array([0., 0., 0., -1.0])) 
        successes += t.reward['success']
    print(f"Success rate: {successes/args.num_trials}")

def peg_insert(args, env):
    successes = 0
    for trial in range(args.num_trials):
        t = env.reset()
        for _ in range(5):
            t = env.step(np.array([0., 0., -1.0, 0.]))
        for _ in range(15):
            t = env.step(np.array([0., 0., 0., 1.0])) 
        for _ in range(50):
            t = env.step(np.array([-1.0, 0., 0., 1.0]))
        successes += t.reward['success']
    print(f"Success rate: {successes/args.num_trials}")

def hammer(args, env):
    successes = 0
    for trial in range(args.num_trials):
        t = env.reset()
        for _ in range(15):
            t = env.step(np.array([0., 0., 0., 1.0])) 
        for _ in range(20):
            t = env.step(np.array([0., 1.0, 0., 1.0]))
        successes += t.reward['success']
    print(f"Success rate: {successes/args.num_trials}")

ENV_POLICIES = {
    'assembly-v2': assembly,
    'disassemble-v2': disassemble,
    'peg-insert-side-v2': peg_insert, 
    'hammer-v2': hammer, 
    'bin-picking-v2': bin_picking,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_videos', type=bool, required=False, help='Generate clean video and video with text.')
    parser.add_argument('--teleport_instead_of_mp', type=bool, required=False, help='Use teleportation instead of motion planner.')
    parser.add_argument('--num_trials', type=int, required=True, help='Number of trials to generate success metric.')
    parser.add_argument('--env_name', type=str, required=True, help='Name of environment.')
    args = parser.parse_args()
    assert args.env_name in ENV_POLICIES, f"Environment {args.env_name} does not exist."
    env = make_metaworld(
        name=args.env_name,
        frame_stack=1,
        action_repeat=1,
        discount=1.0,
        seed=np.random.randint(10000),
        camera_name=ENV_CAMERA_DICT[args.env_name],
        add_segmentation_to_obs=False,
        noisy_mask_drop_prob=0.0,
        mprl=True,
        use_mp=args.teleport_instead_of_mp
    )
    ENV_POLICIES[args.env_name](args, env)
