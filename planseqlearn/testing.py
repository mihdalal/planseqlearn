import numpy as np
import mujoco_py
from planseqlearn.environments.metaworld_dm_env import make_metaworld
from planseqlearn.environments.mopa_dm_env import make_mopa
from rlkit.mprl.mp_env_metaworld import body_check_grasp
import matplotlib.pyplot as plt

# env = make_mopa(
#     name="SawyerLiftObstacle-v0",
#     frame_stack=1,
#     action_repeat=1,
#     discount=1.0,
#     seed=0,
#     horizon=100,
#     mprl=True,
#     is_eval=False,
# )
# o = env.reset()
# plt.imshow(np.transpose(o.observation["pixels"], (1, 2, 0)))
# plt.savefig("start.png")


"""
List of environments to try out:
- assembly-v2 done 
- disassemble-v2 done 
- stick-pull-v2
- bin-picking-v2
- hammer-v2 done 
- peg-insert-side-v2 done 

Things to check for each environment 
- reset looks okay
- grasp gives right thing
- gripper camera looks ok (gripperPOV??)


Check reset stuff + stepping + video
mopa:
assembly
push

"""


def metaworld_test():
    env = make_metaworld(
        "assembly-v2",
        frame_stack=1,
        action_repeat=1,
        discount=1.0,
        seed=1,
        camera_name="corner",
        add_segmentation_to_obs=False,
        noisy_mask_drop_prob=0.0,
        mprl=True,
    )
    # env.name = "stick-pull-v2"

    o = env.reset()
    frame = env.sim.render(camera_name="gripperPOV", height=500, width=500)
    plt.imshow(frame)
    plt.savefig("reset.png")
    for _ in range(25):
        env.step(np.array([0.0, 0.0, -0.1, 1.0]))
    for _ in range(25):
        env.step(np.array([0.0, 0.0, 0.5, 1.0]))
    frame = env.sim.render(camera_name="gripperPOV", height=500, width=500)
    plt.imshow(frame)
    plt.savefig("start.png")


def spa_assembly():
    for i in range(3):
        np.random.seed(np.random.randint(0, 10000))
        results = []
        for _ in range(10):
            env = make_metaworld(
                "assembly-v2",
                frame_stack=1,
                action_repeat=1,
                discount=1.0,
                seed=1,
                camera_name="corner",
                add_segmentation_to_obs=False,
                noisy_mask_drop_prob=0.0,
                mprl=True,
            )
            o = env.reset()
            for _ in range(25):
                env.step(np.array([0.0, 0.0, 0.0, 1.0]))
            for _ in range(10):
                env.step(np.array([0.0, 0.0, 0.5, 1.0]))
            for _ in range(30):
                o = env.step(np.array([0.12, 0.0, -1.0, 1.0]))
            for _ in range(15):
                o = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
            results.append(o.reward["success"])
        print(np.mean(results))


def spa_disassemble():
    for _ in range(3):
        seed = np.random.randint(0, 100000)
        np.random.seed(seed)
        results = []
        for _ in range(10):
            success = 0
            env = make_metaworld(
                "disassemble-v2",
                frame_stack=1,
                action_repeat=1,
                discount=1.0,
                seed=1,
                camera_name="corner",
                add_segmentation_to_obs=False,
                noisy_mask_drop_prob=0.0,
                mprl=True,
            )
            o = env.reset()
            for _ in range(30):
                env.step(np.array([0.0, 0.0, 0.0, 1.0]))
            for _ in range(30):
                o = env.step(np.array([0.0, 0.0, 1.0, 1.0]))
                success = max(success, o.reward["success"])
            results.append(success)
        print(np.mean(results))


def spa_bin_picking():
    for _ in range(3):
        np.random.seed(np.random.randint(0, 10000))
        for _ in range(10):
            results = []
            env = make_metaworld(
                "bin-picking-v2",
                frame_stack=1,
                action_repeat=1,
                discount=1.0,
                seed=1,
                camera_name="corner",
                add_segmentation_to_obs=False,
                noisy_mask_drop_prob=0.0,
                mprl=True,
            )
            o = env.reset()
            for _ in range(20):
                env.step(np.array([0.0, 0.0, 0.0, 1.0]))
            for _ in range(15):
                env.step(np.array([0.0, 0.0, 0.5, 1.0]))
            for _ in range(15):
                o = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
            results.append(o.reward["success"])
        print(np.mean(results))


def spa_peg_insert_side():
    for i in range(3):
        np.random.seed(np.random.randint(0, 100000))
        results = []
        env = make_metaworld(
            "peg-insert-side-v2",
            frame_stack=1,
            action_repeat=1,
            discount=1.0,
            seed=1,
            camera_name="corner",
            add_segmentation_to_obs=False,
            noisy_mask_drop_prob=0.0,
            mprl=True,
        )
        for _ in range(10):
            o = env.reset()
            # for _ in range(25):
            #     env.step([0., 0., 0., 1.0])
            # for _ in range(10):
            #     env.step(np.array([0., 0., 0.5, 1.0]))
            # for _ in range(75):
            #     o = env.step(np.array([-5.0, 0., -0.0, 1.0]))
            # print(o.reward["success"])
            # frame = env.sim.render(camera_name="corner2", height=500, width =500)
            frame = np.transpose(o.observation["pixels"], (1, 2, 0))
            plt.imshow(frame)
            plt.savefig("reset.png")
            assert False
            results.append(o.reward["success"])
        print(np.mean(results))


def spa_stick_pull():
    env = make_metaworld(
        "stick-pull-v2",
        frame_stack=1,
        action_repeat=1,
        discount=1.0,
        seed=1,
        camera_name="corner",
        add_segmentation_to_obs=False,
        noisy_mask_drop_prob=0.0,
        mprl=True,
    )


def spa_hammer():
    for _ in range(3):
        np.random.seed(np.random.randint(0, 10000))
        results = []
        env = make_metaworld(
            "hammer-v2",
            frame_stack=1,
            action_repeat=1,
            discount=1.0,
            seed=1,
            camera_name="corner",
            add_segmentation_to_obs=False,
            noisy_mask_drop_prob=0.0,
            mprl=True,
        )
        print(env.sim.model.camera_names)
        assert False
        for _ in range(10):
            o = env.reset()
            for _ in range(30):
                env.step(np.array([0.0, 0.0, 0.0, 1.0]))
            for _ in range(5):
                env.step(np.array([0.0, 0.0, 0.3, 1.0]))
            for _ in range(30):
                o = env.step(np.array([0.0, 1.0, 0.0, 1.0]))
            results.append(o.reward["success"])
        assert False
        print(np.mean(results))


def mopa_test():
    env = make_mopa(
        name="SawyerAssemblyObstacle-v0",
        frame_stack=1,
        action_repeat=1,
        discount=1.0,
        seed=0,
        horizon=100,
        mprl=True,
        is_eval=False,
    )
    o = env.reset()
    frame = o.observation["pixels"]
    # frame = env.render()
    plt.imshow(np.transpose(frame, (1, 2, 0)))
    plt.savefig("frame.png")


def video_testing_dis():
    import torch

    with open("best_eval_disassemble.pt", "rb") as f:
        policy = torch.load(f)
    agent = policy["agent"]
    env = make_metaworld(
        "disassemble-v2",
        frame_stack=3,
        action_repeat=2,
        discount=1.0,
        seed=1,
        camera_name="corner2",
        add_segmentation_to_obs=False,
        noisy_mask_drop_prob=0.0,
        mprl=True,
    )
    global_step = 0
    from tqdm import tqdm
    import imageio

    success = False
    with torch.no_grad():
        while not success:
            frames = []
            o = env.reset()
            # print(len(env.intermediate_frames))
            frames.extend(env.intermediate_frames)
            for _ in tqdm(range(99)):
                act = agent.act(o.observation, global_step, eval_mode=True)
                global_step += 1
                o = env.step(act)
                frames.extend(env.intermediate_frames)
                frames.append(
                    env.sim.render(camera_name="corner", height=540, width=960)
                )
                if o.reward["success"] == 1.0 or env._reset_next_step:
                    success = True
                    break
            imageio.mimsave("dis_vid.mp4", frames, fps=20)


def video_testing_hammer():
    import torch

    with open("best_eval_hammer.pt", "rb") as f:
        policy = torch.load(f)
    agent = policy["agent"]
    env = make_metaworld(
        "hammer-v2",
        frame_stack=3,
        action_repeat=2,
        discount=1.0,
        seed=1,
        camera_name="corner2",
        add_segmentation_to_obs=False,
        noisy_mask_drop_prob=0.0,
        mprl=True,
    )
    global_step = 0
    frames = []
    o = env.reset()
    # print(len(env.intermediate_frames))
    frames.extend(env.intermediate_frames)
    from tqdm import tqdm
    import imageio

    with torch.no_grad():
        success = False
        for _ in tqdm(range(99)):
            act = agent.act(o.observation, global_step, eval_mode=True)
            global_step += 1
            o = env.step(act)
            frames.extend(env.intermediate_frames)
            frames.append(env.sim.render(camera_name="corner", height=540, width=960))
            if o.reward["success"] == 1.0 or env._reset_next_step:
                break
    imageio.mimsave("ham_vid.mp4", frames, fps=20)


def video_testing_hammer():
    import torch

    with open("best_eval_hammer.pt", "rb") as f:
        policy = torch.load(f)
    agent = policy["agent"]
    env = make_metaworld(
        "hammer-v2",
        frame_stack=3,
        action_repeat=2,
        discount=1.0,
        seed=1,
        camera_name="corner2",
        add_segmentation_to_obs=False,
        noisy_mask_drop_prob=0.0,
        mprl=True,
    )
    global_step = 0
    from tqdm import tqdm
    import imageio

    success = False
    with torch.no_grad():
        while not success:
            frames = []
            o = env.reset()
            # print(len(env.intermediate_frames))
            frames.extend(env.intermediate_frames)
            for _ in tqdm(range(99)):
                act = agent.act(o.observation, global_step, eval_mode=True)
                global_step += 1
                o = env.step(act)
                frames.extend(env.intermediate_frames)
                frames.append(
                    env.sim.render(camera_name="corner", height=540, width=960)
                )
                if o.reward["success"] == 1.0 or env._reset_next_step:
                    success = True
                    break
            imageio.mimsave("ham_vid.mp4", frames, fps=20)


def video_testing_bin():
    import torch

    with open("best_eval_bin_picking.pt", "rb") as f:
        policy = torch.load(f)
    agent = policy["agent"]
    env = make_metaworld(
        "bin-picking-v2",
        frame_stack=3,
        action_repeat=2,
        discount=1.0,
        seed=1,
        camera_name="corner2",
        add_segmentation_to_obs=False,
        noisy_mask_drop_prob=0.0,
        mprl=True,
    )
    global_step = 0
    from tqdm import tqdm
    import imageio

    success = False
    with torch.no_grad():
        while not success:
            frames = []
            o = env.reset()
            # print(len(env.intermediate_frames))
            frames.extend(env.intermediate_frames)
            for _ in tqdm(range(99)):
                act = agent.act(o.observation, global_step, eval_mode=True)
                global_step += 1
                o = env.step(act)
                frames.extend(env.intermediate_frames)
                frames.append(
                    env.sim.render(camera_name="corner", height=540, width=960)
                )
                if o.reward["success"] == 1.0 or env._reset_next_step:
                    success = True
                    break
            imageio.mimsave("bin_vid.mp4", frames, fps=20)


def mopa_rl_video():
    import torch
    from tqdm import tqdm

    with open("best_eval_sawyer_assembly.pt", "rb") as f:
        policy = torch.load(f)
    env = make_mopa(
        "SawyerAssemblyObstacle-v0",
        frame_stack=3,
        action_repeat=2,
        discount=1.0,
        seed=1,
        horizon=200,
        mprl=True,
        is_eval=False,
    )
    o = env.reset()
    global_step = 0
    agent = policy["agent"]
    frame = np.flipud(
        env._wrapped_env.sim.render(camera_name="frontview", height=500, width=500)
    )
    plt.imshow(frame)
    plt.savefig("reset.png")
    with torch.no_grad():
        for _ in tqdm(range(99)):
            act = agent.act(o.observation, global_step, eval_mode=True)
            global_step += 1
            o = env.step(act)
    frame = np.flipud(
        env._wrapped_env.sim.render(camera_name="frontview", height=500, width=500)
    )
    plt.imshow(frame)
    plt.savefig("reset.png")
    return


if __name__ == "__main__":
    import argparse

    mopa_rl_video()
    # parser = argparse.ArgumentParser(
    #                 prog='ProgramName',
    #                 description='What the program does',
    #                 epilog='Text at the bottom of help')
    # parser.add_argument("--env")
    # args = parser.parse_args()
    # if args.env == "dis":
    #     video_testing_dis()
    # if args.env == "ham":
    #     video_testing_hammer()
    # if args.env == "ass":
    #     video_testing_assembly()
    # if args.env == "bin":
    #     video_testing_bin()
