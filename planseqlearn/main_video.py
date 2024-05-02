from copy import copy
import os
import random
import time
import gym
import mujoco_py
import argparse
import nvisii 
from PIL import ImageFont, ImageDraw, Image

from tqdm import tqdm
from planseqlearn.environments.robosuite_dm_env import make_robosuite
from planseqlearn.environments.metaworld_dm_env import make_metaworld
from planseqlearn.environments.mopa_dm_env import make_mopa
from planseqlearn.environments.kitchen_dm_env import make_kitchen
import torch
import numpy as np
from planseqlearn.psl.env_text_plans import *
import pickle
import robosuite as suite
import imageio
import cv2
import metaworld
import xml.etree.ElementTree as ET
from planseqlearn.nvisii_renderer.nvisii_renderer import NVISIIRenderer
from mopa_rl.config.default_configs import (
    LIFT_OBSTACLE_CONFIG,
    LIFT_CONFIG,
    ASSEMBLY_OBSTACLE_CONFIG,
    PUSHER_OBSTACLE_CONFIG,
)

from planseqlearn.utils import make_video
import mujoco

class SimWrapper():
    def __init__(self, model, data):
        self.model = model
        self.data = data
    def forward(self):
        pass

class ModelWrapper():
    def __init__(self, model):
        self.model = model
        
    def get_xml(self):
        # string representation of the xml
        mujoco.mj_saveLastXML('/tmp/temp.xml', self.model)
        xml = open('/tmp/temp.xml').read()
        xml_root = ET.fromstring(xml)
        for elem in xml_root.iter():
            for attr in elem.attrib:
                if attr == 'file':
                    menagerie_path = 'mujoco_menagerie'
                    elem.attrib[attr] = os.path.join(menagerie_path, 'rethink_robotics_sawyer/assets', elem.attrib[attr])
        xml = ET.tostring(xml_root, encoding='unicode')
        return xml
    def body_name2id(self, name):
        return name2id(self.model, "body", name)
def name2id(model, type_name, name):
    obj_id = mujoco.mj_name2id(
        model, mujoco.mju_str2Type(type_name.encode()), name.encode()
    )
    if obj_id < 0:
        raise ValueError('No {} with name "{}" exists.'.format(type_name, name))
    return obj_id
class DataWrapper():
    def __init__(self, data):
        self.data = data
    @property
    def qpos(self):
        return self.data.qpos
    @property
    def qvel(self):
        return self.data.qvel
    def get_geom_xpos(self, name):
        model = self.data.model
        geom_id = name2id(model, "geom", name)
        return self.data.geom_xpos[geom_id]
    def get_body_xpos(self, name):
        model = self.data.model
        body_id = name2id(model, "body", name)
        return self.data.model.body_pos[body_id]
    @property
    def body_xmat(self):
        quat = self.data.model.body_quat
        xmat = []
        for i in range(quat.shape[0]):
            xmat_ = np.zeros((9, 1))
            mujoco.mju_quat2Mat(xmat_,quat[i].reshape(-1, 1))
            xmat.append(xmat_)
        xmat = np.array(xmat)[:, :, 0]
        return xmat
class MenagerieEnv():
    def __init__(self):
        model = mujoco.MjModel.from_xml_path("mujoco_menagerie/rethink_robotics_sawyer/scene.xml")
        data = mujoco.MjData(model)
        self.sim = SimWrapper(ModelWrapper(model), DataWrapper(data))
        self.env_name = 'menagerie'
        self.data = data

    def reset(self):
        mujoco.mj_step(self.data.model, self.data)
    def step(self, action):
        pass

# REPLACE LINE WIDTHS WHEN LAUNCHIGN FULL
IMWIDTH = 1920#240
IMHEIGHT = 1080
def get_geom_segmentation(geom_name, renderer):
    segmentation_array = nvisii.render_data(
        width=IMWIDTH,
        height=IMHEIGHT,
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="entity_id",
        seed=1,
    )
    segmentation_array = np.flipud(
        np.array(segmentation_array).reshape(IMHEIGHT, IMWIDTH, 4)[:, :, 0].astype(np.uint8)
    )
    all_entity_ids = set()
    all_entity_ids.add(renderer.components[geom_name].element_id)
    for i in range(len(segmentation_array)):
            for j in range(len(segmentation_array[0])):
                if segmentation_array[i][j] in renderer.parser.entity_id_class_mapping:
                    segmentation_array[i][j] = renderer.parser.entity_id_class_mapping[segmentation_array[i][j]]
                else:
                    segmentation_array[i][j] = 254
    for id in list(np.unique(segmentation_array)):
        if id not in all_entity_ids:
            segmentation_array[segmentation_array == id] = 0 
        else:
            segmentation_array[segmentation_array == id] = 1
    import matplotlib.pyplot as plt 
    plt.imshow(segmentation_array)
    plt.savefig("micro_seg.png")
    print("Array sum ", np.sum(segmentation_array))
    return segmentation_array

def gen_video(env_name, camera_name, suite):
    np.random.seed(0)
    random.seed(0)
    if suite == 'metaworld':
        # create environment
        mt = metaworld.MT1(env_name, seed=0)
        all_envs = {
            name: env_cls() for name, env_cls in mt.train_classes.items()
        }
        _, env = random.choice(list(all_envs.items()))
        task = random.choice(
                [task for task in mt.train_tasks if task.env_name == env_name]
            )
        env.set_task(task)
        
        def get_xml():
            import metaworld.envs as envs
            base_path = os.path.join(envs.__path__[0], "assets_v2")
            xml_root = ET.fromstring(env.sim.model.get_xml())
            # convert all ../ to absolute path
            for elem in xml_root.iter():
                for attr in elem.attrib:
                    if "../" in elem.attrib[attr]:
                        orig = copy(elem.attrib[attr].split("/")[0])
                        while "../" in elem.attrib[attr]:
                            elem.attrib[attr] = elem.attrib[attr].replace("../", "")
                        # extract first part of the path
                        first_part = elem.attrib[attr].split("/")[0]
                        if first_part == "objects":
                            # get the absolute path
                            abs_path = os.path.join(base_path, elem.attrib[attr])
                        elif first_part == 'scene':
                            abs_path = os.path.join(base_path, elem.attrib[attr])
                        elif first_part == 'scene_textures':
                            abs_path = os.path.join(base_path, 'scene', elem.attrib[attr])
                        elif first_part == 'textures':
                            abs_path = os.path.join(base_path, elem.attrib[attr])
                        else:
                            print(first_part, orig)
                        elem.attrib[attr] = abs_path
                    else:
                        first_part = elem.attrib[attr].split("/")[0]
                        if first_part == 'assets':
                            abs_path = os.path.join(base_path, 'objects', elem.attrib[attr])
                            elem.attrib[attr] = abs_path
                            
            xml = ET.tostring(xml_root, encoding='unicode')
            return xml
        env.get_modified_xml = get_xml
    elif suite == 'kitchen':
        env_kwargs = dict(
            dense=False,
            image_obs=True,
            action_scale=1,
            control_mode="end_effector",
            frame_skip=40,
            max_path_length=280,
        )
        from d4rl.kitchen.env_dict import ALL_KITCHEN_ENVIRONMENTS
        env = ALL_KITCHEN_ENVIRONMENTS[env_name](**env_kwargs)
    elif suite == 'mopa':
        if env_name == "SawyerLift-v0":
            config = LIFT_CONFIG
        elif env_name == "SawyerLiftObstacle-v0":
            config = LIFT_OBSTACLE_CONFIG
        elif env_name == "SawyerAssemblyObstacle-v0":
            config = ASSEMBLY_OBSTACLE_CONFIG
        elif env_name == "SawyerPushObstacle-v0":
            config = PUSHER_OBSTACLE_CONFIG
        config['max_episode_steps'] = 100
        env = gym.make(**config)
        
        def get_xml():
            import mopa_rl.env as envs
            base_path = os.path.join(envs.__path__[0], "assets")
            xml_root = ET.fromstring(env.sim.model.get_xml())
            # convert all ../ to absolute path
            for elem in xml_root.iter():
                for attr in elem.attrib:
                    if "../" in elem.attrib[attr]:
                        orig = copy(elem.attrib[attr].split("/")[0])
                        while "../" in elem.attrib[attr]:
                            elem.attrib[attr] = elem.attrib[attr].replace("../", "")
                        # extract first part of the path
                        first_part = elem.attrib[attr].split("/")[0]
                        if first_part == "meshes":
                            abs_path = os.path.join(base_path, elem.attrib[attr])
                        elif first_part == 'textures':
                            abs_path = os.path.join(base_path, elem.attrib[attr])
                        else:
                            print(first_part, orig)
                        elem.attrib[attr] = abs_path
                    else:
                        first_part = elem.attrib[attr].split("/")[0]
                        if first_part == 'assets':
                            abs_path = os.path.join(base_path, 'objects', elem.attrib[attr])
                            elem.attrib[attr] = abs_path
                        if './' in elem.attrib[attr]:
                            elem.attrib[attr] = elem.attrib[attr].replace('./', '')
                            first_part = elem.attrib[attr].split("/")[0]
                            if first_part == 'common':
                                abs_path = os.path.join(base_path, 'xml', elem.attrib[attr])
                                elem.attrib[attr] = abs_path
                            
            xml = ET.tostring(xml_root, encoding='unicode')
            return xml
        env.get_modified_xml = get_xml
        env.env_name = env_name
    env.reset()
    cfg = {
        "img_path": "images_2/",
        "width": IMWIDTH, # used to be 1980 by 1080
        "height": IMHEIGHT,
        "spp": 512,
        "use_noise": False,
        "debug_mode": False,
        "video_mode": False,
        "video_path": "videos/",
        "video_name": "robosuite_video_0.mp4",
        "video_fps": 30,
        "verbose": 1,
        "vision_modalities": None
    }
    renderer = NVISIIRenderer(env,
                              **cfg)
    states = np.load(f"states/{env_name}_{camera_name}_states_0.npz")
    mp_idxs = list(np.load("mp_idxs/kitchen-ms5-v0_wrist_mp_idxs_0.npz")["mp_idxs"])
    print([mp_idxs[i] for i in range(len(mp_idxs) - 1) if mp_idxs[i + 1] > mp_idxs[i] + 1])
    renderer.reset()
    # clear images folder

    orig_cam_at = np.array((-.2, 1.5, 1.5))
    kettle_cam_at = np.array((-.2, 1.5, 0.75))
    orig_cam_pose = np.array(((-0.5, -1.5, 3.5)))
    kettle_cam_pose = np.array((-0.2, -0.1, 2.5))
    
    orig_to_kettle_at = [orig_cam_at + t * (kettle_cam_at - orig_cam_at) for t in np.linspace(0, 1, 20)]
    orig_to_kettle_pose = [orig_cam_pose + t * (kettle_cam_pose - orig_cam_pose) for t in np.linspace(0, 1, 20)]

    PLAN_TEXT = "Plan: [('microwave', 'grasp'), ('kettle', 'grasp'), ...]"
    # test stuff
    font_path = 'planseqlearn/arial.ttf'
    font = ImageFont.truetype(font_path, 22) # should be 22
    boldfont = ImageFont.truetype(font_path, 44) # used to be 52
    boldfont2 = ImageFont.truetype(font_path, 44) 
    firstfont = ImageFont.truetype(font_path, 42) # should be like 42 or osmething
    bfont_path = 'planseqlearn/Arial Bold.ttf'
    icon_frame = Image.open("icon.png")
    icon_frame = icon_frame.resize((160, 100))
    llmfont = ImageFont.truetype(bfont_path, 40)
    llmfont2 = ImageFont.truetype(bfont_path, 60) # used to be 60
    for file in os.listdir("images_mod2"): 
        os.remove(os.path.join("images_mod2", file))
    solving_idxs = []
    # render starting
    for i in range(140): # used to be 115
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
        frame = cv2.imread(f"images_mod2/image_{renderer.img_cntr}.png")
        frame = (frame * 0.2).astype(np.uint8)
        x = 0.3 * cfg['width']
        y = 0.4 * cfg['height']
        text1 = "Output a high level plan to solve this task:"
        text2 = "• open the microwave door"
        text3 = "• move the kettle"
        text4 = "• flick the light switch"
        text5 = "• adjust the top burner"
        text6 = "• slide the cabinet door"
        if i < 10:
            text1 = text1[:max(int((i) * len(text1) / 10), 0)]
            text2, text3, text4, text5, text6 = "", "", "", "", ""
        if i < 20:
            text2 = text2[:max(int((i - 10 + 1) * len(text2) / 10), 0)]
            text3, text4, text5, text6 = "", "", "", ""
        if i < 30:
            text3 = text3[:max(int((i - 20 + 1) * len(text3) / 10), 0)]
            text4, text5, text6 = "", "", ""
        if i < 40:
            text4 = text4[:max(int((i - 30 + 1) * len(text4) / 10), 0)]
            text5, text6 = "", ""
        if i < 50:
            text5 = text5[:max(int((i - 40 + 1) * len(text5) / 10), 0)]
            text6 = ""
        if i < 60:
            text6 = text6[:max(int((i - 50 + 1) * len(text6) / 10), 0)]
        
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)


        draw.rounded_rectangle(
            (x - cfg['width']/75, y - cfg['height']/50, 
            x + 0.45 * cfg['width'], y + 0.3  * cfg['height']), fill=(255, 255, 255, 1), radius=15
        )
        
        if i > 70 and i <= 100:
            draw.line(
                xy=((0.5 * cfg['width'], max(0.16, 0.39 - (i - 70)/50) * cfg['height']), 
                    (0.5 * cfg['width'], 0.39 * cfg['height'])), width=8
            )
        if i > 80:
            draw.rounded_rectangle(
                xy =[(0.45 * cfg['width'], max(0.05, (0.16 - (i - 80)/40)) * cfg['height']), # x + 0.52 used to be y + 0.42
                (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        if i > 90:
            draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1))
        if i > 100:
            draw.line(
                xy=((0.5 * cfg['width'], 0.16 * cfg['height']), 
                    (0.5 * cfg['width'], max(0.16, 0.39 - (i - 100)/50) * cfg['height'])), width=8
            )


        draw.text((x, y), text1, font=firstfont, fill = (0, 0, 0, 1))
        draw.text((x, y + 0.05 * cfg['height']), text2, font=firstfont, fill = (0, 0, 0, 255))
        draw.text((x, y + 0.1 * cfg['height']), text3, font=firstfont, fill = (0, 0, 0, 255))
        draw.text((x, y + 0.15 * cfg['height']), text4, font=firstfont, fill = (0, 0, 0, 255))
        draw.text((x, y + 0.2 * cfg['height']), text5, font=firstfont, fill = (0, 0, 0, 255))
        draw.text((x, y + 0.25 * cfg['height']), text6, font=firstfont, fill = (0, 0, 0, 255))
        
        img_pil.paste(icon_frame, (int(0.21 * cfg['width']), int(y + 0.10 * cfg['height'])), mask=icon_frame)
        to_save = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", to_save)
    # load and draw llm icon 
    llm_icon = Image.open("llm_icon.jpeg")
    llm_icon = llm_icon.resize((60, 50))
    # render first image 
    for i in range(15):
        renderer.img_cntr += 1
        # load previous image 
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        if i < 8:
            text = ""
        else:
            text = text[:int(((i - 8 + 1)/5) * len(text))]
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        # first 6 frames change size
        if i < 4:
            draw.rounded_rectangle(
                xy =[((0.45 - (i + 1)/300) * cfg['width'], (0.05 - (i + 1)/300) * cfg['height']), # x + 0.52 used to be y + 0.42
                ((0.55 + (i + 1)/300) * cfg['width'], (0.16 + (i + 1)/300) * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        elif i >= 4 and i < 8:
            draw.rounded_rectangle(
                xy =[((0.45 - (8 - (i + 1))/300) * cfg['width'], (0.05 - (8 - (i + 1))/300) * cfg['height']), # x + 0.52 used to be y + 0.42
                ((0.55 + (8 - (i + 1))/300) * cfg['width'], (0.16 + (8 - (i + 1))/300) * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        else:
            draw.rounded_rectangle(
                xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
                (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        # 6
        if i >= 8 and i < 11:
            draw.line(
                xy=((0.55 * cfg['width'], 0.105 * cfg['height']), 
                    ((0.70 - 5 * (11 - (i + 1))/100) * cfg['width'], 0.105 * cfg['height'])), width=10, fill="white",
            )
        if i >= 11 and i < 14:
            draw.line(
                xy=(( (0.70 - 5 * (14 - (i + 1))/100) * cfg['width'], 0.105 * cfg['height']), 
                    ((0.70) * cfg['width'], 0.105 * cfg['height'])), width=10, fill="white",
            )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    # # pause for plan
    for _ in range(15):
        renderer.img_cntr += 1
        # # load previous image 
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)



    microwave_cam_pose = np.array(((-0.7, -0.35, 2.0)))
    orig_to_microwave = [orig_cam_pose + (microwave_cam_pose - orig_cam_pose) * t for t in np.linspace(0, 1, 20)] # should be 20
    # # # # change camera viewpoint and write segment microwave handle
    for i, pose in enumerate(orig_to_microwave):
        
        renderer.img_cntr += 1

        # # # load previous image 
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Segment microwave handle"
        text = text[:int(len(text) * (i + 1) / 5)]
        draw.text((0.0825 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    
    # # # change color of microwave handle 
    orig_microwave_color = np.array((
        renderer._init_component_colors['mchandle'].x,
        renderer._init_component_colors['mchandle'].y,
        renderer._init_component_colors['mchandle'].z
    ))
    blue_color = np.array([0., 1.0, 1.0])
    gray_to_blue = [orig_microwave_color + (blue_color - orig_microwave_color) * t for t in np.linspace(0, 1, 15)]
    # # pause segment microwave handle
    for i in range(10):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Segment microwave handle"
        draw.text((0.0825 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    

    # # # # erase segment 
    for i, color in enumerate(gray_to_blue):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Segment microwave handle"
        draw.text((0.0825 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)


    # # # #renderer.components['mchandle'].obj.get_material().set_base_color_texture(teal_texture)

    # write estimate pose 
    for i in range(56):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Estimate pose from point cloud"
        text = text[:int((i + 1) * len(text) / 5)]
        draw.text((0.065 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")

        if i < 54 and i > 20:
            # top line 
            draw.line(
                ((cfg['width'] * max(0.37, 0.43 - (i - 20)/20), cfg['height'] * 0.32),
                 (cfg['width'] * 0.43, cfg['height'] * 0.32)), 
                fill="white", width=5,
            )
            # left line 
            if i > 22:
                draw.line(
                ((cfg['width'] * 0.37, cfg['height'] * 0.32),
                 (cfg['width'] * 0.37, cfg['height'] * min(0.65, 0.32 + (i - 22)/10))), 
                fill="white", width=5,
                )
            # bottom line
            if i > 25:
                draw.line(
                ((cfg['width'] * 0.37, cfg['height'] * 0.65),
                 (cfg['width'] * min(0.43, 0.37 + (i - 25)/20), cfg['height'] * 0.65)), 
                fill="white", width=5,
                ) 
            # right line 
            if i > 27:
                draw.line(
                ((cfg['width'] * 0.43, cfg['height'] * 0.65),
                 (cfg['width'] * 0.43, cfg['height'] * max(0.32, 0.65 - (i - 27)/20))), 
                fill="white", width=5,
                )
        elif i >= 54:
            draw.rectangle(
                ((cfg['width'] * 0.37, cfg['height'] * 0.32),
                (cfg['width'] * 0.43, cfg['height'] * 0.65)),
                fill=None, outline=(128, 128, 128, 1), width=5
            )
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    
    # pause estimate pose 
    for i in range(20):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Estimate pose from point cloud"
        draw.text((0.065 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")

        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)


    # # # reverse from microwave 
    for pose in orig_to_microwave[::-1]:
        
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Estimate pose from point cloud"
        draw.text((0.065 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")

        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    # pause 
    for _ in range(5):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Estimate pose from point cloud"
        draw.text((0.065 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")

        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    # # # # write motion planner text 
    for _ in range(10):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Motion planner"
        text = text[:int(len(text) * (_ + 1) / 5)]
        draw.text((0.1475 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")

        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    # # # microwave mp
    for step in tqdm(range(0, 50)):        
        renderer.img_cntr += 1
        solving_idxs.append(renderer.img_cntr)
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Motion planner"
        draw.text((0.1475 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")

        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    

    # # # write local policy
    for _ in range(10):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Local policy"
        text = text[:int(len(text) * (_ + 1) / 5)]
        draw.text((0.16 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")

        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    
    # # # # microwave low level 
    for step in tqdm(range(50, 61)):
        renderer.img_cntr += 1
        solving_idxs.append(renderer.img_cntr)
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('microwave', 'grasp')"
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Local policy"
        draw.text((0.16 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")

        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    # # # # write segment kettle and pause 
    for i in range(15):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('kettle', 'grasp')"
        if i < 8:
            text = ""
        else:
            text = text[:int(((i - 8 + 1)/5) * len(text))]
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        # first 6 frames change size
        if i < 4:
            draw.rounded_rectangle(
                xy =[((0.45 - (i + 1)/300) * cfg['width'], (0.05 - (i + 1)/300) * cfg['height']), # x + 0.52 used to be y + 0.42
                ((0.55 + (i + 1)/300) * cfg['width'], (0.16 + (i + 1)/300) * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        elif i >= 4 and i < 8:
            draw.rounded_rectangle(
                xy =[((0.45 - (8 - (i + 1))/300) * cfg['width'], (0.05 - (8 - (i + 1))/300) * cfg['height']), # x + 0.52 used to be y + 0.42
                ((0.55 + (8 - (i + 1))/300) * cfg['width'], (0.16 + (8 - (i + 1))/300) * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        else:
            draw.rounded_rectangle(
                xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
                (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        # 6
        if i >= 8 and i < 11:
            draw.line(
                xy=((0.55 * cfg['width'], 0.105 * cfg['height']), 
                    ((0.70 - 5 * (11 - (i + 1))/100) * cfg['width'], 0.105 * cfg['height'])), width=10, fill="white",
            )
        if i >= 11 and i < 14:
            draw.line(
                xy=(( (0.70 - 5 * (14 - (i + 1))/100) * cfg['width'], 0.105 * cfg['height']), 
                    ((0.70) * cfg['width'], 0.105 * cfg['height'])), width=10, fill="white",
            )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Segment kettle handle"
        text = text[:int(len(text) * (i + 1) / 5)]
        draw.text((0.11 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    
    for at, pose in zip(orig_to_kettle_at, orig_to_kettle_pose):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Segment kettle handle"
        draw.text((0.11 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    # # # change color and write segment kettle handle 
    for color in gray_to_blue:
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Segment kettle handle"
        text = text[:int(len(text) * (i + 1) / 5)]
        draw.text((0.11 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    
    # # # # rectangle bounding box (0.35, 0.55), (0.55, 0.66)

    for i in range(56):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Estimate pose from point cloud"
        text = text[:int(len(text) * (i + 1) / 5)]
        draw.text((0.065 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")

        # draw rectangle 
        if i < 54 and i > 20:
            # top line 
            draw.line(
                ((cfg['width'] * max(0.35, 0.45 - (i - 20)/20), cfg['height'] * 0.55),
                 (cfg['width'] * 0.55, cfg['height'] * 0.55)), 
                fill="white", width=5,
            )
            # left line 
            if i > 22:
                draw.line(
                ((cfg['width'] * 0.35, cfg['height'] * 0.55),
                 (cfg['width'] * 0.35, cfg['height'] * min(0.65, 0.55 + (i - 22)/10))), 
                fill="white", width=5,
                )
            # bottom line
            if i > 25:
                draw.line(
                ((cfg['width'] * 0.35, cfg['height'] * 0.65),
                 (cfg['width'] * min(0.55, 0.35 + (i - 25)/15), cfg['height'] * 0.65)), 
                fill="white", width=5,
                ) 
            # right line 
            if i > 27:
                draw.line(
                ((cfg['width'] * 0.55, cfg['height'] * 0.65),
                 (cfg['width'] * 0.55, cfg['height'] * max(0.55, 0.65 - (i - 27)/10))), 
                fill="white", width=5,
                )
        elif i >= 54:
            draw.rectangle(
                ((cfg['width'] * 0.35, cfg['height'] * 0.55),
                (cfg['width'] * 0.55, cfg['height'] * 0.65)),
                fill=None, outline=(128, 128, 128, 1), width=5
            )
        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    
    for i in range(11):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Estimate pose from point cloud"
        draw.text((0.065 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    # # # # pause written pose 
    for i in range(5):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Estimate pose from point cloud"
        draw.text((0.065 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    # reverse from kettle
    for at, pose in zip(orig_to_kettle_at[::-1], orig_to_kettle_pose[::-1]):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Estimate pose from point cloud"
        draw.text((0.065 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    
    # # # # # write motion planner
    for i in range(11):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Motion planner"
        text = text[:int(len(text) * (i + 1) / 5)]
        draw.text((0.1475 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    # # # # # kettle mp
    for step in tqdm(range(61, 111)):
        renderer.img_cntr += 1
        solving_idxs.append(renderer.img_cntr)
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Motion planner"
        draw.text((0.1475 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    
    # # # # # write low level policy
    for i in range(11):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Local policy"
        text = text[:int(len(text) * (i + 1) / 5)]
        draw.text((0.16 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    
    # # # # # # kettle low level 
    for step in tqdm(range(111, 121)):
        renderer.img_cntr += 1
        solving_idxs.append(renderer.img_cntr)
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Local policy"
        draw.text((0.16 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    # # # # # load png files from images folder

    # # # # erase low level

    # # # # write sequence remaining stages
    for i in range(25):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = "('kettle', 'grasp')"
        draw.rounded_rectangle(
            xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.695 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Sequence remaining stages"
        text = text[:int((i + 1) * len(text) / 5)]
        draw.text((0.08 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)


    for step in tqdm(range(121, 174)):
        renderer.img_cntr += 1
        solving_idxs.append(renderer.img_cntr)
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        i = step - 121
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('light', 'grasp')"
        if i < 8:
            text = ""
        else:
            text = text[:int(((i - 8 + 1)/5) * len(text))]
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        # first 6 frames change size
        if i < 4:
            draw.rounded_rectangle(
                xy =[((0.45 - (i + 1)/300) * cfg['width'], (0.05 - (i + 1)/300) * cfg['height']), # x + 0.52 used to be y + 0.42
                ((0.55 + (i + 1)/300) * cfg['width'], (0.16 + (i + 1)/300) * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        elif i >= 4 and i < 8:
            draw.rounded_rectangle(
                xy =[((0.45 - (8 - (i + 1))/300) * cfg['width'], (0.05 - (8 - (i + 1))/300) * cfg['height']), # x + 0.52 used to be y + 0.42
                ((0.55 + (8 - (i + 1))/300) * cfg['width'], (0.16 + (8 - (i + 1))/300) * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        else:
            draw.rounded_rectangle(
                xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
                (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        # 6
        if i >= 8 and i < 11:
            draw.line(
                xy=((0.55 * cfg['width'], 0.105 * cfg['height']), 
                    ((0.70 - 5 * (11 - (i + 1))/100) * cfg['width'], 0.105 * cfg['height'])), width=10, fill="white",
            )
        if i >= 11 and i < 14:
            draw.line(
                xy=(( (0.70 - 5 * (14 - (i + 1))/100) * cfg['width'], 0.105 * cfg['height']), 
                    ((0.70) * cfg['width'], 0.105 * cfg['height'])), width=10, fill="white",
            )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.705 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        x = 0.1 * cfg['width']
        y = 0.07 * cfg['height']
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Sequence remaining stages"
        draw.text((0.08 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    

    for step in tqdm(range(174, 229)):
        renderer.img_cntr += 1
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        solving_idxs.append(renderer.img_cntr)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        i = step - 174
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('top burner', 'grasp')"
        if i < 8:
            text = ""
        else:
            text = text[:int(((i - 8 + 1)/5) * len(text))]
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        # first 6 frames change size
        if i < 4:
            draw.rounded_rectangle(
                xy =[((0.45 - (i + 1)/300) * cfg['width'], (0.05 - (i + 1)/300) * cfg['height']), # x + 0.52 used to be y + 0.42
                ((0.55 + (i + 1)/300) * cfg['width'], (0.16 + (i + 1)/300) * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        elif i >= 4 and i < 8:
            draw.rounded_rectangle(
                xy =[((0.45 - (8 - (i + 1))/300) * cfg['width'], (0.05 - (8 - (i + 1))/300) * cfg['height']), # x + 0.52 used to be y + 0.42
                ((0.55 + (8 - (i + 1))/300) * cfg['width'], (0.16 + (8 - (i + 1))/300) * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        else:
            draw.rounded_rectangle(
                xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
                (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        # 6
        if i >= 8 and i < 11:
            draw.line(
                xy=((0.55 * cfg['width'], 0.105 * cfg['height']), 
                    ((0.70 - 5 * (11 - (i + 1))/100) * cfg['width'], 0.105 * cfg['height'])), width=10, fill="white",
            )
        if i >= 11 and i < 14:
            draw.line(
                xy=(( (0.70 - 5 * (14 - (i + 1))/100) * cfg['width'], 0.105 * cfg['height']), 
                    ((0.70) * cfg['width'], 0.105 * cfg['height'])), width=10, fill="white",
            )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.67 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        x = 0.1 * cfg['width']
        y = 0.07 * cfg['height']
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Sequence remaining stages"
        draw.text((0.08 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)

    
    for step in tqdm(range(229, len(states["qpos"]))):
        renderer.img_cntr += 1
        solving_idxs.append(renderer.img_cntr)
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        i = step - 229
        frame = cv2.imread(f"images_2/image_{renderer.img_cntr}.png")
        x = 0.1 * cfg['width']
        y = 0.9 * cfg['height']
        text = "('slide cabinet', 'grasp')"
        if i < 8:
            text = ""
        else:
            text = text[:int(((i - 8 + 1)/5) * len(text))]
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        # first 6 frames change size
        if i < 4:
            draw.rounded_rectangle(
                xy =[((0.45 - (i + 1)/300) * cfg['width'], (0.05 - (i + 1)/300) * cfg['height']), # x + 0.52 used to be y + 0.42
                ((0.55 + (i + 1)/300) * cfg['width'], (0.16 + (i + 1)/300) * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        elif i >= 4 and i < 8:
            draw.rounded_rectangle(
                xy =[((0.45 - (8 - (i + 1))/300) * cfg['width'], (0.05 - (8 - (i + 1))/300) * cfg['height']), # x + 0.52 used to be y + 0.42
                ((0.55 + (8 - (i + 1))/300) * cfg['width'], (0.16 + (8 - (i + 1))/300) * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        else:
            draw.rounded_rectangle(
                xy =[(0.45 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
                (0.55 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
            )
        # 6
        if i >= 8 and i < 11:
            draw.line(
                xy=((0.55 * cfg['width'], 0.105 * cfg['height']), 
                    ((0.70 - 5 * (11 - (i + 1))/100) * cfg['width'], 0.105 * cfg['height'])), width=10, fill="white",
            )
        if i >= 11 and i < 14:
            draw.line(
                xy=(( (0.70 - 5 * (14 - (i + 1))/100) * cfg['width'], 0.105 * cfg['height']), 
                    ((0.70) * cfg['width'], 0.105 * cfg['height'])), width=10, fill="white",
            )
        draw.text((0.47 * cfg['width'], 0.07 * cfg['height']), "LLM", font=llmfont2, fill = (0, 0, 0, 1), outline="black")
        draw.rounded_rectangle(
            xy =[(0.60 * cfg['width'], 0.05 * cfg['height']), # x + 0.52 used to be y + 0.42
            (0.95 * cfg['width'], 0.16 * cfg['height'])], fill=(255, 255, 255, 1), radius=15, outline="black" # x + 0.72 used to be y + 0.58
        )
        draw.text((0.66 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont, fill = (0, 0, 0, 1))
        x = 0.1 * cfg['width']
        y = 0.07 * cfg['height']
        draw.rounded_rectangle(
            xy=[(0.05 * cfg['width'], 0.05 * cfg['height']), (0.4 * cfg['width'], 0.16 * cfg['height'])],
            outline="black", fill="white", radius=15,
        )
        text = "Sequence remaining stages"
        draw.text((0.08 * cfg['width'], 0.08 * cfg['height']), text, font=boldfont2, fill="black")
        frame = np.array(img_pil)
        #cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)
        cv2.imwrite(f"images_mod2/image_{renderer.img_cntr}.png", frame)
    
    policy_frames = []
    for idx in solving_idxs:
        frame = cv2.imread(f"images_2/image_{idx}.png")
        policy_frames.append(frame)
        # cv2.imwrite(f"images_policy/image_{idx}.png", frame)
    # multiply frame by 0.2 then fade out text 
    for i in range(1, 4):
        policy_frames[-i] = (policy_frames[-i] * 0.2).astype(np.uint8)
    for i in range(4, 7):
        policy_frames[-i] = (policy_frames[-i] * 0.4).astype(np.uint8)
    for i in range(7, 10):
        policy_frames[-i] = (policy_frames[-i] * 0.6).astype(np.uint8)
    for i in range(10, 13):
        policy_frames[-i] = (policy_frames[-i] * 0.8).astype(np.uint8)
    fade_frames = []
    # 25 to write 10 to pause 
    psl_frames = [policy_frames[-1].copy() for _ in range(85)]
    text1 = "Plan-Seq-Learn: Language Model Guided RL"
    text2 = "for Solving Long Horizon Robotics Tasks"
    font = ImageFont.truetype(bfont_path, 70)
    for i in range(75):
        img_pil = Image.fromarray(psl_frames[i])
        draw = ImageDraw.Draw(img_pil)
        fill = (255, 255, 255, 1)
        if i > 72:
            fill = (128, 128, 128, 1)
        draw.text((0.12 * cfg['width'], 0.4 * cfg['height']), text1[:int(i * len(text)/10)],font=font, fill=fill)
        if i > 10:
            draw.text((0.15 * cfg['width'], 0.5 * cfg['height']), text2[:int((i - 10) * len(text)/10)], font=font, fill=fill)
        psl_frames[i] = np.array(img_pil)
    frames = []
    for idx in range(1, len(os.listdir("images_mod2"))):
        im_path = os.path.join("images_mod2", "image_{}.png".format(idx))
        frames.append(cv2.imread(im_path))

    frames = policy_frames[::3] + fade_frames + psl_frames + frames
    video_filename = f"{env_name}_{camera_name}.mp4"
    make_video(frames, "rendered_videos", video_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, help="Name of the environment")
    parser.add_argument("--camera_name", type=str, help="Camera name")
    parser.add_argument("--suite", type=str, help="Name of the suite")
    parser.add_argument("--clean", action="store_true", help="Generate clean video")
    args = parser.parse_args()
    gen_video(args.env_name, args.camera_name, args.suite)