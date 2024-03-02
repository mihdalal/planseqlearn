import traceback
import xml.etree.ElementTree as ET
from collections import namedtuple
from metaworld.envs.mujoco.mujoco_env import MujocoEnv

import numpy as np
import nvisii

from planseqlearn.nvisii_renderer.base_parser import BaseParser
from planseqlearn.nvisii_renderer.nvisii_utils import load_object
from planseqlearn.nvisii_renderer.mjcf_utils import string_to_array

Components = namedtuple(
    "Components", ["obj", "geom_index", "element_id", "parent_body_name", "geom_pos", "geom_quat", "dynamic"]
)


class Parser(BaseParser):
    def __init__(self, renderer, env):
        """
        Parse the mujoco xml and initialize NVISII renderer objects.
        Args:
            env (Mujoco env): Environment to parse
        """

        super().__init__(renderer, env)
        self.components = {}
        self.env = env

    def parse_textures(self):
        """
        Parse and load all textures and store them
        """

        self.texture_attributes = {}
        self.texture_id_mapping = {}

        for texture in self.xml_root.iter("texture"):
            texture_type = texture.get("type")
            texture_name = texture.get("name")
            texture_file = texture.get("file")
            texture_rgb = texture.get("rgb1")

            if texture_file is not None:
                self.texture_attributes[texture_name] = texture.attrib
            else:
                color = np.array(string_to_array(texture_rgb))
                self.texture_id_mapping[texture_name] = (color, texture_type)

    def parse_materials(self):
        """
        Parse all materials and use texture mapping to initialize materials
        """

        self.material_texture_mapping = {}
        self.material_non_texture_mapping = {}
        for material in self.xml_root.iter("material"):
            material_name = material.get("name")
            texture_name = material.get("texture")
            self.material_texture_mapping[material_name] = texture_name
            if texture_name is None:
                rgba = material.get("rgba")
                self.material_non_texture_mapping[material_name] = string_to_array(rgba)

    def parse_meshes(self):
        """
        Create mapping of meshes.
        """
        self.meshes = {}
        for mesh in self.xml_root.iter("mesh"):
            self.meshes[mesh.get("name")] = mesh.attrib

    def parse_geometries(self):
        """
        Iterate through each goemetry and load it in the NVISII renderer.
        """
        self.parse_meshes()
        element_id = 0
        repeated_names = {}
        block_rendering_objects = ["VisualBread_g0", "VisualCan_g0", "VisualCereal_g0", "VisualMilk_g0"]

        self.entity_id_class_mapping = {}
        for geom_index, geom in enumerate(self.xml_root.iter("geom")):
            parent_body = self.parent_map.get(geom)
            parent_body_name = parent_body.get("name", "worldbody")

            geom_name = geom.get("name")
            geom_type = geom.get("type", None)
            if geom_type is None:
                # check if it is a mesh
                if geom.get("mesh") is not None:
                    geom_type = "mesh"
            rgba_str = geom.get("rgba")
            geom_class = geom.get("class", '')            
            if geom_class == 'sawyer_viz':
                rgba_str = "0.5 0.1 0.1 1"
            geom_rgba = string_to_array(rgba_str) if rgba_str is not None else None
            if geom_name is None:
                if parent_body_name in repeated_names:
                    geom_name = parent_body_name + str(repeated_names[parent_body_name])
                    repeated_names[parent_body_name] += 1
                else:
                    geom_name = parent_body_name + "0"
                    repeated_names[parent_body_name] = 1
            if ("collision" in geom_name) or ("worldbody" in geom_name) or 'col' in geom_class:
                continue
            
            if issubclass(type(self.env), MujocoEnv):
                if ('right_l' in geom_name and not geom_name.startswith('robot')):
                    # removing weird cylinders in metaworld envs
                    continue
        
            if 'indicator' in geom_name or geom_name.endswith('target') or geom_name.endswith('target0') or geom_name.endswith('target1') or 'collison' in geom_name:
                continue
            
            geom_quat = string_to_array(geom.get("quat", "1 0 0 0"))
            geom_quat = [geom_quat[0], geom_quat[1], geom_quat[2], geom_quat[3]]

            # handling special case of bins arena
            geom_pos = string_to_array(geom.get("pos", "0 0 0"))

            if geom_type == "mesh":
                try:
                    geom_scale = string_to_array(self.meshes[geom.get("mesh")].get("scale", "1 1 1"))
                except:
                    geom_scale = [1, 1, 1]
            else:
                geom_scale = [1, 1, 1]
            geom_size = string_to_array(geom.get("size", "1 1 1"))

            geom_mat = geom.get("material")

            dynamic = True

            geom_tex_name, geom_tex_file, geom_rgba = self.parse_material(geom_mat, geom_rgba)

            # manually specify colors/textures for certain objects
            if 'slide' in geom_name and geom_rgba is None and geom_mat is None:
                geom_mat = 'M_slide_blue'
                geom_tex_name, geom_tex_file, geom_rgba = self.parse_material(geom_mat, geom_rgba)
            if ('counters' in geom_name) and geom_rgba is None and geom_mat is None:
                geom_mat = 'counter_metal'
                geom_tex_name, geom_tex_file, geom_rgba = self.parse_material(geom_mat, geom_rgba)
            if 'micro' in geom_name and geom_rgba is None and geom_mat is None:
                geom_mat = 'micro_black'
                geom_tex_name, geom_tex_file, geom_rgba = self.parse_material(geom_mat, geom_rgba)
            if 'oven' in geom_name and geom_rgba is None and geom_mat is None:
                geom_mat = 'oven_metal'
                geom_tex_name, geom_tex_file, geom_rgba = self.parse_material(geom_mat, geom_rgba)
            if 'knob' in geom_name and geom_rgba is None and geom_mat is None:
                geom_mat = 'oven_metal'
                geom_tex_name, geom_tex_file, geom_rgba = self.parse_material(geom_mat, geom_rgba)
            if 'lshandle' in geom_name and geom_rgba is None and geom_mat is None:
                geom_mat = 'oven_metal'
                geom_tex_name, geom_tex_file, geom_rgba = self.parse_material(geom_mat, geom_rgba)
            if 'lightswitchbase' in geom_name and geom_rgba is None and geom_mat is None:
                geom_mat = 'oven_metal'
                geom_tex_name, geom_tex_file, geom_rgba = self.parse_material(geom_mat, geom_rgba)
            if any([x in geom_name for x in ['brb_handle', 'blb_handle', 'tlb_handle', 'trb_handle']]) and geom_rgba is None and geom_mat is None:
                geom_mat = 'oven_metal'
                geom_tex_name, geom_tex_file, geom_rgba = self.parse_material(geom_mat, geom_rgba)
            if 'kettleroot' in geom_name and geom_rgba is None and geom_mat is None:
                geom_mat = 'kettle_white'
                geom_tex_name, geom_tex_file, geom_rgba = self.parse_material(geom_mat, geom_rgba) 
            
            class_id = element_id

            # load obj into nvisii
            try:
                obj, entity_ids = load_object(
                    geom=geom,
                    geom_name=geom_name,
                    geom_type=geom_type,
                    geom_quat=geom_quat,
                    geom_pos=geom_pos,
                    geom_size=geom_size,
                    geom_scale=geom_scale,
                    geom_rgba=geom_rgba,
                    geom_tex_name=geom_tex_name,
                    geom_tex_file=geom_tex_file,
                    class_id=class_id,  # change
                    meshes=self.meshes,
                )
                print(f"Loaded {geom_name} {geom_type} {geom_size} {geom_scale} {geom_pos} {geom_quat} {geom_mat} {geom_tex_name} {geom_tex_file} {geom_rgba}")
            except:
                print(traceback.format_exc())
                print(geom_name, geom_type)
                continue

            element_id += 1

            for entity_id in entity_ids:
                self.entity_id_class_mapping[entity_id] = class_id

            self.components[geom_name] = Components(
                obj=obj,
                geom_index=geom_index,
                element_id=element_id,
                parent_body_name=parent_body_name,
                geom_pos=geom_pos,
                geom_quat=geom_quat,
                dynamic=dynamic,
            )

        self.max_elements = element_id

    def tag_in_name(self, name, tags):
        """
        Checks if one of the tags in body tags in the name

        Args:
            name (str): Name of geom element.

            tags (array): List of keywords to check from.
        """
        for tag in tags:
            if tag in name:
                return True
        return False

    def parse_material(self, geom_mat, geom_rgba):
        geom_tex_name = None
        geom_tex_file = None

        if geom_mat is not None:
            geom_tex_name = self.material_texture_mapping[geom_mat]

            if geom_tex_name in self.texture_attributes:
                geom_tex_file = self.texture_attributes[geom_tex_name]["file"]
            if geom_tex_name is None:
                geom_rgba = self.material_non_texture_mapping[geom_mat]
        return geom_tex_name, geom_tex_file, geom_rgba