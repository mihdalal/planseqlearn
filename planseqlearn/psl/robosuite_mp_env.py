import numpy as np
from planseqlearn.psl.vision_utils import *
from planseqlearn.psl.mp_env import PSLEnv
from robosuite.wrappers.gym_wrapper import GymWrapper
import robosuite.utils.camera_utils as CU
from robosuite.controllers import controller_factory
import robosuite
from robosuite.utils.control_utils import orientation_error
import robosuite.utils.transform_utils as T
from robosuite.utils.transform_utils import *
import copy
from urdfpy import URDF
import trimesh
from robosuite.wrappers.gym_wrapper import GymWrapper


def update_controller_config(env, controller_config):
    controller_config["robot_name"] = env.robots[0].name
    controller_config["sim"] = env.robots[0].sim
    controller_config["eef_name"] = env.robots[0].gripper.important_sites["grip_site"]
    controller_config["eef_rot_offset"] = env.robots[0].eef_rot_offset
    controller_config["joint_indexes"] = {
        "joints": env.robots[0].joint_indexes,
        "qpos": env.robots[0]._ref_joint_pos_indexes,
        "qvel": env.robots[0]._ref_joint_vel_indexes,
    }
    controller_config["actuator_range"] = env.robots[0].torque_limits
    controller_config["policy_freq"] = env.robots[0].control_freq
    controller_config["ndim"] = len(env.robots[0].robot_joints)


def apply_controller(controller, action, robot, policy_step):
    gripper_action = None
    if robot.has_gripper:
        gripper_action = action[
            controller.control_dim :
        ]  # all indexes past controller dimension indexes
        arm_action = action[: controller.control_dim]
    else:
        arm_action = action

    # Update the controller goal if this is a new policy step
    if policy_step:
        controller.set_goal(arm_action)

    # Now run the controller for a step
    torques = controller.run_controller()

    # Clip the torques
    low, high = robot.torque_limits
    torques = np.clip(torques, low, high)

    # Get gripper action, if applicable
    if robot.has_gripper:
        robot.grip_action(gripper=robot.gripper, gripper_action=gripper_action)

    # Apply joint torque control
    robot.sim.data.ctrl[robot._ref_joint_actuator_indexes] = torques


def check_robot_string(string):
    if string is None:
        return False
    return string.startswith("robot") or string.startswith("gripper")


def check_string(string, other_string):
    if string is None:
        return False
    return string.startswith(other_string)


def pcd_collision_check(
    env,
    target_angles,
    gripper_qpos,
    is_grasped,
):
    xyz, object_pts = env.xyz, env.object_pcd
    robot = env.robot
    joints = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    combined = []
    qpos = np.concatenate([target_angles, gripper_qpos])
    base_xpos = env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_link0")]
    fk = robot.collision_trimesh_fk(dict(zip(joints, qpos)))
    link_fk = robot.link_fk(dict(zip(joints, qpos)))
    mesh_base_xpos = link_fk[robot.links[0]][:3, 3]
    for mesh, pose in fk.items():
        pose[:3, 3] = pose[:3, 3] + (base_xpos - mesh_base_xpos)
        homogenous_vertices = np.concatenate(
            [mesh.vertices, np.ones((mesh.vertices.shape[0], 1))], axis=1
        ).astype(np.float32)
        transformed = np.matmul(pose.astype(np.float32), homogenous_vertices.T).T[:, :3]
        mesh_new = trimesh.Trimesh(transformed, mesh.faces)
        combined.append(mesh_new)

    combined_mesh = trimesh.util.concatenate(combined)
    robot_mesh = combined_mesh.as_open3d
    # transform object pcd by amount rotated/moved by eef link
    # compute the transform between the old and new eef poses

    # note: this is just to get the forward kinematics using the sim,
    # faster/easier that way than using trimesh fk
    # implementation detail, not important
    old_eef_xquat = env._eef_xquat.copy()
    old_eef_xpos = env._eef_xpos.copy()
    old_qpos = env.sim.data.qpos.copy()

    env.robots[0].set_robot_joint_positions(target_angles)

    ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
    ee_new_mat = pose2mat((env._eef_xpos, env._eef_xquat))
    transform = ee_new_mat @ np.linalg.inv(ee_old_mat)

    env.robots[0].set_robot_joint_positions(old_qpos[:7])

    # Create a scene and add the triangle mesh

    if is_grasped:
        object_pts = object_pts @ transform[:3, :3].T + transform[:3, 3]
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pts)
        hull, _ = object_pcd.compute_convex_hull()
        # compute pcd distance to xyz
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(hull + robot_mesh)
        )  # we do not need the geometry ID for mesh
        occupancy = scene.compute_occupancy(xyz.astype(np.float32), nthreads=32)
        collision = sum(occupancy.numpy()) > 5
    else:
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(robot_mesh)
        )  # we do not need the geometry ID for mesh
        occupancy = scene.compute_occupancy(xyz.astype(np.float32), nthreads=32)
        collision = sum(occupancy.numpy()) > 5
    return collision


def grasp_pcd_collision_check(
    env,
    obj_idx=0,
):
    xyz = compute_object_pcd(
        env,
        obj_idx=obj_idx,
        grasp_pose=False,
        target_obj=False,
        camera_height=256,
        camera_width=256,
    )
    robot = env.robot
    joints = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    combined = []

    # compute floating gripper mesh at the correct pose
    base_xpos = env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_link0")]
    link_fk = robot.link_fk(dict(zip(joints, env.sim.data.qpos[:9])))
    mesh_base_xpos = link_fk[robot.links[0]][:3, 3]
    combined = []
    for link in robot.links[-2:]:
        pose = link_fk[link]
        pose[:3, 3] = pose[:3, 3] + (base_xpos - mesh_base_xpos)
        homogenous_vertices = np.concatenate(
            [
                link.collision_mesh.vertices,
                np.ones((link.collision_mesh.vertices.shape[0], 1)),
            ],
            axis=1,
        )
        transformed = np.matmul(pose, homogenous_vertices.T).T[:, :3]
        mesh_new = trimesh.Trimesh(transformed, link.collision_mesh.faces)
        combined.append(mesh_new)
    robot_mesh = trimesh.util.concatenate(combined).as_open3d

    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(robot_mesh)
    )  # we do not need the geometry ID for mesh
    sdf = scene.compute_signed_distance(xyz.astype(np.float32), nthreads=32).numpy()
    collision = np.any(sdf < 0.001)
    return collision


class RobosuitePSLEnv(PSLEnv):
    def __init__(
        self,
        env,
        env_name,
        **kwargs,
    ):
        # add cameras
        super().__init__(
            env,
            env_name,
            **kwargs,
        )
        self.max_path_length = self._wrapped_env.horizon
        self.vertical_displacement = kwargs["vertical_displacement"]
        self.estimate_orientation = kwargs["estimate_orientation"]
        self.add_cameras()
        self.controller_configs = dict(
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
        self.mp_bounds_low = (-2.0, -2.0, -2.0)
        self.mp_bounds_high = (3.0, 3.0, 3.0)
        self.grip_ctrl_scale = 0.0025
        self.cache = {}
        self.robot_bodies = [
            "robot0_base",
            "robot0_link0",
            "robot0_link1",
            "robot0_link2",
            "robot0_link3",
            "robot0_link4",
            "robot0_link5",
            "robot0_link6",
            "robot0_link7",
            "robot0_right_hand",
            "gripper0_right_gripper",
            "gripper0_eef",
            "gripper0_leftfinger",
            "gripper0_finger_joint1_tip",
            "gripper0_rightfinger",
            "gripper0_finger_joint2_tip",
        ]
        (
            self.robot_body_ids,
            self.robot_geom_ids,
        ) = self.get_body_geom_ids_from_robot_bodies()
        self.original_colors = [
            self.sim.model.geom_rgba[idx].copy() for idx in self.robot_geom_ids
        ]
        self.pick_place_bin_names = {}
        if self.env_name == "PickPlace":
            idx = 1
            for obj_name in ["Milk", "Bread", "Cereal", "Can"]:
                if obj_name in self.valid_obj_names:
                    self.pick_place_bin_names[obj_name] = idx
                    idx += 1
        self.robot = URDF.load(
            robosuite.__file__[: -len("/__init__.py")]
            + "/models/assets/bullet_data/panda_description/urdf/panda_arm_hand.urdf"
        )
        if (
            self.env_name.startswith("PickPlace")
            or self.env_name.startswith("Nut")
            or self.env_name.startswith("Lift")
        ):
            self.burn_in = True
        self.pick_place_bin_locations = np.array(
            [
                [0.0025, 0.1575, 0.8],
                [0.1975, 0.1575, 0.8],
                [0.0025, 0.4025, 0.8],
                [0.1975, 0.4025, 0.8],
            ]
        )
        self.retry = False
    
    def reset_precompute_sam_poses(self):
        self.sam_object_pose = {}
        if self.env_name == "Lift":
            frame = self.sim.render(camera_name="frontview", width=500, height=500)
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                np.flipud(frame[:, :, ::-1]),
                self.dino,
                self.sam,
                text_prompts=["small red cube"],
                box_threshold=0.35,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            object_mask = obj_masks[-1].cpu().detach().numpy()[0, :, :]
            depth_map = get_camera_depth(
                camera_name="frontview",
                camera_width=500,
                camera_height=500,
                sim=self.sim,
            )
            depth_map = np.expand_dims(
                CU.get_real_depth_map(sim=self.sim, depth_map=depth_map), -1
            )
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.sim,
                camera_name="frontview",
                camera_height=500,
                camera_width=500,
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            object_pixels = np.argwhere(object_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            self.sam_object_pose["red cube"] = np.mean(object_pointcloud, axis=0)
        elif self.env_name == "Door":
            frame = self.sim.render(camera_name="frontview", width=500, height=500)
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                np.flipud(frame[:, :, ::-1]),
                self.dino,
                self.sam,
                text_prompts=["black door"],
                box_threshold=0.30,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            object_mask = obj_masks[-1].cpu().detach().numpy()[0, :, :]
            depth_map = get_camera_depth(
                camera_name="frontview",
                camera_width=500,
                camera_height=500,
                sim=self.sim,
            )
            depth_map = np.expand_dims(
                CU.get_real_depth_map(sim=self.sim, depth_map=depth_map), -1
            )
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.sim,
                camera_name="frontview",
                camera_height=500,
                camera_width=500,
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            object_pixels = np.argwhere(object_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            self.sam_object_pose["door"] = np.mean(object_pointcloud, axis=0) + np.array([0., 0.08, 0.07])
        elif self.env_name == "NutAssemblySquare":
            frame = self.sim.render(camera_name="agentview", width=500, height=500)
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                np.flipud(frame[:, :, ::-1]),
                self.dino,
                self.sam,
                text_prompts=["gold square key"],
                box_threshold=0.30,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            object_mask = obj_masks[-1].cpu().detach().numpy()[0, :, :]
            placement_mask = obj_masks[1].cpu().detach().numpy()[0, :, :]
            depth_map = get_camera_depth(
                camera_name="agentview",
                camera_width=500,
                camera_height=500,
                sim=self.sim,
            )
            depth_map = np.expand_dims(
                CU.get_real_depth_map(sim=self.sim, depth_map=depth_map), -1
            )
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.sim,
                camera_name="agentview",
                camera_height=500,
                camera_width=500,
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            object_pixels = np.argwhere(object_mask)
            placement_pixels = np.argwhere(placement_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            placement_pointcloud = CU.transform_from_pixels_to_world(
                pixels=placement_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            self.sam_object_pose["gold square nut"] = np.mean(object_pointcloud, axis=0)
            self.sam_object_pose["gold peg"] = np.mean(placement_pointcloud, axis=0)
        elif self.env_name == "NutAssemblyRound":
            frame = self.sim.render(camera_name="agentview", width=500, height=500)
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                np.flipud(frame[:, :, ::-1]),
                self.dino,
                self.sam,
                text_prompts=["silver round nut"],
                box_threshold=0.3,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            object_mask = obj_masks[1].cpu().detach().numpy()[0, :, :]
            placement_mask = obj_masks[-1].cpu().detach().numpy()[0, :, :]
            depth_map = get_camera_depth(
                camera_name="agentview",
                camera_width=500,
                camera_height=500,
                sim=self.sim,
            )
            depth_map = np.expand_dims(
                CU.get_real_depth_map(sim=self.sim, depth_map=depth_map), -1
            )
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.sim,
                camera_name="agentview",
                camera_height=500,
                camera_width=500,
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            object_pixels = np.argwhere(object_mask)
            placement_pixels = np.argwhere(placement_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            placement_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            self.sam_object_pose["silver round nut"] = np.mean(object_pointcloud, axis=0)
            self.sam_object_pose["silver peg"] = np.mean(placement_pointcloud, axis=0)
        elif self.env_name == "PickPlaceCan":
            frame = self.sim.render(camera_name="agentview", width=500, height=500)
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                np.flipud(frame[:, :, ::-1]),
                self.dino,
                self.sam,
                text_prompts=["red can"],
                box_threshold=0.30,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            object_mask = obj_masks[-1].cpu().detach().numpy()[0, :, :]
            depth_map = get_camera_depth(
                camera_name="agentview",
                camera_width=500,
                camera_height=500,
                sim=self.sim,
            )
            depth_map = np.expand_dims(
                CU.get_real_depth_map(sim=self.sim, depth_map=depth_map), -1
            )
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.sim,
                camera_name="agentview",
                camera_height=500,
                camera_width=500,
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            object_pixels = np.argwhere(object_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            self.sam_object_pose["red can"] = np.mean(object_pointcloud, axis=0) 
        elif self.env_name == "PickPlaceCereal":
            frame = self.sim.render(camera_name="robot0_robotview", width=500, height=500)
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                np.flipud(frame[:, :, ::-1]),
                self.dino,
                self.sam,
                text_prompts=["red cereal box"],
                box_threshold=0.30,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            object_mask = obj_masks[0].cpu().detach().numpy()[0, :, :]
            depth_map = get_camera_depth(
                camera_name="robot0_robotview",
                camera_width=500,
                camera_height=500,
                sim=self.sim,
            )
            depth_map = np.expand_dims(
                CU.get_real_depth_map(sim=self.sim, depth_map=depth_map), -1
            )
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.sim,
                camera_name="robot0_robotview",
                camera_height=500,
                camera_width=500,
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            object_pixels = np.argwhere(object_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            self.sam_object_pose["cereal box"] = np.mean(object_pointcloud, axis=0)
        elif self.env_name == "PickPlaceBread":
            frame = self.sim.render(camera_name="robot0_robotview", width=500, height=500)
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                np.flipud(frame[:, :, ::-1]),
                self.dino,
                self.sam,
                text_prompts=["small brown box"],
                box_threshold=0.30,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            object_mask = obj_masks[0].cpu().detach().numpy()[0, :, :]
            depth_map = get_camera_depth(
                camera_name="robot0_robotview",
                camera_width=500,
                camera_height=500,
                sim=self.sim,
            )
            depth_map = np.expand_dims(
                CU.get_real_depth_map(sim=self.sim, depth_map=depth_map), -1
            )
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.sim,
                camera_name="robot0_robotview",
                camera_height=500,
                camera_width=500,
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            object_pixels = np.argwhere(object_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            self.sam_object_pose["bread"] = np.mean(object_pointcloud, axis=0) 
        elif self.env_name == "PickPlaceMilk":
            frame = self.sim.render(camera_name="agentview", width=500, height=500)
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                np.flipud(frame[:, :, ::-1]),
                self.dino,
                self.sam,
                text_prompts=["robot, white milk carton"],
                box_threshold=0.30,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            object_mask = obj_masks[1].cpu().detach().numpy()[0, :, :]
            depth_map = get_camera_depth(
                camera_name="agentview",
                camera_width=500,
                camera_height=500,
                sim=self.sim,
            )
            depth_map = np.expand_dims(
                CU.get_real_depth_map(sim=self.sim, depth_map=depth_map), -1
            )
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.sim,
                camera_name="agentview",
                camera_height=500,
                camera_width=500,
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            object_pixels = np.argwhere(object_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            self.sam_object_pose["milk carton"] = np.mean(object_pointcloud, axis=0)
        # placement for nut assembly
        if self.env_name.startswith("NutAssembly"):
            frame = self.sim.render(camera_name="robot0_robotview", width=500, height=500)
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                np.flipud(frame[:, :, ::-1]),
                self.dino,
                self.sam,
                text_prompts=["tall cylinder"],
                box_threshold=0.4,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            gold_mask = obj_masks[0].cpu().detach().numpy()[0, :, :]
            silver_mask = obj_masks[1].cpu().detach().numpy()[0, :, :]
            depth_map = get_camera_depth(
                camera_name="robot0_robotview",
                camera_width=500,
                camera_height=500,
                sim=self.sim,
            )
            depth_map = np.expand_dims(
                CU.get_real_depth_map(sim=self.sim, depth_map=depth_map), -1
            )
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.sim,
                camera_name="robot0_robotview",
                camera_height=500,
                camera_width=500,
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            gold_pixels = np.argwhere(gold_mask)
            gold_pointcloud = CU.transform_from_pixels_to_world(
                pixels=gold_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            silver_pixels = np.argwhere(gold_mask)
            silver_pointcloud = CU.transform_from_pixels_to_world(
                pixels=silver_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            self.sam_object_pose['gold peg'] = np.mean(gold_pointcloud, axis=0)
            self.sam_object_pose['silver peg'] = np.mean(silver_pointcloud, axis=0)
        # placement for pick place
        if self.env_name.startswith("PickPlace"):
            frame = self.sim.render(camera_name="birdview", width=500, height=500)
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                np.flipud(frame[:, :, ::-1]),
                self.dino,
                self.sam,
                text_prompts=["grid"],
                box_threshold=0.4,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            object_mask = obj_masks[0].cpu().detach().numpy()[0, :, :]
            depth_map = get_camera_depth(
                camera_name="birdview",
                camera_width=500,
                camera_height=500,
                sim=self.sim,
            )
            depth_map = np.expand_dims(
                CU.get_real_depth_map(sim=self.sim, depth_map=depth_map), -1
            )
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.sim,
                camera_name="birdview",
                camera_height=500,
                camera_width=500,
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            object_pixels = np.argwhere(object_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            bin_center = np.mean(object_pointcloud, axis=0)
            self.sam_object_pose["bin1"] = bin_center + np.array([-0.075, -0.13, -0.04])
            self.sam_object_pose["bin2"] = bin_center + np.array([-0.075, 0.13, -0.04])
            self.sam_object_pose["bin3"] = bin_center + np.array([0.075, -0.13, -0.04])
            self.sam_object_pose["bin4"] = bin_center + np.array([-0.075, 0.13, -0.04])

    def get_body_geom_ids_from_robot_bodies(self):
        body_ids = [self.sim.model.body_name2id(body) for body in self.robot_bodies]
        geom_ids = []
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            if body_id in body_ids:
                geom_ids.append(geom_id)
        return body_ids, geom_ids

    def get_hardcoded_text_plan(self):
        plan = []
        if self.env_name == "PickPlaceCan":
            return [("red can", "grasp"), ("bin4", "place")]
        if self.env_name == "PickPlaceBread":
            return [("bread", "grasp"), ("bin2", "place")]
        if self.env_name == "PickPlaceMilk":
            return [("milk carton", "grasp"), ("bin1", "place")]
        if self.env_name == "PickPlaceCereal":
            return [("cereal box", "grasp"), ("bin3", "place")]
        if self.env_name == "PickPlace":
            for name in self.valid_obj_names:
                plan.append((name.lower(), "grasp"))
                plan.append((f"bin{self.pick_place_bin_names[name]}", "place"))
            return plan
        if self.env_name == "Lift":
            return [("red cube", "grasp")]
        if self.env_name == "Door":
            return [("door", "grasp")]
        if self.env_name == "NutAssemblyRound":
            return [("silver round nut", "grasp"), ("silver peg", "place")]
        if self.env_name == "NutAssemblySquare":
            return [("gold square nut", "grasp"), ("gold peg", "place")]
        if self.env_name == "NutAssembly":
            return [
                ("gold square nut", "grasp"),
                ("gold peg", "place"),
            ] + [("silver round nut", "grasp"), ("silver peg", "place")] 

    def set_robot_colors(self, colors):
        if type(colors) is np.ndarray:
            colors = [colors] * len(self.robot_geom_ids)
        for idx, geom_id in enumerate(self.robot_geom_ids):
            self.sim.model.geom_rgba[geom_id] = colors[idx]
        self.sim.forward()

    def reset_robot_colors(self):
        self.set_robot_colors(self.original_colors)
        self.sim.forward()

    def get_all_initial_object_poses(self):
        if self.env_name.endswith("NutAssembly"):
            # self.initial_object_pos = [
            #     self.get_object_pose_mp(obj_idx=0)[0].copy(),
            #     self.get_object_pose_mp(obj_idx=1)[0].copy(),
            # ]
            # self.initial_object_pos_dict = {}
            # self.initial_object_pos_dict["gold square nut"] = self.initial_object_pos[0].copy()
            # self.initial_object_pos_dict["silver round nut"] = self.initial_object_pos[1].copy()
            self.initial_object_pos_dict = {}
            for obj_name, action in self.text_plan:
                if action == "grasp":
                    self.initial_object_pos_dict[obj_name] = self.named_get_object_pose_mp(obj_name)[0].copy()
        else:
            self.initial_object_pos_dict = {}
            for obj_name, action in self.text_plan:
                if action == "grasp":
                    self.initial_object_pos_dict[obj_name] = self.named_get_object_pose_mp(obj_name)[0].copy()

    def post_reset_burn_in(self):
        if "NutAssembly" in self.env_name:
            for _ in range(10):
                a = np.zeros(7)
                a[-1] = -1
                self._wrapped_env.step(a)

    def reset(self, get_intermediate_frames=False, **kwargs):
        o = super().reset(get_intermediate_frames=get_intermediate_frames, **kwargs)
        return o

    def add_cameras(self):
        for cam_name, cam_w, cam_h, cam_d, cam_seg in zip(
            self.camera_names,
            self.camera_widths,
            self.camera_heights,
            self.camera_depths,
            self.camera_segmentations,
        ):
            # Add cameras associated to our arrays
            cam_sensors, _ = self._create_camera_sensors(
                cam_name,
                cam_w=cam_w,
                cam_h=cam_h,
                cam_d=cam_d,
                cam_segs=cam_seg,
                modality="image",
            )
            self.cam_sensor = cam_sensors

    def update_controllers(self):
        self.ik_controller_config = {
            "type": "IK_POSE",
            "ik_pos_limit": 0.02,
            "ik_ori_limit": 0.05,
            "interpolation": None,
            "ramp_ratio": 0.2,
            "converge_steps": 100,
        }
        self.osc_controller_config = {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "output_min": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
        }
        self.jp_controller_config = {
            "type": "JOINT_POSITION",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "output_min": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
        }
        update_controller_config(self, self.ik_controller_config)
        self.ik_ctrl = controller_factory("IK_POSE", self.ik_controller_config)
        self.ik_ctrl.update_base_pose(self.robots[0].base_pos, self.robots[0].base_ori)

    def get_image(self, camera_name="frontview", width=960, height=540, depth=False):
        if depth:
            im = self.sim.render(
                camera_name=camera_name, width=width, height=height, depth=True
            )[1]
            return im
        else:
            im = self.sim.render(camera_name=camera_name, width=width, height=height)[
                :, :, ::-1
            ]
            return np.flipud(im)

    # def get_object_string(self, obj_idx=0):
    #     if self.env_name.endswith("Lift"):
    #         obj_string = "cube"
    #     elif self.env_name.startswith("PickPlace"):
    #         if self.env_name.endswith("Bread"):
    #             obj_string = "Bread"
    #         elif self.env_name.endswith("Can"):
    #             obj_string = "Can"
    #         elif self.env_name.endswith("Milk"):
    #             obj_string = "Milk"
    #         elif self.env_name.endswith("Cereal"):
    #             obj_string = "Cereal"
    #         else:
    #             obj_string = self.text_plan[obj_idx * 2][
    #                 0
    #             ]  # self.valid_obj_names[obj_idx - 1]
    #     elif self.env_name.endswith("Door"):
    #         obj_string = "latch"
    #     elif "NutAssembly" in self.env_name:
    #         if self.env_name.endswith("Square"):
    #             nut = self.nuts[0]
    #         elif self.env_name.endswith("Round"):
    #             nut = self.nuts[1]
    #         elif self.env_name.endswith("NutAssembly"):
    #             nut = self.nuts[1 - obj_idx]  # first nut is round, second nut is square
    #         obj_string = nut.name
    #     else:
    #         raise NotImplementedError
    #     return obj_string

    # def compute_correct_obj_idx(self, obj_idx=0):
    #     if self.env_name.startswith("PickPlace"):
    #         valid_obj_names = self.valid_obj_names
    #         print(self.valid_obj_names)
    #         obj_string_to_idx = {}
    #         idx = 0
    #         for obj_name in ["Milk", "Bread", "Cereal", "Can"]:
    #             if obj_name in valid_obj_names:
    #                 obj_string_to_idx[obj_name.lower()] = idx
    #                 idx += 1
    #         new_obj_idx = obj_string_to_idx[self.get_object_string(obj_idx=obj_idx)]
    #     else:
    #         new_obj_idx = obj_idx
    #     return new_obj_idx

    def curr_obj_name_to_env_idx(self):
        if self.env_name.startswith("PickPlace"):
            valid_obj_names = self.valid_obj_names
            obj_string_to_idx = {}
            idx = 0
            for obj_name in valid_obj_names:
                obj_string_to_idx[obj_name] = idx
                if obj_name.lower() in self.curr_obj_name.lower():
                    return idx
                idx += 1
        elif self.env_name.startswith("NutAssembly"):
            if (
                "square" in self.curr_obj_name.lower()
                or "gold" in self.curr_obj_name.lower()
            ):
                return 0
            else:
                return 1
        else:
            return 0
        return None

    def get_object_pose_mp(self, obj_idx=0):
        assert obj_idx is not None, "Object name not available"
        if self.env_name.endswith("Lift"):
            object_pos = self.sim.data.qpos[9:12].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[12:16].copy(), to="xyzw")
        elif self.env_name.endswith("PickPlaceMilk"):
            object_pos = self.sim.data.qpos[9:12].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[12:16].copy(), to="xyzw")
        elif self.env_name.endswith("PickPlaceBread"):
            object_pos = self.sim.data.qpos[16:19].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[19:23].copy(), to="xyzw")
        elif self.env_name.endswith("PickPlaceCereal"):
            object_pos = self.sim.data.qpos[23:26].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[26:30].copy(), to="xyzw")
        elif self.env_name.endswith("PickPlaceCan"):
            object_pos = self.sim.data.qpos[30:33].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[33:37].copy(), to="xyzw")
        elif self.env_name.endswith("PickPlace"):
            new_obj_idx = self.compute_correct_obj_idx(obj_idx=obj_idx)
            object_pos = self.sim.data.qpos[
                9 + 7 * new_obj_idx : 12 + 7 * new_obj_idx
            ].copy()
            object_quat = T.convert_quat(
                self.sim.data.qpos[12 + 7 * new_obj_idx : 16 + 7 * new_obj_idx].copy(),
                to="xyzw",
            )
        elif self.env_name.startswith("Door"):
            object_pos = self.sim.data.site_xpos[self.door_handle_site_id].copy()
            object_quat = np.zeros(4)
        elif "NutAssembly" in self.env_name:
            if self.env_name.endswith("Square"):
                nut = self.nuts[0]
            elif self.env_name.endswith("Round"):
                nut = self.nuts[1]
            elif self.env_name.endswith("NutAssembly"):
                nut = self.nuts[1 - obj_idx]
            nut_name = nut.name
            object_pos = self.sim.data.get_site_xpos(nut.important_sites["handle"])
            object_quat = T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[nut_name]], to="xyzw"
            )
        else:
            raise NotImplementedError
        if self.use_vision_pose_estimation:
            if not self.use_sam_segmentation:
                #object_pcd = compute_object_pcd(self, obj_idx=obj_idx)
                if obj_idx == 0:
                    object_pcd = compute_object_pcd(self, obj_name="gold")
                if obj_idx == 1:
                    object_pcd = compute_object_pcd(self, obj_name="silver")
                object_pos = np.mean(object_pcd, axis=0)
                if self.env_name.startswith("Door"):
                    object_pos[0] -= 0.15
                    object_pos[1] += 0.05
            else:
                object_name = (
                    self.curr_obj_name
                )  # self.text_plan[self.num_high_level_steps][0]
                if self.env_name == "Lift":
                    object_pos = get_pose_from_sam_segmentation(
                        self, object_name, "frontview"
                    )
                if self.env_name.startswith("PickPlace"):
                    object_pos = get_pose_from_sam_segmentation(
                        self, object_name, "agentview", threshold=0.3
                    )
        return object_pos, object_quat

    def get_object_pose(self, obj_idx=0):
        if self.env_name.endswith("Lift"):
            object_pos = self.sim.data.qpos[9:12].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[12:16].copy(), to="xyzw")
        elif self.env_name.startswith("PickPlaceMilk"):
            object_pos = self.sim.data.qpos[9:12].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[12:16].copy(), to="xyzw")
        elif self.env_name.startswith("PickPlaceBread"):
            object_pos = self.sim.data.qpos[16:19].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[19:23].copy(), to="xyzw")
        elif self.env_name.startswith("PickPlaceCereal"):
            object_pos = self.sim.data.qpos[23:26].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[26:30].copy(), to="xyzw")
        elif self.env_name.startswith("PickPlaceCan"):
            object_pos = self.sim.data.qpos[30:33].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[33:37].copy(), to="xyzw")
        elif self.env_name.endswith("PickPlace"):
            new_obj_idx = self.compute_correct_obj_idx(obj_idx=obj_idx)
            object_pos = self.sim.data.qpos[
                9 + 7 * new_obj_idx : 12 + 7 * new_obj_idx
            ].copy()
            object_quat = T.convert_quat(
                self.sim.data.qpos[12 + 7 * new_obj_idx : 16 + 7 * new_obj_idx].copy(),
                to="xyzw",
            )
        elif self.env_name.startswith("Door"):
            object_pos = np.array(
                [self.sim.data.qpos[self.hinge_qpos_addr]]
            )  # this is not what they are, but they will be decoded properly
            object_quat = np.array(
                [self.sim.data.qpos[self.handle_qpos_addr]]
            )  # this is not what they are, but they will be decoded properly
        elif "NutAssembly" in self.env_name:
            if self.env_name.endswith("Square"):
                nut = self.nuts[0]
            elif self.env_name.endswith("Round"):
                nut = self.nuts[1]
            elif self.env_name.endswith("NutAssembly"):
                nut = self.nuts[1 - obj_idx]  # first nut is round, second nut is square
            if nut.name == "SquareNut":
                return np.array(self.sim.data.qpos[9:12]), T.convert_quat(
                    self.sim.data.qpos[12:16], to="xyzw"
                )
            else:
                return np.array(self.sim.data.qpos[16:19]), T.convert_quat(
                    self.sim.data.qpos[19:23], to="xyzw"
                )
        else:
            raise NotImplementedError
        return object_pos, object_quat

    # valid names idx to dict idx
    def get_placement_pose(self, obj_idx=0):
        target_quat = self.reset_ori
        if self.env_name.endswith("Lift"):
            target_pos = np.array([0, 0, 0.1]) + self.initial_object_pos
            return target_pos, target_quat
        elif "PickPlace" in self.env_name:
            bin_num = int(self.text_plan[obj_idx * 2 + 1][0][-1])
            target_pos = self.pick_place_bin_locations[bin_num - 1].copy()
            target_pos[2] += 0.125
        elif "NutAssembly" in self.env_name:
            if obj_idx == 0:
                target_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])
            elif obj_idx == 1:
                target_pos = np.array(self.sim.data.body_xpos[self.peg2_body_id])
            if self.env_name.endswith("Round"):
                target_pos = np.array(self.sim.data.body_xpos[self.peg2_body_id])
            elif self.env_name.endswith("Square"):
                target_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])
            target_pos[2] += 0.15
            target_pos[0] -= 0.065
        if self.use_vision_pose_estimation:
            if not self.use_sam_segmentation:
                if self.env_name == "NutAssembly":
                    if "silver" in self.text_plan[obj_idx * 2 + 1][0]:
                        obj_idx = 1
                    else:
                        obj_idx = 0
                target_pcd = compute_object_pcd(
                    self, grasp_pose=False, target_obj=True, obj_idx=obj_idx
                )
                self.target_pcd = target_pcd
                target_pos_pcd = np.mean(target_pcd, axis=0)
                if "NutAssembly" in self.env_name:
                    target_pos_pcd[2] += 0.065  # 0.1
                    target_pos_pcd[0] -= 0.065
                elif "PickPlace" in self.env_name:
                    target_pos_pcd[2] += 0.125
                target_pos = target_pos_pcd
            else:
                object_name = self.text_plan[self.num_high_level_steps][0]
                if self.env_name == "Lift":
                    return target_pos  # no need to segment anything
                elif self.env_name.startswith("PickPlace"):
                    object_name = (
                        "dark brown bin"  # hardcoding for now, should always be this
                    )
                    pc_mean, object_pointcloud = get_pose_from_sam_segmentation(
                        self, "dark brown bin", "birdview", return_pcd=True
                    )
                    pc_xmin = np.quantile(object_pointcloud[:, 0], 0.1)
                    pc_xmax = np.quantile(object_pointcloud[:, 0], 0.9)
                    pc_ymin = np.quantile(object_pointcloud[:, 1], 0.1)
                    pc_ymax = np.quantile(object_pointcloud[:, 1], 0.9)
                    pc_new = np.array(
                        [
                            [
                                np.mean((pc_xmin, pc_mean[0])),
                                np.mean((pc_ymin, pc_mean[1])),
                                pc_mean[2],
                            ],
                            [
                                np.mean((pc_xmax, pc_mean[0])),
                                np.mean((pc_ymin, pc_mean[1])),
                                pc_mean[2],
                            ],
                            [
                                np.mean((pc_xmin, pc_mean[0])),
                                np.mean((pc_ymax, pc_mean[1])),
                                pc_mean[2],
                            ],
                            [
                                np.mean((pc_xmax, pc_mean[0])),
                                np.mean((pc_ymax, pc_mean[1])),
                                pc_mean[2],
                            ],
                        ]
                    )
                    print(f"PC mean: {pc_mean}")
                    print(f"Pc New: {pc_new}")
                    print(f"target bin placements: {self.target_bin_placements}")
                    target_pos = pc_new[obj_idx - 1].copy()
        return target_pos, target_quat

    def get_object_poses(self):
        object_poses = []
        if (
            self.env_name.endswith("Lift")
            or self.env_name.endswith("PickPlaceBread")
            or self.env_name.endswith("PickPlaceCereal")
            or self.env_name.endswith("PickPlaceCan")
            or self.env_name.endswith("PickPlaceMilk")
            or self.env_name.endswith("NutAssemblyRound")
            or self.env_name.endswith("NutAssemblySquare")
            or self.env_name.endswith("Door")
        ):
            object_poses.append(self.get_object_pose_mp(obj_idx=0))
        elif self.env_name.endswith("PickPlace"):
            for obj_idx in range(len(self.valid_obj_names)):
                object_poses.append(self.get_object_pose_mp(obj_idx=obj_idx))
        elif self.env_name.endswith("NutAssembly"):
            for obj_idx in range(2):
                object_poses.append(self.get_object_pose_mp(obj_idx=obj_idx))
        else:
            raise NotImplementedError
        return object_poses

    def get_placement_poses(self):
        placement_poses = []
        if self.env_name.endswith("Lift"):
            placement_pose, placement_quat = self.get_placement_pose(obj_idx=0)
            placement_poses.append((placement_pose, placement_quat))
        elif self.env_name.startswith("PickPlace"):
            for ob in range(0, len(self.text_plan), 2):
                placement_pose, placement_quat = self.get_placement_pose(
                    obj_idx=ob // 2
                )
                placement_poses.append((placement_pose, placement_quat))
        elif self.env_name.startswith("NutAssembly"):
            for ob in range(0, len(self.text_plan), 2):
                if "gold" in self.text_plan[ob + 1][0]:
                    placement_poses.append(self.get_placement_pose(obj_idx=0))
                else:
                    placement_poses.append(self.get_placement_pose(obj_idx=1))
        else:
            placement_poses = [(None, None)]
        return placement_poses

    def compute_hardcoded_orientation(self, target_pos, quat):
        qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        # compute perpendicular top grasps for the object, pick one that has less error
        orig_ee_quat = self.reset_ori.copy()
        ee_euler = mat2euler(quat2mat(orig_ee_quat))
        obj_euler = mat2euler(quat2mat(quat))
        ee_euler[2] = obj_euler[2] + np.pi / 2
        target_quat1 = mat2quat(euler2mat(ee_euler))

        target_quat = target_quat1
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()
        return target_quat

    def named_get_object_pose_mp(self, obj_name):
        if self.use_sam_segmentation:
            object_pos = self.sam_object_pose[obj_name].copy()
            object_quat = self.named_get_object_pose(obj_name)[1].copy()
        elif self.use_vision_pose_estimation:
            object_pcd = compute_object_pcd(self, obj_name=obj_name)
            object_pos = np.mean(object_pcd, axis=0)
            object_quat = np.zeros(4) if self.text_plan[self.curr_plan_stage][1] == "place" \
                else self.named_get_object_pose(obj_name)[1].copy()
            if "place" == self.text_plan[self.curr_plan_stage][1]:
                object_pos += 0.125
        else:
            if self.env_name.endswith("Lift"):
                assert "cube" in obj_name.lower(), f"Object {obj_name} does not exist in environment!"
                object_pos = self.sim.data.qpos[9:12].copy()
                object_quat = T.convert_quat(self.sim.data.qpos[12:16].copy(), to="xyzw")
            elif self.env_name.startswith("PickPlaceMilk"):
                assert "milk" in obj_name.lower(), f"Object {obj_name} does not exist in environment!"
                object_pos = self.sim.data.qpos[9:12].copy()
                object_quat = T.convert_quat(self.sim.data.qpos[12:16].copy(), to="xyzw")
            elif self.env_name.startswith("PickPlaceBread"):
                assert "bread" in obj_name.lower(), f"Object {obj_name} does not exist in environment!"
                object_pos = self.sim.data.qpos[16:19].copy()
                object_quat = T.convert_quat(self.sim.data.qpos[19:23].copy(), to="xyzw")
            elif self.env_name.startswith("PickPlaceCereal"):
                assert "cereal" in obj_name.lower(), f"Object {obj_name} does not exist in environment!"
                object_pos = self.sim.data.qpos[23:26].copy()
                object_quat = T.convert_quat(self.sim.data.qpos[26:30].copy(), to="xyzw")
            elif self.env_name.startswith("PickPlaceCan"):
                assert "can" in obj_name.lower(), f"Object {obj_name} does not exist in environment!"
                object_pos = self.sim.data.qpos[30:33].copy()
                object_quat = T.convert_quat(self.sim.data.qpos[33:37].copy(), to="xyzw")
            elif self.env_name.endswith("PickPlace"):
                if self.text_plan[self.curr_plan_stage][1] == "grasp":
                    all_obj_names = [name.lower() for name in ["Milk", "Bread", "Cereal", "Can"] if name in self.valid_obj_names]
                    new_obj_idx = [name for name in enumerate(all_obj_names) if name[1] in obj_name][0][0]
                    object_pos = self.sim.data.qpos[
                        9 + 7 * new_obj_idx : 12 + 7 * new_obj_idx
                    ].copy()
                    object_quat = T.convert_quat(
                        self.sim.data.qpos[12 + 7 * new_obj_idx : 16 + 7 * new_obj_idx].copy(),
                        to="xyzw",
                    )
                elif self.text_plan[self.curr_plan_stage][1] == "place":
                    assert self.text_plan[self.curr_plan_stage][0].lower().startswith("bin"), "placement location must be a bin"
                    bin_num = int(self.text_plan[self.curr_plan_stage][0][-1])
                    object_pos = self.pick_place_bin_locations[bin_num - 1].copy()
                    object_pos[2] += 0.125
                    object_quat = np.zeros(4)
            elif self.env_name.startswith("Door"):
                object_pos = np.array(
                    [self.sim.data.qpos[self.hinge_qpos_addr]]
                )  # this is not what they are, but they will be decoded properly
                object_quat = np.array(
                    [self.sim.data.qpos[self.handle_qpos_addr]]
                )  # this is not what they are, but they will be decoded properly
            elif "NutAssembly" in self.env_name:
                if "grasp" in self.text_plan[self.curr_plan_stage][1]:
                    if "gold" in obj_name:
                        nut = self.nuts[0]
                    elif "silver" in obj_name:
                        nut = self.nuts[1]
                    else:
                        raise NotImplementedError
                    object_pos = self.sim.data.get_site_xpos(nut.important_sites["handle"])
                    object_quat = T.convert_quat(
                        self.sim.data.body_xquat[self.obj_body_id[nut.name]], to="xyzw"
                    )
                elif "place" in self.text_plan[self.curr_plan_stage][1]:
                    if "gold" in obj_name:
                        object_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])
                    elif "silver" in obj_name:
                        object_pos = np.array(self.sim.data.body_xpos[self.peg2_body_id])
                    object_pos[2] += 0.15
                    object_pos[0] -= 0.065
                    object_quat = np.zeros(4) # not used
            else:
                raise NotImplementedError
        return object_pos, object_quat
    
    def named_get_object_pose(self, obj_name):
        if self.env_name.endswith("Lift"):
            assert "cube" in obj_name.lower(), f"Object {obj_name} does not exist in environment!"
            object_pos = self.sim.data.qpos[9:12].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[12:16].copy(), to="xyzw")
        elif self.env_name.startswith("PickPlaceMilk"):
            assert "milk" in obj_name.lower(), f"Object {obj_name} does not exist in environment!"
            object_pos = self.sim.data.qpos[9:12].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[12:16].copy(), to="xyzw")
        elif self.env_name.startswith("PickPlaceBread"):
            assert "bread" in obj_name.lower(), f"Object {obj_name} does not exist in environment!"
            object_pos = self.sim.data.qpos[16:19].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[19:23].copy(), to="xyzw")
        elif self.env_name.startswith("PickPlaceCereal"):
            assert "cereal" in obj_name.lower(), f"Object {obj_name} does not exist in environment!"
            object_pos = self.sim.data.qpos[23:26].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[26:30].copy(), to="xyzw")
        elif self.env_name.startswith("PickPlaceCan"):
            assert "can" in obj_name.lower(), f"Object {obj_name} does not exist in environment!"
            object_pos = self.sim.data.qpos[30:33].copy()
            object_quat = T.convert_quat(self.sim.data.qpos[33:37].copy(), to="xyzw")
        elif self.env_name.endswith("PickPlace"):
            all_obj_names = [name.lower() for name in ["Milk", "Bread", "Cereal", "Can"] if name in self.valid_obj_names]
            new_obj_idx = [name for name in enumerate(all_obj_names) if name[1] in obj_name][0][0]
            object_pos = self.sim.data.qpos[
                9 + 7 * new_obj_idx : 12 + 7 * new_obj_idx
            ].copy()
            object_quat = T.convert_quat(
                self.sim.data.qpos[12 + 7 * new_obj_idx : 16 + 7 * new_obj_idx].copy(),
                to="xyzw",
            )
        elif self.env_name.startswith("Door"):
            object_pos = np.array(
                [self.sim.data.qpos[self.hinge_qpos_addr]]
            )  # this is not what they are, but they will be decoded properly
            object_quat = np.array(
                [self.sim.data.qpos[self.handle_qpos_addr]]
            )  # this is not what they are, but they will be decoded properly
        elif "NutAssembly" in self.env_name:
            if "gold" in obj_name:
                nut = self.nuts[0]
            elif "silver" in obj_name:
                nut = self.nuts[1]
            if nut.name == "SquareNut":
                return np.array(self.sim.data.qpos[9:12]), T.convert_quat(
                    self.sim.data.qpos[12:16], to="xyzw"
                )
            else:
                return np.array(self.sim.data.qpos[16:19]), T.convert_quat(
                    self.sim.data.qpos[19:23], to="xyzw"
                )
        else:
            raise NotImplementedError
        return object_pos, object_quat

    def get_target_pos(self):
        pos, obj_quat = self.named_get_object_pose_mp(self.text_plan[self.curr_plan_stage][0])
        if self.estimate_orientation and self.text_plan[self.curr_plan_stage][1] == "grasp": #self.num_high_level_steps % 2 == 0:
            pos += np.array([0.0, 0.0, self.vertical_displacement])
            quat = self.compute_hardcoded_orientation(pos, obj_quat)
        else:
            quat = self.reset_ori
        return pos, quat

    def check_object_placement(self, obj_idx=0):
        if "PickPlace" in self.env_name:
            new_obj_idx = self.compute_correct_obj_idx(obj_idx)
            placed = self.objects_in_bins[new_obj_idx]
        elif "NutAssembly" in self.env_name:
            placed = self.objects_on_pegs[1 - obj_idx]
        else:
            placed = True  # just dummy value if not pickplace/nut
        if self.use_vision_placement_check:
            obj_xyz = compute_object_pcd(
                self,
                obj_idx=obj_idx,
                grasp_pose=False,
                target_obj=False,
                camera_height=256,
                camera_width=256,
            )
            obj_pos = np.mean(obj_xyz, axis=0)
            if "PickPlace" in self.env_name:
                # get bin pcd
                # get extent of pcd (min/max x/y) - bin size
                # get avg of pcd - bin pos
                xyz = self.target_pcd
                new_obj_idx = self.compute_correct_obj_idx(obj_idx=obj_idx)
                bin_size = np.array(
                    [max(xyz[0]) - min(xyz[0]), max(xyz[1]) - min(xyz[1])]
                )
                bin2_pos = np.mean(xyz, axis=0)
                bin_x_low = bin2_pos[0]
                bin_y_low = bin2_pos[1]
                if new_obj_idx == 0 or new_obj_idx == 2:
                    bin_x_low -= bin_size[0] / 2
                if new_obj_idx < 2:
                    bin_y_low -= bin_size[1] / 2

                bin_x_high = bin_x_low + bin_size[0] / 2
                bin_y_high = bin_y_low + bin_size[1] / 2

                new_placed = False
                if (
                    bin_x_low < obj_pos[0] < bin_x_high
                    and bin_y_low < obj_pos[1] < bin_y_high
                    and bin2_pos[2] < obj_pos[2] < bin2_pos[2] + 0.1
                ):
                    new_placed = True
            elif "NutAssembly" in self.env_name:
                if self.env_name.endswith("Square") or self.env_name.endswith("Round"):
                    peg_pos = self.placement_poses[0].copy()
                else:
                    peg_pos = self.placement_poses[obj_idx][0].copy()
                # basically undo the peg pos target pose
                peg_pos[2] -= 0.15
                peg_pos[0] += 0.065
                placed = False
                if (
                    abs(obj_pos[0] - peg_pos[0]) < 0.03
                    and abs(obj_pos[1] - peg_pos[1]) < 0.03
                    and obj_pos[2]
                    < self.table_offset[2] + 0.05  # TODO: don't hardcode table offset
                ):
                    placed = True
            else:
                placed = True
        return placed

    def compute_ik(self, target_pos, target_quat, qpos, qvel, og_qpos, og_qvel):
        # reset to canonical state before doing IK
        self.sim.data.qpos[:7] = qpos[:7]
        self.sim.data.qvel[:7] = qvel[:7]
        self.sim.forward()
        self.ik_ctrl.sync_state()
        cur_rot_inv = quat_conjugate(self._eef_xquat.copy())
        pos_diff = target_pos - self._eef_xpos
        rot_diff = quat2mat(quat_multiply(target_quat, cur_rot_inv))
        joint_pos = np.array(
            self.ik_ctrl.joint_positions_for_eef_command(pos_diff, rot_diff)
        )

        # clip joint positions to be within joint limits
        joint_pos = np.clip(
            joint_pos, self.sim.model.jnt_range[:7, 0], self.sim.model.jnt_range[:7, 1]
        )

        self.sim.data.qpos = og_qpos
        self.sim.data.qvel = og_qvel
        self.sim.forward()
        return joint_pos

    def set_object_pose(self, object_pos, object_quat, obj_name):
        if len(object_quat) == 4: 
            object_quat = T.convert_quat(object_quat, to="wxyz")
        if self.env_name.endswith("Lift"):
            self.sim.data.qpos[9:12] = object_pos
            self.sim.data.qpos[12:16] = object_quat
        elif self.env_name.startswith("PickPlaceBread"):
            self.sim.data.qpos[16:19] = object_pos
            self.sim.data.qpos[19:23] = object_quat
        elif self.env_name.startswith("PickPlaceMilk"):
            self.sim.data.qpos[9:12] = object_pos
            self.sim.data.qpos[12:16] = object_quat
        elif self.env_name.startswith("PickPlaceCereal"):
            self.sim.data.qpos[23:26] = object_pos
            self.sim.data.qpos[26:30] = object_quat
        elif self.env_name.startswith("PickPlaceCan"):
            self.sim.data.qpos[30:33] = object_pos
            self.sim.data.qpos[33:37] = object_quat
        elif self.env_name.endswith("PickPlace"):
            all_obj_names = [name.lower() for name in ["Milk", "Bread", "Cereal", "Can"] if name in self.valid_obj_names]
            new_obj_idx = [name for name in enumerate(all_obj_names) if name[1] in obj_name][0][0]
            self.sim.data.qpos[9 + 7 * new_obj_idx : 12 + 7 * new_obj_idx] = object_pos
            self.sim.data.qpos[
                12 + 7 * new_obj_idx : 16 + 7 * new_obj_idx
            ] = object_quat
        elif self.env_name.startswith("Door"):
            self.sim.data.qpos[self.hinge_qpos_addr] = object_pos
            self.sim.data.qpos[self.handle_qpos_addr] = object_quat
        elif "NutAssembly" in self.env_name:
            if "gold" in obj_name:
                nut = self.nuts[0]
            elif "silver" in obj_name:
                nut = self.nuts[1]
            # elif self.env_name.endswith("NutAssembly"):
            #     nut = self.nuts[1 - obj_idx]  # first nut is round, second nut is square
            self.sim.data.set_joint_qpos(
                nut.joints[0],
                np.concatenate([np.array(object_pos), np.array(object_quat)]),
            )
        else:
            raise NotImplementedError

    def rebuild_controller(self):
        new_args = copy.deepcopy(self.controller_configs)
        update_controller_config(self, new_args)
        osc_ctrl = controller_factory("OSC_POSE", new_args)
        osc_ctrl.update_base_pose(self.robots[0].base_pos, self.robots[0].base_ori)
        osc_ctrl.reset_goal()
        self.robots[0].controller = osc_ctrl

    def set_robot_based_on_ee_pos(
        self,
        target_pos,
        target_quat,
        qpos,
        qvel,
        obj_name=None,
    ):
        # recompute is grasped in here 
        is_grasped = self.named_check_object_grasp(
            self.text_plan[self.curr_plan_stage - (self.curr_plan_stage % 2)][0]
        )()
        object_pos, object_quat = self.named_get_object_pose(obj_name) #self.get_object_pose(obj_idx=obj_idx)
        object_pos = object_pos.copy()
        object_quat = object_quat.copy()
        gripper_qpos = self.sim.data.qpos[7:9].copy()
        gripper_qvel = self.sim.data.qvel[7:9].copy()
        old_eef_xquat = self._eef_xquat.copy()
        old_eef_xpos = self._eef_xpos.copy()
        og_qpos = self.sim.data.qpos.copy()
        og_qvel = self.sim.data.qvel.copy()
        joint_pos = self.compute_ik(
            target_pos, target_quat, qpos, qvel, og_qpos, og_qvel
        )
        self.robots[0].set_robot_joint_positions(joint_pos)
        assert (
            self.sim.data.qpos[:7] - joint_pos
        ).sum() < 1e-10  # ensure we accurately set the sim pose to the ik command
        if is_grasped:
            self.sim.data.qpos[7:9] = gripper_qpos
            self.sim.data.qvel[7:9] = gripper_qvel
            # compute the transform between the old and new eef poses
            ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
            ee_new_mat = pose2mat((self._eef_xpos, self._eef_xquat))
            transform = ee_new_mat @ np.linalg.inv(ee_old_mat)
            # apply the transform to the object
            new_object_pose = mat2pose(
                np.dot(transform, pose2mat((object_pos, object_quat)))
            )
            self.set_object_pose(
                new_object_pose[0], new_object_pose[1], obj_name
            )
            self.sim.forward()
        else:
            self.set_object_pose(object_pos, object_quat, obj_name)

        if not is_grasped:
            try:
                self.sim.data.qpos[7:9] = np.array([0.04, -0.04])
                self.sim.data.qvel[7:9] = np.zeros(2)
                self.sim.forward()
            except:
                pass
        else:
            try:
                self.sim.data.qpos[7:9] = gripper_qpos
                self.sim.data.qvel[7:9] = gripper_qvel
                self.sim.forward()
            except:
                pass

        self.rebuild_controller()

        ee_error = np.linalg.norm(self._eef_xpos - target_pos)
        return ee_error

    def set_robot_based_on_joint_angles(
        self,
        joint_pos,
        qpos,
        qvel,
        is_grasped=False,
        obj_idx=0,
        open_gripper_on_tp=True,
    ):
        obj_idx = self.compute_correct_obj_idx(obj_idx)
        object_pos, object_quat = self.get_object_pose(
            obj_idx=self.compute_correct_obj_idx(obj_idx)
        )
        object_pos = object_pos.copy()
        object_quat = object_quat.copy()
        gripper_qpos = self.sim.data.qpos[7:9].copy()
        gripper_qvel = self.sim.data.qvel[7:9].copy()
        old_eef_xquat = self._eef_xquat.copy()
        old_eef_xpos = self._eef_xpos.copy()
        self.robots[0].set_robot_joint_positions(joint_pos)
        assert (self.sim.data.qpos[:7] - joint_pos).sum() < 1e-10
        if is_grasped:
            self.sim.data.qpos[7:9] = gripper_qpos
            self.sim.data.qvel[7:9] = gripper_qvel

            # compute the transform between the old and new eef poses
            ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
            ee_new_mat = pose2mat((self._eef_xpos, self._eef_xquat))
            transform = ee_new_mat @ np.linalg.inv(ee_old_mat)

            # apply the transform to the object
            new_object_pose = mat2pose(
                np.dot(transform, pose2mat((object_pos, object_quat)))
            )
            self.set_object_pose(
                new_object_pose[0], new_object_pose[1], obj_idx=obj_idx
            )
            self.sim.forward()
        else:
            # make sure the object is back where it started
            self.set_object_pose(object_pos, object_quat, obj_idx=obj_idx)

        if open_gripper_on_tp:
            self.sim.data.qpos[7:9] = np.array([0.04, -0.04])
            self.sim.data.qvel[7:9] = np.zeros(2)
            self.sim.forward()
        else:
            self.sim.data.qpos[7:9] = gripper_qpos
            self.sim.data.qvel[7:9] = gripper_qvel
            self.sim.forward()
        self.rebuild_controller()

    def backtracking_search_from_goal(
        self,
        start_pos,
        start_quat,
        target_pos,
        target_quat,
        qpos,
        qvel,
        is_grasped=False,
        movement_fraction=0.001,
        open_gripper_on_tp=True,
        obj_idx=0,
    ):
        curr_pos = target_pos.copy()
        self.set_robot_based_on_ee_pos(
            curr_pos,
            target_quat,
            qpos,
            qvel,
            is_grasped=is_grasped,
            obj_idx=obj_idx,
            open_gripper_on_tp=open_gripper_on_tp,
        )
        collision = self.check_robot_collision(
            ignore_object_collision=is_grasped, obj_idx=obj_idx
        )
        iters = 0
        max_iters = int(1 / movement_fraction)
        while collision and iters < max_iters:
            curr_pos = curr_pos - movement_fraction * (target_pos - start_pos)
            self.set_robot_based_on_ee_pos(
                curr_pos,
                target_quat,
                qpos,
                qvel,
                is_grasped=is_grasped,
                obj_idx=obj_idx,
                open_gripper_on_tp=open_gripper_on_tp,
            )
            collision = self.check_robot_collision(
                ignore_object_collision=is_grasped, obj_idx=obj_idx
            )
            iters += 1
        if collision:
            return np.concatenate((start_pos, start_quat))
        else:
            return np.concatenate((curr_pos, target_quat))

    def backtracking_search_from_goal_joints(
        self,
        start_angles,
        goal_angles,
        qpos,
        qvel,
        obj_idx=0,
        is_grasped=False,
        movement_fraction=0.001,
    ):
        curr_angles = goal_angles.copy()
        valid = self.check_state_validity_joint(
            curr_angles,
            qpos,
            qvel,
            is_grasped=is_grasped,
            obj_idx=obj_idx,
            ignore_object_collision=is_grasped,
        )
        collision = not valid
        iters = 0
        max_iters = int(1 / movement_fraction)
        while collision and iters < max_iters:
            curr_angles = curr_angles - movement_fraction * (goal_angles - start_angles)
            valid = self.check_state_validity_joint(
                curr_angles,
                qpos,
                qvel,
                is_grasped=is_grasped,
                obj_idx=obj_idx,
                ignore_object_collision=is_grasped,
            )
            collision = not valid
            iters += 1
        if collision:
            return start_angles
        else:
            return curr_angles

    def check_robot_collision(self, ignore_object_collision, obj_idx=0):
        obj_string = self.get_object_string(obj_idx=obj_idx)
        d = self.sim.data
        for coni in range(d.ncon):
            con1 = self.sim.model.geom_id2name(d.contact[coni].geom1)
            con2 = self.sim.model.geom_id2name(d.contact[coni].geom2)
            if check_robot_string(con1) ^ check_robot_string(con2):
                if (
                    check_string(con1, obj_string)
                    or check_string(con2, obj_string)
                    and ignore_object_collision
                ):
                    # if the robot and the object collide, then we can ignore the collision
                    continue
                return True
            elif ignore_object_collision:
                if check_string(con1, obj_string) or check_string(con2, obj_string):
                    # if we are supposed to be "ignoring object collisions" then we assume the
                    # robot is "joined" to the object. so if the object collides with any non-robot
                    # object, then we should call that a collision
                    body1 = self.sim.model.body_id2name(
                        self.sim.model.geom_bodyid[d.contact[coni].geom1]
                    )
                    body2 = self.sim.model.body_id2name(
                        self.sim.model.geom_bodyid[d.contact[coni].geom2]
                    )
                    return True
        return False
    
    def named_check_object_placement(self, obj_name):
        def check_object_placement(obj_name=obj_name):
            is_dropped = \
                not (self.named_check_object_grasp(self.text_plan[self.curr_plan_stage - 1][0]))()
            if self.use_vision_placement_check:
                # get object point cloud 
                obj_xyz = compute_object_pcd(
                    self,
                    obj_name=self.text_plan[self.curr_plan_stage - (self.curr_plan_stage % 2)][0],
                    grasp_pose=False,
                    target_obj=False,
                    camera_height=256,
                    camera_width=256,
                )
                obj_pos = np.mean(obj_xyz, axis=0)
                if "NutAssembly" in self.env_name: 
                    pass 
                elif "PickPlace" in self.env_name:
                    pass 
            else:
                if self.env_name == "Lift" or self.env_name == "Door":
                    return False # not used for these environments
                if "NutAssembly" in self.env_name:
                    #obj_name = self.text_plan[self.curr_plan_stage - (self.curr_plan_stage % 2)][0]
                    if "gold" in obj_name:
                        placed = self.objects_on_pegs[0]
                    if "silver" in obj_name:
                        placed = self.objects_on_pegs[1]
                if "PickPlace" in self.env_name:
                    obj_name = self.text_plan[self.curr_plan_stage - (self.curr_plan_stage % 2)][0]
                    all_obj_names = [name.lower() for name in ["Milk", "Bread", "Cereal", "Can"] if name in self.valid_obj_names]
                    new_obj_idx = [name for name in enumerate(all_obj_names) if name[1] in obj_name][0][0]
                    placed = self.objects_in_bins[new_obj_idx]
            return is_dropped and placed 
        return check_object_placement                 
        
    def named_check_object_grasp(self, obj_name):
        if self.use_vision_grasp_check: # using vision
            raise NotImplementedError
        else: # hardcoded 
            def check_object_grasp():
                if self.env_name.endswith("Lift"):
                    is_grasped = self._check_grasp(
                        gripper=self.robots[0].gripper,
                        object_geoms=self.cube,
                    )
                if self.env_name.endswith("PickPlace"):
                    all_obj_names = [name.lower() for name in ["Milk", "Bread", "Cereal", "Can"] if name in self.valid_obj_names]
                    new_obj_idx = [name for name in enumerate(all_obj_names) if name[1] in obj_name][0][0]
                    is_grasped = self._check_grasp(
                        gripper=self.robots[0].gripper,
                        object_geoms=self.objects[new_obj_idx],
                    )
                if self.env_name.startswith("NutAssembly"):
                    if "gold" in obj_name:
                        is_grasped = self._check_grasp(
                            gripper=self.robots[0].gripper,
                            object_geoms=[g for g in self.nuts[0].contact_geoms],
                        )
                    if "silver" in obj_name:
                        is_grasped = self._check_grasp(
                            gripper=self.robots[0].gripper,
                            object_geoms=[g for g in self.nuts[1].contact_geoms],
                        )
                initial_object_pos = self.initial_object_pos_dict[obj_name]
                pos, quat = self.named_get_object_pose_mp(obj_name)
                is_grasped = is_grasped and (pos[2] - initial_object_pos[2]) > 0.005
                return is_grasped
            return check_object_grasp 
    
    # def check_object_grasp(self, obj_idx=0):
    #     if self.env_name.endswith("Lift"):
    #         is_grasped = self._check_grasp(
    #             gripper=self.robots[0].gripper,
    #             object_geoms=self.cube,
    #         )
    #     elif self.env_name.startswith("PickPlace"):
    #         if self.env_name.endswith("PickPlace"):
    #             is_grasped = self._check_grasp(
    #                 gripper=self.robots[0].gripper,
    #                 object_geoms=self.objects[
    #                     self.compute_correct_obj_idx(obj_idx=obj_idx)
    #                 ],
    #             )
    #         else:
    #             is_grasped = self._check_grasp(
    #                 gripper=self.robots[0].gripper,
    #                 object_geoms=self.objects[self.object_id],
    #             )
    #     elif self.env_name.endswith("NutAssemblySquare"):
    #         nut = self.nuts[0]
    #         is_grasped = self._check_grasp(
    #             gripper=self.robots[0].gripper,
    #             object_geoms=[g for g in nut.contact_geoms],
    #         )
    #     elif self.env_name.endswith("NutAssemblyRound"):
    #         nut = self.nuts[1]
    #         is_grasped = self._check_grasp(
    #             gripper=self.robots[0].gripper,
    #             object_geoms=[g for g in nut.contact_geoms],
    #         )
    #     elif self.env_name.endswith("NutAssembly"):
    #         nut = self.nuts[1 - obj_idx]
    #         is_grasped = self._check_grasp(
    #             gripper=self.robots[0].gripper,
    #             object_geoms=[g for g in nut.contact_geoms],
    #         )
    #     elif self.env_name.endswith("Door"):
    #         is_grasped = self._check_grasp(
    #             gripper=self.robots[0].gripper,
    #             object_geoms=[self.door],
    #         )
    #     else:
    #         raise NotImplementedError
    #     if self.use_vision_grasp_check:
    #         is_grasped = grasp_pcd_collision_check(self, obj_idx=obj_idx)
    #     return is_grasped

    # def check_grasp(self):
    #     obj_idx = None
    #     curr_obj_name = self.curr_obj_name
    #     for i in range(len(self.text_plan)):
    #         if self.text_plan[i][0] == curr_obj_name:
    #             obj_idx = i // 2
    #     assert obj_idx is not None
    #     is_grasped = self.check_object_grasp(obj_idx=obj_idx)
    #     if is_grasped:
    #         pos, quat = self.get_object_pose_mp(obj_idx=obj_idx)
    #         init_object_pos = self.initial_object_pos[obj_idx]
    #         is_grasped = is_grasped and (pos[2] - init_object_pos[2]) > 0.005
    #     return is_grasped

    def get_poses_from_obj_name(self, curr_obj_name):
        if self.env_name.startswith("PickPlace"):
            idx = self.valid_obj_names.index(curr_obj_name)
            placement_pose, placement_quat = self.get_placement_pose(obj_idx=idx)
            placement_pose += np.array([0.0, 0.0, self.vertical_displacement])
            return self.get_object_pose_mp(obj_idx=idx), (
                placement_pose,
                placement_quat,
            )
        else:
            raise NotImplementedError

    def get_observation(self):
        di = self._wrapped_env._get_observations(force_update=True)
        if type(self._wrapped_env) == GymWrapper:
            return self._wrapped_env._flatten_obs(di)
        else:
            return di

    def check_state_validity_joint(
        self,
        joint_pos,
        qpos,
        qvel,
        is_grasped,
        obj_idx,
        ignore_object_collision=False,
    ):
        if self.use_pcd_collision_check:
            raise NotImplementedError
        else:
            self.set_robot_based_on_joint_angles(
                joint_pos,
                qpos,
                qvel,
                is_grasped=is_grasped,
                obj_idx=obj_idx,
                open_gripper_on_tp=not is_grasped,
            )
            valid = not self.check_robot_collision(
                ignore_object_collision=ignore_object_collision,
                obj_idx=obj_idx,
            )
        return valid

    def get_joint_bounds(self):
        return self.sim.model.jnt_range[:7, :].copy().astype(np.float64)

    def update_mp_controllers(self):
        if self.use_joint_space_mp:
            update_controller_config(self, self.jp_controller_config)
            self.jp_ctrl = controller_factory(
                "JOINT_POSITION", self.jp_controller_config
            )
            self.jp_ctrl.update_base_pose(
                self.robots[0].base_pos, self.robots[0].base_ori
            )
            self.jp_ctrl.reset_goal()
        else:
            update_controller_config(self, self.osc_controller_config)
            self.osc_ctrl = controller_factory("OSC_POSE", self.osc_controller_config)
            self.osc_ctrl.update_base_pose(
                self.robots[0].base_pos, self.robots[0].base_ori
            )
            self.osc_ctrl.reset_goal()

    def take_mp_step(self, state, is_grasped):
        if self.use_joint_space_mp:
            policy_step = True
            if is_grasped:
                grip_val = self.grip_ctrl_scale
            else:
                grip_val = -1
            action = np.concatenate([(state - self.sim.data.qpos[:7]), [grip_val]])
            if np.linalg.norm(action) < 1e-5:
                self.break_mp = True
                return
            for i in range(int(self.control_timestep // self.model_timestep)):
                self.sim.forward()
                apply_controller(self.jp_ctrl, action, self.robots[0], policy_step)
                self.sim.step()
                self._update_observables()
        else:
            desired_rot = quat2mat(state[3:])
            current_rot = quat2mat(self._eef_xquat)
            rot_delta = orientation_error(desired_rot, current_rot)
            pos_delta = state[:3] - self._eef_xpos
            if is_grasped:
                grip_ctrl = self.grip_ctrl_scale
            else:
                grip_ctrl = -1
            action = np.concatenate((pos_delta, rot_delta, [grip_ctrl]))
            if np.linalg.norm(action[:-4]) < 1e-3:
                return
            policy_step = True
            for i in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                apply_controller(self.osc_ctrl, action, self.robots[0], policy_step)
                self.sim.step()
                self._update_observables()
                policy_step = False

    def get_eef_xpos(self):
        return self._eef_xpos

    def get_eef_xquat(self):
        return self._eef_xquat

    def process_state_frames(self, frames):
        raise NotImplementedError

    def get_robot_mask(self):
        sim = self.sim
        segmentation_map = CU.get_camera_segmentation(
            camera_name="frontview",
            camera_width=960,
            camera_height=540,
            sim=sim,
        )
        geom_ids = np.unique(segmentation_map[:, :, 1])
        robot_ids = []
        for geom_id in geom_ids:
            geom_name = sim.model.geom_id2name(geom_id)
            if geom_name is None or geom_name.startswith("Visual"):
                continue
            if geom_name.startswith("robot0") or geom_name.startswith("gripper"):
                robot_ids.append(geom_id)
        robot_mask = np.expand_dims(
            np.any(
                [segmentation_map[:, :, 1] == robot_id for robot_id in robot_ids],
                axis=0,
            ),
            -1,
        )
        return robot_mask
