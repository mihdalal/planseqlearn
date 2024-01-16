import numpy as np
from planseqlearn.psl.inverse_kinematics import qpos_from_site_pose_kitchen
from planseqlearn.psl.mp_env import PSLEnv
from planseqlearn.psl.vision_utils import *
import robosuite.utils.transform_utils as T
from robosuite.utils.transform_utils import *


def check_object_grasp(env, obj_idx=0):
    element = env.TASK_ELEMENTS[obj_idx]
    is_grasped = False

    if element == "slide cabinet":
        for i in range(1, 6):
            obj_pos = env.get_site_xpos("schandle{}".format(i))
            left_pad = env.get_site_xpos("leftpad")
            right_pad = env.get_site_xpos("rightpad")
            within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.07
            within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.07
            right = right_pad[0] < obj_pos[0]
            left = obj_pos[0] < left_pad[0]
            if right and left and within_sphere_right and within_sphere_left:
                is_grasped = True
    if element == "top burner":
        obj_pos = env.get_site_xpos("tlbhandle")
        left_pad = env.get_site_xpos("leftpad")
        right_pad = env.get_site_xpos("rightpad")
        within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.035
        within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.04
        right = right_pad[0] < obj_pos[0]
        left = obj_pos[0] < left_pad[0]
        if within_sphere_right and within_sphere_left and right and left:
            is_grasped = True
    if element == "microwave":
        for i in range(1, 6):
            obj_pos = env.get_site_xpos("mchandle{}".format(i))
            left_pad = env.get_site_xpos("leftpad")
            right_pad = env.get_site_xpos("rightpad")
            within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.05
            within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.05
            if (
                right_pad[0] < obj_pos[0]
                and obj_pos[0] < left_pad[0]
                and within_sphere_right
                and within_sphere_left
            ):
                is_grasped = True
    if element == "hinge cabinet":
        for i in range(1, 6):
            obj_pos = env.get_site_xpos("hchandle{}".format(i))
            left_pad = env.get_site_xpos("leftpad")
            right_pad = env.get_site_xpos("rightpad")
            within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.06
            within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.06
            if (
                right_pad[0] < obj_pos[0]
                and obj_pos[0] < left_pad[0]
                and within_sphere_right
            ):
                is_grasped = True
    if element == "left hinge cabinet":
        for i in range(1, 6):
            obj_pos = env.get_site_xpos("hchandle_left{}".format(i))
            left_pad = env.get_site_xpos("leftpad")
            right_pad = env.get_site_xpos("rightpad")
            within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.06
            within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.06
            if (
                right_pad[0] < obj_pos[0]
                and obj_pos[0] < left_pad[0]
                and within_sphere_right
            ):
                is_grasped = True
    if element == "light switch":
        for i in range(1, 4):
            obj_pos = env.get_site_xpos("lshandle{}".format(i))
            left_pad = env.get_site_xpos("leftpad")
            right_pad = env.get_site_xpos("rightpad")
            within_sphere_left = np.linalg.norm(obj_pos - left_pad) < 0.045
            within_sphere_right = np.linalg.norm(obj_pos - right_pad) < 0.03
            if within_sphere_right and within_sphere_left:
                is_grasped = True
    if element == "kettle":
        # TODO: check if kettle is grasped
        pass
    return is_grasped


def check_robot_string(string):
    if string is None:
        return False
    return string.startswith("panda0") or string.startswith("gripper")


def get_object_string(env, obj_idx=0):
    element = env.TASK_ELEMENTS[obj_idx]
    return element


def check_string(string, other_string):
    if string is None:
        return False
    return string.startswith(other_string)


class KitchenPSLEnv(PSLEnv):
    def __init__(
        self,
        env,
        env_name,
        **kwargs,
    ):
        super().__init__(
            env,
            env_name,
            **kwargs,
        )
        self.eef_id = self._wrapped_env.sim.model.site_name2id("end_effector")
        self.robot_bodies = [
            "panda0_link0",
            "panda0_link1",
            "panda0_link2",
            "panda0_link3",
            "panda0_link4",
            "panda0_link5",
            "panda0_link6",
            "panda0_link7",
            "panda0_leftfinger",
            "panda0_rightfinger",
        ]
        (
            self.robot_body_ids,
            self.robot_geom_ids,
        ) = self.get_body_geom_ids_from_robot_bodies()
        self.original_colors = [
            env.sim.model.geom_rgba[idx].copy() for idx in self.robot_geom_ids
        ]
        if self.use_sam_segmentation:
            self.precalculate_poses()
    
    def precalculate_poses(self):
        #self._wrapped_env.reset()
        # this function is mainly used to precalculate poses
        # with sam - the reason we are precalculating is because
        # we are assuming sam is being used for inference with a learned policy
        # so collisions will be minimal - in this case since sam is 
        # quite brittle with grounding dino, precalculating enables using the 
        # same prompts as during testing without issues
        pos_dict = {}
        # get poses for kettle, microwave, and slide
        frame = self.mjpy_sim.render(camera_name="leftview", width=500, height=500)
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            self.dino,
            self.sam,
            text_prompts=["vertical bar"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        # list should be [slide cabinet, ___, kettle, ___, ____, microwave]
        assert len(pred_phrases) == 5
        slide_mask = obj_masks[0].detach().cpu().numpy()[0, :, :]
        kettle_mask = obj_masks[2].detach().cpu().numpy()[0, :, :]
        microwave_mask = obj_masks[-1].detach().cpu().numpy()[0, :, :]
        depth_map = get_camera_depth(
            sim=self.mjpy_sim,
            camera_name="leftview",
            camera_height=500,
            camera_width=500,
        )
        depth_map = np.expand_dims(
            CU.get_real_depth_map(sim=self.mjpy_sim, depth_map=depth_map), -1
        )
        world_to_camera = CU.get_camera_transform_matrix(
            sim=self.mjpy_sim,
            camera_name="leftview",
            camera_height=500,
            camera_width=500,
        )
        camera_to_world = np.linalg.inv(world_to_camera)
        slide_pixels = np.argwhere(slide_mask)
        slide_pointcloud = CU.transform_from_pixels_to_world(
            pixels=slide_pixels,
            depth_map=depth_map[..., 0],
            camera_to_world_transform=camera_to_world,
        )
        kettle_pixels = np.argwhere(kettle_mask)
        kettle_pointcloud = CU.transform_from_pixels_to_world(
            pixels=kettle_pixels,
            depth_map=depth_map[..., 0],
            camera_to_world_transform=camera_to_world,
        )
        microwave_pixels = np.argwhere(microwave_mask)
        microwave_pointcloud = CU.transform_from_pixels_to_world(
            pixels=microwave_pixels,
            depth_map=depth_map[..., 0],
            camera_to_world_transform=camera_to_world,
        )
        pos_dict['schandle1'] = np.mean(slide_pointcloud, axis=0)
        pos_dict['khandle1'] = np.mean(kettle_pointcloud, axis=0) + np.array([0.09, 0., 0.])
        pos_dict['mchandle1'] = np.mean(microwave_pointcloud, axis=0)

        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            self.dino,
            self.sam,
            text_prompts=["small knob"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        light_mask = obj_masks[-2].detach().cpu().numpy()[0, :, :]
        light_pixels = np.argwhere(light_mask)
        light_pointcloud = CU.transform_from_pixels_to_world(
            pixels=light_pixels,
            depth_map=depth_map[..., 0],
            camera_to_world_transform=camera_to_world,
        )
        pos_dict['lschandle1'] = np.mean(light_pointcloud, axis=0)
        self.pos_dict = pos_dict

    def get_body_geom_ids_from_robot_bodies(self):
        body_ids = [self.sim.model.body_name2id(body) for body in self.robot_bodies]
        geom_ids = []
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            if body_id in body_ids:
                geom_ids.append(geom_id)
        return body_ids, geom_ids

    @property
    def _eef_xpos(self):
        return self.sim.data.site_xpos[self.eef_id]

    @property
    def _eef_xquat(self):
        quat = T.convert_quat(
            T.mat2quat(self.sim.data.site_xmat[self.eef_id].reshape(3, 3)), to="wxyz"
        )
        return quat

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def check_grasp(self, obj_idx=0):
        return check_object_grasp(self, obj_idx=obj_idx)

    def check_object_placement(self, **kwargs):
        pass

    def check_robot_collision(
        self,
        ignore_object_collision=False,
        obj_idx=0,
    ):
        obj_string = get_object_string(self, obj_idx=obj_idx)
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
                    return True
        return False

    def get_site_xpos(self, name):
        site_id = self.sim.model.site_name2id(name)
        return self.sim.data.site_xpos[site_id]

    def get_site_xquat(self, name):
        site_id = self.sim.model.site_name2id(name)
        quat = T.convert_quat(
            T.mat2quat(self.sim.data.site_xmat[site_id].reshape(3, 3)), to="wxyz"
        )
        return quat

    def get_site_xmat(self, name):
        site_id = self.sim.model.site_name2id(name)
        return self.sim.data.site_xmat[site_id]

    def get_object_pose_mp(self, obj_idx=0):
        element = self.TASK_ELEMENTS[obj_idx]
        if element == "slide cabinet":
            object_pos = self.get_site_xpos("schandle1")
            if self.use_sam_segmentation:
                object_pos = self.pos_dict['schandle1']
            object_quat = np.zeros(4)  # doesn't really matter
        elif element == "top burner":
            object_pos = self.get_site_xpos("tlbhandle")
            object_quat = np.zeros(4)  # doesn't really matter
        elif element == "bottom left burner":
            object_pos = self.get_site_xpos("blbhandle")
            object_quat = np.zeros(4)
        elif element == "bottom right burner":
            object_pos = self.get_site_xpos("brbhandle")
            object_quat = np.zeros(4)
        elif element == "top right burner":
            object_pos = self.get_site_xpos("trbhandle")
            object_quat = np.zeros(4)
        elif element == "hinge cabinet":
            object_pos = self.get_site_xpos("hchandle1")
            object_quat = np.zeros(4)  # doesn't really matter
        elif element == "left hinge cabinet":
            object_pos = self.get_site_xpos("hchandle_left1")
            object_quat = np.zeros(4)
        elif element == "light switch":
            object_pos = self.get_site_xpos("lshandle1")
            object_quat = np.zeros(4)  # doesn't really matter
        elif element == "microwave":
            object_pos = self.get_site_xpos("mchandle1")
            if self.use_sam_segmentation:
                object_pos = self.pos_dict['mchandle1']
            object_quat = np.zeros(4)  # doesn't really matter
        elif element == "kettle":
            object_pos = self.get_site_xpos("khandle1")
            if self.use_sam_segmentation:
                object_pos = self.pos_dict['khandle1']
            object_quat = np.zeros(4)  # doesn't really matter
        return object_pos, object_quat

    def get_object_pose(self, obj_idx=0):
        element = self.TASK_ELEMENTS[obj_idx]
        object_qpos = self.sim.data.qpos[-21:]
        object_pos = object_qpos[self.OBS_ELEMENT_INDICES[element] - 9]
        object_quat = np.zeros(4)  # doesn't really matter
        return object_pos, object_quat

    def get_placement_poses(self):
        return []

    def get_object_poses(self):
        return []  # return none since

    def get_target_pose_list(self):
        pose_list = []
        # init target pos (object pos + vertical displacement)
        # final target positions, depending on the task
        for idx, element in enumerate(self.TASK_ELEMENTS):
            object_pos, object_quat = self.get_object_pose_mp(obj_idx=idx)
            target_pos = object_pos + np.array([0, -0.05, 0])
            target_quat = self.reset_ori
            pose_list.append((target_pos, target_quat))
        return pose_list

    def get_target_pos(self):
        target_pose_list = self.get_target_pose_list()
        if self.num_high_level_steps > len(target_pose_list) - 1:
            return target_pose_list[-1]
        return self.get_target_pose_list()[self.num_high_level_steps]

    def set_robot_based_on_ee_pos(
        self,
        target_pos,
        target_quat,
        qpos,
        qvel,
        is_grasped,
        obj_idx=0,
        open_gripper_on_tp=False,
    ):
        """
        Set robot joint positions based on target ee pose. Uses IK to solve for joint positions.
        If grasping an object, ensures the object moves with the arm in a consistent way.
        """
        # cache quantities from prior to setting the state
        object_pos, object_quat = self.get_object_pose(obj_idx=obj_idx)
        object_pos = object_pos.copy()
        object_quat = object_quat.copy()
        gripper_qpos = self.sim.data.qpos[7:9].copy()
        gripper_qvel = self.sim.data.qvel[7:9].copy()
        old_eef_xquat = self._eef_xquat.copy()
        old_eef_xpos = self._eef_xpos.copy()

        # reset to canonical state before doing IK
        self.sim.data.qpos[:7] = qpos[:7]
        self.sim.data.qvel[:7] = qvel[:7]
        self.sim.forward()
        qpos_from_site_pose_kitchen(
            self,
            "end_effector",
            target_pos=target_pos,
            target_quat=target_quat.astype(np.float64),
            joint_names=[
                "panda0_joint0",
                "panda0_joint1",
                "panda0_joint2",
                "panda0_joint3",
                "panda0_joint4",
                "panda0_joint5",
                "panda0_joint6",
            ],
            tol=1e-14,
            rot_weight=1.0,
            regularization_threshold=0.1,
            regularization_strength=3e-2,
            max_update_norm=2.0,
            progress_thresh=20.0,
            max_steps=1000,
        )
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

        if open_gripper_on_tp:
            self.sim.data.qpos[7:9] = np.array([0.04, 0.04])
            self.sim.data.qvel[7:9] = np.zeros(2)
            self.sim.forward()
        else:
            self.sim.data.qpos[7:9] = gripper_qpos
            self.sim.data.qvel[7:9] = gripper_qvel
            self.sim.forward()

        ee_error = np.linalg.norm(self._eef_xpos - target_pos)
        return ee_error

    def set_object_pose(self, object_pos, object_quat, obj_idx=0):
        element = self.TASK_ELEMENTS[obj_idx]
        self.sim.data.qpos[-21 + self.OBS_ELEMENT_INDICES[element] - 9] = object_pos
        self.sim.forward()

    def set_robot_based_on_joint_angles(
        self,
        joint_pos,
        qpos,
        qvel,
        is_grasped=False,
        obj_idx=0,
        open_gripper_on_tp=False,
    ):
        object_pos, object_quat = self.get_object_pose(obj_idx=obj_idx)
        object_pos = object_pos.copy()
        object_quat = object_quat.copy()
        gripper_qpos = self.sim.data.qpos[7:9].copy()
        gripper_qvel = self.sim.data.qvel[7:9].copy()
        old_eef_xquat = self._eef_xquat.copy()
        old_eef_xpos = self._eef_xpos.copy()

        self.sim.data.qpos[:7] = joint_pos
        self.sim.forward()
        assert (self.sim.data.qpos[:7] - joint_pos).sum() < 1e-10
        # error = np.linalg.norm(env._eef_xpos - pos[:3])

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
            self.sim.data.qpos[7:9] = np.array([0.04, 0.04])
            self.sim.data.qvel[7:9] = np.zeros(2)
            self.sim.forward()
        else:
            self.sim.data.qpos[7:9] = gripper_qpos
            self.sim.data.qvel[7:9] = gripper_qvel
            self.sim.forward()

    def check_state_validity_joint(
        self,
        curr_pos,
        qpos,
        qvel,
        is_grasped=False,
        obj_idx=0,
        ignore_object_collision=False,
        open_gripper_on_tp=False,
    ):
        if self.use_pcd_collision_check:
            raise NotImplementedError
        else:
            self.set_robot_based_on_joint_angles(
                curr_pos,
                qpos,
                qvel,
                is_grasped=is_grasped,
                obj_idx=obj_idx,
                open_gripper_on_tp=open_gripper_on_tp,
            )
            valid = not self.check_robot_collision(
                ignore_object_collision=ignore_object_collision, obj_idx=obj_idx
            )
        return valid

    def get_robot_mask(self):
        segmentation_map = self.sim.render(segmentation=True, height=540, width=960)
        geom_ids = np.unique(segmentation_map[:, :, 1])
        robot_ids = []
        object_ids = []
        object_string = get_object_string(self, obj_idx=0)
        for geom_id in geom_ids:
            if geom_id == -1:
                continue
            geom_name = self.sim.model.geom_id2name(geom_id)
            if geom_name is None or geom_name.startswith("Visual"):
                continue
            if geom_name.startswith("panda0") or geom_name.startswith("gripper"):
                robot_ids.append(geom_id)
            if object_string in geom_name:
                object_ids.append(geom_id)
        robot_mask = np.any(
            [segmentation_map[:, :, 1] == robot_id for robot_id in robot_ids], axis=0
        )[:, :, None]
        return robot_mask

    def process_state_frames(self):
        raise NotImplementedError

    def compute_hardcoded_orientation(self, *args, **kwargs):
        pass

    def backtracking_search_from_goal(
        self,
        start_pos,
        start_quat,
        target_pos,
        target_quat,
        qpos,
        qvel,
        is_grasped=False,
        open_gripper_on_tp=True,
        ignore_object_collision=False,
        obj_idx=0,
        movement_fraction=0.001,
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
            ignore_object_collision=ignore_object_collision, obj_idx=obj_idx
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
                ignore_object_collision=ignore_object_collision, obj_idx=obj_idx
            )
            iters += 1
        if collision:
            return np.concatenate((start_pos, start_quat))
        else:
            return np.concatenate((curr_pos, target_quat))

    def backtracking_search_from_goal_joints(
        self,
        start_pos,
        target_pos,
        qpos,
        qvel,
        is_grasped=False,
        open_gripper_on_tp=True,
        ignore_object_collision=False,
        obj_idx=0,
        movement_fraction=0.001,
    ):
        raise NotImplementedError

    def get_observation(self):
        return np.zeros_like(self.observation_space.low)

    def get_joint_bounds(self):
        env_bounds = self.sim.model.jnt_range[:7, :].copy().astype(np.float64)
        env_bounds[:, 0] -= np.pi
        env_bounds[:, 1] += np.pi
        return env_bounds

    def process_state_frames(self, frames):
        raise NotImplementedError

    def get_image(self):
        im = self.sim.render(
            width=960,
            height=540,
        )
        return im

    def set_robot_colors(self, colors):
        if type(colors) is np.ndarray:
            colors = [colors] * len(self.robot_geom_ids)
        for idx, geom_id in enumerate(self.robot_geom_ids):
            self.sim.model.geom_rgba[geom_id] = colors[idx]
        self.sim.forward()

    def reset_robot_colors(self):
        self.set_robot_colors(self.original_colors)
        self.sim.forward()

    def compute_ik(self, target_pos, target_quat, qpos, qvel, og_qpos, og_qvel):
        target_angles = qpos_from_site_pose_kitchen(
            self,
            "end_effector",
            target_pos=target_pos,
            target_quat=target_quat.astype(np.float64),
            joint_names=[
                "panda_joint0",
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
            ],
            tol=1e-14,
            rot_weight=1.0,
            regularization_threshold=0.1,
            regularization_strength=3e-2,
            max_update_norm=2.0,
            progress_thresh=20.0,
            max_steps=1000,
        ).qpos.copy()[:7]
        self.sim.data.qpos[:] = og_qpos
        self.sim.data.qvel[:] = og_qvel
        self.sim.forward()
        return target_angles

    def update_controllers(self):
        pass

    def update_mp_controllers(self):
        pass

    @property
    def obj_idx(self):
        return self.num_high_level_steps

    def rebuild_controller(self):
        pass

    def take_mp_step(
        self,
        state,
        is_grasped,
    ):
        curr = self.sim.data.qpos[:7]
        new_state = curr + (state - curr) / 5
        self.set_robot_based_on_joint_angles(
            new_state,
            self.reset_qpos,
            self.reset_qvel,
            is_grasped=is_grasped,
            obj_idx=self.obj_idx,
            open_gripper_on_tp=not is_grasped,
        )

    def step(self, action, get_intermediate_frames=False, **kwargs):
        o, r, d, i = self._wrapped_env.step(action)
        take_planner_step = r > 0  # if succeeded at subtask, jump to next subtask
        if take_planner_step:
            target_pos, target_quat = self.get_target_pos()
            if self.teleport_instead_of_mp:
                error = self.set_robot_based_on_ee_pos(
                    target_pos,
                    target_quat,
                    self.reset_qpos,
                    self.reset_qvel,
                    is_grasped=False,
                    obj_idx=0,
                    open_gripper_on_tp=True,
                )
                o = self.get_observation()
            else:
                error = self.mp_to_point(
                    target_pos,
                    target_quat,
                    self.reset_qpos,
                    self.reset_qvel,
                    is_grasped=False,
                    obj_idx=0,
                    open_gripper_on_tp=True,
                    get_intermediate_frames=get_intermediate_frames,
                )
                o = self.get_observation()
            self.num_high_level_steps += 1
        info = {}
        if "microwave" in self.env_name:
            info["microwave_success"] = i["microwave success"]
            info["microwave_distance_to_goal"] = i["microwave distance to goal"]
        if "hinge" in self.env_name:
            info["hinge_success"] = i["hinge cabinet success"]
            info["distance_to_goal"] = i["hinge cabinet distance to goal"]
        if "tlb" in self.env_name:
            info["tlb_success"] = i["top burner success"]
            info["tlb_distance_to_goal"] = i["top burner distance to goal"]
        if "trb" in self.env_name:
            info["trb_success"] = i["top right burner success"]
            info["trb_distance_to_goal"] = i["top right burner distance to goal"]
        if "blb" in self.env_name:
            info["blb_success"] = i["bottom left burner success"]
            info["blb_distance_to_goal"] = i["bottom left burner distance to goal"]
        if "brb" in self.env_name:
            info["brb_success"] = i["bottom right burner success"]
            info["brb_distance_to_goal"] = i["bottom right burner distance to goal"]
        if "kettle" in self.env_name:
            info["kettle_success"] = i["kettle success"]
            info["kettle_distance_to_goal"] = i["kettle distance to goal"]
        if "light" in self.env_name:
            info["light_success"] = i["light switch success"]
            info["light_distance_to_goal"] = i["light switch distance to goal"]
        if "slider" in self.env_name:
            info["slider_success"] = i["slide cabinet success"]
            info["slider_distance_to_goal"] = i["slide cabinet distance to goal"]
        info["score"] = i["score"]
        return o, r, d, info
