import numpy as np
from planseqlearn.mnm.inverse_kinematics import qpos_from_site_pose
from planseqlearn.mnm.mp_env import MPEnv
from planseqlearn.mnm.vision_utils import *
from robosuite.utils.transform_utils import (
    mat2pose,
    pose2mat,
    quat2mat,
)



def set_robot_based_on_ee_pos(
    env,
    target_pos,
    target_quat,
    qpos,
    qvel,
    is_grasped,
):
    """
    Set robot joint positions based on target ee pose. Uses IK to solve for joint positions.
    If grasping an object, ensures the object moves with the arm in a consistent way.
    """
    # cache quantities from prior to setting the state
    object_pose = env.get_object_pose()
    gripper_qpos = env.sim.data.qpos[7:9].copy()
    gripper_qvel = env.sim.data.qvel[7:9].copy()
    old_eef_xquat = env._eef_xquat.copy()
    old_eef_xpos = env._eef_xpos.copy()

    # reset to canonical state before doing IK
    env.sim.data.qpos[:7] = qpos[:7]
    env.sim.data.qvel[:7] = qvel[:7]
    env.sim.forward()
    result = qpos_from_site_pose(
        env,
        "endEffector",
        target_pos=target_pos,
        target_quat=target_quat.astype(np.float64),
        joint_names=[
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
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
        env.sim.data.qpos[7:9] = gripper_qpos
        env.sim.data.qvel[7:9] = gripper_qvel

        # compute the transform between the old and new eef poses
        ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
        ee_new_mat = pose2mat((env._eef_xpos, env._eef_xquat))
        transform = ee_new_mat @ np.linalg.inv(ee_old_mat)

        # apply the transform to the object
        new_object_pose = mat2pose(
            np.dot(transform, pose2mat((object_pose[0], object_pose[1])))
        )
        env.set_object_pose(new_object_pose[0], new_object_pose[1])
        env.sim.forward()
    else:
        # make sure the object is back where it started
        env.set_object_pose(object_pose[0], object_pose[1])

    env.sim.data.qpos[7:9] = gripper_qpos
    env.sim.data.qvel[7:9] = gripper_qvel
    env.sim.forward()

    ee_error = np.linalg.norm(env._eef_xpos - target_pos)
    # need to update the mocap pos post teleport
    env.reset_mocap2body_xpos(env.sim)
    return ee_error


def check_robot_string(string):
    if string is None:
        return False
    return (
        string.startswith("robot")
        or string.startswith("leftclaw")
        or string.startswith("rightclaw")
        or string.startswith("rightpad")
        or string.startswith("leftpad")
    )


def check_string(string, other_string):
    if string is None:
        return False
    return string.startswith(other_string)


class MetaworldMPEnv(MPEnv):
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
        self.ROBOT_BODIES = [
            "right_hand",
            "hand",
            "rightclaw",
            "rightpad",
            "leftclaw",
            "leftpad",
        ]
        self.mp_bounds_low = (-2.0, -2.0, -2.0)
        self.mp_bounds_high = (2.0, 2.0, 2.0)
        self.use_joint_space_mp = False
        self.geom_bodies = [
            "base",
            "controller_box",
            "pedestal_feet",
            "torso",
            "pedestal",
            "right_arm_base_link",
            "right_l0",
            "head",
            "screen",
            "head_camera",
            "right_torso_itb",
            "right_l1",
            "right_l2",
            "right_l3",
            "right_l4",
            "right_arm_itb",
            "right_l5",
            "right_hand_camera",
            "right_wrist",
            "right_l6",
            "right_hand",
            "hand",
            "rightclaw",
            "rightpad",
            "leftclaw",
            "leftpad",
            "right_l4_2",
            "right_l2_2",
            "right_l1_2",
        ]
        (
            self.robot_body_ids,
            self.robot_geom_ids,
        ) = self.get_body_geom_ids_from_robot_bodies()
        self.original_colors = [
            env.sim.model.geom_rgba[idx].copy() for idx in self.robot_geom_ids
        ]
        self.max_path_length = kwargs["max_path_length"]
        self.hardcoded_text_plan = self.get_hardcoded_text_plan()
        self.initial_object_pos = self.get_object_pose()[0].copy()

    def get_body_geom_ids_from_robot_bodies(self):
        body_ids = [self.sim.model.body_name2id(body) for body in self.geom_bodies]
        geom_ids = []
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            if body_id in body_ids:
                geom_ids.append(geom_id)
        return body_ids, geom_ids

    def get_hardcoded_text_plan(self):
        if self.env_name == "assembly-v2":
            return [("green wrench", "grasp"), ("small maroon peg", "place")]
        if self.env_name == "hammer-v2":
            return [("red hammer handle", "grasp"), ("gray nail", "place")]
        if self.env_name == "bin-picking-v2":
            return [("green cube", "grasp"), ("blue bin", "place")]
        if self.env_name == "disassemble-v2":
            return [("green wrench", "grasp"), (None, None)]

    def get_image(self):
        im = self.sim.render(
            camera_name="corner",
            width=960,
            height=540,
        )
        return im

    def get_object_pose_mp(self, obj_idx=0):
        if self.env_name == "assembly-v2":
            object_pos = self._get_pos_objects().copy() + np.array([0.03, 0.0, 0.05])
            object_quat = self._get_quat_objects().copy()
        elif self.env_name == "disassemble-v2":
            object_pos = self._get_pos_objects().copy() + np.array([0.0, 0.0, 0.06])
            object_quat = self._get_quat_objects().copy()
        elif self.env_name == "hammer-v2":
            object_pos = self._get_pos_objects()[:3] + np.array([0.09, 0.0, 0.05])
            object_quat = self._get_quat_objects().copy()
        elif self.env_name == "bin-picking-v2":
            object_pos = self._get_pos_objects().copy() + np.array([0.0, 0.0, 0.02])
            object_quat = self._get_quat_objects().copy()
        if self.use_vision_pose_estimation:
            object_name = (
                self.text_plan[0][0]
                if self.use_llm_plan
                else self.hardcoded_text_plan[0][0]
            )
            if self.env_name == "assembly-v2" or self.env_name == "disassemble-v2":
                if not self.use_sam_segmentation:
                    object_pos = get_geom_pose_from_seg(
                        self,
                        self.sim.model.geom_name2id("WrenchHandle"),
                        ["corner", "corner2"],
                        500,
                        500,
                        self.sim,
                    )
                    object_pos += np.array([0.02, 0.02, 0.03])
                else:
                    object_pos = get_pose_from_sam_segmentation(
                        self, object_name, "corner"
                    )
                    object_pos += np.array([0.08, 0.02, 0.07])
                object_quat = self._get_quat_objects().copy()
            elif self.env_name == "hammer-v2":
                if not self.use_sam_segmentation:
                    object_pos = get_geom_pose_from_seg(
                        self,
                        self.sim.model.geom_name2id("HammerHandle"),
                        ["topview", "corner2"],
                        500,
                        500,
                        self.sim,
                    )
                    object_pos += np.array([0.0, 0.0, 0.05])
                else:
                    object_pos = get_pose_from_sam_segmentation(
                        self, object_name, "corner"
                    )
                    object_pos += np.array([0.0, 0.0, 0.05])
                object_quat = self._get_quat_objects()[:4]
            elif self.env_name == "bin-picking-v2":
                if not self.use_sam_segmentation:
                    object_pos = get_geom_pose_from_seg(
                        self, 36, ["topview", "corner2"], 500, 500, self.sim
                    )
                else:
                    # saving image first
                    object_pos = get_pose_from_sam_segmentation(
                        self, object_name, "corner3"
                    )
                object_pos += np.array([-0.00, 0.0, 0.02])
                object_quat = self._get_quat_objects().copy()
        return object_pos, object_quat

    def get_object_poses(self):
        object_poses = []
        object_poses.append((self.get_object_pose_mp()))
        return object_poses

    def get_placement_pose(self):
        placement_quat = None
        if not self.use_vision_pose_estimation:
            if self.env_name == "assembly-v2":
                placement_pos = self.sim.data.body_xpos[
                    self.sim.model.body_name2id("peg")
                ] + np.array([0.13, 0.0, 0.15])
            elif self.env_name == "bin-picking-v2":
                placement_pos = self._target_pos + np.array([0, 0, 0.15])
            elif self.env_name == "disassemble-v2":
                placement_pos = None
            elif self.env_name == "hammer-v2":
                placement_pos = self._wrapped_env._get_pos_objects()[3:] + np.array(
                    [-0.05, -0.20, 0.05]
                )
        else:
            object_name = (
                self.text_plan[1][0]
                if self.use_llm_plan
                else self.hardcoded_text_plan[1][0]
            )
            if self.env_name == "disassemble-v2":
                placement_pos = None
            elif self.env_name == "hammer-v2":
                if not self.use_sam_segmentation:
                    placement_pos = get_geom_pose_from_seg(
                        self, 53, ["corner2", "topview", "corner3"], 500, 500, self.sim
                    )
                    placement_pos += np.array([-0.15, -0.15, 0.0])
                else:
                    placement_pos = get_pose_from_sam_segmentation(
                        self, object_name, "corner"
                    )
                    placement_pos += np.array([-0.15, -0.15, 0.0])
            elif self.env_name == "bin-picking-v2":
                if not self.use_sam_segmentation:
                    placement_pos = get_geom_pose_from_seg(
                        self, 44, ["corner", "corner2"], 500, 500, self.sim
                    )
                else:
                    placement_pos = get_pose_from_sam_segmentation(
                        self, object_name, "corner3"
                    )
                placement_pos += np.array([0.03, 0.03, 0.10])
            elif self.env_name == "assembly-v2":
                if not self.use_sam_segmentation:
                    placement_pos = get_geom_pose_from_seg(
                        self, 49, ["corner", "corner2"], 500, 500, self.sim
                    )
                else:
                    placement_pos = get_pose_from_sam_segmentation(
                        self, object_name, "corner"
                    )
                placement_pos += np.array([0.13, 0.0, 0.15])
        return placement_pos, placement_quat

    def get_placement_poses(self):
        return [self.get_placement_pose()]

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    def get_observation(self):
        return self._get_obs()

    def get_object_pose(self):
        if self.env_name == "hammer-v2":
            object_pos = self._get_pos_objects()[:3] + np.array([0.0, 0.0, 0.016])
            object_quat = self._get_quat_objects()[:4]
        elif self.env_name == "assembly-v2" or self.env_name == "disassemble-v2":
            object_pos = self._get_pos_objects().copy() - np.array(
                [
                    0.13,
                    0.0,
                    0.0,
                ]
            )
            object_quat = self._get_quat_objects().copy()
        else:
            object_pos = self._get_pos_objects().copy()
            object_quat = self._get_quat_objects().copy()
        return object_pos, object_quat

    def set_object_pose(self, object_pos, object_quat):
        self._set_obj_pose(np.concatenate((object_pos, object_quat)))

    def get_object_string(self):
        if self.env_name == "hammer-v2":
            obj_name = "hammer"
        elif self.env_name == "assembly-v2" or self.env_name == "disassemble-v2":
            obj_name = "asmbly_peg"
        elif self.env_name == "bin-picking-v2":
            obj_name = "objA"
        return obj_name

    def body_check_grasp(self):
        obj_name = self.get_object_string()
        d = self.sim.data
        left_gripper_contact, right_gripper_contact = (False, False)
        for coni in range(d.ncon):
            con1 = self.sim.model.geom_id2name(d.contact[coni].geom1)
            con2 = self.sim.model.geom_id2name(d.contact[coni].geom2)
            body1 = self.sim.model.body_id2name(
                self.sim.model.geom_bodyid[d.contact[coni].geom1]
            )
            body2 = self.sim.model.body_id2name(
                self.sim.model.geom_bodyid[d.contact[coni].geom2]
            )
            if (
                body1 == "leftpad"
                and body2 == obj_name
                or body2 == "leftpad"
                and body1 == obj_name
            ):
                left_gripper_contact = True
            if (
                body1 == "rightpad"
                and body2 == obj_name
                or body2 == "rightpad"
                and body1 == obj_name
            ):
                right_gripper_contact = True
        return left_gripper_contact and right_gripper_contact

    def check_grasp(self):
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        contact_grasp = self.body_check_grasp()
        if self.env_name == "hammer-v2":
            return contact_grasp
        return (
            contact_grasp
            and abs(self.get_object_pose()[0][2] - self.initial_object_pos[2]) > 0.03
        )

    def set_object_pose(self, object_pos, object_quat):
        self._set_obj_pose(np.concatenate((object_pos, object_quat)))

    def get_target_pos(self):
        if self.num_high_level_steps % 2 == 0:
            # pos = self.object_poses[self.num_high_level_steps // 2][0]
            pos = self.get_object_pose_mp()[0]
        else:
            pos = self.placement_poses[(self.num_high_level_steps - 1) // 2][0]
        return pos, self._eef_xquat

    def check_robot_collision(self, ignore_object_collision, obj_idx=0, verbose=False):
        obj_string = self.get_object_string()
        d = self.sim.data
        for coni in range(d.ncon):
            con1 = self.sim.model.geom_id2name(d.contact[coni].geom1)
            con2 = self.sim.model.geom_id2name(d.contact[coni].geom2)
            body1 = self.sim.model.body_id2name(
                self.sim.model.geom_bodyid[d.contact[coni].geom1]
            )
            body2 = self.sim.model.body_id2name(
                self.sim.model.geom_bodyid[d.contact[coni].geom2]
            )
            if verbose:
                print(f"Con1: {con1} Con2: {con2} Body1: {body1} Body2: {body2}")
            if (check_robot_string(con1) ^ check_robot_string(con2)) or (
                (body1 in self.ROBOT_BODIES) ^ (body2 in self.ROBOT_BODIES)
            ):
                if (
                    check_string(con1, obj_string)
                    or check_string(con2, obj_string)
                    and ignore_object_collision
                ):
                    # if the robot and the object collide, then we can ignore the collision
                    continue
                # check using bodies
                if (
                    (body1 == obj_string and body2 in self.ROBOT_BODIES)
                    or (body2 == obj_string and body1 in self.ROBOT_BODIES)
                ) and ignore_object_collision:
                    continue
                return True
            elif ignore_object_collision:
                if check_string(con1, obj_string) or check_string(con2, obj_string):
                    # if we are supposed to be "ignoring object collisions" then we assume the
                    # robot is "joined" to the object. so if the object collides with any non-robot
                    # object, then we should call that a collision
                    return True
                if (body1 == obj_string and body2 not in self.ROBOT_BODIES) or (
                    body2 == obj_string and body1 not in self.ROBOT_BODIES
                ):
                    return True
        return False

    def check_object_placement(self, **kwargs):
        return False

    def compute_hardcoded_orientation(self, *args, **kwargs):
        pass

    def set_robot_based_on_ee_pos(
        self,
        target_pos,
        target_quat,
        qpos,
        qvel,
        is_grasped,
        open_gripper_on_tp=False,  # placeholder argument
        obj_idx=0,  # placeholder argument for robosuite
    ):
        if target_pos is not None:
            set_robot_based_on_ee_pos(
                self,
                target_pos,
                target_quat,
                qpos,
                qvel,
                is_grasped,
            )
        else:
            return -np.inf

    def backtracking_search_from_goal(
        self,
        start_pos,
        start_quat,
        target_pos,
        target_quat,
        qpos,
        qvel,
        is_grasped,
        movement_fraction=0.001,
        open_gripper_on_tp=True,
        obj_idx=0,
    ):
        curr_pos = target_pos.copy()
        self.set_robot_based_on_ee_pos(curr_pos, target_quat, qpos, qvel, is_grasped)
        collision = self.check_robot_collision(ignore_object_collision=is_grasped)
        iters = 0
        max_iters = int(1 / movement_fraction)
        while collision and iters < max_iters:
            curr_pos = curr_pos - movement_fraction * (target_pos - start_pos)
            self.set_robot_based_on_ee_pos(
                curr_pos,
                target_quat,
                qpos,
                qvel,
                is_grasped,
                open_gripper_on_tp=open_gripper_on_tp,
            )
            collision = self.check_robot_collision(ignore_object_collision=is_grasped)
            iters += 1
        if collision:
            return np.concatenate((start_pos, start_quat))
        else:
            return np.concatenate((curr_pos, target_quat))

    def get_observation(self):
        return self._get_obs()

    # robosuite functions
    def update_controllers(self):
        pass

    @property
    def _eef_xpos(self):
        return self.get_endeff_pos()

    @property
    def _eef_xquat(self):
        return self.get_endeff_quat()

    def set_robot_colors(self, colors):
        if type(colors) is np.ndarray:
            colors = [colors] * len(self.robot_geom_ids)
        for idx, geom_id in enumerate(self.robot_geom_ids):
            self.sim.model.geom_rgba[geom_id] = colors[idx]
        self.sim.forward()

    def reset_robot_colors(self):
        self.set_robot_colors(self.original_colors)
        self.sim.forward()

    def get_poses_from_obj_name(self, obj_name):
        if self.env_name == "assembly-v2" or self.env_name == "disassemble-v2":
            if "wrench" in obj_name:
                return self.get_object_pose_mp(), self.get_placement_pose()
            else:
                raise NotImplementedError
        elif self.env_name == "bin-picking-v2":
            if "cube" in obj_name:
                return self.get_object_pose_mp(), self.get_placement_pose()
            else:
                raise NotImplementedError
        elif self.env_name == "hammer-v2":
            if "hammer" in obj_name:
                return self.get_object_pose_mp(), self.get_placement_pose()
            else:
                raise NotImplementedError

    def update_controllers(self, *args, **kwargs):
        pass

    def update_mp_controllers(self, *args, **kwargs):
        pass

    def rebuild_controller(self, *args, **kwargs):
        pass

    def process_state_frames(self, frames):
        if len(frames) < 50 and not self.break_mp:
            return frames
        else:
            self.reset_robot_colors()
            im = self.sim.render(
                camera_name="corner",
                width=960,
                height=540,
            )
            self.reset_robot_colors()
            segmentation_map = np.flipud(
                CU.get_camera_segmentation(
                    camera_name="corner",
                    camera_width=960,
                    camera_height=540,
                    sim=self.sim,
                )
            )
            # get robot segmentation mask
            geom_ids = np.unique(segmentation_map[:, :, 1])
            robot_ids = []
            for geom_id in geom_ids:
                if geom_id != -1:
                    geom_name = self.sim.model.geom_id2name(geom_id)
                    if geom_name == None:
                        continue
                    if (
                        geom_name.startswith("robot")
                        or geom_name.startswith("left")
                        or geom_name.startswith("right")
                        or geom_id == 27
                    ):
                        robot_ids.append(geom_id)
            robot_ids.append(27)
            robot_ids.append(28)
            robot_mask = np.expand_dims(
                np.any(
                    [segmentation_map[:, :, 1] == robot_id for robot_id in robot_ids],
                    axis=0,
                ),
                -1,
            )
            waypoint_mask = robot_mask
            waypoint_img = robot_mask * im
            for i in range(len(frames)):
                frames[i] = (
                    0.5 * (frames[i] * robot_mask)
                    + 0.5 * (waypoint_img)
                    + frames[i] * (1.0 - robot_mask)
                )
                frames[i] = frames[i][:, :, ::-1]
            self.set_robot_colors(np.array([0.1, 0.3, 0.7, 1.0]))
            return frames

    def get_mp_waypoints(self, *args, **kwargs):
        return None, None

    def take_mp_step(
        self,
        state,
        is_grasped,
        *args,
    ):
        desired_rot = quat2mat(state[3:])
        for s in range(50):
            if np.linalg.norm(state[:3] - self._eef_xpos) < 1e-5:
                self.break_mp = True
                return
            self.set_xyz_action((state[:3] - self._eef_xpos))
            if is_grasped:
                self.do_simulation([1.0, -1.0], n_frames=self.frame_skip)
            else:
                self.do_simulation([0.0, -0.0], n_frames=self.frame_skip)
            for site in self._target_site_config:  # taken from metaworld repo
                self._set_pos_site(*site)
