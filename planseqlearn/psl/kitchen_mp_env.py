import numpy as np
from planseqlearn.psl.inverse_kinematics import qpos_from_site_pose_kitchen
from planseqlearn.psl.mp_env import PSLEnv
from planseqlearn.psl.vision_utils import *
import robosuite.utils.transform_utils as T
from robosuite.utils.transform_utils import *
from planseqlearn.psl.env_text_plans import KITCHEN_PLANS

OBS_ELEMENT_INDICES = {
    "bottom left burner": np.array([11]), #correct
    "bottom right burner": np.array([9]),
    "top burner": np.array([15]), #correct
    "top right burner": np.array([13]), #correct
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "left hinge cabinet": np.array([20]),
    "hinge cabinet": np.array([21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
    "close hinge cabinet": np.array([13, 21])
}
OBS_ELEMENT_GOALS = {
    "bottom left burner": np.array([-0.92]),
    "bottom right burner": np.array([-0.92]),
    "top burner": np.array([-0.92]),
    "top right burner": np.array([-0.92]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "left hinge cabinet": np.array([-1.45]),
    "hinge cabinet": np.array([1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
    "close hinge cabinet": np.array([-0.92, 0.0])
}
BONUS_THRESH = 0.3

def check_robot_string(string):
    if string is None:
        return False
    return string.startswith("panda0") or string.startswith("gripper")


def get_object_string(env, obj_name):
    if "microwave" in obj_name:
        return "microwave"
    if "kettle" in obj_name:
        return "kettle"
    if "light" in obj_name:
        return "light switch"
    if "slide" in obj_name:
        return "slide cabinet"
    if "hinge" in obj_name:
        return "hinge cabinet"
    if "burner" in obj_name:
        return "top burner"

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
        if len(self.text_plan) == 0:
            self.text_plan = KITCHEN_PLANS[self.env_name]
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

    def get_sam_kwargs(self, obj_name):
        if "microwave" in obj_name:
            return {
                "camera_name": "leftview",
                "text_prompts": ["vertical bar"],
                "idx": -1,
                "offset": np.zeros(3),
                "box_threshold": 0.3,
                "flip_image": True,
                "flip_channel": True,
                "flip_dm": False,
            }
        if "slide" in obj_name:
            return {
                "camera_name": "leftview",
                "text_prompts": ["vertical bar"],
                "idx": 0,
                "offset": np.zeros(3),
                "box_threshold": 0.3,
                "flip_image": True,
                "flip_channel": True,
                "flip_dm": False,
            }
        if "top burner" in obj_name:
            return {
                "camera_name": "leftview",
                "text_prompts": ["red tips"],
                "idx": 5,
                "offset": np.array([-0.12, 0., 0.05]),
                "box_threshold": 0.3,
                "flip_image": True,
                "flip_channel": True,
                "flip_dm": False,
            }
        if "kettle" in obj_name:
            return {
                "camera_name": "leftview",
                "text_prompts": ["vertical bar"],
                "idx": 2,
                "offset": np.array([0.09, 0., 0.]),
                "box_threshold": 0.3,
                "flip_image": True,
                "flip_channel": True,
                "flip_dm": False,
            }
        if "light" in obj_name: 
            return {
                "camera_name": "leftview",
                "text_prompts": ["small knob"],
                "idx": -2,
                "offset": np.zeros(3),
                "box_threshold": 0.3,
                "flip_image": True,
                "flip_channel": True,
                "flip_dm": False,
            }
        if "top right burner" in obj_name:
            return {
                "camera_name": "leftview",
                "text_prompts": ["red tips"],
                "idx": 5,
                "offset": np.array([0.0, 0., 0.05]),
                "box_threshold": 0.3,
                "flip_image": True,
                "flip_channel": True,
                "flip_dm": False,
            }
        if "bottom left burner" in obj_name:
            return {
                "camera_name": "leftview",
                "text_prompts": ["red tips"],
                "idx": 6,
                "offset": np.array([-0.12, 0., -0.07]),
                "box_threshold": 0.3,
                "flip_image": True,
                "flip_channel": True,
                "flip_dm": False,
            }
        if "bottom right burner" in obj_name:
            return {
                "camera_name": "leftview",
                "text_prompts": ["red tips"],
                "idx": 5,
                "offset": np.array([0.00, 0., -0.07]),
                "box_threshold": 0.3,
                "flip_image": True,
                "flip_channel": True,
                "flip_dm": False,
            }
        if "hinge" in obj_name and "close" not in obj_name:
            return {
                "camera_name": "leftview",
                "text_prompts": ["vertical bar"],
                "idx": 1,
                "offset": np.array([0.16, 0., 0.,]),
                "box_threshold": 0.3,
                "flip_image": True,
                "flip_channel": True,
                "flip_dm": False,
            }
        if "close" in obj_name and "hinge" in obj_name:
            return {
                "camera_name": "leftview",
                "text_prompts": ["vertical bar"],
                "idx": 1,
                "offset": np.array([0., 0., 0.,]),
                "box_threshold": 0.3,
                "flip_image": True,
                "flip_channel": True,
                "flip_dm": False,
            }

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

    def check_object_placement(self, **kwargs):
        def check_object_placement(*args, **kwargs):
            return False 
        return check_object_placement 

    def check_robot_collision(
        self,
        ignore_object_collision=False,
        obj_name="",
    ):
        obj_string = get_object_name(env, obj_name)
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
    
    def get_sim_object_pose(self, obj_name):
        if "slide" in obj_name:
            object_pos = self.get_site_xpos("schandle1")
            object_quat = np.zeros(4)  # doesn't really matter
        elif "top burner" in obj_name:
            object_pos = self.get_site_xpos("tlbhandle")
            object_quat = np.zeros(4)  # doesn't really matter
        elif "bottom left burner" in obj_name:
            object_pos = self.get_site_xpos("blbhandle")
            object_quat = np.zeros(4)
        elif "bottom right burner" in obj_name:
            object_pos = self.get_site_xpos("brbhandle")
            object_quat = np.zeros(4)
        elif "top right burner" in obj_name:
            object_pos = self.get_site_xpos("trbhandle")
            object_quat = np.zeros(4)
        elif "hinge cabinet" in obj_name:
            object_pos = self.get_site_xpos("hchandle1")
            object_quat = np.zeros(4)  # doesn't really matter
        elif "left hinge cabinet" in obj_name:
            object_pos = self.get_site_xpos("hchandle_left1")
            object_quat = np.zeros(4)
        elif "light" in obj_name:
            object_pos = self.get_site_xpos("lshandle1")
            object_quat = np.zeros(4)  # doesn't really matter
        elif "microwave" in obj_name:
            object_pos = self.get_site_xpos("mchandle1")
            object_quat = np.zeros(4)  # doesn't really matter
        elif "kettle" in obj_name:
            object_pos = self.get_site_xpos("khandle1")
            object_quat = np.zeros(4)  # doesn't really matter
        return object_pos, object_quat
    
    def get_mp_target_pose(self, obj_name):
        if "slide" in obj_name:
            object_pos = self.get_site_xpos("schandle1")
            object_quat = np.zeros(4)  # doesn't really matter
        elif "top burner" in obj_name:
            object_pos = self.get_site_xpos("tlbhandle")
            object_quat = np.zeros(4)  # doesn't really matter
        elif "bottom left burner" in obj_name:
            object_pos = self.get_site_xpos("blbhandle")
            object_quat = np.zeros(4)
        elif "bottom right burner" in obj_name:
            object_pos = self.get_site_xpos("brbhandle")
            object_quat = np.zeros(4)
        elif "top right burner" in obj_name:
            object_pos = self.get_site_xpos("trbhandle")
            object_quat = np.zeros(4)
        elif "left hinge cabinet" in obj_name:
            object_pos = self.get_site_xpos("hchandle_left1")
            object_quat = np.zeros(4)  # doesn't really matter
        elif "hinge cabinet" in obj_name:
            object_pos = self.get_site_xpos("hchandle1")
            object_quat = np.zeros(4)
        elif "light" in obj_name:
            object_pos = self.get_site_xpos("lshandle1")
            object_quat = np.zeros(4)  # doesn't really matter
        elif "microwave" in obj_name:
            object_pos = self.get_site_xpos("mchandle1") + np.array([0, -0.05, 0])
            object_quat = np.zeros(4)  # doesn't really matter
        elif "kettle" in obj_name:
            object_pos = self.get_site_xpos("khandle1")
            object_quat = np.zeros(4)  # doesn't really matter
        if self.use_sam_segmentation:
            object_pos = self.sam_object_pose[obj_name] + np.array([0., -0.05, 0])
            object_quat = np.zeros(4)
        return object_pos, object_quat 

    def get_target_pos(self):
        pos, obj_quat = self.get_mp_target_pose(self.text_plan[self.curr_plan_stage][0])
        return pos, obj_quat 

    def set_robot_based_on_ee_pos(
        self,
        target_pos,
        target_quat,
        qpos,
        qvel,
        obj_name="",
        open_gripper_on_tp=False,
    ):
        """
        Set robot joint positions based on target ee pose. Uses IK to solve for joint positions.
        If grasping an object, ensures the object moves with the arm in a consistent way.
        """
        # cache quantities from prior to setting the state
        is_grasped = self.named_check_object_grasp(
            self.text_plan[self.curr_plan_stage - (self.curr_plan_stage % 2)][0]
        )()
        object_pos, object_quat = self.get_sim_object_pose(obj_name)
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

        if not is_grasped:
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
        obj_name="",
    ):
        object_pos, object_quat = self.get_sim_object_pose(obj_name=obj_name)
        is_grasped = False # always true for kitchen environment
        open_gripper_on_tp = True 
        object_pos = object_pos.copy()
        object_quat = object_quat.copy()
        gripper_qpos = self.sim.data.qpos[7:9].copy()
        gripper_qvel = self.sim.data.qvel[7:9].copy()
        old_eef_xquat = self._eef_xquat.copy()
        old_eef_xpos = self._eef_xpos.copy()

        self.sim.data.qpos[:7] = joint_pos
        self.sim.forward()
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
        obj_name="",
    ):
        if self.use_pcd_collision_check:
            raise NotImplementedError
        else:
            self.set_robot_based_on_joint_angles(
                curr_pos,
                qpos,
                qvel,
                obj_name=obj_name
            )
            valid = not self.check_robot_collision(
                ignore_object_collision=False, obj_name=obj_name
            )
        return valid

    def get_robot_mask(self):
        segmentation_map = self.sim.render(segmentation=True, height=540, width=960)
        geom_ids = np.unique(segmentation_map[:, :, 1])
        robot_ids = []
        object_ids = []
        object_string = get_object_string(self, self.text_plan[self.curr_plan_stage][0])
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

    def named_check_object_grasp(self, obj_name):
        def check_object_grasp(info={}, *args, **kwargs):
            # copy code from https://github.com/mihdalal/d4rl/blob/primitives/d4rl/kitchen/kitchen_envs.py
            if info != {}:
                for obj in self.tasks_to_complete:
                    if "microwave" in obj and "microwave" in obj_name:
                        return False 
                    if "top burner" in obj and "top burner" in obj_name:
                        return False 
                    if "top right burner" in obj and "top right burner" in obj_name:
                        return False 
                    if "bottom right burner" in obj and "bottom right burner" in obj_name:
                        return False 
                    if "bottom left burner" in obj and "bottom left burner" in obj_name:
                        return False 
                    if "light" in obj and "light" in obj_name:
                        return False 
                    if "slide" in obj and "slide" in obj_name:
                        return False 
                    if "kettle" in obj and "kettle" in obj_name:
                        return False
                    if "close hinge cabinet" in obj and "close hinge cabinet" in obj_name:
                        return False
                    if "hinge cabinet" in obj and "hinge cabinet" in obj_name \
                        and "close" not in obj and "close" not in obj_name:
                        return False 
            return True
        return check_object_grasp 
        
    def get_image(self):
        im = self.sim.render(
            width=960,
            height=540,
        )
        return im

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