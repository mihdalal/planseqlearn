import numpy as np
import copy
from gym import spaces
import collections
from robosuite.utils.transform_utils import (
    convert_quat,
    euler2mat,
    mat2pose,
    mat2quat,
    pose2mat,
    quat2mat,
    mat2quat,
    quat_conjugate,
    quat_multiply,
)
from mopa_rl.config.default_configs import *
import mopa_rl.env
from planseqlearn.mnm.inverse_kinematics import qpos_from_site_pose
from planseqlearn.mnm.mp_env import MPEnv, ProxyEnv
from planseqlearn.mnm.vision_utils import *


def save_img(env, camera_name, filename, flip=False):
    # frame = env.sim.render(camera_name=camera_name, width=500, height=500)
    frame = env.sim.render(width=500, camera_name=camera_name, height=500)
    if flip:
        frame = np.flipud(frame)
    plt.imshow(frame)
    plt.savefig(filename)


def check_grasp(env, name):
    """
    Checks grasp of object in environment.
    Args:
        env: Gym environment
        name: name of environment
    Returns:
        boolean corresponding to grasped object
    """
    if name == "SawyerLift-v0" or name == "SawyerLiftObstacle-v0":
        touch_left_finger = False
        touch_right_finger = False
        for i in range(env.sim.data.ncon):
            c = env.sim.data.contact[i]
            if c.geom1 == env.cube_geom_id:
                if c.geom2 in env.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom2 in env.r_finger_geom_ids:
                    touch_right_finger = True
            elif c.geom2 == env.cube_geom_id:
                if c.geom1 in env.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in env.r_finger_geom_ids:
                    touch_right_finger = True
        return touch_left_finger and touch_right_finger
    if name == "SawyerAssemblyObstacle-v0":
        return False  # there is no sense of grasping in the assembly environment so return false
    if name == "SawyerPushObstacle-v0":
        return False
    else:
        raise NotImplementedError


def cart2joint_ac(
    env,
    ik_env,
    ac,
    qpos,
    qvel,
    config,
):
    curr_qpos = env.sim.data.qpos.copy()
    target_cart = np.clip(
        env.sim.data.get_site_xpos(config["ik_target"])[: len(env.min_world_size)]
        + config["action_range"] * ac["default"],
        env.min_world_size,
        env.max_world_size,
    )
    if len(env.min_world_size) == 2:
        target_cart = np.concatenate(
            (
                target_cart,
                np.array([env.sim.data.get_site_xpos(config["ik_target"])[2]]),
            )
        )
    if "quat" in ac.keys():
        target_quat = ac["quat"]
    else:
        target_quat = None
    ik_env.set_state(curr_qpos.copy(), env.data.qvel.copy())
    result = qpos_from_site_pose(
        ik_env,
        config["ik_target"],
        target_pos=target_cart,
        target_quat=target_quat,
        rot_weight=2.0,
        joint_names=env.robot_joints,
        max_steps=100,
        tol=1e-2,
    )
    target_qpos = env.sim.data.qpos.copy()
    target_qpos[env.ref_joint_pos_indexes] = result.qpos[
        env.ref_joint_pos_indexes
    ].copy()
    pre_converted_ac = (
        target_qpos[env.ref_joint_pos_indexes] - curr_qpos[env.ref_joint_pos_indexes]
    ) / env._ac_scale
    if "gripper" in ac.keys():
        pre_converted_ac = np.concatenate((pre_converted_ac, ac["gripper"]))
    converted_ac = collections.OrderedDict([("default", pre_converted_ac)])
    return converted_ac


def check_collisions(env, allowed_collision_pairs, env_name, verbose=False, *args, **kwargs):
    mjcontacts = env.sim.data.contact
    ncon = env.sim.data.ncon
    for i in range(ncon):
        ct = mjcontacts[i]
        ct1 = ct.geom1
        ct2 = ct.geom2
        b1 = env.sim.model.geom_bodyid[ct1]
        b2 = env.sim.model.geom_bodyid[ct2]
        bn1 = env.sim.model.body_id2name(b1)
        bn2 = env.sim.model.body_id2name(b2)
        if verbose:
            print(f"ct1:{ct1} ct2:{ct2} b1:{bn1} b2:{bn2}")
        # robot bodies checking allows robot to collide with itself
        # useful for going up when grasping
        if env_name == "SawyerLift-v0" or env_name == "SawyerLiftObstacle-v0":
            if (
                ((bn1 in ROBOT_BODIES) and (bn2 in ROBOT_BODIES))
                or ((bn1 in ROBOT_BODIES) and (bn2 == "cube"))
                or ((bn2 in ROBOT_BODIES) and (bn1 == "cube"))
                or ((ct1, ct2) in allowed_collision_pairs)
                or ((ct2, ct1) in allowed_collision_pairs)
            ):
                continue
            else:
                return True
        elif env_name == "SawyerAssemblyObstacle-v0" or env_name == "SawyerPushObstacle-v0":
            if ((ct1, ct2) not in allowed_collision_pairs) and (
                (ct2, ct1) not in allowed_collision_pairs
            ):
                if verbose:
                    print(f"Case")
                return True
    return False


def set_object_pose(env, name, new_xpos, new_xquat):
    """
    Gets pose of desired object.
    Args:
        env: Gym environment
        name: name of manipulation object from body_names in sim
        new_xpos: (3,) numpy.ndarray of new xyz coordinates
        new_xquat: (4,) numpy.ndarary of new quaternion, in xyzw format
    Returns:
        None
    """
    start = env.sim.model.body_jntadr[env.sim.model.body_name2id(name)]
    new_xquat = convert_quat(new_xquat, to="wxyz")
    env.sim.data.qpos[start : start + 3] = new_xpos
    env.sim.data.qpos[start + 3 : start + 7] = new_xquat


ROBOT_BODIES = [
    "base",
    "controller_box",
    "pedestal_feet",
    "torso",
    "pedestal",
    "right_arm_base_link",
    "right_l0",
    "head",
    "screen",
    "right_l1",
    "right_l2",
    "right_l3",
    "right_l4",
    "right_l5",
    "right_l6",
    "right_ee_attchment",
    "clawGripper",
    "right_gripper_base",
    "right_gripper",
    "rightclaw",
    "r_gripper_l_finger_tip",
    "leftclaw",
    "r_gripper_r_finger_tip",
]


def get_site_pose(env, name):
    """
    Gets pose of site.
    Args:
        env: Gym environment
        name: name of site in sim
    Returns:
        (pos, orn) tuple where pos is xyz location of site, orn
            is (4,) numpy.ndarray corresponding to quat in xyzw format
    """
    xpos = env.sim.data.get_site_xpos(name)[: len(env.min_world_size)].copy()
    model = env.sim.model
    xquat = mat2quat(env.sim.data.get_site_xmat(name).copy())
    return xpos, xquat


class MoPAWrapper(ProxyEnv):
    def __init__(self, env, ik_env, env_name, config):
        super().__init__(env)
        self.env_name = env_name
        self.ik_env = ik_env
        self.config = config

    def _convert_observation(self, obs):
        observation = np.array([])
        for k in obs.keys():
            observation = np.concatenate((observation, obs[k].copy()))
        return observation

    def reset(self):
        o = self._wrapped_env.reset()
        return self._convert_observation(o)
    
    def _check_success(self):
        if self.env_name == "SawyerLift-v0" or self.env_name == "SawyerLiftObstacle-v0":
            # copied from sawyer_lift_obstacle.py
            info = {}
            reward = 0
            reach_mult = 0.1
            grasp_mult = 0.35
            lift_mult = 0.5
            hover_mult = 0.7
            cube_body_id = self._wrapped_env.sim.model.body_name2id("cube")
            reward_reach = 0.0
            gripper_site_pos = self._wrapped_env.sim.data.get_site_xpos("grip_site")
            cube_pos = np.array(self._wrapped_env.sim.data.body_xpos[cube_body_id])
            gripper_to_cube = np.linalg.norm(cube_pos - gripper_site_pos)
            reward_reach = (1 - np.tanh(10 * gripper_to_cube)) * reach_mult
            touch_left_finger = False
            touch_right_finger = False
            for i in range(self._wrapped_env.sim.data.ncon):
                c = self._wrapped_env.sim.data.contact[i]
                if c.geom1 == self._wrapped_env.cube_geom_id:
                    if c.geom2 in self._wrapped_env.l_finger_geom_ids:
                        touch_left_finger = True
                    if c.geom2 in self._wrapped_env.r_finger_geom_ids:
                        touch_right_finger = True
                elif c.geom2 == self._wrapped_env.cube_geom_id:
                    if c.geom1 in self._wrapped_env.l_finger_geom_ids:
                        touch_left_finger = True
                    if c.geom1 in self._wrapped_env.r_finger_geom_ids:
                        touch_right_finger = True
            has_grasp = touch_right_finger and touch_left_finger
            reward_grasp = int(has_grasp) * grasp_mult
            reward_lift = 0.0
            object_z_locs = self._wrapped_env.sim.data.body_xpos[cube_body_id][2]
            if reward_grasp > 0.0:
                z_target = self._wrapped_env._get_pos("bin1")[2] + 0.45
                z_dist = np.maximum(z_target - object_z_locs, 0.0)
                reward_lift = grasp_mult + (1 - np.tanh(15 * z_dist)) * (
                    lift_mult - grasp_mult
                )
                print(f"Dist: {np.abs(object_z_locs - z_target)}")
            reward += max(reward_reach, reward_grasp, reward_lift)
            info = dict(
                reward_reach=reward_reach,
                reward_grasp=reward_grasp,
                reward_lift=reward_lift,
            )
            success = False
            
            if reward_grasp > 0.0 and np.abs(object_z_locs - z_target) < 0.05:
                reward += 150.0 # this is the success reward
                success = True
            return success
        elif self.env_name == "SawyerAssemblyObstacle-v0":
            info = {}
            reward = 0
            pegHeadPos = self._wrapped_env.sim.data.get_site_xpos("pegHead")
            hole = self._wrapped_env.sim.data.get_site_xpos("hole")
            dist = np.linalg.norm(pegHeadPos - hole)
            hole_bottom = self._wrapped_env.sim.data.get_site_xpos("hole_bottom")
            dist_to_hole_bottom = np.linalg.norm(pegHeadPos - hole_bottom)
            dist_to_hole = np.linalg.norm(pegHeadPos - hole)
            reward_reach = 0
            if dist < 0.3:
                reward_reach += 0.4 * (1 - np.tanh(15 * dist_to_hole))
            reward += reward_reach
            success = False
            if dist_to_hole_bottom < 0.025:
                success = True
                terminal = True
            return success
        elif self.env_name == "SawyerPushObstacle-v0":
            self._wrapped_env.compute_reward(None)
            success = self._wrapped_env._success
            return success

    @property
    def observation_space(self):
        obs_space = self._wrapped_env.observation_space
        low = np.array([])
        high = np.array([])
        for k in obs_space.spaces.keys():
            low = np.concatenate((low, obs_space[k].low))
            high = np.concatenate((high, obs_space[k].high))
        return spaces.Box(
            low=low,
            high=high,
        )

    @property
    def action_space(self):
        low = np.array([-np.inf for _ in range(7)])
        high = np.array([np.inf for _ in range(7)])
        return spaces.Box(
            low=low,
            high=high,
        )

    def step(self, action):
        delta_pos = action[:3]
        gripper_ac = np.array([action[-1]])
        ac = collections.OrderedDict()
        delta_rot_mat = euler2mat(action[3:6])
        target_rot = delta_rot_mat @ self._wrapped_env.sim.data.get_site_xmat(
            "grip_site"
        )
        ac["quat"] = mat2quat(target_rot)
        ac["quat"] = convert_quat(ac["quat"], to="wxyz")
        ac["default"] = delta_pos
        if self.env_name == "SawyerLift-v0" or self.env_name == "SawyerLiftObstacle-v0":
            ac["gripper"] = gripper_ac
        converted_ac = cart2joint_ac(
            self._wrapped_env,
            self.ik_env,
            ac,
            self._wrapped_env.sim.data.qpos.copy(),
            self._wrapped_env.sim.data.qvel.copy(),
            self.config,
        )
        o, r, d, i = self._wrapped_env.step(converted_ac)
        o = self._convert_observation(o)
        return o, r, d, i


class MoPAMPEnv(MPEnv):
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
        ik_env = kwargs["ik_env"]
        self.allowed_collision_pairs = []
        for manipulation_geom_id in self._wrapped_env.manipulation_geom_ids:
            for geom_id in self._wrapped_env.static_geom_ids:
                self.allowed_collision_pairs.append((manipulation_geom_id, geom_id))
        if self.env_name == "SawyerLift-v0" or self.env_name == "SawyerLiftObstacle-v0":
            for manipulation_geom_id in self._wrapped_env.manipulation_geom_ids:
                for lf in self._wrapped_env.left_finger_geoms:
                    self.allowed_collision_pairs.append(
                        (
                            manipulation_geom_id,
                            self._wrapped_env.sim.model.geom_name2id(lf),
                        )
                    )
            for manipulation_geom_id in self._wrapped_env.manipulation_geom_ids:
                for rf in self._wrapped_env.right_finger_geoms:
                    self.allowed_collision_pairs.append(
                        (
                            manipulation_geom_id,
                            self._wrapped_env.sim.model.geom_name2id(rf),
                        )
                    )
        if self.env_name == "SawyerLift-v0":
            config = LIFT_CONFIG
            config["camera_name"] = "eye_in_hand"
        elif self.env_name == "SawyerLiftObstacle-v0":
            config = LIFT_OBSTACLE_CONFIG
            config["camera_name"] = "eye_in_hand"
        elif self.env_name == "SawyerAssemblyObstacle-v0":
            config = ASSEMBLY_OBSTACLE_CONFIG
            config["camera_name"] = "eye_in_hand"
        elif self.env_name == "SawyerPushObstacle-v0":
            config = PUSHER_OBSTACLE_CONFIG
            config["camera_name"] = "eye_in_hand"
        self.config = config
        # wrap env in mopa env class to make observation and action space translation easier
        self._wrapped_env = MoPAWrapper(
            self._wrapped_env,
            ik_env,
            self.env_name,
            self.config,
        )
        self.robot_bodies = [body for body in self.sim.model.body_names
                            if body.startswith("left") or body.startswith("right") and not body.endswith("target") and not body.endswith("indicator")]
        self.body_ids = [self.sim.model.body_name2id(body) for body in self.robot_bodies]
        self.robot_geom_ids = []
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            if body_id in self.body_ids:
                self.robot_geom_ids.append(geom_id)
        self.original_colors = [self.sim.model.geom_rgba[idx].copy() for idx in self.robot_geom_ids]
        self.retry = False
        self.text_plan = [("null", "null")]

    def check_robot_collision(self, **kwargs):
        return check_collisions(self, self.allowed_collision_pairs, self.env_name, **args, **kwargs)
    
    def set_robot_colors(self, colors):
        if type(colors) is np.ndarray:
            colors = [colors] * len(self.robot_geom_ids)
        for idx, geom_id in enumerate(self.robot_geom_ids):
            self.sim.model.geom_rgba[geom_id] = colors[idx]
        self.sim.forward()
    
    def reset_robot_colors(self):
        self.set_robot_colors(self.original_colors)
        self.sim.forward()

    def get_object_pose(self, **kwargs):
        if self.env_name == "SawyerLiftObstacle-v0" or self.env_name == "SawyerLift-v0":
            start = self.sim.model.body_jntadr[self.sim.model.body_name2id("cube")]
            object_pos = self.sim.model.body_jntadr[self.sim.model.body_name2id("cube")]
            object_pos = self.sim.data.qpos[start : start + 3].copy()
            object_quat = self.sim.data.qpos[start + 3 : start + 7].copy()
            object_quat = convert_quat(object_quat, to="xyzw")
        elif self.env_name == "SawyerAssemblyObstacle-v0":
            object_pos = get_site_pose(self._wrapped_env, "hole")[0]
            object_quat = None
        elif self.env_name == "SawyerPushObstacle-v0":
            object_pos = np.array(
                self._wrapped_env.sim.data.body_xpos[self._wrapped_env.cube_body_id]
            )
            object_quat = None
        else:
            raise NotImplementedError
        return object_pos, object_quat

    def get_object_pose_mp(self, **kwargs):
        if self.env_name == "SawyerLiftObstacle-v0" or self.env_name == "SawyerLift-v0":
            object_pos, object_quat = self.get_object_pose()
            if self.use_vision_pose_estimation:
                object_pos = get_object_pose_from_seg(
                    env=self,
                    object_string="cube",
                    camera_name="topview",
                    camera_width=500,
                    camera_height=500,
                    sim=self._wrapped_env.sim,
                )
            object_pos += np.array([0.0, 0.0, 0.07])
            object_quat = np.array([-0.1268922, 0.21528646, 0.96422245, -0.08846001])
            object_quat /= np.linalg.norm(object_quat)
        elif self.env_name == "SawyerAssemblyObstacle-v0":
            object_pos, object_quat = self.get_object_pose()
            object_pos = get_site_pose(self._wrapped_env, "hole")[0]
            object_pos += np.array([-0.12, 0.05, 0.45])
            if self.use_vision_pose_estimation:
                object_pos = get_object_pose_from_seg(
                    self,
                    "4_part4_mesh",
                    "topview",
                    500,
                    500,
                    self._wrapped_env.sim,
                    self, "4_part4_mesh", "topview", 500, 500, self._wrapped_env.sim
                )
                object_pos += np.array([0, -0.3, 0.45]) 
            object_quat = np.array([-0.50258679, -0.61890813, -0.49056324, 0.35172])
            object_quat /= np.linalg.norm(object_quat)
        elif self.env_name == "SawyerPushObstacle-v0":
            object_pos, object_quat = self.get_object_pose()
            if self.use_vision_pose_estimation:
                object_pos = get_object_pose_from_seg(
                    self, "cube", "frontview", 500, 500, self._wrapped_env.sim
                )
            object_pos += np.array([-0.11, 0.045, 0.11])
            object_quat = np.array([-0.57194288, 0.00869415, -0.77486997, 0.26903954])
            object_quat = None
        else:
            raise NotImplementedError
        return object_pos, object_quat

    def get_object_poses(self):
        object_pos, object_quat = self.get_object_pose_mp()
        return [(object_pos, object_quat)]

    def get_placement_poses(self, **kwargs):
        return [(None, None)]

    def get_target_pos(self):
        if self.num_high_level_steps % 2 == 0:
            return self.get_object_pose_mp()
        else:
            return self.placement_poses[(self.num_high_level_steps - 1) // 2]

    def update_controllers(self):
        pass

    def get_observation(self):
        obs = self._wrapped_env._get_obs()
        observation = np.array([])
        for k in obs.keys():
            observation = np.concatenate((observation, obs[k]))
        return observation

    @property
    def _eef_xpos(self):
        return self.sim.data.site_xpos[self.sim.model.site_name2id("grip_site")]

    @property
    def _eef_xquat(self):
        xmat = self.sim.data.site_xmat[self.sim.model.site_name2id("grip_site")]
        quat = mat2quat(xmat.reshape(3, 3))
        return convert_quat(quat, to="wxyz")

    def check_object_placement(self, **kwargs):
        return True
    
    def compute_hardcoded_orientation(self, *args, **kwargs):
        pass 

    def compute_ik(self, target_pos, target_quat, qpos, qvel, og_qpos, og_qvel):
        self.ik_env.set_state(qpos.copy(), qvel.copy())
        result = qpos_from_site_pose(
            self.ik_env,
            self.config["ik_target"],
            target_pos=target_pos,
            target_quat=target_quat,
            rot_weight=2.0,
            joint_names=self.robot_joints,
            max_steps=100,
            tol=1e-2,
        )
        return self.ik_env.sim.data.qpos[self.ref_joint_pos_indexes].copy()

    def get_joint_bounds(self):
        return self.sim.model.jnt_range[self.ref_joint_pos_indexes]

    def check_state_validity_joint(
        self,
        joint_pos,
        qpos,
        qvel,
        is_grasped=False,
        **kwargs,
    ):
        assert not is_grasped  # for mopa tasks we never consider teleporting with grasp
        self.sim.data.qpos[self.ref_joint_pos_indexes] = joint_pos
        self.sim.forward()
        return not check_collisions(self, self.allowed_collision_pairs, self.env_name)

    def backtracking_search_from_goal_joints(
        self,
        start_angles,
        goal_angles,
        qpos,
        qvel,
        movement_fraction=0.001,
        *args,
        **kwargs,
    ):
        curr_angles = goal_angles.copy()
        collision = not self.check_state_validity_joint(curr_angles, qpos, qvel)
        iters = 0
        max_iters = int(1 / movement_fraction)
        while collision and iters < max_iters:
            curr_angles = curr_angles - movement_fraction * (goal_angles - start_angles)
            valid = self.check_state_validity_joint(curr_angles, qpos, qvel)
            collision = not valid
            iters += 1
        if collision:
            return start_angles
        else:
            return curr_angles

    def update_mp_controllers(self):
        pass

    def set_robot_based_on_ee_pos(
        self,
        target_pos,
        target_quat,
        qpos,
        qvel,
        is_grasped=False,
        obj_idx=0,
        open_gripper_on_tp=False,
    ):
        if target_pos is None:
            return -np.inf
        # keep track of gripper pos, etc
        gripper_qpos = self.sim.data.qpos[self.ref_gripper_joint_pos_indexes].copy()
        gripper_qvel = self.sim.data.qvel[self.ref_gripper_joint_pos_indexes].copy()
        old_eef_xpos, old_eef_xquat = get_site_pose(self, self.config["ik_target"])

        object_pose, object_quat = self.get_object_pose()

        target_cart = np.clip(
            target_pos,
            self.min_world_size,
            self.max_world_size,
        )
        self.ik_env.set_state(qpos, qvel)  # check this doesn't break anything
        result = qpos_from_site_pose(
            self.ik_env,
            self.config["ik_target"],
            target_pos=target_cart,
            target_quat=target_quat,
            rot_weight=2.0,
            joint_names=self.robot_joints,
            max_steps=100,
            tol=1e-2,
        )
        if result.success == False or check_collisions(
            self.ik_env, self.allowed_collision_pairs, self.env_name
        ):
            return (
                result.success
                and not check_collisions(
                    self.ik_env, self.allowed_collision_pairs, self.env_name
                ),
                result.err_norm,
            )
        # set state here
        self._wrapped_env.set_state(
            self.ik_env.sim.data.qpos.copy(), self.ik_env.sim.data.qvel.copy()
        )

        if is_grasped:
            self.sim.data.qpos[self.ref_gripper_joint_pos_indexes] = gripper_qpos
            self.sim.data.qvel[self.ref_gripper_joint_pos_indexes] = gripper_qvel

            # compute transform between new and old
            ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
            new_eef_xpos, new_eef_xquat = get_site_pose(self, self.config["ik_target"])
            ee_new_mat = pose2mat((new_eef_xpos, new_eef_xquat))
            transform = ee_new_mat @ np.linalg.inv(ee_old_mat)

            # get new object pose
            new_object_pose = mat2pose(
                np.dot(transform, pose2mat((object_pose[:3], object_pose[3:])))
            )
            if "Lift" in self.env_name:
                set_object_pose(self, "cube", new_object_pose[0], new_object_pose[1])
            self.sim.forward()

        return result.err_norm

    def backtracking_search_from_goal_pos(
        self,
        start_pos,
        start_quat,
        target_quat,
        goal_pos,
        qpos,
        qvel,
        movement_fraction=0.001,
        *args,
        **kwargs,
    ):
        curr_pos = goal_pos.copy()
        self.set_robot_based_on_ee_pos(
            goal_pos,
            target_quat,
            qpos,
            qvel,
            is_grasped=False,
        )
        collision = check_collisions(self, self.allowed_collision_pairs, self.env_name)
        iters = 0
        max_iters = int(1 / movement_fraction)
        while collision and iters < max_iters:
            curr_pos = curr_pos - movement_fraction * (goal_pos - start_pos)
            self.set_robot_based_on_ee_pos(
                curr_pos,
                target_quat,
                qpos,
                qvel,
                is_grasped=False,
            )
            collision = check_collisions(self, self.allowed_collision_pairs, self.env_name)
            iters += 1
            print(f"Iters: {iters, collision, curr_pos}")
        if collision:
            return start_pos, start_quat
        else:
            return curr_pos, target_quat
    def set_robot_based_on_joint_angles(
        self,
        joint_pos,
        qpos,
        qvel,
        is_grasped=False,
        obj_idx=0,
        open_gripper_on_tp=False,
    ):
        object_pos, object_quat = self.get_object_pose()
        gripper_qpos = self.sim.data.qpos[self.ref_gripper_joint_pos_indexes].copy()
        gripper_qvel = self.sim.data.qvel[self.ref_gripper_joint_pos_indexes].copy()
        old_eef_xpos, old_eef_xquat = get_site_pose(self, self.config['ik_target'])
        self.sim.data.qpos[self.ref_joint_pos_indexes] = joint_pos[:]
        self.sim.forward()
        assert (self.sim.data.qpos[:7] - joint_pos).sum() < 1e-10
        if is_grasped:
            self.sim.data.qpos[self.ref_gripper_joint_pos_indexes] = gripper_qpos
            self.sim.data.qvel[self.ref_gripper_joint_pos_indexes] = gripper_qvel
            ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
            ee_new_mat = pose2mat((self._eef_xpos, self._eef_xquat))
            transform = ee_new_mat @ np.linalg.inv(ee_old_mat)
            new_object_pose = mat2pose(
                np.dot(transform, pose2mat((object_pos, object_quat)))
            )
            if "Lift" in self.env_name:
                set_object_pose(self, "cube", new_object_pose[0], new_object_pose[1])
            self.sim.forward()
        return 0

    def check_grasp(self, **kwargs):
        return check_grasp(self, self.env_name)

    def process_state_frames(self, frames):
        raise NotImplementedError

    def get_image(self):
        return self.sim.render(
            camera_name=self.config["camera_name"],
            width=self.config["screen_width"],
            height=self.config["screen_height"],
        )

    def get_vid_image(self):
        return np.flipud(self.sim.render(camera_name="frontview", width=960, height=540))[:, :, ::-1]

    def take_mp_step(
        self,
        state,
        start_pos,
        is_grasped=False,
    ):
        ac_to_do = self._wrapped_env.form_action(state)
        self._wrapped_env._wrapped_env.step(ac_to_do, is_planner=True)

    def get_robot_mask(self):
        sim = self.sim
        self.sim.forward()
        segmentation_map = CU.get_camera_segmentation(
            camera_name="frontview",
            camera_width=960,
            camera_height=540,
            sim=sim,
        )
        geom_ids = np.unique(segmentation_map[:, :, 1])
        robot_mask = np.expand_dims(np.any(
            [segmentation_map[:, :, 1] == robot_id for robot_id in self.robot_geom_ids], axis=0
        ), -1)
        return robot_mask

    def rebuild_controller(self):
        pass

    def take_mp_step(self, state, is_grasped=False):
        curr_qpos = self.sim.data.qpos.copy()
        self.sim.data.qpos[self.ref_joint_pos_indexes] = state[:]
        self.sim.forward()