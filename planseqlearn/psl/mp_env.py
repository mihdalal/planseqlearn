import numpy as np
from robosuite.utils.transform_utils import *
from planseqlearn.psl.vision_utils import reset_precompute_sam_poses
from planseqlearn.psl.sam_utils import build_models
from rlkit.envs.wrappers import ProxyEnv as RlkitProxyEnv
from collections import OrderedDict
import sys

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    pass


class ProxyEnv(RlkitProxyEnv):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env


class PSLEnv(ProxyEnv):
    def __init__(
        self,
        env,
        env_name,
        # llm
        text_plan,
        # mp
        teleport_instead_of_mp=True,
        teleport_on_grasp=True,
        use_joint_space_mp=True,
        # vision
        use_vision_pose_estimation=False,
        use_sam_segmentation=False,
        use_vision_placement_check=False,
        **kwargs,
    ):
        """
        Base class for Plan-Seq-Learn Environments.
        Args:
            env: The environment to wrap.
            teleport_instead_of_mp (bool): Whether to teleport instead of motion planning.
            teleport_on_grasp (bool): Whether to teleport on grasp.
            use_joint_space_mp (bool): Whether to use joint space motion planning instead of end-effector space motion planning.
            use_vision_pose_estimation (bool): Whether to use vision pose estimation.
            use_sam_segmentation (bool): Whether to use SAM for segmentation or the simulator.
            use_vision_placement_check (bool): Whether to use vision-based placement checking.
            use_llm_plan (bool): Whether to use LLM high-level plan.
            text_plan (list): Text plan.
        """
        super().__init__(env)
        self.env_name = env_name
        # mp and teleporting
        self.teleport_instead_of_mp = teleport_instead_of_mp
        self.use_joint_space_mp = use_joint_space_mp
        self.use_vision_pose_estimation = use_vision_pose_estimation
        self.use_sam_segmentation = use_sam_segmentation
        self.use_vision_placement_check = use_vision_placement_check
        self.teleport_on_grasp = teleport_on_grasp
        if self.use_sam_segmentation:
            self.dino, self.sam = build_models(
                config_file="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                grounded_checkpoint="Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
                sam_checkpoint="Grounded-Segment-Anything/sam_vit_h_4b8939.pth",
                sam_hq_checkpoint=None,
                use_sam_hq=False,
            )
        print(f"Text plan: {text_plan}")
        self.text_plan = text_plan
        # trajectory information
        self.num_high_level_steps = 0
        self.object_idx = 0
        self.curr_ll_step = 0
    
    def set_robot_colors(self, colors):
        if type(colors) is np.ndarray:
            colors = [colors] * len(self.robot_geom_ids)
        for idx, geom_id in enumerate(self.robot_geom_ids):
            self.sim.model.geom_rgba[geom_id] = colors[idx]
        self.sim.forward()

    def reset_robot_colors(self):
        self.set_robot_colors(self.original_colors)
        self.sim.forward()

    def get_observation(self):
        raise NotImplementedError

    def get_image(self):
        raise NotImplementedError

    def check_grasp(self, **kwargs):
        raise NotImplementedError

    def check_place(self, **kwargs):
        raise NotImplementedError

    def check_robot_collision(self, is_grasped=False):
        raise NotImplementedError

    def post_reset_burn_in(self):
        pass

    def get_object_pose_mp(self, obj_idx=0):
        raise NotImplementedError

    def get_all_initial_object_poses(self):
        self.initial_object_pos_dict = {}
        for obj_name, action in self.text_plan:
            if action == "grasp":
                self.initial_object_pos_dict[obj_name] = self.get_mp_target_pose(obj_name)[0].copy()

    def get_curr_postcondition_function(self):
        if self.text_plan[self.curr_plan_stage][1].lower() == "grasp":
            return self.named_check_object_grasp(self.text_plan[self.curr_plan_stage][0])
        elif self.text_plan[self.curr_plan_stage][1].lower() == "place":
            return self.named_check_object_placement(self.text_plan[self.curr_plan_stage][0])
        else:
            raise NotImplementedError("Currently only supporting grasp and place postconditions")

    def reset(self, get_intermediate_frames=False, **kwargs):
        self.num_high_level_steps = 0
        self.object_idx = 0
        # reset wrapped env
        obs = self._wrapped_env.reset(**kwargs)
        self.curr_plan_stage = 0
        self.curr_ll_step = 0
        self.curr_postcondition = self.get_curr_postcondition_function()

        self.post_reset_burn_in()
        if self.use_sam_segmentation:
            reset_precompute_sam_poses(self)
        # reset 
        self.reset_pos, self.reset_ori = self._eef_xpos.copy(), self._eef_xquat.copy()
        self.reset_qpos, self.reset_qvel = (
            self.sim.data.qpos.copy(),
            self.sim.data.qvel.copy(),
        )
        self.get_all_initial_object_poses()
        self.update_controllers()
        # go to initial position 
        target_pos, target_quat = self.get_target_pos()
        if self.teleport_instead_of_mp:
            self.set_robot_based_on_ee_pos(
                target_pos,
                target_quat,
                self.reset_qpos,
                self.reset_qvel,
                obj_name=self.text_plan[0][0],  # first step
            )
        else:
            self.mp_to_point(
                target_pos,
                target_quat,
                self.reset_qpos,
                self.reset_qvel,
                obj_name=self.text_plan[0][0],
                get_intermediate_frames=get_intermediate_frames,
            )
        try:
            obs = self.get_observation()
        except:
            obs = np.zeros_like(self.observation_space.low, dtype=np.float64) 
            # doesn't matter since all training is with vision
        return obs

    def step(self, action, get_intermediate_frames=False, **kwargs):
        o, r, d, i = self._wrapped_env.step(action)
        curr_plan_stage_finished = \
            self.curr_postcondition(info=i) and (self.curr_plan_stage) != len(self.text_plan) - 1
        if curr_plan_stage_finished: # advance plan to next stage using motion planner
            self.curr_plan_stage += 1
            target_pos, target_quat = self.get_target_pos()
            if self.teleport_instead_of_mp:
                error = self.set_robot_based_on_ee_pos(
                    target_pos.copy(),
                    target_quat.copy(),
                    self.reset_qpos,
                    self.reset_qvel,
                    obj_name=self.text_plan[self.curr_plan_stage - (self.curr_plan_stage % 2)][0],
                )
            else:
                error = self.mp_to_point(
                    target_pos.copy(),
                    target_quat.copy(),
                    self.reset_qpos,
                    self.reset_qvel,
                    obj_name=self.text_plan[self.curr_plan_stage - (self.curr_plan_stage % 2)][0],
                    get_intermediate_frames=get_intermediate_frames,
                )
            self.curr_postcondition = self.get_curr_postcondition_function()
        self.curr_ll_step += 1
        if self.env_name.startswith("kitchen"):
            d |= (self.curr_ll_step >= (self.curr_plan_stage + 1) * 25)
        return o, r, d, i

    def construct_mp_problem(
        self,
        target_pos,
        target_quat,
        start_pos,
        start_quat,
        qpos,
        qvel,
        og_qpos,
        og_qvel,
        is_grasped,
        obj_name,
        check_valid,
        backtrack_movement_fraction=0.001,
    ):
        if self.use_joint_space_mp:
            target_angles = self.compute_ik(
                target_pos, target_quat, qpos, qvel, og_qpos, og_qvel
            ).astype(np.float64)
            space = ob.RealVectorStateSpace(7)
            bounds = ob.RealVectorBounds(7)
            env_bounds = self.get_joint_bounds()
            for i in range(7):
                bounds.setLow(i, env_bounds[i, 0])
                bounds.setHigh(i, env_bounds[i, 1])
            space.setBounds(bounds)
            si = ob.SpaceInformation(space)
            si.setStateValidityChecker(ob.StateValidityCheckerFn(check_valid))
            si.setStateValidityCheckingResolution(0.01)
            start = ob.State(space)
            for i in range(7):
                start()[i] = self.sim.data.qpos[i].astype(np.float64)
            goal = ob.State(space)
            for i in range(7):
                goal()[i] = target_angles[i]
            goal_valid = check_valid(goal())
            # do manual check
            if not goal_valid:
                if self.env_name.startswith("Sawyer"):
                    target_pos, target_quat = self.backtracking_search_from_goal_pos(
                        start_pos=start_pos,
                        start_quat=start_quat,
                        target_quat=target_quat,
                        goal_pos=target_pos,
                        qpos=qpos,
                        qvel=qvel,
                        obj_name=obj_name,
                    )
                    target_angles = self.compute_ik(
                        target_pos, target_quat, qpos, qvel, og_qpos, og_qvel
                    ).astype(np.float64)
                else:
                    target_angles = self.backtracking_search_from_goal_joints(
                        start_angles=og_qpos[:7],
                        goal_angles=target_angles,
                        qpos=qpos,
                        qvel=qvel,
                        movement_fraction=backtrack_movement_fraction,
                        is_grasped=is_grasped,
                        obj_name=obj_name,
                    )
                for i in range(7):
                    goal()[i] = target_angles[i]
                assert check_valid(goal())
        else:
            space = ob.SE3StateSpace()
            # set lower and upper bounds
            bounds = ob.RealVectorBounds(3)
            # compare bounds to start state
            bounds_low = self.mp_bounds_low
            bounds_high = self.mp_bounds_high

            bounds_low = np.minimum(self.mp_bounds_low, start_pos)
            bounds_high = np.maximum(self.mp_bounds_high, start_pos)
            target_pos[:3] = np.clip(target_pos, bounds_low, bounds_high)

            bounds.setLow(0, bounds_low[0])
            bounds.setLow(1, bounds_low[1])
            bounds.setLow(2, bounds_low[2])
            bounds.setHigh(0, bounds_high[0])
            bounds.setHigh(1, bounds_high[1])
            bounds.setHigh(2, bounds_high[2])
            space.setBounds(bounds)

            # construct an instance of space information from this state space
            si = ob.SpaceInformation(space)
            # set state validity checking for this space
            si.setStateValidityChecker(ob.StateValidityCheckerFn(check_valid))
            si.setStateValidityCheckingResolution(0.001)
            # create a random start state
            start = ob.State(space)
            start().setXYZ(*start_pos)
            start().rotation().x = start_quat[0]
            start().rotation().y = start_quat[1]
            start().rotation().z = start_quat[2]
            start().rotation().w = start_quat[3]
            start_valid = check_valid(start())
            goal = ob.State(space)
            goal().setXYZ(*target_pos)
            goal().rotation().x = target_quat[0]
            goal().rotation().y = target_quat[1]
            goal().rotation().z = target_quat[2]
            goal().rotation().w = target_quat[3]
            goal_valid = check_valid(goal())
            if not goal_valid:
                pos = self.backtracking_search_from_goal(
                    start_pos,
                    start_quat,
                    target_pos,
                    target_quat,
                    qpos,
                    qvel,
                    is_grasped=is_grasped,
                    open_gripper_on_tp=open_gripper_on_tp,
                    obj_idx=obj_idx,
                )
                goal = ob.State(space)
                goal().setXYZ(*pos[:3])
                goal().rotation().x = pos[3]
                goal().rotation().y = pos[4]
                goal().rotation().z = pos[5]
                goal().rotation().w = pos[6]
                goal_error = self.set_robot_based_on_ee_pos(
                    pos[:3],
                    pos[3:],
                    qpos,
                    qvel,
                    obj_name=obj_name,
                )
        return start, goal, si

    def get_mp_waypoints(self, path, qpos, qvel, is_grasped, obj_name):
        waypoint_imgs, waypoint_masks = [], []
        for i, state in enumerate(path):
            if self.use_joint_space_mp:
                self.set_robot_based_on_joint_angles(
                    state,
                    qpos,
                    qvel,
                    obj_name=obj_name
                )
            else:
                self.set_robot_based_on_ee_pos(
                    state[:3],
                    state[3:],
                    qpos,
                    qvel,
                    obj_name=obj_name,
                )
            if hasattr(self, "get_vid_image"):
                im = self.get_vid_image()
            else:
                im = self.get_image()
            mask = self.get_robot_mask()
            waypoint_imgs.append(im * mask)
            waypoint_masks.append(mask)
        return [waypoint_imgs[-1] for i in range(len(waypoint_imgs))], [
            waypoint_masks[-1] for i in range(len(waypoint_masks))
        ]

    def mp_to_point(
        self,
        target_pos,
        target_quat,
        qpos,
        qvel,
        is_grasped=False,
        obj_name="",
        open_gripper_on_tp=False,
        planning_time=50.0,
        get_intermediate_frames=False,
    ):
        if "place" in self.text_plan[self.curr_plan_stage][1]:
            is_grasped = self.named_check_object_grasp(self.text_plan[self.curr_plan_stage - 1][0])()
        else:
            is_grasped = self.named_check_object_grasp(self.text_plan[self.curr_plan_stage][0])()
        if target_pos is None:
            return -np.inf
        get_intermediate_frames = True
        og_qpos = self.sim.data.qpos.copy()
        og_qvel = self.sim.data.qvel.copy()
        og_eef_xpos = self._eef_xpos.copy().astype(np.float64)
        og_eef_xquat = self._eef_xquat.copy().astype(np.float64)
        og_eef_xquat /= np.linalg.norm(og_eef_xquat)
        try:
            target_quat = target_quat.astype(np.float64)
            target_quat /= np.linalg.norm(target_quat)
        except:
            pass
        target_pos = target_pos.astype(np.float64)
        self.update_controllers()

        def isStateValid(state):
            if self.use_joint_space_mp:
                joint_pos = np.zeros(7)
                for i in range(7):
                    joint_pos[i] = state[i]
                if all(joint_pos == og_qpos[:7]):
                    return True
                valid = self.check_state_validity_joint(
                    joint_pos,
                    qpos,
                    qvel,
                    is_grasped=is_grasped,
                    obj_name=obj_name 
                )
                return valid
            else:
                pos = np.array([state.getX(), state.getY(), state.getZ()])
                quat = np.array(
                    [
                        state.rotation().x,
                        state.rotation().y,
                        state.rotation().z,
                        state.rotation().w,
                    ]
                )
                if all(pos == og_eef_xpos) and all(quat == og_eef_xquat):
                    # start state is always valid.
                    return True
                else:
                    self.set_robot_based_on_ee_pos(
                        pos,
                        quat,
                        qpos,
                        qvel,
                        obj_name=obj_name,
                    )
                    valid = not self.check_robot_collision(
                        ignore_object_collision=is_grasped,
                        obj_name=obj_name,
                    )
                return valid
        start, goal, si = self.construct_mp_problem(
            target_pos=target_pos,
            target_quat=target_quat,
            start_pos=og_eef_xpos,
            start_quat=og_eef_xquat,
            qpos=qpos,
            qvel=qvel,
            og_qpos=og_qpos,
            og_qvel=og_qvel,
            is_grasped=is_grasped,
            obj_name=obj_name,
            check_valid=isStateValid,
        )

        # create a problem instance
        curr_sol = None
        ct = 0
        while "Exact" not in str(curr_sol):
            pdef = ob.ProblemDefinition(si)
            # set the start and goal states
            pdef.setStartAndGoalStates(start, goal)
            # create a planner for the defined space
            pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
            planner = og.AITstar(si)
            # set the problem we are trying to solve for the planner
            planner.setProblemDefinition(pdef)
            # perform setup steps for the planner
            planner.setup()
            # attempt to solve the problem within planning_time seconds of planning time
            solved = planner.solve(planning_time)
            print(f"Solved: {solved}")
            curr_sol = solved
            ct += 1
            if not self.retry:
                break
            if ct >= 5:
                assert False
        intermediate_frames = []
        clean_frames = []
        if solved:
            path = pdef.getSolutionPath()
            success = og.PathSimplifier(si).simplify(path, 0.001)
            converted_path = []
            for s, state in enumerate(path.getStates()):
                if self.use_joint_space_mp:
                    new_state = np.array([path.getStates()[s][j] for j in range(7)])
                else:
                    new_state = np.array(
                        [
                            state.getX(),
                            state.getY(),
                            state.getZ(),
                            state.rotation().x,
                            state.rotation().y,
                            state.rotation().z,
                            state.rotation().w,
                        ]
                    )
                converted_path.append(new_state)
            #converted_path = converted_path[1:]
            if get_intermediate_frames:
                waypoint_imgs, waypoint_masks = self.get_mp_waypoints(
                    converted_path, qpos, qvel, is_grasped, obj_name=obj_name
                )
            self._wrapped_env.reset()
            self.sim.data.qpos[:] = og_qpos.copy()
            self.sim.data.qvel[:] = og_qvel.copy()
            self.sim.forward()
            self.update_mp_controllers()
            self.break_mp = False
            self.set_robot_colors(np.array([0.1, 0.3, 0.7, 1.0]))
            self.intermediate_qposes, self.intermediate_qvels = [], []
            for state_idx, state in enumerate(converted_path):
                state_frames = []
                try:
                    state_frames = self.process_state_frames(state_frames)
                except:
                    pass
                start_qpos = self.sim.data.qpos[:].copy()
                if state_idx == 0:
                    self.intermediate_qposes = [start_qpos.copy()]
                    self.intermediate_qvels = [self.sim.data.qvel[:].copy()]
                print(f"First state: {state}")
                for step in range(20):
                    self.take_mp_step(
                        state,
                        is_grasped,
                    )
                    self.intermediate_qposes.append(self.sim.data.qpos[:].copy())
                    self.intermediate_qvels.append(self.sim.data.qvel[:].copy())
                    if get_intermediate_frames:
                        if hasattr(self, "get_vid_image"):
                            im = self.get_vid_image()
                        else:
                            im = self.get_image()
                        self.reset_robot_colors()
                        if hasattr(self, "get_vid_image"):
                            im2 = self.get_vid_image()
                        else:
                            im2 = self.get_image()
                        self.set_robot_colors(np.array([0.1, 0.3, 0.7, 1.0]))
                        if self.env_name.endswith("v2"):
                            clean_frames.append(im2[:, :, ::-1])
                        else:
                            clean_frames.append(im2)
                        try:
                            state_frames.append(im)
                            state_frames = self.process_state_frames(state_frames)
                        except:
                            robot_mask = waypoint_masks[state_idx]
                            im = (
                                0.5 * (im * robot_mask)
                                + 0.5 * waypoint_imgs[state_idx]
                                + im * (1 - robot_mask)
                            )
                            intermediate_frames.append(im)
                try:
                    state_frames = self.process_state_frames(state_frames)
                    intermediate_frames.extend(state_frames)
                except:
                    pass

            self.reset_robot_colors()
            self.rebuild_controller()
            self.intermediate_frames = intermediate_frames
            self.clean_frames = clean_frames
            print(f"Error: {np.linalg.norm(self._eef_xpos - target_pos)}")
            return np.linalg.norm(self._eef_xpos - target_pos)
