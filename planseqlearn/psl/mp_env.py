import numpy as np
from robosuite.utils.transform_utils import *
from tqdm import tqdm
from planseqlearn.psl.vision_utils import reset_precompute_sam_poses
from planseqlearn.psl.sam_utils import build_models
from rlkit.envs.wrappers import ProxyEnv as RlkitProxyEnv

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    pass

from atob.caelan_smoothing import *


def smooth_cubic(
    path,
    collision_fn,
    resolutions,
    v_max=None,
    a_max=None,
    limit_lower=None,
    limit_upper=None,
    time_step=1e-2,
    parabolic=True,
    sample=False,
    intermediate=True,
    max_iterations=1000,
    max_time=np.inf,
    min_improve=0.0,
    verbose=False,
):
    start_time = time.time()
    if path is None:
        return None
    assert (v_max is not None) or (a_max is not None)
    assert path and (max_iterations < np.inf) or (max_time < np.inf)

    def curve_collision_fn(segment, t0=None, t1=None):
        _, samples = sample_discretize_curve(
            segment, resolutions, start_t=t0, end_t=t1, time_step=time_step
        )
        if any(map(collision_fn, default_selector(samples))):
            return True
        # Check for joint limits
        if limit_lower is not None and limit_upper is not None:
            if any((s < limit_lower).any() or (s > limit_upper).any() for s in samples):
                return True
        return False

    start_positions = waypoints_from_path(
        path
    )
    if len(start_positions) == 1:
        start_positions.append(start_positions[-1])

    start_durations = [0] + [
        solve_linear(np.subtract(p2, p1), v_max, a_max, t_min=1e-3, only_duration=True)
        for p1, p2 in get_pairs(start_positions)
    ] 
    start_times = np.cumsum(start_durations)  
    start_velocities = [np.zeros(len(start_positions[0])) for _ in range(len(start_positions))]
    start_curve = CubicHermiteSpline(start_times, start_positions, dydx=start_velocities)
    
    if len(start_positions) <= 2:
        return start_curve

    curve = start_curve
    for iteration in range(max_iterations):
        if elapsed_time(start_time) >= max_time:
            break
        times = curve.x
        durations = [0.0] + [t2 - t1 for t1, t2 in get_pairs(times)]
        positions = [curve(t) for t in times]
        velocities = [curve(t, nu=1) for t in times]

        t1, t2 = np.random.uniform(times[0], times[-1], 2)
        if t1 > t2:
            t1, t2 = t2, t1
        ts = [t1, t2]
        i1 = find(lambda i: times[i] <= t1, reversed(range(len(times))))  # index before t1
        i2 = find(lambda i: times[i] >= t2, range(len(times)))  # index after t2
        assert i1 != i2

        local_positions = [curve(t) for t in ts]
        local_velocities = [curve(t, nu=1) for t in ts]
        # Before continuing, check if the new local_positions violate joint limits
        if limit_lower is not None and limit_upper is not None:
            if any(
                (pos < limit_lower).any() or (pos > limit_upper).any() for pos in local_positions
            ):
                continue  # Skip this iteration if any position is outside the limits

        if not all(
            np.less_equal(np.absolute(v), np.array(v_max) + EPSILON).all() for v in local_velocities
        ):
            continue

        x1, x2 = local_positions
        v1, v2 = local_velocities

        current_t = (t2 - t1) - min_improve 
        # min_t = 0
        min_t = find_lower_bound(x1, x2, v1, v2, v_max=v_max, a_max=a_max)
        if parabolic:
            # Softly applies limits
            min_t = solve_multivariate_ramp(
                x1, x2, v1, v2, v_max, a_max
            )  
            if min_t is None:
                continue
        if min_t >= current_t:
            continue
        best_t = random.uniform(min_t, current_t) if sample else min_t

        local_durations = [t1 - times[i1], best_t, times[i2] - t2]
        # local_times = [0, best_t]
        local_times = [
            t1,
            (t1 + best_t),
        ]  # Good if the collision function is time varying

        if intermediate:
            local_curve = CubicHermiteSpline(local_times, local_positions, dydx=local_velocities)
            if curve_collision_fn(local_curve, t0=None, t1=None):  # check_spline
                continue
            # local_positions = [local_curve(x) for x in local_curve.x]
            # local_velocities = [local_curve(x, nu=1) for x in local_curve.x]
            local_durations = (
                [t1 - times[i1]]
                + [x - local_curve.x[0] for x in local_curve.x[1:]]
                + [times[i2] - t2]
            )

        new_durations = np.concatenate([durations[: i1 + 1], local_durations, durations[i2 + 1 :]])
        new_times = np.cumsum(new_durations)
        new_positions = positions[: i1 + 1] + local_positions + positions[i2:]
        new_velocities = velocities[: i1 + 1] + local_velocities + velocities[i2:]

        new_curve = CubicHermiteSpline(new_times, new_positions, dydx=new_velocities)
        if not intermediate and curve_collision_fn(new_curve, t0=None, t1=None):
            continue
        if verbose:
            print(
                "Iterations: {} | Current time: {:.3f} | New time: {:.3f} | Elapsed time: {:.3f}".format(
                    iteration,
                    spline_duration(curve),
                    spline_duration(new_curve),
                    elapsed_time(start_time),
                )
            )
        curve = new_curve
    if verbose:
        print(
            "Iterations: {} | Start time: {:.3f} | End time: {:.3f} | Elapsed time: {:.3f}".format(
                max_iterations,
                spline_duration(start_curve),
                spline_duration(curve),
                elapsed_time(start_time),
            )
        )
    return curve



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
        use_joint_space_mp=True,
        planning_time=5.0,
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
            use_joint_space_mp (bool): Whether to use joint space motion planning instead of end-effector space motion planning.
            planning_time (float): The time to plan for.
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
        self.planning_time = planning_time
        self.use_vision_pose_estimation = use_vision_pose_estimation
        self.use_sam_segmentation = use_sam_segmentation
        self.use_vision_placement_check = use_vision_placement_check
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
            raise NotImplementedError
        
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
                planning_time=self.planning_time,
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
            if self.env_name.startswith("kitchen"):
                obj_name = self.text_plan[self.curr_plan_stage][0]
                self.sim.data.qpos[7:9] = .04
                self.sim.forward()
            else:
                obj_name = self.text_plan[self.curr_plan_stage - (self.curr_plan_stage % 2)][0]
            print(self.text_plan[self.curr_plan_stage])
            if self.teleport_instead_of_mp:
                error = self.set_robot_based_on_ee_pos(
                    target_pos.copy(),
                    target_quat.copy(),
                    self.reset_qpos,
                    self.reset_qvel,
                    obj_name=obj_name,
                )
            else:
                error = self.mp_to_point(
                    target_pos.copy(),
                    target_quat.copy(),
                    self.reset_qpos,
                    self.reset_qvel,
                    obj_name=obj_name,
                    get_intermediate_frames=get_intermediate_frames,
                    planning_time=self.planning_time,
                )
            self.curr_postcondition = self.get_curr_postcondition_function()
        self.curr_ll_step += 1
        if self.env_name.startswith("kitchen"):
            d |= (self.curr_ll_step >= (self.curr_plan_stage + 1) * 25)
        return o, r, d, i

    def backtracking_search_from_goal_pos(
        self,
        start_pos,
        start_quat,
        target_pos,
        target_quat,
        qpos,
        qvel,
        is_grasped=False,
        movement_fraction=0.001,
        obj_name="",
    ):
        curr_pos = target_pos.copy()
        collision = not self.check_state_validity_ee(
            curr_pos,
            target_quat,
            qpos,
            qvel,
            is_grasped=is_grasped,
            obj_name=obj_name,
        )
        iters = 0
        max_iters = int(1 / movement_fraction)
        while collision and iters < max_iters:
            curr_pos = curr_pos - movement_fraction * (target_pos - start_pos)
            valid = self.check_state_validity_ee(
                curr_pos,
                target_quat,
                qpos,
                qvel,
                is_grasped=is_grasped,
                obj_name=obj_name,
            )
            collision = not valid
            iters += 1
        if collision:
            return np.concatenate((start_pos, start_quat))
        else:
            return np.concatenate((curr_pos, target_quat))
    
    def check_state_validity_joint(
        self,
        joint_pos,
        qpos,
        qvel,
        is_grasped,
        obj_name="",
    ):
        self.set_robot_based_on_joint_angles(joint_pos, qpos, qvel, obj_name=obj_name, is_grasped=is_grasped)
        valid = not self.check_robot_collision(
            ignore_object_collision=is_grasped,
            obj_name=obj_name,
        )
        return valid

    def check_state_validity_ee(
        self, 
        pos, 
        quat,
        qpos,
        qvel,
        is_grasped,
        obj_name="",
    ):
        self.set_robot_based_on_ee_pos(pos, quat, qpos, qvel, obj_name=obj_name)
        valid = not self.check_robot_collision(
            ignore_object_collision=is_grasped,
            obj_name=obj_name,
        )
        return valid
    
    def backtracking_search_from_goal_joints(
        self,
        start_angles,
        goal_angles,
        qpos,
        qvel,
        obj_name="",
        is_grasped=False,
        movement_fraction=0.001,
    ):
        curr_angles = goal_angles.copy()
        valid = self.check_state_validity_joint(
            curr_angles,
            qpos,
            qvel,
            is_grasped=is_grasped,
            obj_name=obj_name,
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
                obj_name=obj_name,
            )
            collision = not valid
            iters += 1
        if collision:
            return start_angles
        else:
            return curr_angles


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
        target_qpos=None
    ):
        if self.use_joint_space_mp:
            target_angles = target_qpos.astype(np.float64)
            space = ob.RealVectorStateSpace(7)
            bounds = ob.RealVectorBounds(7)
            env_bounds = self.get_joint_bounds()
            for i in range(7):
                bounds.setLow(i, env_bounds[i][0])
                bounds.setHigh(i, env_bounds[i][1])
            space.setBounds(bounds)
            si = ob.SpaceInformation(space)
            si.setStateValidityChecker(ob.StateValidityCheckerFn(check_valid))
            si.setStateValidityCheckingResolution(0.005)
            start = ob.State(space)
            for i in range(7):
                start()[i] = self.sim.data.qpos[i].astype(np.float64)
            goal = ob.State(space)
            for i in range(7):
                goal()[i] = target_angles[i]
            goal_valid = check_valid(goal())
            start_valid = check_valid(start())
            if not goal_valid: 
                # raise end-effector to avoid collision
                for _ in range(100):
                    self.wrapped_env.step(np.array([0, 0, .01, 0, 0, 0, 0]))
                    collision = self.check_robot_collision(
                        ignore_object_collision=is_grasped,
                        obj_name=obj_name,
                    )
                    if not collision:
                        break
                angles = self.sim.data.qpos[:7].copy()
                    
                print(f"Goal valid: {goal_valid}")
                goal = ob.State(space)
                for i in range(7):
                    goal()[i] = angles[i]
                goal_valid = check_valid(goal())
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
            si.setStateValidityCheckingResolution(0.005)
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
            goal().rotation().x = start_quat[0]
            goal().rotation().y = start_quat[1]
            goal().rotation().z = start_quat[2]
            goal().rotation().w = start_quat[3]
            goal_valid = check_valid(goal())
            if not goal_valid:
                pos = self.backtracking_search_from_goal_pos(
                    start_pos,
                    start_quat,
                    target_pos,
                    target_quat,
                    qpos,
                    qvel,
                    is_grasped=is_grasped,
                    obj_name=obj_name,
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

    def get_mp_waypoints(self, path, qpos, qvel, obj_name):
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
    
    def check_robot_collision_at_state(self, state, qpos, qvel, obj_name, is_grasped):
        if self.use_joint_space_mp:
            collision = not self.check_state_validity_joint(
                state,
                qpos,
                qvel,
                is_grasped=is_grasped,
                obj_name=obj_name,
            )
        else:
            collision = not self.check_state_validity_ee(
                state[:3],
                state[3:],
                qpos,
                qvel,
                is_grasped=is_grasped,
                obj_name=obj_name,
            )
        return collision    

    def smooth(self, path, timesteps, qpos, qvel, obj_name, is_grasped):
        if self.use_joint_space_mp:
            # FRANKA_LOWER_LIMITS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            FRANKA_LOWER_LIMITS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.05, -2.8973])
            # FRANKA_UPPER_LIMITS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
            FRANKA_UPPER_LIMITS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.75, 2.8973])

            FRANKA_VELOCITY_LIMIT = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
            FRANKA_ACCELERATION_LIMIT = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])

            curve = smooth_cubic(
                path,
                lambda q: self.check_robot_collision_at_state(q, qpos, qvel, obj_name, is_grasped),
                np.radians(0.1) * np.ones(7),
                FRANKA_VELOCITY_LIMIT,
                FRANKA_ACCELERATION_LIMIT,
                limit_lower=FRANKA_LOWER_LIMITS,
                limit_upper=FRANKA_UPPER_LIMITS,
            )
            ts = (curve.x[-1] - curve.x[0]) / (timesteps - 1)
            return np.array([curve(ts * i) for i in range(timesteps)])
        else:
            return path

    def mp_to_point(
        self,
        target_pos,
        target_quat,
        qpos,
        qvel,
        is_grasped=False,
        obj_name="",
        planning_time=5.0,
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
        if self.use_joint_space_mp:
            target_qpos = self.compute_ik(
                target_pos, target_quat, qpos, qvel, og_qpos, og_qvel
            )
        else:
            target_qpos = None

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
                    obj_name=obj_name,
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
                    valid = self.check_state_validity_ee(
                        pos,
                        quat,
                        qpos,
                        qvel,
                        is_grasped=is_grasped,
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
            target_qpos=target_qpos,
        )
        # create a problem instance
        curr_sol = None
        ct = 0
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
        self.intermediate_frames = []
        self.clean_frames = []
        self.intermediate_qposes, self.intermediate_qvels, self.intermediate_states = [], [], []
        if solved:
            path = pdef.getSolutionPath()
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
            og.PathSimplifier(si).simplify(path, 1)
            converted_path = self.smooth(converted_path, 50,  qpos, qvel, obj_name, is_grasped)
            self.waypoint_imgs, self.waypoint_masks = self.get_mp_waypoints(
                converted_path, qpos, qvel, obj_name=obj_name
            )
            self.sim.data.qpos[:] = og_qpos.copy()
            self.sim.data.qvel[:] = og_qvel.copy()
            self.sim.forward()
            self.update_mp_controllers()
            self.break_mp = False
            self.set_robot_colors(np.array([0.1, 0.3, 0.7, 1.0]))
            self.get_intermediate_frames = get_intermediate_frames
            
            converted_path = converted_path[1:] # remove the start state
            for state_idx, state in tqdm(enumerate(converted_path)):
                start_qpos = self.sim.data.qpos[:].copy()
                if state_idx == 0:
                    self.intermediate_qposes = [start_qpos.copy()]
                    self.intermediate_qvels = [self.sim.data.qvel[:].copy()]
                    self.intermediate_states = [self.get_env_state() if hasattr(self, "get_env_state") else None]
                for step in (range(self.num_mp_execution_steps)):
                    self.take_mp_step(
                        state,
                        is_grasped,
                        state_idx,
                        start_qpos,
                        step,
                        50
                    )

            self.reset_robot_colors()
            self.rebuild_controller()
            print(f"Error: {np.linalg.norm(self._eef_xpos - target_pos)}")
            print(self._eef_xpos, target_pos)
            return np.linalg.norm(self._eef_xpos - target_pos)
        else:
            self.sim.data.qpos[:] = og_qpos.copy()
            self.sim.data.qvel[:] = og_qvel.copy()
            self.sim.forward()
            return -np.inf

    def update_intermediate_frames(self, state_idx):
        if self.get_intermediate_frames:
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
                self.clean_frames.append(im2[:, :, ::-1])
            else:
                self.clean_frames.append(im2)
            robot_mask = self.waypoint_masks[state_idx]
            im = (
                0.5 * (im * robot_mask)
                + 0.5 * self.waypoint_imgs[state_idx]
                + im * (1 - robot_mask)
            )
            self.intermediate_frames.append(im)
        