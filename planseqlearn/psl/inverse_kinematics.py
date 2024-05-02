# mostly taken from rlkit
import collections

import numpy as np
from dm_control.mujoco.wrapper import mjbindings

IKResult = collections.namedtuple("IKResult", ["qpos", "err_norm", "steps", "success"])
import robosuite.utils.transform_utils as T


def indexer(env, names):
    indices = []
    for name in names:
        indices.append(env.sim.model.joint_name2id(name))

    return indices


def qpos_from_site_pose(
    env,
    site,
    target_pos=None,
    target_quat=None,
    joint_names=None,
    max_steps=100,
    rot_weight=1.0,
    tol=1e-14,
    max_update_norm=2.0,
    progress_thresh=20.0,
    regularization_threshold=0.1,
    regularization_strength=3e-2,
):
    mjlib = mjbindings.mjlib
    dtype = env.sim.data.qpos.dtype

    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, len(joint_names)), dtype=dtype)
        err = np.empty(6, dtype=dtype)

        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, len(joint_names)), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, _ = jac, None
            err_pos, _ = err, None
        elif target_quat is not None:
            _, jac_rot = None, jac
            _, err_rot = None, err
        else:
            raise ValueError(
                "At least one of `target_pos` or `target_quat` must be specified"
            )

    non_movable_joint_names = list(set(env.sim.model.joint_names) - set(joint_names))
    non_movable_joint_indices = indexer(env, non_movable_joint_names)

    update_nv = np.zeros(len(env.sim.data.qpos), dtype=dtype)
    # for i, name in zip(non_movable_joint_indices, non_movable_joint_names):
    #    update_nv[i] = env._get_qpos(name)

    site_xpos = env.data.get_site_xpos(site)
    site_xmat = env.data.get_site_xmat(site).ravel()

    if target_quat is not None:
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)
        site_xquat = np.empty(4, dtype=dtype)

    if joint_names is None:
        dof_indices = slice(None)
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        dof_indices = indexer(env, joint_names)

    steps = 0
    success = False

    for steps in range(max_steps):
        err_norm = 0.0

        if target_pos is not None:
            err_pos[:] = target_pos - site_xpos
            # print('steps=', steps, '   ', np.linalg.norm(err_pos))
            err_norm += np.linalg.norm(err_pos)

        if target_quat is not None:
            mjlib.mju_mat2Quat(site_xquat, site_xmat)
            mjlib.mju_negQuat(neg_site_xquat, site_xquat)
            mjlib.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mjlib.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            # print('IK success')
            success = True
            break
        else:
            jac_pos = env.data.get_site_jacp(site).reshape((3, env.sim.model.nv))
            if target_quat is not None:
                jac_rot = env.data.get_site_jacr(site).reshape((3, env.sim.model.nv))
                jac = np.concatenate((jac_pos, jac_rot))
            else:
                jac = jac_pos
            jac_joints = jac[:, dof_indices]

            # reg_strength later
            reg_strength = (
                regularization_strength if err_norm > regularization_threshold else 0.0
            )

            collision = env.sim.data.ncon
            update_joints = nullspace_method(
                jac_joints, err, regularization_strength, collision
            )
            update_norm = np.linalg.norm(update_joints)

            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            update_nv[dof_indices] = update_joints

            env.set_state(
                env.sim.data.qpos.copy() + update_nv, env.sim.data.qvel.ravel().copy()
            )
            ##env.set_state(env.sim.data.qpos+update_nv, np.ones(len(env.sim.data.qvel))*0.01)
            # env.step(update_nv[:-2])
    return IKResult(
        qpos=env.sim.data.qpos, err_norm=err_norm, steps=steps, success=success
    )


def qpos_from_site_pose_kitchen(
    env,
    site,
    target_pos=None,
    target_quat=None,
    joint_names=None,
    max_steps=100,
    rot_weight=1.0,
    tol=1e-14,
    max_update_norm=2.0,
    progress_thresh=20.0,
    regularization_threshold=0.1,
    regularization_strength=3e-2,
):
    mjlib = mjbindings.mjlib
    dtype = env.sim.data.qpos.dtype

    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, len(joint_names)), dtype=dtype)
        err = np.empty(6, dtype=dtype)

        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, len(joint_names)), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, _ = jac, None
            err_pos, _ = err, None
        elif target_quat is not None:
            _, jac_rot = None, jac
            _, err_rot = None, err
        else:
            raise ValueError(
                "At least one of `target_pos` or `target_quat` must be specified"
            )
    update_nv = np.zeros(len(env.sim.data.qpos), dtype=dtype)

    site_xpos = env.get_site_xpos(site)
    site_xmat = env.get_site_xmat(site).ravel()

    if target_quat is not None:
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)
        site_xquat = np.empty(4, dtype=dtype)

    if joint_names is None:
        dof_indices = slice(None)
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        dof_indices = list(range(7))

    steps = 0
    success = False

    for steps in range(max_steps):
        err_norm = 0.0

        if target_pos is not None:
            err_pos[:] = target_pos - site_xpos
            # print('steps=', steps, '   ', np.linalg.norm(err_pos))
            err_norm += np.linalg.norm(err_pos)

        if target_quat is not None:
            mjlib.mju_mat2Quat(site_xquat, site_xmat)
            mjlib.mju_negQuat(neg_site_xquat, site_xquat)
            mjlib.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mjlib.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight
        if err_norm < tol:
            # print('IK success')
            success = True
            break
        else:
            jac_pos = env.data.get_site_jacp(site).reshape((3, env.sim.model.nv))
            if target_quat is not None:
                jac_rot = env.data.get_site_jacr(site).reshape((3, env.sim.model.nv))
                jac = np.concatenate((jac_pos, jac_rot))
            else:
                jac = jac_pos
            jac_joints = jac[:, dof_indices]

            # reg_strength later
            reg_strength = (
                regularization_strength if err_norm > regularization_threshold else 0.0
            )

            collision = env.sim.data.ncon
            update_joints = nullspace_method(
                jac_joints, err, regularization_strength, collision
            )
            update_norm = np.linalg.norm(update_joints)

            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            update_nv[dof_indices] = update_joints

            # env.set_state(
            #     env.sim.data.qpos.copy() + update_nv, env.sim.data.qvel.ravel().copy()
            # )
            env.sim.data.qpos[:] = env.sim.data.qpos.copy() + update_nv
            env.sim.data.qvel[:] = env.sim.data.qvel.ravel().copy()
            env.sim.forward()
            ##env.set_state(env.sim.data.qpos+update_nv, np.ones(len(env.sim.data.qvel))*0.01)
            # env.step(update_nv[:-2])
    return IKResult(
        qpos=env.sim.data.qpos, err_norm=err_norm, steps=steps, success=success
    )


def qpos_from_site_pose_sampling(
    env,
    site,
    target_pos=None,
    target_quat=None,
    joint_names=None,
    max_steps=100,
    rot_weight=1.0,
    tol=1e-4,
    max_update_norm=2.0,
    progress_thresh=20.0,
    regularization_threshold=0.1,
    regularization_strength=3e-2,
    logging=False,
    trials=10,
):
    mjlib = mjbindings.mjlib
    dtype = env.sim.data.qpos.dtype
    tried = 0
    while True:
        if target_pos is not None and target_quat is not None:
            jac = np.empty((6, len(joint_names)), dtype=dtype)
            err = np.empty(6, dtype=dtype)

            jac_pos, jac_rot = jac[:3], jac[3:]
            err_pos, err_rot = err[:3], err[3:]
        else:
            jac = np.empty((3, len(joint_names)), dtype=dtype)
            err = np.empty(3, dtype=dtype)
            if target_pos is not None:
                jac_pos, _ = jac, None
                err_pos, _ = err, None
            elif target_quat is not None:
                _, jac_rot = None, jac
                _, err_rot = None, err
            else:
                raise ValueError(
                    "At least one of `target_pos` or `target_quat` must be specified"
                )

        non_movable_joint_names = list(
            set(env.sim.model.joint_names) - set(joint_names)
        )
        non_movable_joint_indices = indexer(env, non_movable_joint_names)

        update_nv = np.zeros(len(env.sim.data.qpos), dtype=dtype)
        # for i, name in zip(non_movable_joint_indices, non_movable_joint_names):
        #    update_nv[i] = env._get_qpos(name)

        site_xpos = env.data.get_site_xpos(site)
        site_xmat = env.data.get_site_xmat(site).ravel()

        if target_quat is not None:
            neg_site_xquat = np.empty(4, dtype=dtype)
            err_rot_quat = np.empty(4, dtype=dtype)
            site_xquat = np.empty(4, dtype=dtype)

        if joint_names is None:
            dof_indices = slice(None)
        elif isinstance(joint_names, (list, np.ndarray, tuple)):
            if isinstance(joint_names, tuple):
                joint_names = list(joint_names)
            dof_indices = indexer(env, joint_names)

        steps = 0
        success = False

        for steps in range(max_steps):
            err_norm = 0.0

            if target_pos is not None:
                err_pos[:] = target_pos - site_xpos
                # print('steps=', steps, '   ', np.linalg.norm(err_pos))
                err_norm += np.linalg.norm(err_pos)

            if target_quat is not None:
                mjlib.mju_mat2Quat(site_xquat, site_xmat)
                mjlib.mju_negQuat(neg_site_xquat, site_xquat)
                mjlib.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
                mjlib.mju_quat2Vel(err_rot, err_rot_quat, 1)
                err_norm += np.linalg.norm(err_rot) * rot_weight

            if err_norm < tol:
                # print('IK success')
                success = True
                break
            else:
                jac_pos = env.data.get_site_jacp(site).reshape((3, env.sim.model.nv))
                jac_rot = env.data.get_site_jacr(site).reshape((3, env.sim.model.nv))
                jac = np.concatenate((jac_pos, jac_rot))
                jac_joints = jac[:, dof_indices]

                # reg_strength later
                reg_strength = (
                    regularization_strength
                    if err_norm > regularization_threshold
                    else 0.0
                )

                collision = env.sim.data.ncon
                update_joints = nullspace_method(
                    jac_joints, err, regularization_strength, collision
                )
                update_norm = np.linalg.norm(update_joints)

                progress_criterion = err_norm / update_norm
                if progress_criterion > progress_thresh:
                    break

                if update_norm > max_update_norm:
                    update_joints *= max_update_norm / update_norm

                update_nv[dof_indices] = update_joints

                env.set_state(
                    env.sim.data.qpos.copy() + update_nv,
                    env.sim.data.qvel.ravel().copy(),
                )
                ##env.set_state(env.sim.data.qpos+update_nv, np.ones(len(env.sim.data.qvel))*0.01)
                # env.step(update_nv[:-2])
                if steps % 10 == 0 and logging:
                    print(
                        "Step %2i: err_norm=%-10.3f update_norm=%-10.3f"
                        % (steps, err_norm, update_norm)
                    )

        if env.sim.data.ncon == 0 or tried > trials:
            return IKResult(
                qpos=env.sim.data.qpos, err_norm=err_norm, steps=steps, success=success
            )
        else:
            env.initialize_joints()
            tried += 1


def nullspace_method(jac_joints, delta, regularization_strength=0.0, collision=0.0):
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
