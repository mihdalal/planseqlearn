import numpy as np
import robosuite.utils.camera_utils as CU
from planseqlearn.mnm.sam_utils import get_seg_mask
import matplotlib.pyplot as plt
import open3d as o3d
from plantcv import plantcv as pcv


def get_camera_depth(sim, camera_name, camera_height, camera_width):
    """
    Obtains depth image.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        im (np.array): the depth image b/w 0 and 1
    """
    return sim.render(
        camera_name=camera_name, height=camera_height, width=camera_width, depth=True
    )[1][::-1]


def get_sam_segmentation(env, camera_name, camera_width, camera_height, geom, **kwargs):
    # metaworld environments
    frame = env.sim.render(
        camera_name=camera_name, width=camera_width, height=camera_height
    )
    if env.env_name == "assembly-v2":
        if env.sim.model.body_id2name(env.sim.model.geom_bodyid[geom]) == "asmbly_peg":
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                frame[:, :, ::-1],
                env.dino,
                env.sam,
                text_prompts=["green wrench on table"],
                box_threshold=0.3,
                text_threshold=0.25,
                device="cuda",
                debug=False,
                output_dir="sam_outputs",
            )
            obj_mask = None
            for i in range(len(pred_phrases)):
                if "green wrench" in pred_phrases[i]:
                    obj_mask = obj_masks[i].cpu().detach().numpy()
            assert obj_mask is not None, "Unable to segment wrench"
        elif env.sim.model.body_id2name(env.sim.model.geom_bodyid[geom]) == "peg":
            obj_masks, _, _, pred_phrases, _ = get_seg_mask(
                frame[:, :, ::-1],
                env.dino,
                env.sam,
                text_prompts=["small maroon peg", "robot", "table", "wrench"],
                box_threshold=0.3,
                text_threshold=0.25,
                device="cuda",
                debug=True,
                output_dir="sam_outputs",
            )
            obj_mask = None
            for i in range(len(pred_phrases)):
                if "small maroon peg" in pred_phrases[i]:
                    obj_mask = obj_masks[i].cpu().detach().numpy()
            assert obj_mask is not None, "Unable to segment peg"
        else:
            raise NotImplementedError
    if env.env_name == "disassemble-v2":
        if env.sim.model.body_name2id(env.sim.model.geom_bodyid[geom]) == "asmbly_peg":
            obj_masks, _, _, pred_phases, _ = get_seg_mask(
                frame[:, :, ::-1],
                env.dino,
                env.sam,
                text_prompts=["green wrench on table"],
                box_threshold=0.3,
                text_threshold=0.25,
                device="cuda",
                debug=False,
                output_dir="sam_outputs",
            )
            obj_mask = None
            for i in range(len(pred_phrases)):
                if "green wrench" in pred_phrases[i]:
                    obj_mask = np.transpose(obj_masks[i], (1, 2, 0))
            assert obj_mask is not None, "Unable to segment wrench"
    if env.env_name == "hammer-v2":
        pass
    if env.env_name == "bin-picking-v2":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            frame[:, :, ::-1],
            env.dino,
            env.sam,
            text_prompts=["green cube in red bin"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=False,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "green cube" in pred_phrases[i]:
                obj_mask = obj_masks[i].cpu().detach().numpy()
        assert obj_mask is not None, "Unable to segment wrench"
    # robosuite environments
    if env.env_name == "Lift":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            frame[:, :, ::-1],
            env.dino,
            env.sam,
            text_prompts=["red cube on table"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=False,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "red cube" in pred_phrases[i]:
                obj_mask = obj_masks[i].cpu().detach().numpy()
        assert obj_mask is not None, "Unable to segment wrench"

    if env.env_name == "PickPlaceBread":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            env.dino,
            env.sam,
            text_prompts=["brown package in bin"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "package" in pred_phrases[i]:
                obj_mask = np.transpose(obj_masks[i].cpu().detach().numpy(), (1, 2, 0))

    if env.env_name == "PickPlaceMilk":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            env.dino,
            env.sam,
            text_prompts=["milk carton in bin"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "milk" in pred_phrases[i]:
                obj_mask = np.transpose(obj_masks[i].cpu().detach().numpy(), (1, 2, 0))

    if env.env_name == "PickPlaceCan":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            env.dino,
            env.sam,
            text_prompts=["red can in bin"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "can" in pred_phrases[i]:
                obj_mask = np.transpose(obj_masks[i].cpu().detach().numpy(), (1, 2, 0))

    if env.env_name == "PickPlaceCereal":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            env.dino,
            env.sam,
            text_prompts=["red cereal box in bin"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "cereal" in pred_phrases[i]:
                obj_mask = np.transpose(obj_masks[i].cpu().detach().numpy(), (1, 2, 0))

    if env.env_name == "Door":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            env.dino,
            env.sam,
            text_prompts=["door handle"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "handle" in pred_phrases[i]:
                obj_mask = np.transpose(obj_masks[i].cpu().detach().numpy(), (1, 2, 0))

    if env.env_name == "NutAssemblySquare":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            env.dino,
            env.sam,
            text_prompts=["small brown square with hole"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "square with hole" in pred_phrases[i]:
                obj_mask = np.transpose(obj_masks[i].cpu().detach().numpy(), (1, 2, 0))

    if env.env_name == "NutAssemblyRound":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            env.dino,
            env.sam,
            text_prompts=["small silver circle with hole"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "square with hole" in pred_phrases[i]:
                obj_mask = np.transpose(obj_masks[i].cpu().detach().numpy(), (1, 2, 0))
    # mopa environments
    if env.env_name == "SawyerLiftObstacle-v0":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            env.dino,
            env.sam,
            text_prompts=["red cylinder and robot"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "red cylinder" in pred_phrases[i]:
                obj_mask = np.transpose(obj_masks[i].cpu().detach().numpy(), (1, 2, 0))
        assert obj_mask is not None, "Unable to segment red cylinder"

    if env.env_name == "SawyerAssemblyObstacle-v0":
        raise NotImplementedError

    if env.env_name == "SawyerPushObstacle-v0":
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            np.flipud(frame[:, :, ::-1]),
            env.dino,
            env.sam,
            text_prompts=["small red cube in front of green circle"],
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        obj_mask = None
        for i in range(len(pred_phrases)):
            if "red cube" in pred_phrases[i]:
                obj_mask = np.transpose(obj_masks[i].cpu().detach().numpy(), (1, 2, 0))
        assert obj_mask is not None, "Unable to segment cube"

    if obj_mask is None:
        return np.zeros((1, camera_height, camera_width))
    return obj_mask


def get_geom_pose_from_seg(env, geom, camera_names, camera_width, camera_height, sim):
    pointclouds = []
    for camera_name in camera_names:
        if env.use_sam_segmentation:
            obj_mask = get_sam_segmentation(
                env,
                camera_name,
                camera_width,
                camera_height,
                geom,
            )
            obj_mask = np.flipud(np.transpose(obj_mask, (1, 2, 0)))
        else:
            segmentation_map = CU.get_camera_segmentation(
                camera_name=camera_name,
                camera_width=camera_width,
                camera_height=camera_height,
                sim=sim,
            )
            obj_mask = segmentation_map == geom
        depth_map = get_camera_depth(
            camera_name=camera_name,
            camera_width=camera_width,
            camera_height=camera_height,
            sim=sim,
        )
        depth_map = np.expand_dims(
            CU.get_real_depth_map(sim=env.sim, depth_map=depth_map), -1
        )
        world_to_camera = CU.get_camera_transform_matrix(
            camera_name=camera_name,
            camera_width=camera_width,
            camera_height=camera_height,
            sim=sim,
        )
        camera_to_world = np.linalg.inv(world_to_camera)
        obj_pointcloud = CU.transform_from_pixels_to_world(
            pixels=np.argwhere(obj_mask),
            depth_map=depth_map[..., 0],
            camera_to_world_transform=camera_to_world,
        )
        pointclouds.append(obj_pointcloud)
    print(f"Position: {np.mean(np.concatenate(pointclouds, axis=0), axis = 0)}")
    return np.mean(np.concatenate(pointclouds, axis=0), axis=0)


def compute_object_pcd(
    env,
    camera_height=480,
    camera_width=640,
    grasp_pose=True,
    target_obj=False,
    obj_idx=0,
):
    name = env.env_name
    object_pts = []
    if target_obj:
        camera_names = ["agentview", "birdview"] # "frontview"]
        # need birdview to properly estimate bin position
    else:
        camera_names = ["agentview", "sideview"]
    for camera_name in camera_names:
        sim = env.sim
        segmentation_map = CU.get_camera_segmentation(
            camera_name=camera_name,
            camera_width=camera_width,
            camera_height=camera_height,
            sim=sim,
        )
        # obj_mask = get_sam_segmentation(
        #     env,
        #     camera_name,
        #     camera_width,
        #     camera_height,
        #     obj_idx,
        # )
        depth_map = get_camera_depth(
            sim=sim,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        depth_map = np.expand_dims(
            CU.get_real_depth_map(sim=env.sim, depth_map=depth_map), -1
        )

        # get camera matrices
        world_to_camera = CU.get_camera_transform_matrix(
            sim=env.sim,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        camera_to_world = np.linalg.inv(world_to_camera)

        # get robot segmentation mask
        geom_ids = np.unique(segmentation_map[:, :, 1])
        object_ids = []
        object_string = env.get_object_string(obj_idx=obj_idx)
        if grasp_pose or not target_obj:
            object_string = env.get_object_string(obj_idx=obj_idx)
            if "Door" in name:
                object_string = "handle"
        else:
            if "NutAssembly" in name:
                if name.endswith("Square"):
                    object_string = "peg1"
                elif name.endswith("Round"):
                    object_string = "peg2"
                else:
                    if obj_idx == 0:
                        object_string = "peg2"
                    else:
                        object_string = "peg1"
            if "PickPlace" in name:
                object_string = "full_bin"
        for i, geom_id in enumerate(geom_ids):
            geom_name = sim.model.geom_id2name(geom_id)
            if geom_name is None or geom_name.startswith("Visual"):
                continue
            if object_string in geom_name:
                if "NutAssembly" in name and grasp_pose:
                    if name.endswith("Square"):
                        target_geom_id = "g4_visual"
                    elif name.endswith("Round"):
                        target_geom_id = "g8_visual"
                    elif name.endswith("NutAssembly"):
                        if obj_idx == 0:
                            target_geom_id = "g8_visual"
                        else:
                            target_geom_id = "g4_visual"
                    if geom_name.endswith(target_geom_id):
                        object_ids.append(geom_id)
                else:
                    object_ids.append(geom_id)
        if len(object_ids) > 0:
            if target_obj and "PickPlace" in name:
                full_bin_mask = segmentation_map[:, :, 1] == object_ids[0]
                clust_img, clust_masks = pcv.spatial_clustering(
                    full_bin_mask.astype(np.uint8) * 255,
                    algorithm="DBSCAN",
                    min_cluster_size=5,
                    max_distance=None,
                )
                new_obj_idx = env.compute_correct_obj_idx(obj_idx=obj_idx)
                clust_masks = [clust_masks[i] for i in [0, 2, 1, 3]]
                object_mask = clust_masks[new_obj_idx]
            else:
                object_mask = np.any(
                    [
                        segmentation_map[:, :, 1] == object_id
                        for object_id in object_ids
                    ],
                    axis=0,
                )
            if env.use_sam_segmentation:
                object_mask = get_sam_segmentation(
                    env=env,
                    camera_name=camera_name,
                    camera_width=camera_width,
                    camera_height=camera_height,
                    geom=object_ids[0],  # doesn't matter in this case
                )
                print(object_mask.shape)
            object_pixels = np.argwhere(object_mask)
            # get object mask from sam if using sam segmentation
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            object_pts.append(object_pointcloud)

    # if object_pts is empty, return the value from the last time this function was called with the same args
    # this is a bit of a hack, but necessary since the object may be occluded sometimes
    if len(object_pts) > 0:
        env.cache[
            (camera_height, camera_width, grasp_pose, target_obj, obj_idx)
        ] = object_pts
    else:
        object_pts = env.cache[
            (camera_height, camera_width, grasp_pose, target_obj, obj_idx)
        ]
    object_pointcloud = np.concatenate(object_pts, axis=0)
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_pointcloud)
    cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    object_pcd = object_pcd.select_by_index(ind)
    object_xyz = np.array(object_pcd.points)
    return object_xyz


def get_object_pose_from_seg(
    env, object_string, camera_name, camera_width, camera_height, sim
):
    if not env.use_sam_segmentation:
        segmentation_map = CU.get_camera_segmentation(
            camera_name=camera_name,
            camera_width=camera_width,
            camera_height=camera_height,
            sim=sim,
        )
        obj_id = sim.model.geom_name2id(object_string)
        obj_mask = segmentation_map == obj_id
    else:
        obj_mask = get_sam_segmentation(
            env, camera_name, camera_width, camera_height, sim
        )
    depth_map = get_camera_depth(
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        sim=sim,
    )
    depth_map = np.expand_dims(
        CU.get_real_depth_map(sim=env.sim, depth_map=depth_map), -1
    )
    world_to_camera = CU.get_camera_transform_matrix(
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        sim=sim,
    )
    camera_to_world = np.linalg.inv(world_to_camera)
    obj_pointcloud = CU.transform_from_pixels_to_world(
        pixels=np.argwhere(obj_mask),
        depth_map=depth_map[..., 0],
        camera_to_world_transform=camera_to_world,
    )
    return np.mean(obj_pointcloud, axis=0)


def compute_pcd(
    env,
    obj_idx=0,
    camera_height=480,
    camera_width=640,
    is_grasped=False,
):
    pts = []
    object_pts = []
    camera_names = ["agentview", "birdview", "frontview"]
    for camera_name in camera_names:
        sim = env.sim
        segmentation_map = CU.get_camera_segmentation(
            camera_name=camera_name,
            camera_width=camera_width,
            camera_height=camera_height,
            sim=sim,
        )
        depth_map = get_camera_depth(
            sim=sim,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        depth_map = np.expand_dims(
            CU.get_real_depth_map(sim=env.sim, depth_map=depth_map), -1
        )

        # get camera matrices
        world_to_camera = CU.get_camera_transform_matrix(
            sim=env.sim,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        camera_to_world = np.linalg.inv(world_to_camera)

        # get robot segmentation mask
        geom_ids = np.unique(segmentation_map[:, :, 1])
        robot_ids = []
        object_ids = []
        object_string = env.get_object_string(obj_idx=obj_idx)
        for geom_id in geom_ids:
            geom_name = sim.model.geom_id2name(geom_id)
            if geom_name is None or geom_name.startswith("Visual"):
                continue
            if geom_name.startswith("robot0") or geom_name.startswith("gripper"):
                robot_ids.append(geom_id)
            if object_string in geom_name:
                object_ids.append(geom_id)
        robot_mask = np.any(
            [segmentation_map[:, :, 1] == robot_id for robot_id in robot_ids], axis=0
        )
        if is_grasped and len(object_ids) > 0:
            object_mask = np.any(
                [segmentation_map[:, :, 1] == object_id for object_id in object_ids],
                axis=0,
            )
            # only remove object from scene if it is grasped
            all_img_pixels = np.argwhere(
                1 - robot_mask - object_mask
            )  # remove robot from scene pcd
            object_pixels = np.argwhere(object_mask)
            object_pointcloud = CU.transform_from_pixels_to_world(
                pixels=object_pixels,
                depth_map=depth_map[..., 0],
                camera_to_world_transform=camera_to_world,
            )
            object_pts.append(object_pointcloud)
        else:
            all_img_pixels = np.argwhere(1 - robot_mask)
        # transform from camera pixel back to world position
        pointcloud = CU.transform_from_pixels_to_world(
            pixels=all_img_pixels,
            depth_map=depth_map[..., 0],
            camera_to_world_transform=camera_to_world,
        )
        pts.append(pointcloud)

    pointcloud = np.concatenate(pts, axis=0)
    pointcloud = pointcloud[pointcloud[:, -1] > 0.75]
    pointcloud = pointcloud[pointcloud[:, -1] < 1.0]
    pointcloud = pointcloud[pointcloud[:, 0] > -0.3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    xyz = np.array(pcd.points)
    if is_grasped and len(object_pts) > 0:
        object_pointcloud = np.concatenate(object_pts, axis=0)
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pointcloud)
        cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        object_pcd = object_pcd.select_by_index(ind)
        object_xyz = np.array(object_pcd.points)
    else:
        object_xyz = None

    return xyz, object_xyz