import numpy as np
import robosuite.utils.camera_utils as CU
from planseqlearn.psl.sam_utils import get_seg_mask
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
    return np.mean(np.concatenate(pointclouds, axis=0), axis=0)

def reset_precompute_sam_poses(env):
    env.sam_object_pose = {}
    for obj_name, action in env.text_plan:
        sam_kwargs = env.get_sam_kwargs(obj_name)
        frame = env.sim.render(camera_name=sam_kwargs["camera_name"], width=500, height=500)
        if sam_kwargs.get("flip_channel", True):
            frame = frame[:, :, ::-1]
        if sam_kwargs.get("flip_image", True):
            frame = np.flipud(frame)
        obj_masks, _, _, pred_phrases, _ = get_seg_mask(
            frame,
            env.dino,
            env.sam,
            text_prompts=sam_kwargs["text_prompts"],
            box_threshold=sam_kwargs["box_threshold"],
            text_threshold=0.25,
            device="cuda",
            debug=True,
            output_dir="sam_outputs",
        )
        object_mask = obj_masks[sam_kwargs["idx"]].cpu().detach().numpy()[0, :, :]
        depth_map = get_camera_depth(
            camera_name=sam_kwargs["camera_name"],
            camera_width=500,
            camera_height=500,
            sim=env.sim,
        )
        depth_map = np.expand_dims(
            CU.get_real_depth_map(sim=env.sim, depth_map=depth_map), -1
        )
        if sam_kwargs.get("flip_dm", True):
            depth_map = np.flipud(depth_map)
        world_to_camera = CU.get_camera_transform_matrix(
            sim=env.sim,
            camera_name=sam_kwargs["camera_name"],
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
        env.sam_object_pose[obj_name] = np.mean(object_pointcloud, axis=0) + sam_kwargs["offset"]


def compute_object_pcd(
    env,
    camera_height=480,
    camera_width=640,
    grasp_pose=True,
    obj_name="",
):
    name = env.env_name
    object_pts = []
    if "bin" in obj_name:
        camera_names = ["agentview", "birdview"]  # "frontview"]
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
        object_string = obj_name
        if "Door" in name:
            object_string = "handle"
        else:
            if "NutAssembly" in name and "peg" in object_string:
                if "gold" in object_string:
                    object_string = "peg1"
                elif name.endswith("Round"):
                    object_string = "peg2"
            if "PickPlace" in name and "bin" in object_string:
                object_string = "full_bin"
        for i, geom_id in enumerate(geom_ids):
            geom_name = sim.model.geom_id2name(geom_id)
            if geom_name is None or geom_name.startswith("Visual"):
                continue
            if "NutAssembly" in name and grasp_pose:
                if "gold" in obj_name:
                    target_geom_id = "SquareNut_g4_visual"
                elif "silver" in obj_name:
                    target_geom_id = "RoundNut_g8_visual"
                if geom_name.endswith(target_geom_id):
                    object_ids.append(geom_id)
            else:
                if object_string in geom_name.lower() or geom_name.split('_')[0].lower() in object_string:
                    object_ids.append(geom_id)
        if len(object_ids) > 0:
            if "bin" in obj_name and "PickPlace" in name:
                full_bin_mask = segmentation_map[:, :, 1] == object_ids[0]
                clust_img, clust_masks = pcv.spatial_clustering(
                    full_bin_mask.astype(np.uint8) * 255,
                    algorithm="DBSCAN",
                    min_cluster_size=5,
                    max_distance=None,
                )
                new_obj_idx = int(obj_name[-1])
                clust_masks = [clust_masks[i] for i in range(4)] #[0, 2, 1, 3]]
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
            (camera_height, camera_width, grasp_pose, obj_name)
        ] = object_pts
    else:
        object_pts = env.cache[
            (camera_height, camera_width, grasp_pose, obj_name)
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
