import open3d as o3d
import numpy as np
import copy
import random

print("DEBUG: alignment.py top level executed")

# Set random seed for deterministic RANSAC behavior
random.seed(42)
np.random.seed(42)

def execute_global_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, params):
    """
    Performs global registration using RANSAC on FPFH features.
    Args:
        source_pcd (o3d.geometry.PointCloud): Source point cloud.
        target_pcd (o3d.geometry.PointCloud): Target point cloud.
        source_fpfh (o3d.pipelines.registration.Feature): FPFH features of source.
        target_fpfh (o3d.pipelines.registration.Feature): FPFH features of target.
        params (dict): Configuration parameters.
    Returns:
        o3d.pipelines.registration.RegistrationResult: RANSAC registration result.
    """
    voxel_size = params["voxel_downsample_size"]
    distance_threshold = voxel_size * params.get("ransac_distance_threshold_factor", 1.5)
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        params.get("ransac_n_points", 4),  # RANSAC n points
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(params.get("ransac_edge_length_factor",0.9)),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(params.get("ransac_iterations", 100000), 
                                                              params.get("ransac_confidence", 0.999))
    )
    return result

def refine_registration_icp(source_pcd, target_pcd, initial_transform, params):
    """
    Refines registration using Iterative Closest Point (ICP).
    Args:
        source_pcd (o3d.geometry.PointCloud): Source point cloud.
        target_pcd (o3d.geometry.PointCloud): Target point cloud.
        initial_transform (np.ndarray): Initial 4x4 transformation guess.
        params (dict): Configuration parameters.
    Returns:
        o3d.pipelines.registration.RegistrationResult: ICP registration result.
    """
    voxel_size = params["voxel_downsample_size"]
    distance_threshold_icp = voxel_size * params.get("icp_max_correspondence_distance_factor", 2.0)
    
    # Ensure point clouds have normals for point-to-plane ICP
    if not source_pcd.has_normals():
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=params["normal_estimation_radius"], max_nn=params["normal_estimation_max_nn"]))
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=params["normal_estimation_radius"], max_nn=params["normal_estimation_max_nn"]))

    result_icp = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, distance_threshold_icp, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), # Point-to-plane is often better
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=params.get("icp_relative_fitness", 1e-7),
            relative_rmse=params.get("icp_relative_rmse", 1e-7),
            max_iteration=params.get("icp_max_iteration", 50)
        )
    )
    return result_icp

def align_fragments_pcd(source_pcd, target_pcd, source_fpfh, target_fpfh, params, debug=False, source_fragment=None, target_fragment=None):
    """
    Aligns two point clouds (fragments) using global RANSAC + local ICP.
    Args:
        source_pcd, target_pcd: o3d.geometry.PointCloud
        source_fpfh, target_fpfh: o3d.pipelines.registration.Feature
        params: dict of config parameters
        debug: bool, whether to visualize before alignment
        source_fragment, target_fragment: dict, optional fragment data for visualization
    Returns:
        tuple: (transformation_matrix, fitness_score, inlier_rmse)
               Returns (None, 0, 0) if alignment fails or is poor.
    """
    if not source_pcd.has_points() or not target_pcd.has_points():
        print("Error: One or both point clouds are empty for alignment.")
        return None, 0.0, 0.0
    if source_fpfh is None or target_fpfh is None:
        print("Error: FPFH features are missing for alignment.")
        return None, 0.0, 0.0
    if source_fpfh.num() == 0 or target_fpfh.num() == 0:
        print("Error: FPFH features are empty.")
        return None, 0.0, 0.0

    # 1. Global registration (RANSAC on FPFH)
    result_ransac = execute_global_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, params)
    # DEBUG: VISUALIZE AFTER RANSAC
    if debug and source_fragment is not None and target_fragment is not None:
        tgt_mesh = copy.deepcopy(target_fragment['original_mesh'])
        tgt_mesh.paint_uniform_color([0.7,0.7,0.7])
        tgt_fract = target_fragment.get('fracture_surface_mesh')
        vis_geoms = [tgt_mesh]
        if tgt_fract is not None and tgt_fract.has_triangles():
            tgt_fract_vis = copy.deepcopy(tgt_fract)
            tgt_fract_vis.paint_uniform_color([0,1,0])
            vis_geoms.append(tgt_fract_vis)
        # Only show the transformed source mesh and its fracture surface
        src_mesh_ransac = copy.deepcopy(source_fragment['original_mesh'])
        src_mesh_ransac.paint_uniform_color([0.7,0.7,0.7])
        src_mesh_ransac.transform(result_ransac.transformation)
        vis_geoms.append(src_mesh_ransac)
        src_fract = source_fragment.get('fracture_surface_mesh')
        if src_fract is not None and src_fract.has_triangles():
            src_fract_vis = copy.deepcopy(src_fract)
            src_fract_vis.paint_uniform_color([1,0,0])
            src_fract_vis.transform(result_ransac.transformation)
            vis_geoms.append(src_fract_vis)
        o3d.visualization.draw_geometries(vis_geoms, window_name=f"[DEBUG] After RANSAC Alignment (Gray=Full Mesh, Red=Source Fracture, Green=Target Fracture)")
    # RANSAC heuristic
    if result_ransac.fitness < 0.1 and result_ransac.inlier_rmse > params["voxel_downsample_size"] * 5:
        return None, result_ransac.fitness, result_ransac.inlier_rmse
    # 2. Local refinement (ICP)
    result_icp = refine_registration_icp(source_pcd, target_pcd, result_ransac.transformation, params)
    # DEBUG: VISUALIZE AFTER ICP
    if debug and source_fragment is not None and target_fragment is not None:
        tgt_mesh = copy.deepcopy(target_fragment['original_mesh'])
        tgt_mesh.paint_uniform_color([0.7,0.7,0.7])
        tgt_fract = target_fragment.get('fracture_surface_mesh')
        vis_geoms = [tgt_mesh]
        if tgt_fract is not None and tgt_fract.has_triangles():
            tgt_fract_vis = copy.deepcopy(tgt_fract)
            tgt_fract_vis.paint_uniform_color([0,1,0])
            vis_geoms.append(tgt_fract_vis)
        # Only show the transformed source mesh and its fracture surface
        src_mesh_icp = copy.deepcopy(source_fragment['original_mesh'])
        src_mesh_icp.paint_uniform_color([0.7,0.7,0.7])
        src_mesh_icp.transform(result_icp.transformation)
        vis_geoms.append(src_mesh_icp)
        src_fract = source_fragment.get('fracture_surface_mesh')
        if src_fract is not None and src_fract.has_triangles():
            src_fract_vis = copy.deepcopy(src_fract)
            src_fract_vis.paint_uniform_color([1,0,0])
            src_fract_vis.transform(result_icp.transformation)
            vis_geoms.append(src_fract_vis)
        o3d.visualization.draw_geometries(vis_geoms, window_name=f"[DEBUG] After ICP Alignment (Gray=Full Mesh, Red=Source Fracture, Green=Target Fracture)")
    min_fitness = params.get("min_match_score", 0.7)
    if result_icp.fitness > min_fitness and result_icp.inlier_rmse < params["voxel_downsample_size"] * 2.0:
        return result_icp.transformation, result_icp.fitness, result_icp.inlier_rmse
    else:
        return None, result_icp.fitness, result_icp.inlier_rmse
