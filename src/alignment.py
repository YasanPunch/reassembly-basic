import open3d as o3d
import numpy as np

print("DEBUG: alignment.py top level executed")

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

def align_fragments_pcd(source_pcd, target_pcd, source_fpfh, target_fpfh, params):
    """
    Aligns two point clouds (fragments) using global RANSAC + local ICP.
    Args:
        source_pcd, target_pcd: o3d.geometry.PointCloud
        source_fpfh, target_fpfh: o3d.pipelines.registration.Feature
        params: dict of config parameters
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
    # print("Performing global registration (RANSAC)...")
    result_ransac = execute_global_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, params)
    # print(f"RANSAC Fitness: {result_ransac.fitness:.4f}, RMSE: {result_ransac.inlier_rmse:.4f}")

    if result_ransac.fitness < 0.1 and result_ransac.inlier_rmse > params["voxel_downsample_size"] * 5: # Heuristic
        # print("RANSAC alignment poor, skipping ICP.")
        return None, result_ransac.fitness, result_ransac.inlier_rmse


    # 2. Local refinement (ICP)
    # print("Performing local refinement (ICP)...")
    result_icp = refine_registration_icp(source_pcd, target_pcd, result_ransac.transformation, params)
    # print(f"ICP Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.4f}")

    # Determine if the alignment is good enough
    min_fitness = params.get("min_match_score", 0.7) # This score is crucial
    if result_icp.fitness > min_fitness and result_icp.inlier_rmse < params["voxel_downsample_size"] * 2.0: # Heuristic
        return result_icp.transformation, result_icp.fitness, result_icp.inlier_rmse
    else:
        # print(f"ICP alignment insufficient (Fitness: {result_icp.fitness:.3f}, RMSE: {result_icp.inlier_rmse:.3f})")
        return None, result_icp.fitness, result_icp.inlier_rmse


if __name__ == '__main__':
    from io_utils import load_fragment
    from preprocessing import preprocess_fragment
    from feature_extraction import extract_features_from_pcd
    import json
    import copy

    # Create a dummy config for testing
    dummy_params = {
        "voxel_downsample_size": 0.02, # Smaller for better chance with simple shapes
        "normal_estimation_radius": 0.04,
        "normal_estimation_max_nn": 30,
        "fpfh_feature_radius": 0.1,
        "fpfh_feature_max_nn": 100,
        "ransac_distance_threshold_factor": 1.5,
        "ransac_edge_length_factor": 0.9,
        "ransac_iterations": 50000, # Low for quick test, real values much higher
        "ransac_n_points": 3,
        "ransac_confidence": 0.99,
        "icp_max_correspondence_distance_factor": 2.0,
        "icp_relative_fitness": 1e-6,
        "icp_relative_rmse": 1e-6,
        "icp_max_iteration": 30,
        "min_match_score": 0.5 # Lower for testing
    }

    # Load two slightly different/transformed versions of a mesh
    # Assuming cube1.obj exists from io_utils test in ../dummy_data
    frag1_mesh = load_fragment('../dummy_data/input_fragments/cube1.obj')
    frag2_mesh = load_fragment('../dummy_data/input_fragments/cube1.obj') # Load same for test

    if frag1_mesh and frag2_mesh:
        # Apply a known transformation to frag2 to test alignment
        # Create a slight rotation and translation
        # angle = np.pi / 10  # 18 degrees
        # R = np.array([[np.cos(angle), -np.sin(angle), 0],
        #               [np.sin(angle), np.cos(angle), 0],
        #               [0, 0, 1]])
        # T_manual = np.eye(4)
        # T_manual[:3, :3] = R
        # T_manual[:3, 3] = [0.1, 0.05, 0.0] # Small translation
        # frag2_mesh.transform(T_manual) # This transforms the mesh itself

        # For a better test, let's take two parts of a more complex object
        # For now, we'll use the same object slightly transformed
        # (This is a "perfect match" scenario, easier than real fragments)
        transform = np.asarray([[0.862, 0.011, -0.507, 0.5],
                                [-0.139, 0.967, -0.215, 0.7],
                                [0.487, 0.255, 0.835, -0.4],
                                [0.0, 0.0, 0.0, 1.0]])
        frag2_mesh_transformed = copy.deepcopy(frag1_mesh) # Use original as target
        frag1_mesh_source = copy.deepcopy(frag1_mesh) # Use transformed as source
        frag1_mesh_source.transform(transform)


        pcd1 = preprocess_fragment(frag1_mesh_source, dummy_params)
        pcd2 = preprocess_fragment(frag2_mesh_transformed, dummy_params) # Target (original)

        if pcd1.has_points() and pcd2.has_points():
            fpfh1, pcd1_feat = extract_features_from_pcd(pcd1, dummy_params)
            fpfh2, pcd2_feat = extract_features_from_pcd(pcd2, dummy_params)

            if fpfh1 and fpfh2:
                print("Attempting alignment...")
                # Note: pcd1_feat and pcd2_feat are the point clouds FPFH were computed on.
                # These are what should be passed to align_fragments_pcd.
                transformation, fitness, rmse = align_fragments_pcd(pcd1_feat, pcd2_feat, fpfh1, fpfh2, dummy_params)

                if transformation is not None:
                    print("Alignment successful!")
                    print("Transformation matrix:\n", transformation)
                    print(f"Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")

                    # Visualize
                    # Before alignment:
                    # pcd1_feat.paint_uniform_color([1,0,0]) # Source red
                    # pcd2_feat.paint_uniform_color([0,1,0]) # Target green
                    # o3d.visualization.draw_geometries([pcd1_feat, pcd2_feat], window_name="Before Alignment")
                    
                    # After alignment:
                    pcd1_aligned = copy.deepcopy(pcd1_feat)
                    pcd1_aligned.transform(transformation)
                    # pcd1_aligned.paint_uniform_color([1,0,0])
                    # pcd2_feat.paint_uniform_color([0,1,0])
                    # o3d.visualization.draw_geometries([pcd1_aligned, pcd2_feat], window_name="After Alignment")
                else:
                    print("Alignment failed or quality too low.")
            else:
                print("FPFH feature extraction failed for one or both fragments.")
        else:
            print("Preprocessing failed for one or both fragments.")
    else:
        print("Failed to load fragments for alignment test.")