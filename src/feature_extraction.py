import open3d as o3d

print("DEBUG: feature_extraction.py top level executed") 


def extract_features_from_pcd(pcd_from_fracture_surface, params):
    """
    Extracts FPFH features from a preprocessed point cloud,
    which is assumed to be derived from segmented fracture surfaces.

    Args:
        pcd_from_fracture_surface (o3d.geometry.PointCloud): Input point cloud,
                                   assumed to be downsampled and have normals.
        params (dict): Dictionary of parameters from config.
    Returns:
        tuple: (o3d.pipelines.registration.Feature, o3d.geometry.PointCloud)
               FPFH features and the point cloud they correspond to.
    """
    if pcd_from_fracture_surface is None or not pcd_from_fracture_surface.has_points():
        print("    FeatureExtraction: Input PCD is None or empty. Cannot extract features.")
        return None, None

    if not pcd_from_fracture_surface.has_normals():
        # This should ideally be handled in preprocessing, but as a fallback:
        print("    FeatureExtraction: Warning - Input PCD has no normals. Estimating now.")
        voxel_size = params["voxel_downsample_size"] # Get for radii calculation
        radius_normal = params.get("normal_estimation_radius", voxel_size * 2.0) # Use absolute or factor
        max_nn_normal = params.get("normal_estimation_max_nn", 30)
        pcd_from_fracture_surface.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normal))
        # Orientation should also ideally be done in preprocessing
        try:
            pcd_from_fracture_surface.orient_normals_consistent_tangent_plane(k=params.get("orient_normals_k", 15))
        except RuntimeError:
            pass # Ignore if orientation fails here, normals are still there

    # FPFH computation parameters
    voxel_size = params["voxel_downsample_size"] # Basis for FPFH radius if using factor
    # Use absolute radius from config, or calculate if factor is preferred
    radius_feature = params.get("fpfh_feature_radius", voxel_size * 5.0) 
    max_nn_feature = params.get("fpfh_feature_max_nn", 100)

    print(f"    FeatureExtraction: Computing FPFH with radius_feature={radius_feature:.3f} on {len(pcd_from_fracture_surface.points)} points.")

    # The compute_fpfh_features in geometry_utils already handles KDTreeSearchParamHybrid
    # but it uses its own internal normal estimation. We want to use the normals we already computed.
    # So, we call o3d.pipelines.registration.compute_fpfh_feature directly.

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_from_fracture_surface,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_feature))

    if fpfh is None or fpfh.num() == 0:
        print("    FeatureExtraction: FPFH computation failed or yielded empty features.")
        return None, pcd_from_fracture_surface # Return the PCD even if features are None

    print(f"    FeatureExtraction: Extracted {fpfh.num()} FPFH features (dim: {fpfh.dimension()}).")
    return fpfh, pcd_from_fracture_surface  # Return the pcd it was computed on
