import open3d as o3d
import numpy as np
from utils.geometry_utils import compute_fpfh_features  # , get_mesh_boundary_vertices

print("DEBUG: feature_extraction.py top level executed")


def extract_features_from_pcd(pcd, params):
    """
    Extracts FPFH features from a preprocessed point cloud.
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud, assumed to be preprocessed
                                       (downsampled, normals estimated).
        params (dict): Dictionary of parameters from config.
    Returns:
        tuple: (o3d.pipelines.registration.Feature, o3d.geometry.PointCloud)
               FPFH features and the point cloud they correspond to (which might be
               further processed, e.g., boundary points only).
    """
    if not pcd.has_points():
        print("Warning: Point cloud is empty, cannot extract features.")
        return None, None
    if not pcd.has_normals():
        print("Warning: Point cloud has no normals, estimating them now for FPFH.")
        radius_normal = params.get(
            "normal_estimation_radius", params["voxel_downsample_size"] * 2
        )
        max_nn_normal = params.get("normal_estimation_max_nn", 30)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=max_nn_normal
            )
        )

    # For a "basic" application, we'll compute FPFH features on the entire downsampled point cloud.
    # A more advanced approach might try to identify "break surfaces" or use `get_mesh_boundary_vertices`
    # on the original mesh, then sample points on those surfaces and compute features.
    # However, `get_mesh_boundary_vertices` can be tricky with complex/noisy meshes.

    # Example for using boundary points (can be unstable):
    # boundary_pcd = get_mesh_boundary_vertices(original_mesh) # original_mesh would need to be passed
    # if boundary_pcd and boundary_pcd.has_points():
    #     print(f"Using {len(boundary_pcd.points)} boundary points for feature extraction.")
    #     # Downsample boundary points if too many, estimate normals, then FPFH
    #     target_pcd_for_features = boundary_pcd.voxel_down_sample(params["voxel_downsample_size"])
    #     if not target_pcd_for_features.has_points(): target_pcd_for_features = pcd # fallback
    # else:
    #     print("No boundary points found or used, using full preprocessed PCD for features.")
    target_pcd_for_features = pcd  # Use the whole preprocessed PCD

    if not target_pcd_for_features.has_points():
        print("Target PCD for features is empty.")
        return None, None

    voxel_size = params["voxel_downsample_size"]  # Use this as a basis for radii
    radius_feature = params.get("fpfh_feature_radius", voxel_size * 5)

    fpfh = compute_fpfh_features(
        target_pcd_for_features,
        voxel_size,
        params["normal_estimation_radius"],
        radius_feature,
    )
    return fpfh, target_pcd_for_features


if __name__ == "__main__":
    from io_utils import load_fragment
    from preprocessing import preprocess_fragment
    import json

    # Create a dummy config for testing
    dummy_params = {
        "voxel_downsample_size": 0.05,
        "normal_estimation_radius": 0.1,
        "normal_estimation_max_nn": 30,
        "fpfh_feature_radius": 0.25,
        "fpfh_feature_max_nn": 100,
    }
    # Ensure dummy_data/input_fragments/cube1.obj exists
    # Adjust path if running from src directory
    cube_mesh = load_fragment("../dummy_data/input_fragments/cube1.obj")

    if cube_mesh:
        print("Original cube mesh vertices:", len(cube_mesh.vertices))
        processed_pcd = preprocess_fragment(cube_mesh, dummy_params)
        print("Processed PCD points for feature extraction:", len(processed_pcd.points))

        if processed_pcd.has_points():
            features, pcd_for_features = extract_features_from_pcd(
                processed_pcd, dummy_params
            )
            if features:
                print("FPFH features extracted.")
                print(
                    " - Number of features (should match points in pcd_for_features):",
                    features.num(),
                )
                print(
                    " - Feature dimensionality (e.g., 33 for FPFH):",
                    features.dimension(),
                )
                print(
                    " - Point cloud used for features has points:",
                    len(pcd_for_features.points),
                )
                # o3d.visualization.draw_geometries([pcd_for_features])
            else:
                print("Feature extraction failed.")
        else:
            print("Processed PCD is empty, cannot extract features.")
    else:
        print("Failed to load cube mesh for feature extraction test.")

