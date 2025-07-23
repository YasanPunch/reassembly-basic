import open3d as o3d
import numpy as np
import copy
from src.segmentation import extract_fracture_surface_mesh
from src.feature_extraction import extract_features_from_pcd
from src.utils.visualization_utils import debug_visualize_voxel_downsampling

print("\nDEBUG: preprocessing.py top level executed")

np.random.seed(42)


def preprocess_fragment(fragment_info, params):
    """
    Preprocesses a single fragment:
    1. Identifies fracture surfaces using normal-based segmentation.
    2. Samples points densely from these fracture surfaces.
    3. Downsamples this point cloud.
    4. Estimates normals.

    Args:
        fragment_info (dict): Dict containing 'mesh' (original o3d.geometry.TriangleMesh)
                              and 'name'.
        params (dict): Dictionary of parameters from config.

    Returns:
        tuple: (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh or None)
               - Preprocessed point cloud (from fracture surface, downsampled, with normals).
               - The extracted fracture surface mesh itself (for visualization/debug).
               Returns (None, None) if processing fails.
    """
    original_mesh = fragment_info["mesh"]  # Get mesh from fragment_info
    fragment_name = fragment_info["name"]  # Get name from fragment_info

    if not original_mesh.has_vertices():
        print(f"    Preprocessing: Original mesh {fragment_name} has no vertices.")
        return [], [], []

    # --- Step 1: Identify and Extract Fracture Surface Meshes (now a list) ---
    print(f"    Preprocessing: Segmenting fracture surfaces for {fragment_name}...")
    fracture_surfaces = extract_fracture_surface_mesh(
        original_mesh, fragment_name, params
    )
    if not isinstance(fracture_surfaces, list):
        fracture_surfaces = [fracture_surfaces] if fracture_surfaces is not None else []

    # --- Step 2: For each surface, sample points, extract features, and store as lists ---
    features_list = []
    pcds_for_features_list = []
    for surf in fracture_surfaces:
        if surf is None or not surf.has_triangles():
            continue
        num_dense_sample_points = params.get(
            "fracture_surface_dense_sample_points", 5000
        )
        if len(surf.vertices) < 3:
            continue
        pcd = surf.sample_points_poisson_disk(number_of_points=num_dense_sample_points)
        if not pcd.has_points():
            continue
        voxel_size = params.get("voxel_downsample_size", 0.01)
        if voxel_size > 0 and len(pcd.points) > 0:
            pcd_downsampled = pcd.voxel_down_sample(voxel_size)
            print(
                f"      Surface {len(features_list)+1}: Downsampled from {len(pcd.points)} to {len(pcd_downsampled.points)} points."
            )
            pcd = pcd_downsampled

            # DEBUG: VISUALIZE VOXEL DOWNSAMPLED POINT CLOUD FOR THIS SURFACE
            if params.get("debug_voxel_downsampling", False):
                debug_visualize_voxel_downsampling(
                    original_mesh, surf, pcd, fragment_name, len(features_list) + 1
                )

        if not pcd.has_points():
            continue

        # Estimate normals
        radius_normal = params.get(
            "normal_estimation_radius",
            voxel_size * params.get("normal_radius_factor", 2.0),
        )
        max_nn_normal = params.get("normal_estimation_max_nn", 30)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=max_nn_normal
            )
        )
        try:
            pcd.orient_normals_consistent_tangent_plane(
                k=params.get("orient_normals_k", 15)
            )
        except RuntimeError as e:
            print(
                f"    Warning: orient_normals_consistent_tangent_plane failed for {fragment_name}: {e}"
            )

        # Feature extraction
        features, _ = extract_features_from_pcd(pcd, params)
        if features is not None and features.num() > 0:
            features_list.append(features)
            pcds_for_features_list.append(pcd)

    # Return lists for downstream processing
    return pcds_for_features_list, features_list, fracture_surfaces
