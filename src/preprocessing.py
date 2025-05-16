import open3d as o3d
import numpy as np

print("DEBUG: preprocessing.py top level executed")

def preprocess_fragment(mesh, params):
    """
    Preprocesses a single fragment mesh.
    - Voxel downsamples the mesh to create a point cloud.
    - Removes statistical outliers from the point cloud.
    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        params (dict): Dictionary of parameters from config.
    Returns:
        o3d.geometry.PointCloud: Preprocessed point cloud.
    """
    if not mesh.has_vertices():
        print("Warning: Mesh has no vertices, cannot preprocess.")
        return o3d.geometry.PointCloud()

    # 1. Convert mesh to point cloud (e.g., by sampling or using vertices)
    # For simplicity, we'll use vertex positions, but sampling might be better for dense meshes.
    # If the mesh is very dense, voxel downsampling on the mesh first, then converting to PCD is also an option.
    
    # pcd = mesh.sample_points_poisson_disk(number_of_points=5000) # Alternative
    # Or, more simply if we want to preserve some original structure from vertices:
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices # Use mesh vertices directly

    # 2. Voxel downsampling of the point cloud
    voxel_size = params.get("voxel_downsample_size", 0.01)
    if voxel_size > 0 and len(pcd.points) > 0:
        # print(f"Before downsampling: {len(pcd.points)} points")
        pcd = pcd.voxel_down_sample(voxel_size)
        # print(f"After downsampling: {len(pcd.points)} points")
    
    # 3. (Optional) Statistical outlier removal
    # This can help clean up noisy scans but might remove valid sparse geometry.
    # Use with caution.
    # if len(pcd.points) > 0:
    #     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #     pcd = pcd.select_by_index(ind)
    #     print(f"After outlier removal: {len(pcd.points)} points")

    # 4. Estimate normals (important for FPFH and ICP)
    if len(pcd.points) > 0 :
        radius_normal = params.get("normal_estimation_radius", voxel_size * 2)
        max_nn_normal = params.get("normal_estimation_max_nn", 30)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normal))
        pcd.orient_normals_consistent_tangent_plane(k=15) # Make normals consistent

    return pcd


if __name__ == '__main__':
    from io_utils import load_fragment
    import json

    # Create a dummy config for testing
    dummy_params = {
        "voxel_downsample_size": 0.05,
        "normal_estimation_radius": 0.1,
        "normal_estimation_max_nn": 30
    }
    
    # Ensure dummy_data/input_fragments/cube1.obj exists from io_utils test
    cube_mesh = load_fragment('../../dummy_data/input_fragments/cube1.obj') # Assuming called from src
    if not cube_mesh:
        # If io_utils.py was run, it created dummy_data at project root.
        # If this script is run from src/, path should be ../dummy_data
        cube_obj_content = """
                            v 0.0 0.0 0.0
                            v 1.0 0.0 0.0
                            v 1.0 1.0 0.0
                            v 0.0 1.0 0.0
                            v 0.0 0.0 1.0
                            v 1.0 0.0 1.0
                            v 1.0 1.0 1.0
                            v 0.0 1.0 1.0
                            f 1 2 3 4
                            f 5 6 7 8
                            f 1 2 6 5
                            f 2 3 7 6
                            f 3 4 8 7
                            f 4 1 5 8
                            """
        import os
        if not os.path.exists('../dummy_data/input_fragments'):
            os.makedirs('../dummy_data/input_fragments')
        with open('../dummy_data/input_fragments/cube1.obj', 'w') as f:
            f.write(cube_obj_content)
        cube_mesh = load_fragment('../dummy_data/input_fragments/cube1.obj')


    if cube_mesh:
        print("Original cube mesh vertices:", len(cube_mesh.vertices))
        processed_pcd = preprocess_fragment(cube_mesh, dummy_params)
        print("Processed PCD points:", len(processed_pcd.points))
        if len(processed_pcd.points) > 0:
            print("Processed PCD has normals:", processed_pcd.has_normals())
            # o3d.visualization.draw_geometries([processed_pcd])
        else:
            print("Processed PCD is empty.")
    else:
        print("Failed to load cube mesh for preprocessing test.")