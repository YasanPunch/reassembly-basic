import open3d as o3d
import numpy as np
import copy
from src.segmentation import extract_fracture_surface_mesh

print("DEBUG: preprocessing.py top level executed")

def preprocess_fragment(fragment_info, params, viz_collector=None): 
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
        viz_collector (list, optional): List to append visualization data to.

    Returns:
        tuple: (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh or None)
               - Preprocessed point cloud (from fracture surface, downsampled, with normals).
               - The extracted fracture surface mesh itself (for visualization/debug).
               Returns (None, None) if processing fails.
    """
    original_mesh = fragment_info['mesh'] # Get mesh from fragment_info
    fragment_name = fragment_info['name'] # Get name from fragment_info
    original_index = fragment_info['original_index'] # Get for logging

    if not original_mesh.has_vertices():
        print(f"    Preprocessing: Original mesh {fragment_name} has no vertices.")
        if viz_collector is not None:
            viz_collector.append({
                'step': 'preprocessing_failed_no_vertices', 
                'name': fragment_name
            })
        return None, None

    # --- Step 1: Identify and Extract Fracture Surface Mesh ---
    print(f"    Preprocessing: Segmenting fracture surface for {fragment_name}...")
    fracture_surface_mesh_o3d = extract_fracture_surface_mesh(original_mesh, fragment_name, params)

    # Visualization parameters for interactive verification
    if params.get('visualize_segmentation', False):
        from src.segmentation import visualize_segmentation
        vis_geometries = visualize_segmentation(original_mesh, fracture_surface_mesh_o3d, fragment_name)
        o3d.visualization.draw_geometries(vis_geometries, 
                                         window_name=f"Segmentation Result: {fragment_name}")

    if viz_collector is not None:
        log_entry = {
            'step': 'segmentation_result', 
            'name': fragment_name,
            'original_index': original_index,
            'original_mesh_type': 'mesh',
            'original_mesh_vertices': np.asarray(original_mesh.vertices),
            'original_mesh_triangles': np.asarray(original_mesh.triangles),
        }
        if fracture_surface_mesh_o3d and fracture_surface_mesh_o3d.has_triangles():
            log_entry.update({
                'fracture_mesh_type': 'mesh',
                'fracture_mesh_vertices': np.asarray(fracture_surface_mesh_o3d.vertices),
                'fracture_mesh_triangles': np.asarray(fracture_surface_mesh_o3d.triangles),
            })
        else:
            log_entry['fracture_mesh_type'] = None # Indicate no fracture mesh found
        viz_collector.append(log_entry)

    # ... (rest of the logic: pcd_target_sampling, dense sampling) ...
    if fracture_surface_mesh_o3d is None or not fracture_surface_mesh_o3d.has_triangles():
        print(f"    Preprocessing: No usable fracture surface found for {fragment_name}. Using whole mesh as fallback.")
        pcd_target_sampling = original_mesh
    else:
        pcd_target_sampling = fracture_surface_mesh_o3d
    
    # --- Step 2: Sample points DENSELY from the target surface ---
    num_dense_sample_points = params.get("fracture_surface_dense_sample_points", 5000)
    print(f"    Preprocessing: Densely sampling {num_dense_sample_points} points from target surface of {fragment_name}...")
    if len(pcd_target_sampling.vertices) < 3:
        print(f"    Preprocessing: Target surface for {fragment_name} has < 3 vertices. Cannot sample points.")
        pcd = o3d.geometry.PointCloud()
    else:
        if not pcd_target_sampling.has_triangles():
             print(f"    Preprocessing: Target surface for {fragment_name} has no triangles. Using vertices directly.")
             pcd = o3d.geometry.PointCloud()
             pcd.points = pcd_target_sampling.vertices
        else:
            pcd = pcd_target_sampling.sample_points_poisson_disk(number_of_points=num_dense_sample_points)

    if not pcd.has_points():
        print(f"    Preprocessing: Dense sampling yielded no points for {fragment_name}.")
        if viz_collector is not None:
             viz_collector.append({'step': 'dense_sampling_failed', 'name': fragment_name, 'original_index': original_index})
        return None, fracture_surface_mesh_o3d

    if viz_collector is not None:
        viz_collector.append({
            'step': 'dense_sampling_result',
            'name': fragment_name,
            'original_index': original_index,
            'type': 'pointcloud',
            'points': np.asarray(pcd.points),
            'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
            # Normals not computed yet
        })
        
    # --- Step 3: Voxel Downsampling ---
    voxel_size = params.get("voxel_downsample_size", 0.01)
    print(f"    Preprocessing: Voxel downsampling points for {fragment_name} with voxel_size={voxel_size}...")
    if voxel_size > 0 and len(pcd.points) > 0:
        pcd_downsampled = pcd.voxel_down_sample(voxel_size)
        print(f"      Downsampled from {len(pcd.points)} to {len(pcd_downsampled.points)} points.")
        pcd = pcd_downsampled
    else:
        print(f"      Skipping voxel downsampling for {fragment_name}.")

    if not pcd.has_points():
        print(f"    Preprocessing: Point cloud became empty after downsampling for {fragment_name}.")
        if viz_collector is not None:
             viz_collector.append({'step': 'downsampling_failed', 'name': fragment_name, 'original_index': original_index})
        return None, fracture_surface_mesh_o3d # Return the original fracture_surface_mesh_o3d
        
    if viz_collector is not None:
        viz_collector.append({
            'step': 'downsampled_pcd_for_features_result',
            'name': fragment_name,
            'original_index': original_index,
            'type': 'pointcloud',
            'points': np.asarray(pcd.points),
            'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
        })

    # --- Step 4: Add Noise ---
    if params.get("add_preprocessing_noise", True) and len(pcd.points) > 0:
        noise_factor = params.get("preprocessing_noise_factor", 0.01)
        noise_magnitude = voxel_size * noise_factor
        noise = np.random.uniform(-noise_magnitude, noise_magnitude, size=np.asarray(pcd.points).shape)
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + noise)

    # --- Step 5: Estimate Normals ---
    print(f"    Preprocessing: Estimating normals for {fragment_name} ({len(pcd.points)} points)...")
    if len(pcd.points) > 0 :
        radius_normal_factor = params.get("normal_radius_factor", 2.0) # Ensure this is in config or use absolute
        radius_normal = params.get("normal_estimation_radius", voxel_size * params.get("normal_radius_factor", 2.0))
        max_nn_normal = params.get("normal_estimation_max_nn", 30)
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normal))
        
        try:
            pcd.orient_normals_consistent_tangent_plane(k=params.get("orient_normals_k", 15))
        except RuntimeError as e:
            print(f"    Warning: orient_normals_consistent_tangent_plane failed for {fragment_name}: {e}")
    else:
        print(f"    Preprocessing: No points to estimate normals for {fragment_name}.")
        if viz_collector is not None:
             viz_collector.append({'step': 'normal_estimation_failed_no_points', 'name': fragment_name, 'original_index': original_index})
        return None, fracture_surface_mesh_o3d # Return original fracture_surface_mesh_o3d

    # After normals are estimated, log the PCD again IF it's substantially different for viz
    # Or assume the 'downsampled_pcd_for_features_result' is what gets features computed on.
    # For simplicity, we assume the PCD logged at 'downsampled_pcd_for_features_result' is what goes to feature extraction.
    
    print(f"    Preprocessing: Finished for {fragment_name}. Final PCD for features has {len(pcd.points)} points.")
    return pcd, fracture_surface_mesh_o3d