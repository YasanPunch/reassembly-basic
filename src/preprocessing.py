import open3d as o3d
import numpy as np
import copy
from src.segmentation import extract_fracture_surface_mesh

print("DEBUG: preprocessing.py top level executed")

def validate_input_mesh(fragment_info, viz_collector=None):
    """
    Validates that the input mesh has vertices and is usable.
    
    Args:
        fragment_info (dict): Dict containing 'mesh' and 'name'.
        viz_collector (list, optional): List to append visualization data to.
        
    Returns:
        bool: True if mesh is valid, False otherwise.
    """
    original_mesh = fragment_info['mesh']
    fragment_name = fragment_info['name']
    
    if not original_mesh.has_vertices():
        print(f"    Preprocessing: Original mesh {fragment_name} has no vertices.")
        if viz_collector is not None:
            viz_collector.append({
                'step': 'preprocessing_failed_no_vertices', 
                'name': fragment_name
            })
        return False
    return True

def extract_fracture_surfaces(original_mesh, fragment_name, params):
    """
    Extracts fracture surfaces from the original mesh.
    
    Args:
        original_mesh (o3d.geometry.TriangleMesh): The original mesh.
        fragment_name (str): Name of the fragment.
        params (dict): Dictionary of parameters from config.
        
    Returns:
        list: List of fracture surface meshes.
    """
    print(f"    Preprocessing: Segmenting fracture surfaces for {fragment_name}...")
    fracture_surfaces = extract_fracture_surface_mesh(original_mesh, fragment_name, params)
    if not isinstance(fracture_surfaces, list):
        fracture_surfaces = [fracture_surfaces] if fracture_surfaces is not None else []
    return fracture_surfaces

def process_single_surface(surf, fragment_name, params):
    """
    Processes a single fracture surface: samples points, downsamples, estimates normals, and extracts features.
    
    Args:
        surf (o3d.geometry.TriangleMesh): Single fracture surface mesh.
        fragment_name (str): Name of the fragment.
        params (dict): Dictionary of parameters from config.
        
    Returns:
        tuple: (features, pcd) or (None, None) if processing fails.
    """
    if surf is None or not surf.has_triangles():
        return None, None
    
    num_dense_sample_points = params.get("fracture_surface_dense_sample_points", 5000)
    if len(surf.vertices) < 3:
        return None, None
    
    # Sample points densely
    pcd = surf.sample_points_poisson_disk(number_of_points=num_dense_sample_points)
    if not pcd.has_points():
        return None, None
    
    # Voxel downsampling
    voxel_size = params.get("voxel_downsample_size", 0.01)
    if voxel_size > 0 and len(pcd.points) > 0:
        pcd_downsampled = pcd.voxel_down_sample(voxel_size)
        pcd = pcd_downsampled
    if not pcd.has_points():
        return None, None
    
    # Estimate normals
    radius_normal = params.get("normal_estimation_radius", voxel_size * params.get("normal_radius_factor", 2.0))
    max_nn_normal = params.get("normal_estimation_max_nn", 30)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normal))
    
    try:
        pcd.orient_normals_consistent_tangent_plane(k=params.get("orient_normals_k", 15))
    except RuntimeError as e:
        print(f"    Warning: orient_normals_consistent_tangent_plane failed for {fragment_name}: {e}")
    
    # Feature extraction
    from src.feature_extraction import extract_features_from_pcd
    features, _ = extract_features_from_pcd(pcd, params)
    
    if features is not None and features.num() > 0:
        return features, pcd
    return None, None

def process_fracture_surfaces(fracture_surfaces, fragment_name, params):
    """
    Processes all fracture surfaces and extracts features.
    
    Args:
        fracture_surfaces (list): List of fracture surface meshes.
        fragment_name (str): Name of the fragment.
        params (dict): Dictionary of parameters from config.
        
    Returns:
        tuple: (pcds_for_features_list, features_list)
    """
    features_list = []
    pcds_for_features_list = []
    
    for surf in fracture_surfaces:
        features, pcd = process_single_surface(surf, fragment_name, params)
        if features is not None:
            features_list.append(features)
            pcds_for_features_list.append(pcd)
    
    return pcds_for_features_list, features_list

def log_segmentation_result(fragment_info, fracture_surfaces, viz_collector):
    """
    Logs segmentation results for visualization.
    
    Args:
        fragment_info (dict): Dict containing mesh and metadata.
        fracture_surfaces (list): List of fracture surface meshes.
        viz_collector (list): List to append visualization data to.
    """
    if viz_collector is None:
        return
    
    original_mesh = fragment_info['mesh']
    fragment_name = fragment_info['name']
    original_index = fragment_info['original_index']
    
    log_entry = {
        'step': 'segmentation_result', 
        'name': fragment_name,
        'original_index': original_index,
        'original_mesh_type': 'mesh',
        'original_mesh_vertices': np.asarray(original_mesh.vertices),
        'original_mesh_triangles': np.asarray(original_mesh.triangles),
    }
    
    # Handle fracture surfaces (could be list or single mesh)
    if isinstance(fracture_surfaces, list) and fracture_surfaces:
        # For now, log the first valid fracture surface
        for surf in fracture_surfaces:
            if surf and surf.has_triangles():
                log_entry.update({
                    'fracture_mesh_type': 'mesh',
                    'fracture_mesh_vertices': np.asarray(surf.vertices),
                    'fracture_mesh_triangles': np.asarray(surf.triangles),
                })
                break
        else:
            log_entry['fracture_mesh_type'] = None
    elif fracture_surfaces and fracture_surfaces.has_triangles():
        log_entry.update({
            'fracture_mesh_type': 'mesh',
            'fracture_mesh_vertices': np.asarray(fracture_surfaces.vertices),
            'fracture_mesh_triangles': np.asarray(fracture_surfaces.triangles),
        })
    else:
        log_entry['fracture_mesh_type'] = None
    
    viz_collector.append(log_entry)

def visualize_segmentation_interactive(original_mesh, fracture_surfaces, fragment_name, params):
    """
    Handles interactive visualization of segmentation results.
    
    Args:
        original_mesh (o3d.geometry.TriangleMesh): Original mesh.
        fracture_surfaces (list): List of fracture surface meshes.
        fragment_name (str): Name of the fragment.
        params (dict): Dictionary of parameters from config.
    """
    if not params.get('visualize_segmentation', False):
        return
    
    from src.segmentation import visualize_segmentation
    
    # Handle fracture_surfaces as list or single mesh
    if isinstance(fracture_surfaces, list) and fracture_surfaces:
        fracture_surface_mesh_o3d = fracture_surfaces[0]  # Use first for visualization
    else:
        fracture_surface_mesh_o3d = fracture_surfaces
    
    vis_geometries = visualize_segmentation(original_mesh, fracture_surface_mesh_o3d, fragment_name)
    o3d.visualization.draw_geometries(vis_geometries, 
                                     window_name=f"Segmentation Result: {fragment_name}")

def debug_visualize_segmentation(original_mesh, fracture_surfaces, fragment_name, params):
    """
    Handles debug visualization of segmentation results.
    
    Args:
        original_mesh (o3d.geometry.TriangleMesh): Original mesh.
        fracture_surfaces (list): List of fracture surface meshes.
        fragment_name (str): Name of the fragment.
        params (dict): Dictionary of parameters from config.
    """
    if not params.get('debug_pairwise_matching', False):
        return
    
    mesh_vis = copy.deepcopy(original_mesh)
    mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])
    vis_geoms = [mesh_vis]
    
    if isinstance(fracture_surfaces, list):
        any_found = False
        for surf in fracture_surfaces:
            if surf and surf.has_triangles():
                surf_vis = copy.deepcopy(surf)
                surf_vis.paint_uniform_color([1, 0, 0])
                vis_geoms.append(surf_vis)
                any_found = True
        if not any_found:
            print(f"WARNING: No fracture surface found for {fragment_name} during segmentation debug visualization.")
    elif fracture_surfaces and fracture_surfaces.has_triangles():
        fracture_vis = copy.deepcopy(fracture_surfaces)
        fracture_vis.paint_uniform_color([1, 0, 0])
        vis_geoms.append(fracture_vis)
    else:
        print(f"WARNING: No fracture surface found for {fragment_name} during segmentation debug visualization.")
    
    o3d.visualization.draw_geometries(vis_geoms, 
                                     window_name=f"[DEBUG] Segmentation: {fragment_name} (Gray=Full Mesh, Red=Fracture Surface)")

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

    # Validate input mesh
    if not validate_input_mesh(fragment_info, viz_collector):
        return [], [], []

    # Extract fracture surfaces
    fracture_surfaces = extract_fracture_surfaces(original_mesh, fragment_name, params)
    
    # Process fracture surfaces and extract features
    pcds_for_features_list, features_list = process_fracture_surfaces(fracture_surfaces, fragment_name, params)
    
    # Log segmentation results for visualization
    log_segmentation_result(fragment_info, fracture_surfaces, viz_collector)
    
    # Handle user interactions (visualization)
    visualize_segmentation_interactive(original_mesh, fracture_surfaces, fragment_name, params)
    debug_visualize_segmentation(original_mesh, fracture_surfaces, fragment_name, params)
    
    # Return lists for downstream processing
    return pcds_for_features_list, features_list, fracture_surfaces