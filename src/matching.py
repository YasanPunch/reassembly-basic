import numpy as np
from itertools import combinations
from src.alignment import align_fragments_pcd
from concurrent.futures import ThreadPoolExecutor, as_completed
import trimesh
import copy

def boolean_intersection_penetration_test(mesh1_o3d, mesh1_name, mesh2_o3d, mesh2_name, params, viz_collector=None):
    """Boolean intersection test for penetration detection during pairwise matching."""
    try:
        # Convert Open3D meshes to Trimesh
        mesh1_tri = trimesh.Trimesh(
            vertices=np.asarray(mesh1_o3d.vertices),
            faces=np.asarray(mesh1_o3d.triangles)
        )
        mesh2_tri = trimesh.Trimesh(
            vertices=np.asarray(mesh2_o3d.vertices),
            faces=np.asarray(mesh2_o3d.triangles)
        )
        
        # Ensure meshes are watertight
        if not mesh1_tri.is_watertight and len(mesh1_tri.faces) > 0:
            mesh1_tri.fill_holes()
        if not mesh2_tri.is_watertight and len(mesh2_tri.faces) > 0:
            mesh2_tri.fill_holes()
            
        # Check if meshes are valid
        if len(mesh1_tri.faces) == 0 or len(mesh2_tri.faces) == 0:
            return True, 0.0, None
            
        # Calculate volumes
        vol1 = mesh1_tri.volume
        vol2 = mesh2_tri.volume
        
        if vol1 <= 0 or vol2 <= 0:
            return True, 0.0, None
            
        # Perform boolean intersection
        try:
            intersection_mesh = trimesh.boolean.intersection([mesh1_tri, mesh2_tri])
            
            if intersection_mesh is None or len(intersection_mesh.faces) == 0:
                return True, 0.0, None
                
            # Calculate intersection volume and ratio
            intersection_volume = intersection_mesh.volume
            total_volume = min(vol1, vol2)  # Use smaller volume for ratio calculation
            intersection_ratio = (intersection_volume / total_volume) if total_volume > 0 else 0.0
            
            # Get penetration threshold from params (default to 0.1 = 10%)
            penetration_threshold = params.get("boolean_penetration_threshold", 0.1)
            
            # Check if penetration ratio is acceptable
            if intersection_ratio <= penetration_threshold:
                # Acceptable penetration - match is valid
                return True, intersection_ratio, intersection_mesh
            else:
                # Too much penetration - reject match
                return False, intersection_ratio, intersection_mesh
            
        except Exception as bool_error:
            print(f"Boolean intersection failed: {bool_error}")
            return None, 0.0, None
            
    except Exception as e:
        print(f"Boolean test error: {e}")
        return None, 0.0, None

def test_proposed_pairwise_match(source_fragment, target_fragment, transformation, params):
    """
    Test if applying a transformation would cause penetration between two fragments.
    
    This function:
    1. Takes the original meshes of two fragments
    2. Applies the transformation to one fragment
    3. Tests if the transformed fragment penetrates the other fragment
    4. Returns True if NO penetration, False if penetration detected
    
    Args:
        source_fragment: Fragment data dict with 'original_mesh' (the fragment to be transformed)
        target_fragment: Fragment data dict with 'original_mesh' (the reference fragment)
        transformation: 4x4 transformation matrix to apply to source_fragment
        params: Configuration parameters
        
    Returns:
        bool: True if transformation is valid (no penetration), False if penetration detected
    """
    # Get the original meshes in their initial positions
    mesh1_original = target_fragment['original_mesh']      # Reference fragment (stays in place)
    mesh2_original = source_fragment['original_mesh']      # Fragment to be transformed
    
    # Apply the transformation to see what the alignment would look like
    mesh2_transformed = copy.deepcopy(mesh2_original)
    mesh2_transformed.transform(transformation)
    
    # Test for penetration between the reference fragment and the transformed fragment
    result = boolean_intersection_penetration_test(
        mesh1_original, target_fragment['name'],
        mesh2_transformed, source_fragment['name'],
        params
    )
    
    is_valid, ratio, intersection_mesh = result
    
    if is_valid:
        if ratio > 0:
            print(f"    ⚠️  Minor penetration detected (ratio: {ratio:.3f}), but within acceptable threshold")
        return True  # Penetration is acceptable - this transformation is valid
    else:
        print(f"    ❌ Excessive penetration detected (ratio: {ratio:.3f}), exceeds threshold")
        return False  # Too much penetration - this transformation would cause excessive overlap

def _match_fragment_pair(i, j, frag_i_data, frag_j_data, params, debug=False):
    matches = []
    # Loop over all surface pairs
    for idx_i, (target_pcd, target_fpfh) in enumerate(zip(frag_i_data['pcds_for_features'], frag_i_data['features_list'])):
        for idx_j, (source_pcd, source_fpfh) in enumerate(zip(frag_j_data['pcds_for_features'], frag_j_data['features_list'])):
            if source_pcd is None or target_pcd is None or source_fpfh is None or target_fpfh is None:
                continue
            if not source_pcd.has_points() or not target_pcd.has_points() or \
               source_fpfh.num() == 0 or target_fpfh.num() == 0:
                continue
            if debug:
                transform_j_to_i, fitness_ji, rmse_ji = align_fragments_pcd(
                    source_pcd, target_pcd, source_fpfh, target_fpfh, params, debug=debug,
                    source_fragment=frag_j_data, target_fragment=frag_i_data
                )
            else:
                transform_j_to_i, fitness_ji, rmse_ji = align_fragments_pcd(
                    source_pcd, target_pcd, source_fpfh, target_fpfh, params
                )
            if transform_j_to_i is not None and fitness_ji >= params.get("min_match_score", 0.6):
                # Step 3: Test if the transformation causes penetration
                # This checks if applying the transformation would make the fragments penetrate each other
                if params.get("use_boolean_intersection_test", False):
                    is_valid = test_proposed_pairwise_match(
                        frag_j_data, frag_i_data, transform_j_to_i, params
                    )
                    if not is_valid:
                        print(f"  ❌ Excessive penetration detected: {frag_j_data['name']} -> {frag_i_data['name']}. Rejecting match.")
                        continue
                    else:
                        print(f"  ✅ Penetration test passed: {frag_j_data['name']} -> {frag_i_data['name']} is a valid match.")
                
                confidence_ji = float(fitness_ji) / (rmse_ji + 1e-6)
                matches.append({
                    'source_idx': j, 'target_idx': i,
                    'source_surface_idx': idx_j, 'target_surface_idx': idx_i,
                    'transformation': transform_j_to_i,
                    'score': fitness_ji, 'rmse': rmse_ji,
                    'confidence': confidence_ji,
                    'source_name': frag_j_data['name'], 'target_name': frag_i_data['name']
                })
            # Also try the reverse direction (i to j)
            if debug:
                transform_i_to_j, fitness_ij, rmse_ij = align_fragments_pcd(
                    target_pcd, source_pcd, target_fpfh, source_fpfh, params, debug=debug,
                    source_fragment=frag_i_data, target_fragment=frag_j_data
                )
            else:
                transform_i_to_j, fitness_ij, rmse_ij = align_fragments_pcd(
                    target_pcd, source_pcd, target_fpfh, source_fpfh, params
                )
            if transform_i_to_j is not None and fitness_ij >= params.get("min_match_score", 0.6):
                # Step 3: Test if the transformation causes penetration
                # This checks if applying the transformation would make the fragments penetrate each other
                if params.get("use_boolean_intersection_test", False):
                    is_valid = test_proposed_pairwise_match(
                        frag_i_data, frag_j_data, transform_i_to_j, params
                    )
                    if not is_valid:
                        print(f"  ❌ Excessive penetration detected: {frag_i_data['name']} -> {frag_j_data['name']}. Rejecting match.")
                        continue
                    else:
                        print(f"  ✅ Penetration test passed: {frag_i_data['name']} -> {frag_j_data['name']} is a valid match.")
                
                confidence_ij = float(fitness_ij) / (rmse_ij + 1e-6)
                matches.append({
                    'source_idx': i, 'target_idx': j,
                    'source_surface_idx': idx_i, 'target_surface_idx': idx_j,
                    'transformation': transform_i_to_j,
                    'score': fitness_ij, 'rmse': rmse_ij,
                    'confidence': confidence_ij,
                    'source_name': frag_i_data['name'], 'target_name': frag_j_data['name']
                })
    return matches

def find_pairwise_matches(fragments_data, params, debug=False, top_n_per_pair=3):
    """
    Finds potential pairwise alignments between all unique pairs of fragments.
    Each item in fragments_data is a dict:
    {'name': str, 'original_index': int, 'mesh': o3d.geometry.TriangleMesh,
     'pcd': o3d.geometry.PointCloud, 'features': o3d.pipelines.registration.Feature,
     'pcd_for_features': o3d.geometry.PointCloud}

    Args:
        fragments_data (list of dict): List of fragment data, including precomputed PCDs and features.
        params (dict): Configuration parameters.
        top_n_per_pair (int): Number of top matches to keep per fragment pair.

    Returns:
        list of dict: Each dict represents a potential match:
                      {'source_idx': int, 'target_idx': int,
                       'transformation': np.ndarray, 'score': float (fitness), 'rmse': float, 'confidence': float}
    """
    potential_matches = []
    num_fragments = len(fragments_data)

    if num_fragments < 2:
        print("Not enough fragments to find matches.")
        return []

    print(f"\nFinding pairwise matches among {num_fragments} fragments...")
    pairs = list(combinations(range(num_fragments), 2))
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_pair = {
            executor.submit(_match_fragment_pair, i, j, fragments_data[i], fragments_data[j], params, debug): (i, j)
            for i, j in pairs
        }
        for future in as_completed(future_to_pair):
            i, j = future_to_pair[future]
            matches = future.result()
            if matches:
                # Only keep top N matches for this pair (by score)
                matches_sorted = sorted(matches, key=lambda x: x['score'], reverse=True)
                results.extend(matches_sorted[:top_n_per_pair])
    # Sort all results by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"Found {len(results)} potential pairwise matches above threshold (top {top_n_per_pair} per pair).")
    return results

if __name__ == '__main__':
    from io_utils import load_fragments_from_directory
    from preprocessing import preprocess_fragment
    from feature_extraction import extract_features_from_pcd
    import json
    import os
    import copy
    import open3d as o3d


    # Create a dummy config for testing
    dummy_params = {
        "voxel_downsample_size": 0.05,
        "normal_estimation_radius": 0.1,
        "normal_estimation_max_nn": 30,
        "fpfh_feature_radius": 0.25,
        "fpfh_feature_max_nn": 100,
        "ransac_distance_threshold_factor": 1.5,
        "ransac_edge_length_factor": 0.9,
        "ransac_iterations": 10000, # Low for test
        "ransac_n_points": 3,
        "ransac_confidence": 0.99,
        "icp_max_correspondence_distance_factor": 2.0,
        "icp_relative_fitness": 1e-6,
        "icp_relative_rmse": 1e-6,
        "icp_max_iteration": 30,
        "min_match_score": 0.3 # Lower for testing
    }

    # Setup dummy data: two slightly transformed cubes
    base_dir = '../dummy_data_matching' # Relative to src/
    input_dir = os.path.join(base_dir, 'input_fragments')
    os.makedirs(input_dir, exist_ok=True)

    # Create a simple cube OBJ
    cube_obj_content = """
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
f 1 2 3 4
f 8 7 6 5
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 4 8 5 1
"""
    with open(os.path.join(input_dir, 'partA.obj'), 'w') as f:
        f.write(cube_obj_content)
    
    # Create a transformed version for partB
    mesh_a = o3d.io.read_triangle_mesh(os.path.join(input_dir, 'partA.obj'))
    mesh_b = copy.deepcopy(mesh_a)
    # Apply a known simple transformation (e.g., shift along X)
    # This is an easy case, real fragments are much harder
    transform_b = np.eye(4)
    transform_b[0, 3] = 0.8 # Shift by 0.8 along X (almost one full width of the cube)
    mesh_b.transform(transform_b)
    o3d.io.write_triangle_mesh(os.path.join(input_dir, 'partB.obj'), mesh_b)

    loaded_frags_info = load_fragments_from_directory(input_dir)
    
    processed_fragments_data = []
    for frag_info in loaded_frags_info:
        mesh = frag_info['mesh']
        pcd = preprocess_fragment(mesh, dummy_params)
        features, pcd_for_features = extract_features_from_pcd(pcd, dummy_params)
        
        processed_fragments_data.append({
            'name': frag_info['name'],
            'original_index': frag_info['original_index'],
            'mesh': mesh, # Original mesh for final assembly
            'pcd': pcd,   # Preprocessed PCD
            'features': features, # FPFH features
            'pcd_for_features': pcd_for_features # PCD used for FPFH (might be boundary, etc.)
        })

    if len(processed_fragments_data) >= 2:
        pairwise_matches = find_pairwise_matches(processed_fragments_data, dummy_params)

        print(f"\nFound {len(pairwise_matches)} potential matches:")
        for match in pairwise_matches:
            print(f"  {match['source_name']} (idx {match['source_idx']}) -> {match['target_name']} (idx {match['target_idx']}) "
                  f"Score: {match['score']:.3f}, RMSE: {match['rmse']:.3f}")
            # print("  Transformation:\n", match['transformation'])
        
        if pairwise_matches:
            # Visualize the best match
            best_match = pairwise_matches[0]
            source_frag_data = processed_fragments_data[best_match['source_idx']]
            target_frag_data = processed_fragments_data[best_match['target_idx']]

            # Use pcd_for_features for visualization as they were used for alignment
            source_pcd_vis = copy.deepcopy(source_frag_data['pcd_for_features'])
            target_pcd_vis = copy.deepcopy(target_frag_data['pcd_for_features'])
            
            source_pcd_vis.paint_uniform_color([1,0,0]) # Source Red
            target_pcd_vis.paint_uniform_color([0,1,0]) # Target Green
            
            # o3d.visualization.draw_geometries([source_pcd_vis, target_pcd_vis], window_name="Pairwise - Before Alignment")

            source_pcd_vis.transform(best_match['transformation'])
            # o3d.visualization.draw_geometries([source_pcd_vis, target_pcd_vis], window_name="Pairwise - After Alignment")
        
    else:
        print("Not enough fragments processed for matching test.")

    # import shutil
    # shutil.rmtree(base_dir)