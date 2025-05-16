import numpy as np
from itertools import combinations
from src.alignment import align_fragments_pcd

print("DEBUG: matching.py top level executed") # <--- ADD THIS

def find_pairwise_matches(fragments_data, params):
    """
    Finds potential pairwise alignments between all unique pairs of fragments.
    Each item in fragments_data is a dict:
    {'name': str, 'original_index': int, 'mesh': o3d.geometry.TriangleMesh,
     'pcd': o3d.geometry.PointCloud, 'features': o3d.pipelines.registration.Feature,
     'pcd_for_features': o3d.geometry.PointCloud}

    Args:
        fragments_data (list of dict): List of fragment data, including precomputed PCDs and features.
        params (dict): Configuration parameters.

    Returns:
        list of dict: Each dict represents a potential match:
                      {'source_idx': int, 'target_idx': int,
                       'transformation': np.ndarray, 'score': float (fitness), 'rmse': float}
    """
    potential_matches = []
    num_fragments = len(fragments_data)

    if num_fragments < 2:
        print("Not enough fragments to find matches.")
        return []

    print(f"\nFinding pairwise matches among {num_fragments} fragments...")
    
    # Iterate over all unique pairs of fragments
    for i, j in combinations(range(num_fragments), 2):
        frag_i_data = fragments_data[i]
        frag_j_data = fragments_data[j]

        print(f"  Attempting to match: {frag_i_data['name']} (idx {i}) <-> {frag_j_data['name']} (idx {j})")

        # Align j to i (j is source, i is target)
        # Use the 'pcd_for_features' as these are what FPFH were computed on
        source_pcd = frag_j_data['pcd_for_features']
        target_pcd = frag_i_data['pcd_for_features']
        source_fpfh = frag_j_data['features']
        target_fpfh = frag_i_data['features']
        
        if source_pcd is None or target_pcd is None or source_fpfh is None or target_fpfh is None:
            print(f"    Skipping pair ({i}, {j}) due to missing PCD/features.")
            continue
        if not source_pcd.has_points() or not target_pcd.has_points() or \
           source_fpfh.num() == 0 or target_fpfh.num() == 0:
            print(f"    Skipping pair ({i}, {j}) due to empty PCD/features.")
            continue


        transform_j_to_i, fitness_ji, rmse_ji = align_fragments_pcd(
            source_pcd, target_pcd, source_fpfh, target_fpfh, params
        )
        if transform_j_to_i is not None and fitness_ji >= params.get("min_match_score", 0.6):
            potential_matches.append({
                'source_idx': j, 'target_idx': i, # j is transformed to align with i
                'transformation': transform_j_to_i,
                'score': fitness_ji, 'rmse': rmse_ji,
                'source_name': frag_j_data['name'], 'target_name': frag_i_data['name']
            })
            print(f"    Match found ({j} to {i}): Score={fitness_ji:.3f}, RMSE={rmse_ji:.3f}")
        else:
            # print(f"    No good match from j to i (Score: {fitness_ji:.3f})")
            pass


        # Align i to j (i is source, j is target)
        source_pcd = frag_i_data['pcd_for_features']
        target_pcd = frag_j_data['pcd_for_features']
        source_fpfh = frag_i_data['features']
        target_fpfh = frag_j_data['features']

        transform_i_to_j, fitness_ij, rmse_ij = align_fragments_pcd(
            source_pcd, target_pcd, source_fpfh, target_fpfh, params
        )
        if transform_i_to_j is not None and fitness_ij >= params.get("min_match_score", 0.6):
            potential_matches.append({
                'source_idx': i, 'target_idx': j, # i is transformed to align with j
                'transformation': transform_i_to_j,
                'score': fitness_ij, 'rmse': rmse_ij,
                'source_name': frag_i_data['name'], 'target_name': frag_j_data['name']
            })
            print(f"    Match found ({i} to {j}): Score={fitness_ij:.3f}, RMSE={rmse_ij:.3f}")
        else:
            # print(f"    No good match from i to j (Score: {fitness_ij:.3f})")
            pass


    # Sort matches by score (descending)
    potential_matches.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Found {len(potential_matches)} potential pairwise matches above threshold.")
    return potential_matches

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