from itertools import combinations
from src.alignment import align_fragments_pcd
import copy
from src.utils.geometry_utils import boolean_intersection_penetration_test

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


def _match_fragment_pair(
    i, j, frag_i_data, frag_j_data, params, debug=False, processing_panel=None
):
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
                    source_pcd,
                    target_pcd,
                    source_fpfh,
                    target_fpfh,
                    params,
                    debug=debug,
                    source_fragment=frag_j_data,
                    target_fragment=frag_i_data,
                    processing_panel=processing_panel,
                )
            else:
                transform_j_to_i, fitness_ji, rmse_ji = align_fragments_pcd(
                    source_pcd,
                    target_pcd,
                    source_fpfh,
                    target_fpfh,
                    params,
                    processing_panel=processing_panel,
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
                    target_pcd,
                    source_pcd,
                    target_fpfh,
                    source_fpfh,
                    params,
                    debug=debug,
                    source_fragment=frag_i_data,
                    target_fragment=frag_j_data,
                    processing_panel=processing_panel,
                )
            else:
                transform_i_to_j, fitness_ij, rmse_ij = align_fragments_pcd(
                    target_pcd,
                    source_pcd,
                    target_fpfh,
                    source_fpfh,
                    params,
                    processing_panel=processing_panel,
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


def find_pairwise_matches(
    fragments_data, params, debug=False, top_n_per_pair=3, processing_panel=None
):
    """
    Finds potential pairwise alignments between all unique pairs of fragments.
    Each item in fragments_data is a dict:
    {'name': str, 'original_index': int, 'mesh': o3d.geometry.TriangleMesh,
     'pcd': o3d.geometry.PointCloud, 'features': o3d.pipelines.registration.Feature,
     'pcd_for_features': o3d.geometry.PointCloud}

    Args:
        fragments_data (list of dict): List of fragment data, including precomputed PCDs and features.
        params (dict): Configuration parameters.
        debug (bool): Whether to enable debug visualization.
        top_n_per_pair (int): Number of top matches to keep per fragment pair.
        processing_panel: Optional processing panel for GUI visualization.

    Returns:
        list of dict: Each dict represents a potential match:
                      {'source_idx': int, 'target_idx': int,
                       'transformation': np.ndarray, 'score': float (fitness), 'rmse': float, 'confidence': float}
    """
    num_fragments = len(fragments_data)

    if num_fragments < 2:
        print("Not enough fragments to find matches.")
        return []

    print(f"\nFinding pairwise matches among {num_fragments} fragments...")
    pairs = list(combinations(range(num_fragments), 2))
    results = []

    # Use deterministic sequential processing instead of parallel processing
    # This ensures consistent results across runs
    for i, j in pairs:
        matches = _match_fragment_pair(
            i, j, fragments_data[i], fragments_data[j], params, debug, processing_panel
        )
        if matches:
            # Only keep top N matches for this pair (by score)
            matches_sorted = sorted(matches, key=lambda x: x['score'], reverse=True)
            results.extend(matches_sorted[:top_n_per_pair])

    # Sort all results by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"Found {len(results)} potential pairwise matches above threshold (top {top_n_per_pair} per pair).")
    return results
