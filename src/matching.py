"""
Pairwise fragment matching with parallel processing support.

This module implements parallel processing for pairwise fragment matching using Open3D.
The multiprocessing start method is set to 'spawn' or 'forkserver' to avoid hanging
issues that occur when forking multithreaded processes (Open3D uses OpenMP internally).

IMPORTANT: Parallel processing is now enabled using file-based loading to avoid Open3D object pickling limitations.
Open3D objects (TriangleMesh, PointCloud, etc.) cannot be pickled and passed between processes.
The solution implemented here:
1. Pass only file paths and indices to worker processes
2. Load Open3D objects from files in each worker process
3. Use spawn/forkserver start method to avoid hanging issues
4. Preprocess fragments in each worker to get required features

Common Open3D operations that hang without proper start method:
- remove_statistical_outlier()
- estimate_normals()
- cluster_connected_triangles()
- Point cloud registration algorithms
- Feature extraction (FPFH, etc.)

References:
- Python multiprocessing docs: https://docs.python.org/3/library/multiprocessing.html
- Open3D community solutions for similar issues
"""

from itertools import combinations
from src.alignment import align_fragments_pcd
import copy
from src.utils.geometry_utils import boolean_intersection_penetration_test
import multiprocessing as mp
from functools import partial
import os
import json
import numpy as np
import hashlib
import platform

# Set multiprocessing start method to avoid issues with Open3D's OpenMP threads
# This prevents hanging when forking multithreaded processes
# Common Open3D operations that hang without this: remove_statistical_outlier,
# estimate_normals, cluster_connected_triangles, and various point cloud operations
# See: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
if platform.system() != "Windows":
    try:
        # Try spawn first (most reliable), fallback to forkserver
        mp.set_start_method("spawn", force=True)
        print(
            "  ℹ️  Set multiprocessing start method to 'spawn' for Open3D compatibility"
        )
    except RuntimeError:
        try:
            # Fallback to forkserver if spawn fails
            mp.set_start_method("forkserver", force=True)
            print(
                "  ℹ️  Set multiprocessing start method to 'forkserver' for Open3D compatibility"
            )
        except RuntimeError:
            # Method already set, ignore
            pass

# Global variable to store fragments data for parallel processing
_global_fragments_data = None
_global_params = None
_global_top_n_per_pair = 3
_global_fragments_paths = None


def _init_worker(fragments_paths, params, top_n_per_pair=3):
    """
    Initialize worker process with fragments file paths and parameters.
    Loads fragments from files to avoid pickling Open3D objects.
    """
    global _global_fragments_data, _global_params, _global_top_n_per_pair, _global_fragments_paths

    # Store file paths for later loading
    _global_fragments_paths = fragments_paths
    _global_params = params
    _global_top_n_per_pair = top_n_per_pair
    _global_fragments_data = None  # Will be loaded on demand


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


def _load_fragments_from_paths(fragments_paths):
    """
    Load fragments from file paths. This function is used in worker processes
    to avoid pickling Open3D objects.

    Args:
        fragments_paths (list): List of file paths to load fragments from

    Returns:
        list: List of fragment data dictionaries with loaded Open3D objects
    """
    import src.io_utils

    fragments_data = []
    for i, file_path in enumerate(fragments_paths):
        # Load the fragment using the existing io_utils function
        fragment_info = {
            "mesh": src.io_utils.load_fragment(file_path),
            "name": os.path.basename(file_path),
            "original_index": i,
            "path": file_path,
        }

        if fragment_info["mesh"] is not None:
            # Preprocess the fragment to get the required data structure
            from src.preprocessing import preprocess_fragment

            # Create the fragment data structure expected by the matching pipeline
            fragment_data = {
                "name": fragment_info["name"],
                "original_index": fragment_info["original_index"],
                "original_mesh": fragment_info["mesh"],
                "fracture_surfaces": [],
                "pcds_for_features": [],
                "features_list": [],
            }

            # Preprocess to get features
            pcds_for_features, features_list, fracture_surfaces = preprocess_fragment(
                fragment_info, _global_params
            )

            fragment_data["pcds_for_features"] = pcds_for_features
            fragment_data["features_list"] = features_list
            fragment_data["fracture_surfaces"] = fracture_surfaces

            fragments_data.append(fragment_data)
        else:
            print(f"    Warning: Failed to load fragment from {file_path}")

    return fragments_data


def _match_fragment_pair_wrapper(pair_indices):
    """
    Wrapper function for parallel processing of fragment pair matching.
    Uses file paths to load fragments in each worker process, avoiding pickling issues.

    Note: This function uses Open3D operations that require spawn/forkserver
    multiprocessing start method to avoid hanging (e.g., point cloud operations,
    feature extraction, registration algorithms).

    Args:
        pair_indices: Tuple containing (i, j) indices of fragments to match

    Returns:
        list: Matches for this fragment pair
    """
    global _global_fragments_data, _global_params, _global_top_n_per_pair, _global_fragments_paths

    i, j = pair_indices

    # Load fragments data if not already loaded
    if _global_fragments_data is None:
        _global_fragments_data = _load_fragments_from_paths(_global_fragments_paths)

    if i >= len(_global_fragments_data) or j >= len(_global_fragments_data):
        print(f"    Error: Invalid fragment indices ({i}, {j})")
        return []

    frag_i_data = _global_fragments_data[i]
    frag_j_data = _global_fragments_data[j]
    params = _global_params
    top_n_per_pair = _global_top_n_per_pair

    # Set process name for debugging
    process_name = f"Worker-{os.getpid()}"

    try:
        print(
            f"    {process_name}: Processing pair ({i}, {j}) - {frag_i_data['name']} vs {frag_j_data['name']}"
        )

        matches = _match_fragment_pair(
            i, j, frag_i_data, frag_j_data, params, debug=False
        )

        if matches:
            # Only keep top N matches for this pair (by score)
            matches_sorted = sorted(matches, key=lambda x: x["score"], reverse=True)
            result = matches_sorted[:top_n_per_pair]
            print(
                f"    {process_name}: Found {len(result)} matches for pair ({i}, {j})"
            )
            return result
        else:
            print(f"    {process_name}: No matches found for pair ({i}, {j})")
            return []

    except Exception as e:
        print(f"    {process_name}: Error processing pair ({i}, {j}): {e}")
        return []


def find_pairwise_matches(
    fragments_data,
    params,
    debug=False,
    top_n_per_pair=3,
    n_jobs=None,
    use_cached_matches=True,
    save_matches=True,
    matches_cache_dir="matches_cache",
    disable_parallel=False,
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
        debug (bool): Whether to enable debug mode.
        top_n_per_pair (int): Number of top matches to keep per fragment pair.
        n_jobs (int): Number of parallel jobs. If None, uses all available CPU cores.
        use_cached_matches (bool): Whether to try loading cached matches first.
        save_matches (bool): Whether to save computed matches for future use.
        matches_cache_dir (str): Directory to store/load cached matches.

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

    # Try to load cached matches first
    if use_cached_matches:
        cached_matches = load_pairwise_matches(
            fragments_data, params, top_n_per_pair, matches_cache_dir
        )
        if cached_matches is not None:
            print(f"  ✅ Using cached pairwise matches ({len(cached_matches)} matches)")
            return cached_matches
        else:
            print(f"  ℹ️  No compatible cached matches found, computing new matches...")

    # Generate all unique pairs (i,j) where i < j to avoid redundancy
    # This produces pairs like: (0,1), (0,2), (1,2) for 3 fragments
    pairs = []
    for i in range(num_fragments):
        for j in range(i + 1, num_fragments):
            pairs.append((i, j))

    # Determine number of jobs
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), len(pairs))  # Use all available CPU cores
    else:
        n_jobs = min(n_jobs, len(pairs))

    print(f"Processing {len(pairs)} fragment pairs using {n_jobs} parallel workers...")

    results = []

    # Use parallel processing for better performance
    # Note: Open3D operations are now safe with spawn/forkserver start method
    # and file-based loading to avoid pickling issues
    if n_jobs > 1 and len(pairs) > 1 and not debug and not disable_parallel:
        try:
            print(f"  Attempting parallel processing with {n_jobs} workers...")

            # Extract file paths from fragments data to avoid pickling Open3D objects
            fragments_paths = [
                frag["path"]
                for frag in fragments_data
                if "path" in frag and frag["path"]
            ]

            if len(fragments_paths) != len(fragments_data):
                print(
                    f"  ⚠️  Some fragments missing file paths ({len(fragments_paths)}/{len(fragments_data)}), falling back to sequential processing..."
                )
                print(
                    f"     Missing paths for fragments: {[frag['name'] for frag in fragments_data if 'path' not in frag or not frag['path']]}"
                )
                raise ValueError("Not all fragments have file paths")

            # Use file paths to avoid pickling Open3D objects
            with mp.Pool(
                processes=n_jobs,
                initializer=_init_worker,
                initargs=(fragments_paths, params, top_n_per_pair),
            ) as pool:
                # Pass only the pair indices to avoid pickling issues
                # Use map_async with timeout to prevent hanging
                async_result = pool.map_async(_match_fragment_pair_wrapper, pairs)

                # Wait for results with a reasonable timeout (30 seconds per pair)
                timeout = len(pairs) * 30  # 30 seconds per pair
                try:
                    parallel_results = async_result.get(timeout=timeout)

                    # Flatten results
                    for result in parallel_results:
                        if result:
                            results.extend(result)

                    print(f"  ✅ Parallel processing completed successfully")

                except mp.TimeoutError:
                    print(f"  ⚠️  Parallel processing timed out after {timeout} seconds")
                    print(f"  Falling back to sequential processing...")
                    # Fall back to sequential processing
                    results = []
                    for i, j in pairs:
                        matches = _match_fragment_pair(
                            i, j, fragments_data[i], fragments_data[j], params, debug
                        )
                        if matches:
                            matches_sorted = sorted(
                                matches, key=lambda x: x["score"], reverse=True
                            )
                            results.extend(matches_sorted[:top_n_per_pair])

        except Exception as e:
            print(f"  ❌ Parallel processing failed: {e}")
            print(f"  Falling back to sequential processing...")
            # Fallback to sequential processing
            results = []
            for i, j in pairs:
                matches = _match_fragment_pair(
                    i, j, fragments_data[i], fragments_data[j], params, debug
                )
                if matches:
                    matches_sorted = sorted(
                        matches, key=lambda x: x["score"], reverse=True
                    )
                    results.extend(matches_sorted[:top_n_per_pair])
    else:
        # Sequential processing for small numbers of pairs, single job, debug mode, or disabled parallel
        if disable_parallel:
            print(f"  Using sequential processing (parallel processing disabled)...")
        else:
            print(f"  Using sequential processing...")
        for i, j in pairs:
            matches = _match_fragment_pair(
                i, j, fragments_data[i], fragments_data[j], params, debug
            )
            if matches:
                matches_sorted = sorted(matches, key=lambda x: x["score"], reverse=True)
                results.extend(matches_sorted[:top_n_per_pair])

    # Sort all results by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"Found {len(results)} potential pairwise matches above threshold (top {top_n_per_pair} per pair).")

    # Save matches for future use if requested
    if save_matches and results:
        save_pairwise_matches(
            results, fragments_data, params, top_n_per_pair, matches_cache_dir
        )

    return results


def get_matches_filename(fragments_data, params, top_n_per_pair=3):
    """
    Generate a filename for saving/loading pairwise matches data.

    Args:
        fragments_data (list): List of fragment data
        params (dict): Configuration parameters
        top_n_per_pair (int): Number of top matches per pair

    Returns:
        str: Filename for the matches cache
    """
    # Create a hash of the parameters and fragment names to ensure consistency
    param_str = str(sorted(params.items()))
    fragment_names = [frag["name"] for frag in fragments_data]
    fragment_str = str(sorted(fragment_names))

    # Create a hash from parameters, fragment names, and top_n_per_pair
    hash_input = f"{param_str}_{fragment_str}_{top_n_per_pair}"
    param_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    return f"pairwise_matches_{param_hash}.json"


def save_pairwise_matches(
    matches, fragments_data, params, top_n_per_pair=3, output_dir="matches_cache"
):
    """
    Save pairwise matches to a JSON file for subsequent runs.

    Args:
        matches (list): List of pairwise match dictionaries
        fragments_data (list): List of fragment data used for matching
        params (dict): Configuration parameters used for matching
        top_n_per_pair (int): Number of top matches per pair
        output_dir (str): Directory to save the matches file

    Returns:
        str: Path to the saved file, or None if saving failed
    """
    try:
        # Create matches directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        filename = get_matches_filename(fragments_data, params, top_n_per_pair)
        filepath = os.path.join(output_dir, filename)

        # Prepare data for saving
        matches_data = {
            "metadata": {
                "num_fragments": len(fragments_data),
                "num_matches": len(matches),
                "top_n_per_pair": top_n_per_pair,
                "fragment_names": [frag["name"] for frag in fragments_data],
                "params_hash": hashlib.md5(
                    str(sorted(params.items())).encode()
                ).hexdigest()[:8],
            },
            "params": params,
            "matches": [],
        }

        # Convert matches to JSON-serializable format
        for match in matches:
            serializable_match = {
                "source_idx": match["source_idx"],
                "target_idx": match["target_idx"],
                "source_surface_idx": match["source_surface_idx"],
                "target_surface_idx": match["target_surface_idx"],
                "transformation": match[
                    "transformation"
                ].tolist(),  # Convert numpy array to list
                "score": float(match["score"]),
                "rmse": float(match["rmse"]),
                "confidence": float(match["confidence"]),
                "source_name": match["source_name"],
                "target_name": match["target_name"],
            }
            matches_data["matches"].append(serializable_match)

        with open(filepath, "w") as f:
            json.dump(matches_data, f, indent=2)

        print(f"  ✅ Pairwise matches saved to: {filepath}")
        print(f"     - {len(matches)} matches for {len(fragments_data)} fragments")
        return filepath

    except Exception as e:
        print(f"  ❌ Could not save pairwise matches: {e}")
        return None


def load_pairwise_matches(
    fragments_data, params, top_n_per_pair=3, matches_dir="matches_cache"
):
    """
    Load pairwise matches from a JSON file if available and compatible.

    Args:
        fragments_data (list): List of fragment data
        params (dict): Configuration parameters
        top_n_per_pair (int): Number of top matches per pair
        matches_dir (str): Directory containing matches files

    Returns:
        list: Loaded matches, or None if loading failed or data incompatible
    """
    try:
        filename = get_matches_filename(fragments_data, params, top_n_per_pair)
        filepath = os.path.join(matches_dir, filename)

        if not os.path.exists(filepath):
            print(f"  ℹ️  No cached matches found at: {filepath}")
            return None

        with open(filepath, "r") as f:
            matches_data = json.load(f)

        # Validate that the loaded data is compatible
        current_fragment_names = [frag["name"] for frag in fragments_data]
        saved_fragment_names = matches_data["metadata"]["fragment_names"]

        if (
            matches_data["metadata"]["num_fragments"] != len(fragments_data)
            or current_fragment_names != saved_fragment_names
            or matches_data["metadata"]["top_n_per_pair"] != top_n_per_pair
        ):
            print(f"  ⚠️  Cached matches incompatible with current fragments/parameters")
            print(
                f"     Current: {len(fragments_data)} fragments, {top_n_per_pair} top matches"
            )
            print(
                f"     Cached: {matches_data['metadata']['num_fragments']} fragments, {matches_data['metadata']['top_n_per_pair']} top matches"
            )
            return None

        # Convert back to the original format
        matches = []
        for match_dict in matches_data["matches"]:
            match = {
                "source_idx": match_dict["source_idx"],
                "target_idx": match_dict["target_idx"],
                "source_surface_idx": match_dict["source_surface_idx"],
                "target_surface_idx": match_dict["target_surface_idx"],
                "transformation": np.array(
                    match_dict["transformation"]
                ),  # Convert back to numpy array
                "score": match_dict["score"],
                "rmse": match_dict["rmse"],
                "confidence": match_dict["confidence"],
                "source_name": match_dict["source_name"],
                "target_name": match_dict["target_name"],
            }
            matches.append(match)

        print(f"  ✅ Loaded {len(matches)} pairwise matches from: {filepath}")
        return matches

    except Exception as e:
        print(f"  ❌ Could not load pairwise matches: {e}")
        return None
