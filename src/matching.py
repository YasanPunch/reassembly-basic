import numpy as np
from itertools import combinations
from src.alignment import align_fragments_pcd
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import trimesh
import copy
from src.utils.geometry_utils import get_adjacent_faces, compute_adjacent_face_normal_similarity, compute_bumpiness_similarity, compute_curvature_similarity
import open3d as o3d
import math

print("DEBUG: matching.py top level executed")

def sample_points_on_surface(mesh, n_points=500):
    # Uniformly sample points on the mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return np.asarray(pcd.points)

def min_distance_between_surfaces(points_a, points_b):
    # Compute minimum distance between two sets of points
    from scipy.spatial import cKDTree
    if points_a.size == 0 or points_b.size == 0:
        print(f"[Warning] min_distance_between_surfaces called with empty array: points_a size {points_a.size}, points_b size {points_b.size}")
        return float('inf'), np.array([])
    tree_b = cKDTree(points_b)
    dists, _ = tree_b.query(points_a)
    return np.min(dists), dists

def surface_coverage(points_a, points_b, threshold=0.5):
    # Fraction of points in A within threshold distance of any point in B
    from scipy.spatial import cKDTree
    if points_a.size == 0 or points_b.size == 0:
        print(f"[Warning] surface_coverage called with empty array: points_a size {points_a.size}, points_b size {points_b.size}")
        return 0.0
    tree_b = cKDTree(points_b)
    dists, _ = tree_b.query(points_a)
    covered = np.sum(dists < threshold)
    return covered / len(points_a) if len(points_a) > 0 else 0.0

def robust_intersection_check(points_a, mesh_b, deep_penetration=1.0):
    # Check if points_a are inside mesh_b, and how deep
    # mesh_b: open3d.geometry.TriangleMesh
    # Returns: fraction_inside, fraction_deep_inside
    try:
        import trimesh
        mesh_b_tri = trimesh.Trimesh(
            vertices=np.asarray(mesh_b.vertices),
            faces=np.asarray(mesh_b.triangles)
        )
        inside = mesh_b_tri.contains(points_a)
        if np.any(inside):
            surface = mesh_b_tri.nearest.signed_distance(points_a[inside])
            deep = np.abs(surface) > deep_penetration
            return np.mean(inside), np.mean(deep)
        else:
            return 0.0, 0.0
    except Exception as e:
        print(f"Intersection check failed: {e}")
        return 0.0, 0.0

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
            intersection_ratio = intersection_volume / total_volume if total_volume > 0 else 0.0
            
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

def _is_valid_mesh(mesh):
    try:
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        if verts.size == 0 or tris.size == 0:
            return False
        if np.any(np.isnan(verts)) or np.any(np.isnan(tris)):
            return False
        if np.any(np.isinf(verts)) or np.any(np.isinf(tris)):
            return False
        if len(verts.shape) != 2 or verts.shape[1] != 3:
            return False
        if len(tris.shape) != 2 or tris.shape[1] not in [3, 4]:
            return False
        return True
    except Exception as e:
        print(f"[Validation Error] Mesh validation failed: {e}")
        return False

def _match_fragment_pair(i, j, frag_i_data, frag_j_data, params, debug=False):
    matches = []
    w1 = params.get('composite_score_icp_weight', 0.5)
    w2 = params.get('composite_score_adjacent_weight', 0.2)
    w3 = params.get('composite_score_bumpiness_weight', 0.15)
    w4 = params.get('composite_score_curvature_weight', 0.15)
    sampled_points_cache = {}
    for idx_i, (target_pcd, target_fpfh) in enumerate(zip(frag_i_data['pcds_for_features'], frag_i_data['features_list'])):
        for idx_j, (source_pcd, source_fpfh) in enumerate(zip(frag_j_data['pcds_for_features'], frag_j_data['features_list'])):
            try:
                if source_pcd is None or target_pcd is None or source_fpfh is None or target_fpfh is None:
                    print(f"[Reject] {frag_j_data['name']} -> {frag_i_data['name']} | Missing PCD or features.")
                    continue
                if not source_pcd.has_points() or not target_pcd.has_points() or \
                   source_fpfh.num() == 0 or target_fpfh.num() == 0:
                    print(f"[Reject] {frag_j_data['name']} -> {frag_i_data['name']} | Empty PCD or features.")
                    continue
                print(f"[Step] ICP alignment {frag_j_data['name']} -> {frag_i_data['name']}")
                if debug:
                    transform_j_to_i, fitness_ji, rmse_ji = align_fragments_pcd(
                        source_pcd, target_pcd, source_fpfh, target_fpfh, params, debug=debug,
                        source_fragment=frag_j_data, target_fragment=frag_i_data
                    )
                else:
                    transform_j_to_i, fitness_ji, rmse_ji = align_fragments_pcd(
                        source_pcd, target_pcd, source_fpfh, target_fpfh, params
                    )
                print(f"[Step] ICP done {frag_j_data['name']} -> {frag_i_data['name']} | Fitness: {fitness_ji:.4f}")
                # Remove hard fitness threshold, just penalize low fitness
                fitness_penalty = max(0.05, fitness_ji)  # Never zero, but low fitness is heavily penalized
                if transform_j_to_i is None:
                    print(f"[Reject] {frag_j_data['name']} -> {frag_i_data['name']} | ICP failed to produce a transformation.")
                    continue
                if params.get("use_boolean_intersection_test", False):
                    print(f"[Step] Boolean intersection test {frag_j_data['name']} -> {frag_i_data['name']}")
                    is_valid = test_proposed_pairwise_match(
                        frag_j_data, frag_i_data, transform_j_to_i, params
                    )
                    if not is_valid:
                        print(f"[Reject] {frag_j_data['name']} -> {frag_i_data['name']} | Boolean intersection test failed.")
                        # Instead of continue, apply a heavy penalty
                        intersection_penalty = 0.05
                    else:
                        print(f"  ✅ Penetration test passed: {frag_j_data['name']} -> {frag_i_data['name']} is a valid match.")
                source_fracture_mesh = frag_j_data.get('fracture_surfaces', [None])[idx_j]
                target_fracture_mesh = frag_i_data.get('fracture_surfaces', [None])[idx_i]
                if source_fracture_mesh is not None and target_fracture_mesh is not None:
                    if not (_is_valid_mesh(source_fracture_mesh) and _is_valid_mesh(target_fracture_mesh)):
                        print(f"[Reject] {frag_j_data['name']} -> {frag_i_data['name']} | Invalid fracture mesh.")
                        continue
                    key_source = id(source_fracture_mesh)
                    key_target = id(target_fracture_mesh)
                    try:
                        print(f"[Step] Sampling source mesh {frag_j_data['name']} fracture {idx_j}")
                        if key_source not in sampled_points_cache:
                            source_fracture_mesh_trans = copy.deepcopy(source_fracture_mesh)
                            source_fracture_mesh_trans.transform(transform_j_to_i)
                            sampled_points_cache[key_source] = sample_points_on_surface(source_fracture_mesh_trans)
                        pts_source = sampled_points_cache[key_source]
                        print(f"[Step] Sampling target mesh {frag_i_data['name']} fracture {idx_i}")
                        if key_target not in sampled_points_cache:
                            sampled_points_cache[key_target] = sample_points_on_surface(target_fracture_mesh)
                        pts_target = sampled_points_cache[key_target]
                    except Exception as e:
                        print(f"[Error] Sampling points failed: {e}")
                        continue
                    try:
                        print(f"[Step] Distance calculation {frag_j_data['name']} -> {frag_i_data['name']}")
                        min_dist, dists = min_distance_between_surfaces(pts_source, pts_target)
                        distance_penalty = np.exp(-min_dist/2)
                        # Surface-to-surface distance penalty (with tolerance)
                        epsilon = 0.05  # Allow up to 0.05 units of overlap
                        if min_dist < -epsilon:
                            print(f"[Reject] {frag_j_data['name']} -> {frag_i_data['name']} | Intersecting (min_dist={min_dist:.4f} < -{epsilon}), skipping match.")
                            continue
                        if min_dist < 0:
                            surface_distance_penalty = np.exp(-5 * abs(min_dist))  # Softer penalty for tiny overlaps
                            print(f"[Penalty] {frag_j_data['name']} -> {frag_i_data['name']} | min_dist={min_dist:.4f} < 0, soft penalty: {surface_distance_penalty:.4f}")
                        else:
                            surface_distance_penalty = np.exp(-min_dist)
                        print(f"[Scoring] {frag_j_data['name']} -> {frag_i_data['name']} | Min distance: {min_dist:.4f} | Distance penalty: {distance_penalty:.4f} | Surface distance penalty: {surface_distance_penalty:.4f}")
                        if min_dist > 3.0:
                            print(f"[Reject] {frag_j_data['name']} -> {frag_i_data['name']} | Min distance {min_dist:.4f} > 3.0, skipping expensive checks.")
                            continue
                    except Exception as e:
                        print(f"[Error] Distance calculation failed: {e}")
                        continue
                    try:
                        print(f"[Step] Intersection check {frag_j_data['name']} -> {frag_i_data['name']}")
                        frac_inside, frac_deep = robust_intersection_check(pts_source, frag_i_data['original_mesh'])
                        # Harsher intersection penalty
                        if frac_deep > 0.1:
                            print(f"[Severe Penalty] {frag_j_data['name']} -> {frag_i_data['name']} | Deep intersection detected (frac_deep={frac_deep:.3f}), setting intersection_penalty to 0.01")
                            intersection_penalty = 0.01
                        else:
                            intersection_penalty = max(0.1, 1.0 - min(1.0, frac_deep * 10))
                        print(f"[Scoring] {frag_j_data['name']} -> {frag_i_data['name']} | Fraction inside: {frac_inside:.4f} | Fraction deep: {frac_deep:.4f} | Intersection penalty: {intersection_penalty:.4f}")
                    except Exception as e:
                        print(f"[Error] Intersection check failed: {e}")
                        intersection_penalty = 0.01
                    try:
                        print(f"[Step] Coverage calculation {frag_j_data['name']} -> {frag_i_data['name']}")
                        coverage_A_by_B = surface_coverage(pts_source, pts_target)
                        coverage_B_by_A = surface_coverage(pts_target, pts_source)
                        print(f"[Scoring] {frag_j_data['name']} -> {frag_i_data['name']} | Coverage A by B: {coverage_A_by_B:.4f} | Coverage B by A: {coverage_B_by_A:.4f}")
                    except Exception as e:
                        print(f"[Error] Coverage calculation failed: {e}")
                        coverage_A_by_B = 0.0
                        coverage_B_by_A = 0.0
                else:
                    distance_penalty = 1.0
                    intersection_penalty = 1.0
                    coverage_A_by_B = 0.0
                    coverage_B_by_A = 0.0
                    print(f"[Scoring] {frag_j_data['name']} -> {frag_i_data['name']} | Fracture surfaces missing, using default penalties and zero coverage.")
                # --- Composite scoring: adjacent face similarity, bumpiness, curvature ---
                source_fracture_faces = getattr(frag_j_data.get('fracture_surfaces', [None])[idx_j], 'triangles', None)
                target_fracture_faces = getattr(frag_i_data.get('fracture_surfaces', [None])[idx_i], 'triangles', None)
                if source_fracture_faces is not None and target_fracture_faces is not None:
                    source_fracture_faces = np.asarray(source_fracture_faces)
                    target_fracture_faces = np.asarray(target_fracture_faces)
                    def find_face_indices_in_mesh(mesh, submesh_triangles):
                        mesh_tris = np.asarray(mesh.triangles)
                        submesh_tris = np.asarray(submesh_triangles)
                        indices = []
                        for tri in submesh_tris:
                            matches = np.where(np.all(mesh_tris == tri, axis=1))[0]
                            if len(matches) > 0:
                                indices.append(matches[0])
                        return indices
                    source_fracture_indices = find_face_indices_in_mesh(frag_j_data['original_mesh'], source_fracture_faces)
                    target_fracture_indices = find_face_indices_in_mesh(frag_i_data['original_mesh'], target_fracture_faces)
                    adj_source = get_adjacent_faces(frag_j_data['original_mesh'], source_fracture_indices)
                    adj_target = get_adjacent_faces(frag_i_data['original_mesh'], target_fracture_indices)
                    source_mesh_transformed = copy.deepcopy(frag_j_data['original_mesh'])
                    source_mesh_transformed.transform(transform_j_to_i)
                    if len(adj_source) == 0 or len(adj_target) == 0:
                        print(f"[Penalty] {frag_j_data['name']} -> {frag_i_data['name']} | Adjacent faces empty, applying penalty but not skipping match.")
                        adjacent_similarity = 0.0
                        bumpiness_similarity = 0.0
                        curvature_similarity = 0.0
                    else:
                        adjacent_similarity = compute_adjacent_face_normal_similarity(
                            source_mesh_transformed, adj_source,
                            frag_i_data['original_mesh'], adj_target
                        )
                        bumpiness_similarity = compute_bumpiness_similarity(
                            source_mesh_transformed, adj_source,
                            frag_i_data['original_mesh'], adj_target
                        )
                        curvature_similarity = compute_curvature_similarity(
                            source_mesh_transformed, adj_source,
                            frag_i_data['original_mesh'], adj_target
                        )
                else:
                    adjacent_similarity = 0.0
                    bumpiness_similarity = 0.0
                    curvature_similarity = 0.0
                composite_score = (
                    w1 * fitness_ji +
                    w2 * adjacent_similarity +
                    w3 * bumpiness_similarity +
                    w4 * curvature_similarity +
                    1.5 * min(coverage_A_by_B, coverage_B_by_A) +  # promote full coverage
                    1.0 * max(coverage_A_by_B, coverage_B_by_A)    # promote partial coverage
                )
                composite_score *= distance_penalty
                composite_score *= surface_distance_penalty
                composite_score *= intersection_penalty
                # Compute total surface coverage as the average of both directions
                total_coverage = (coverage_A_by_B + coverage_B_by_A) / 2.0
                # Intersection penalty (softer)
                if min_dist < -0.1:
                    intersection_penalty = 0.01
                    print(f"[Penalty] {frag_j_data['name']} -> {frag_i_data['name']} | min_dist={min_dist:.4f} < -0.1, strong penalty.")
                elif min_dist < 0:
                    intersection_penalty = np.exp(-5 * abs(min_dist))
                    print(f"[Penalty] {frag_j_data['name']} -> {frag_i_data['name']} | min_dist={min_dist:.4f} < 0, soft penalty: {intersection_penalty:.4f}")
                else:
                    intersection_penalty = 1.0
                # Use precomputed average normals for each fracture surface
                avg_normal_src = None
                avg_normal_tgt = None
                if 'fracture_surface_normals' in frag_j_data and 'fracture_surface_normals' in frag_i_data:
                    if idx_j < len(frag_j_data['fracture_surface_normals']) and idx_i < len(frag_i_data['fracture_surface_normals']):
                        avg_normal_src = frag_j_data['fracture_surface_normals'][idx_j]
                        avg_normal_tgt = frag_i_data['fracture_surface_normals'][idx_i]
                if avg_normal_src is not None and avg_normal_tgt is not None:
                    dot = np.clip(np.dot(avg_normal_src, avg_normal_tgt), -1.0, 1.0)
                    angle = np.degrees(np.arccos(dot))
                    if angle < 90 or angle > 270:
                        normal_alignment_penalty = 0.01
                        print(f"[Normal Penalty] {frag_j_data['name']} -> {frag_i_data['name']} | Angle between normals: {angle:.2f}°, strong penalty.")
                    else:
                        normal_alignment_penalty = max(0.01, 1 - abs(angle - 180) / 90)
                        print(f"[Normal Penalty] {frag_j_data['name']} -> {frag_i_data['name']} | Angle between normals: {angle:.2f}°, penalty: {normal_alignment_penalty:.4f}")
                else:
                    normal_alignment_penalty = 1.0
                # Composite score: ICP fitness * total_coverage * normal_alignment_penalty * intersection_penalty
                composite_score = fitness_ji * total_coverage * normal_alignment_penalty * intersection_penalty
                print(f"[Score] {frag_j_data['name']} -> {frag_i_data['name']} | ICP: {fitness_ji:.4f} | Coverage: {total_coverage:.4f} | Normal penalty: {normal_alignment_penalty:.4f} | Intersection penalty: {intersection_penalty:.4f} | Composite: {composite_score:.4f}")
                confidence_ji = float(fitness_ji) / (rmse_ji + 1e-6)
                matches.append({
                    'source_idx': j, 'target_idx': i,
                    'source_surface_idx': idx_j, 'target_surface_idx': idx_i,
                    'transformation': transform_j_to_i,
                    'score': composite_score, 'rmse': rmse_ji,
                    'confidence': confidence_ji,
                    'source_name': frag_j_data['name'], 'target_name': frag_i_data['name'],
                    'icp_fitness': fitness_ji,
                    'total_coverage': total_coverage,
                    'normal_alignment_penalty': normal_alignment_penalty,
                    'intersection_penalty': intersection_penalty
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
                if transform_i_to_j is None or fitness_ij < 0.2:
                    print(f"[Reject] {frag_i_data['name']} -> {frag_j_data['name']} | ICP fitness too low: {fitness_ij:.4f}")
                    continue
                source_fracture_mesh = frag_i_data.get('fracture_surfaces', [None])[idx_i]
                target_fracture_mesh = frag_j_data.get('fracture_surfaces', [None])[idx_j]
                if source_fracture_mesh is not None and target_fracture_mesh is not None:
                    key_source = id(source_fracture_mesh)
                    key_target = id(target_fracture_mesh)
                    if key_source not in sampled_points_cache:
                        source_fracture_mesh_trans = copy.deepcopy(source_fracture_mesh)
                        source_fracture_mesh_trans.transform(transform_i_to_j)
                        sampled_points_cache[key_source] = sample_points_on_surface(source_fracture_mesh_trans)
                    pts_source = sampled_points_cache[key_source]
                    if key_target not in sampled_points_cache:
                        sampled_points_cache[key_target] = sample_points_on_surface(target_fracture_mesh)
                    pts_target = sampled_points_cache[key_target]
                    min_dist, dists = min_distance_between_surfaces(pts_source, pts_target)
                    distance_penalty = np.exp(-min_dist/2)
                    epsilon = 0.05
                    if min_dist < -0.1:
                        intersection_penalty = 0.01
                        print(f"[Penalty] {frag_i_data['name']} -> {frag_j_data['name']} | min_dist={min_dist:.4f} < -0.1, strong penalty.")
                    elif min_dist < 0:
                        intersection_penalty = np.exp(-5 * abs(min_dist))
                        print(f"[Penalty] {frag_i_data['name']} -> {frag_j_data['name']} | min_dist={min_dist:.4f} < 0, soft penalty: {intersection_penalty:.4f}")
                    else:
                        intersection_penalty = 1.0
                    coverage_A_by_B = surface_coverage(pts_source, pts_target)
                    coverage_B_by_A = surface_coverage(pts_target, pts_source)
                    total_coverage = (coverage_A_by_B + coverage_B_by_A) / 2.0
                    avg_normal_src = None
                    avg_normal_tgt = None
                    if 'fracture_surface_normals' in frag_i_data and 'fracture_surface_normals' in frag_j_data:
                        if idx_i < len(frag_i_data['fracture_surface_normals']) and idx_j < len(frag_j_data['fracture_surface_normals']):
                            avg_normal_src = frag_i_data['fracture_surface_normals'][idx_i]
                            avg_normal_tgt = frag_j_data['fracture_surface_normals'][idx_j]
                    if avg_normal_src is not None and avg_normal_tgt is not None:
                        dot = np.clip(np.dot(avg_normal_src, avg_normal_tgt), -1.0, 1.0)
                        angle = np.degrees(np.arccos(dot))
                        if angle < 90 or angle > 270:
                            normal_alignment_penalty = 0.01
                            print(f"[Normal Penalty] {frag_i_data['name']} -> {frag_j_data['name']} | Angle between normals: {angle:.2f}°, strong penalty.")
                        else:
                            normal_alignment_penalty = max(0.01, 1 - abs(angle - 180) / 90)
                            print(f"[Normal Penalty] {frag_i_data['name']} -> {frag_j_data['name']} | Angle between normals: {angle:.2f}°, penalty: {normal_alignment_penalty:.4f}")
                    else:
                        normal_alignment_penalty = 1.0
                    composite_score = fitness_ij * total_coverage * normal_alignment_penalty * intersection_penalty
                    print(f"[Score] {frag_i_data['name']} -> {frag_j_data['name']} | ICP: {fitness_ij:.4f} | Coverage: {total_coverage:.4f} | Normal penalty: {normal_alignment_penalty:.4f} | Intersection penalty: {intersection_penalty:.4f} | Composite: {composite_score:.4f}")
                    confidence_ij = float(fitness_ij) / (rmse_ij + 1e-6)
                    matches.append({
                        'source_idx': i, 'target_idx': j,
                        'source_surface_idx': idx_i, 'target_surface_idx': idx_j,
                        'transformation': transform_i_to_j,
                        'score': composite_score, 'rmse': rmse_ij,
                        'confidence': confidence_ij,
                        'source_name': frag_i_data['name'], 'target_name': frag_j_data['name'],
                        'icp_fitness': fitness_ij,
                        'total_coverage': total_coverage,
                        'normal_alignment_penalty': normal_alignment_penalty,
                        'intersection_penalty': intersection_penalty
                    })
            except Exception as e:
                print(f"[Error] Exception in _match_fragment_pair for {frag_j_data['name']} -> {frag_i_data['name']}: {e}")
                continue
    return matches

def find_pairwise_matches(fragments_data, params, debug=False, top_n_per_pair=3):
    potential_matches = []
    num_fragments = len(fragments_data)

    if num_fragments < 2:
        print("Not enough fragments to find matches.")
        return []

    print(f"\nFinding pairwise matches among {num_fragments} fragments...")
    pairs = list(combinations(range(num_fragments), 2))
    all_matches = []
    with ThreadPoolExecutor() as executor:
        future_to_pair = {
            executor.submit(_match_fragment_pair, i, j, fragments_data[i], fragments_data[j], params, debug): (i, j)
            for i, j in pairs
        }
        for future in as_completed(future_to_pair):
            i, j = future_to_pair[future]
            try:
                matches = future.result(timeout=30)
            except TimeoutError:
                print(f"[Timeout] Pairwise match {i}-{j} took too long and was skipped.")
                continue
            except Exception as e:
                print(f"[Error] Exception in future for pair {i}-{j}: {e}")
                continue
            if matches:
                all_matches.extend(matches)
    all_matches.sort(key=lambda x: x['score'], reverse=True)
    print(f"Found {len(all_matches)} total pairwise matches. Returning top {top_n_per_pair}.")
    return all_matches[:top_n_per_pair]

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
        "min_match_score": 0.3, # Lower for testing
        "composite_score_icp_weight": 0.7,
        "composite_score_adjacent_weight": 0.3
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