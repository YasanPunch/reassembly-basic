import open3d as o3d
import trimesh
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA # For PCA on normals
import copy

def calculate_face_roughness(mesh, face_cluster, neighborhood_size=5):
    """
    Calculate surface roughness for a cluster of faces.
    Args:
        mesh (trimesh.Trimesh): Input mesh
        face_cluster (np.ndarray): Indices of faces in the cluster
        neighborhood_size (int): Size of neighborhood for local roughness calculation
    Returns:
        float: Roughness metric (higher means more irregular)
    """
    # For each face, calculate the deviation of its normal from its neighbors
    face_normals = mesh.face_normals[face_cluster]
    total_roughness = 0.0
    
    # Get adjacency information for the whole mesh
    # This returns an array of face pairs that share an edge
    face_adjacency = trimesh.graph.face_adjacency(mesh.faces)
    
    for i, face_idx in enumerate(face_cluster):
        # Find faces adjacent to this face
        adjacent_faces = []
        for adj_pair in face_adjacency:
            if adj_pair[0] == face_idx and adj_pair[1] in face_cluster:
                adjacent_faces.append(adj_pair[1])
            elif adj_pair[1] == face_idx and adj_pair[0] in face_cluster:
                adjacent_faces.append(adj_pair[0])
                
        if len(adjacent_faces) < 2:  # Not enough neighbors found
            continue
            
        # Limit to neighborhood_size neighbors if we have more
        if len(adjacent_faces) > neighborhood_size:
            adjacent_faces = adjacent_faces[:neighborhood_size]
            
        # Get normals of neighbors
        neighbor_normals = mesh.face_normals[adjacent_faces]
        
        # Calculate average normal of neighbors
        avg_normal = np.mean(neighbor_normals, axis=0)
        if np.linalg.norm(avg_normal) > 1e-10:  # Avoid division by zero
            avg_normal = avg_normal / np.linalg.norm(avg_normal)  # Normalize
        else:
            continue
        
        # Calculate angular deviation from average
        deviations = np.array([np.arccos(np.clip(np.dot(mesh.face_normals[face_idx], avg_normal), -1.0, 1.0))])
        
        # Use standard deviation as a measure of roughness
        local_roughness = np.std(deviations) if len(deviations) > 1 else 0
        total_roughness += local_roughness
    
    # Average roughness over all faces in cluster
    avg_roughness = total_roughness / len(face_cluster) if len(face_cluster) > 0 else 0
    return avg_roughness

def cluster_faces_by_normal(mesh, eps, min_samples, face_subset_indices=None):
    """
    Cluster faces by normal similarity using DBSCAN.
    Args:
        mesh (trimesh.Trimesh): Input mesh.
        eps (float): DBSCAN epsilon parameter.
        min_samples (int): DBSCAN min_samples parameter.
        face_subset_indices (np.ndarray, optional): If provided, cluster only these faces.
                                                    The returned cluster indices will be relative
                                                    to this subset, but mapped back to original indices.
    Returns:
        tuple: (list of np.ndarray of original face indices, np.ndarray of labels for all faces in subset)
    """
    all_mesh_face_normals = mesh.face_normals

    if face_subset_indices is None:
        target_faces_for_clustering = np.arange(len(all_mesh_face_normals))
        current_face_normals = all_mesh_face_normals
    else:
        if len(face_subset_indices) == 0:
            return [], np.array([])
        target_faces_for_clustering = face_subset_indices
        current_face_normals = all_mesh_face_normals[target_faces_for_clustering]

    if len(current_face_normals) < min_samples:
        if len(current_face_normals) > 0:
            # Treat all as one cluster if too few for DBSCAN's min_samples
            # Return original indices for this single cluster
            return [target_faces_for_clustering.copy()], np.zeros(len(current_face_normals), dtype=int)
        return [], np.array([])

    norms = np.linalg.norm(current_face_normals, axis=1, keepdims=True)
    valid_mask = norms.flatten() > 1e-10
    
    if not np.any(valid_mask): # No valid normals in the subset
        return [], np.array([])

    normalized_normals_subset = np.zeros_like(current_face_normals)
    normalized_normals_subset[valid_mask] = current_face_normals[valid_mask] / norms[valid_mask]
    
    # Only fit DBSCAN on valid normals if there are any non-valid ones
    # This assumes DBSCAN handles non-finite values poorly if any slip through.
    # For simplicity, we assume all normals in current_face_normals are usable after normalization if norm > 0.

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(normalized_normals_subset)
    labels_for_subset = db.labels_ # These labels are for the `current_face_normals` (i.e., for the subset)

    unique_labels = set(labels_for_subset)
    final_clusters_orig_indices = []
    for label in unique_labels:
        cluster_mask_in_subset = (labels_for_subset == label)
        # Get original face indices from the target_faces_for_clustering array
        original_indices_for_this_cluster = target_faces_for_clustering[cluster_mask_in_subset]
        if len(original_indices_for_this_cluster) > 0:
            final_clusters_orig_indices.append(original_indices_for_this_cluster)
            
    return final_clusters_orig_indices, labels_for_subset

def get_segment_connected_components(tri_mesh, segment_face_indices):
    """
    Splits a segment (defined by original face indices) into its connected components
    using tri_mesh.face_adjacency.
    Returns a list of np.ndarrays, each containing original face indices.
    """
    if len(segment_face_indices) == 0:
        return []
    if len(segment_face_indices) == 1:
        return [segment_face_indices.copy()]

    idx_map_orig_to_local = {face_idx: i for i, face_idx in enumerate(segment_face_indices)}
    idx_map_local_to_orig = {i: face_idx for face_idx, i in idx_map_orig_to_local.items()}
    
    num_local_faces = len(segment_face_indices)
    adj_local = [[] for _ in range(num_local_faces)]

    # Ensure face_adjacency is computed
    if not hasattr(tri_mesh, 'face_adjacency') or tri_mesh.face_adjacency is None:
        # print("      get_segment_connected_components: tri_mesh.face_adjacency not found, calling refresh().")
        tri_mesh.refresh() # This should compute face_adjacency
        if not hasattr(tri_mesh, 'face_adjacency') or tri_mesh.face_adjacency is None:
            print("      CRITICAL: tri_mesh.face_adjacency still not available after refresh.")
            # Fallback: return as single component if adjacency cannot be determined
            return [segment_face_indices.copy()]

    # Iterate through all face adjacencies in the original mesh
    for face1_orig, face2_orig in tri_mesh.face_adjacency:
        # Check if both faces are part of the current segment
        if face1_orig in idx_map_orig_to_local and face2_orig in idx_map_orig_to_local:
            u_local, v_local = idx_map_orig_to_local[face1_orig], idx_map_orig_to_local[face2_orig]
            adj_local[u_local].append(v_local)
            adj_local[v_local].append(u_local) # Adjacency is symmetric

    visited_local = [False] * num_local_faces
    final_connected_components_orig_indices = []
    
    for i_local_start in range(num_local_faces):
        if not visited_local[i_local_start]:
            current_component_local_indices_list = []
            q = [i_local_start]
            visited_local[i_local_start] = True
            head = 0
            while head < len(q):
                u_local = q[head]; head += 1
                current_component_local_indices_list.append(u_local)
                for v_local in adj_local[u_local]:
                    # Check if adj_local[u_local] might have duplicates if face_adjacency had them
                    # Though set conversion below should handle it.
                    if not visited_local[v_local]:
                        visited_local[v_local] = True
                        q.append(v_local)
            
            # Make sure adj_local lists are unique before BFS to avoid redundant checks
            # Though visited_local handles cycles, cleaner adj list is better.
            # This is usually handled by how adj_local is built.

            component_orig_indices = np.array([idx_map_local_to_orig[local_idx] for local_idx in current_component_local_indices_list], dtype=int)
            final_connected_components_orig_indices.append(component_orig_indices)
            
    return final_connected_components_orig_indices


def is_segment_normals_coherent_pca(tri_mesh, segment_face_indices, params):
    if len(segment_face_indices) < params.get("refinement_min_faces_for_coherency_check", 10):
        return True 
    
    normals = tri_mesh.face_normals[segment_face_indices]
    # Ensure normals are unit vectors before PCA
    norm_values = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_normals_mask = norm_values.flatten() > 1e-10
    if np.sum(valid_normals_mask) < params.get("refinement_min_faces_for_coherency_check", 10): return True
    
    unit_normals = normals[valid_normals_mask] / norm_values[valid_normals_mask]

    pca = PCA(n_components=min(3, unit_normals.shape[1])) # unit_normals.shape[1] should be 3
    try:
        pca.fit(unit_normals)
    except ValueError: # Can happen if not enough samples for n_components
        return True # Assume coherent if too few valid normals for robust PCA
        
    explained_variance_ratio = pca.explained_variance_ratio_
    ratio_threshold = params.get("refinement_coherency_normal_pca_ratio_threshold", 0.9)
    
    # For a "flat" or "gently curved single side", 1st PCA component should dominate, or 1st+2nd for curves
    if explained_variance_ratio[0] >= ratio_threshold: # Very flat
        return True 
    # if len(explained_variance_ratio) > 1 and (explained_variance_ratio[0] + explained_variance_ratio[1]) >= 0.95: # Gently curved
    #     return True
    # print(f"      PCA coherency fail: {explained_variance_ratio}")
    return False

def is_segment_planar_by_fit(tri_mesh, segment_face_indices, params):
    if len(segment_face_indices) < params.get("refinement_min_faces_for_coherency_check", 10):
        return True

    face_centroids = tri_mesh.triangles_center[segment_face_indices]
    face_normals_segment = tri_mesh.face_normals[segment_face_indices]

    # Robust plane normal: average normal of the segment
    plane_normal = np.mean(face_normals_segment, axis=0)
    norm_of_plane_normal = np.linalg.norm(plane_normal)
    if norm_of_plane_normal < 1e-9: return True # All normals cancelled out - highly non-planar or error
    plane_normal /= norm_of_plane_normal
    plane_point = np.mean(face_centroids, axis=0)

    distances = np.abs(np.dot(face_centroids - plane_point, plane_normal))
    avg_distance_dev = np.mean(distances) if len(distances) > 0 else 0.0

    cos_angles = np.clip(np.dot(face_normals_segment, plane_normal), -1.0, 1.0)
    angle_devs_rad = np.arccos(cos_angles)
    avg_angle_dev_deg = np.mean(np.degrees(angle_devs_rad)) if len(angle_devs_rad) > 0 else 0.0
    
    max_extent = np.max(np.ptp(face_centroids, axis=0)) if len(face_centroids) > 1 else 1.0
    dist_dev_threshold = max_extent * params.get("refinement_coherency_plane_fit_dist_dev_factor", 0.1) # 10% of max extent
    angle_dev_threshold_deg = params.get("refinement_coherency_plane_fit_normal_angle_dev_deg", 25.0)

    if avg_distance_dev < dist_dev_threshold and avg_angle_dev_deg < angle_dev_threshold_deg:
        return True
    # print(f"      Planar fit fail: D_avg={avg_distance_dev:.3f} (T:{dist_dev_threshold:.3f}), A_avg={avg_angle_dev_deg:.1f} (T:{angle_dev_threshold_deg:.1f})")
    return False

def calculate_per_face_roughness_metric(tri_mesh, face_indices, neighborhood_k=5):
    """ A simple per-face roughness: std dev of angles to N adjacent face normals."""
    if len(face_indices) == 0: return np.array([])
    roughness_values = np.zeros(len(face_indices))
    
    for i, face_idx in enumerate(face_indices):
        current_normal = tri_mesh.face_normals[face_idx]
        neighbor_face_indices = tri_mesh.face_neighbors[face_idx]
        
        if len(neighbor_face_indices) == 0:
            roughness_values[i] = 0.0
            continue
            
        # Consider only up to K neighbors for consistency, or all if fewer than K
        actual_neighbors_indices = neighbor_face_indices[:neighborhood_k] if len(neighbor_face_indices) > neighborhood_k else neighbor_face_indices
        if len(actual_neighbors_indices) == 0 : # Should not happen if len(neighbor_face_indices)>0
            roughness_values[i] = 0.0
            continue

        neighbor_normals = tri_mesh.face_normals[actual_neighbors_indices]
        
        cos_angles = np.clip(np.dot(neighbor_normals, current_normal), -1.0, 1.0)
        angles_rad = np.arccos(cos_angles)
        roughness_values[i] = np.std(angles_rad) if len(angles_rad) > 1 else (angles_rad[0] if len(angles_rad)==1 else 0.0)
    return roughness_values


def is_segment_roughness_homogeneous(tri_mesh, segment_face_indices, params):
    if len(segment_face_indices) < params.get("refinement_min_faces_for_roughness_check", 20):
        return True
        
    per_face_roughness = calculate_per_face_roughness_metric(tri_mesh, segment_face_indices, 
                                                             neighborhood_k=params.get("roughness_neighbor_k", 5))
    if len(per_face_roughness) == 0: return True

    std_dev_roughness = np.std(per_face_roughness)
    threshold = params.get("refinement_roughness_std_dev_threshold", 0.2) # Radians
    if std_dev_roughness < threshold:
        return True
    # print(f"      Roughness homogeneity fail: std_dev={std_dev_roughness:.3f} (T:{threshold:.3f})")
    return False

def split_segment_by_recluster(tri_mesh, segment_face_indices, params, original_eps, original_min_samples):
    if len(segment_face_indices) < original_min_samples * params.get("refinement_split_min_samples_factor", 0.75) * 2 : # Not enough to split meaningfully
        return [segment_face_indices.copy()] # Return as is

    strict_eps = original_eps * params.get("refinement_split_strict_eps_factor", 0.7)
    strict_min_samples = max(3, int(original_min_samples * params.get("refinement_split_min_samples_factor", 0.75)))
    
    # print(f"      Splitting segment of {len(segment_face_indices)} faces with eps={strict_eps:.3f}, min_samples={strict_min_samples}")

    # cluster_faces_by_normal expects original_face_indices for its `face_subset_indices` argument
    # and returns list of clusters, where each cluster is an array of original face indices.
    sub_clusters_orig_indices, _ = cluster_faces_by_normal(
        tri_mesh, 
        strict_eps, 
        strict_min_samples, 
        face_subset_indices=segment_face_indices 
    )
            
    if not sub_clusters_orig_indices or (len(sub_clusters_orig_indices) == 1 and len(sub_clusters_orig_indices[0]) == len(segment_face_indices)):
         # print(f"      Re-clustering did not split segment of {len(segment_face_indices)} faces effectively.")
         return [segment_face_indices.copy()]
         
    return [sc.copy() for sc in sub_clusters_orig_indices if len(sc)>0]

def calculate_pca_badness_score(tri_mesh, segment_face_indices, params):
    """
    Calculates a 'badness' score based on PCA of normals.
    Higher score means normals are more spread out (less coherent/planar).
    Score = 1.0 - ratio_of_variance_explained_by_1st_component.
    """
    # Min faces check: if too few, assume "good" (score 0) to avoid splitting tiny segments based on this.
    if len(segment_face_indices) < params.get("refinement_min_faces_for_coherency_check", 10): # Use existing param
        return 0.0 
    
    normals = tri_mesh.face_normals[segment_face_indices]
    norm_values = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_normals_mask = norm_values.flatten() > 1e-10

    if np.sum(valid_normals_mask) < params.get("refinement_min_faces_for_coherency_check", 10):
        return 0.0 # Not enough valid normals, assume good for this check

    unit_normals = normals[valid_normals_mask] / norm_values[valid_normals_mask]

    # Ensure n_components is not greater than number of features or samples
    n_components_pca = min(unit_normals.shape[0], unit_normals.shape[1], 3)
    if n_components_pca == 0: return 0.0 # Should not happen if sum(valid_normals_mask) is checked

    pca = PCA(n_components=n_components_pca)
    try:
        pca.fit(unit_normals)
    except ValueError:
        return 0.0 # Assume coherent if PCA fails (e.g. singular matrix for very few points)
        
    explained_variance_ratio = pca.explained_variance_ratio_
    
    if len(explained_variance_ratio) == 0: # Should not happen if fit succeeds
        return 0.0 
        
    # Badness is 1 minus how much the first component explains.
    # If 1st component explains everything (ratio=1.0), badness = 0.
    # If 1st component explains little (e.g., ratio=0.33 for spherical), badness = 0.67.
    return 1.0 - explained_variance_ratio[0]


def calculate_planar_badness_score(tri_mesh, segment_face_indices, params):
    """
    Calculates a 'badness' score based on fitting a plane.
    Higher score means poorer fit (less planar).
    Score is a sum of normalized deviations (distance and angle).
    """
    if len(segment_face_indices) < params.get("refinement_min_faces_for_coherency_check", 10): # Use existing param
        return 0.0

    face_centroids = tri_mesh.triangles_center[segment_face_indices]
    face_normals_segment = tri_mesh.face_normals[segment_face_indices]

    if len(face_centroids) == 0: return 0.0 # Should be caught by segment_face_indices check

    plane_normal = np.mean(face_normals_segment, axis=0)
    norm_of_plane_normal = np.linalg.norm(plane_normal)
    if norm_of_plane_normal < 1e-9:
        return params.get("refinement_planar_split_badness_thresh", 1.5) + 1.0 # Max badness if normals cancel

    plane_normal /= norm_of_plane_normal
    plane_point = np.mean(face_centroids, axis=0)

    distances = np.abs(np.dot(face_centroids - plane_point, plane_normal))
    avg_distance_dev = np.mean(distances) if len(distances) > 0 else 0.0

    # Ensure dot product is within [-1, 1] for arccos
    cos_angles = np.clip(np.dot(face_normals_segment, plane_normal), -1.0, 1.0)
    angle_devs_rad = np.arccos(cos_angles)
    avg_angle_dev_deg = np.mean(np.degrees(angle_devs_rad)) if len(angle_devs_rad) > 0 else 0.0
    
    # Normalize deviations by their thresholds to get a combined score
    # A score around 1.0 means it's at the edge of one threshold, around 2.0 means it's at edge of both or double one
    max_extent = np.max(np.ptp(face_centroids, axis=0)) if len(face_centroids) > 1 else 1.0
    if max_extent < 1e-6: max_extent = 1.0 # Avoid division by zero for tiny segments

    # Use the general coherency thresholds for normalization here
    # The splitting decision will use the *specific* planar_split_badness_thresh
    dist_dev_thresh_for_norm = max_extent * params.get("refinement_coherency_plane_fit_dist_dev_factor", 0.1)
    angle_dev_thresh_for_norm_deg = params.get("refinement_coherency_plane_fit_normal_angle_dev_deg", 25.0)

    # Avoid division by zero for thresholds
    if dist_dev_thresh_for_norm < 1e-6: dist_dev_thresh_for_norm = 1e-6
    if angle_dev_thresh_for_norm_deg < 1e-6: angle_dev_thresh_for_norm_deg = 1e-6
    
    norm_dist_dev = avg_distance_dev / dist_dev_thresh_for_norm
    norm_angle_dev = avg_angle_dev_deg / angle_dev_thresh_for_norm_deg
    
    # Combine: could be sum, max, or weighted sum. Sum is simple.
    badness_score = norm_dist_dev + norm_angle_dev
    return badness_score

# Remove or comment out the old boolean coherency checks:
# is_segment_normals_coherent_pca
# is_segment_planar_by_fit


def refine_segment(tri_mesh, segment_orig_face_indices, params, original_clustering_eps, original_clustering_min_samples, recursion_depth=0):
    if len(segment_orig_face_indices) == 0 or recursion_depth > params.get("refinement_max_recursion_depth", 3):
        return [seg.copy() for seg in ([segment_orig_face_indices] if len(segment_orig_face_indices) > 0 else [])]

    # 1. Connectivity
    components = get_segment_connected_components(tri_mesh, segment_orig_face_indices)
    if len(components) > 1:
        all_refined_sub_components = []
        for comp_faces in components:
            all_refined_sub_components.extend(
                refine_segment(tri_mesh, comp_faces, params, original_clustering_eps, original_clustering_min_samples, recursion_depth + 1)
            )
        return all_refined_sub_components
    
    current_segment_faces = components[0] 

    # Early exit for segments too small to be meaningfully checked or split further by re-clustering
    # This threshold should be related to the min_samples used in split_segment_by_recluster
    strict_min_samples_for_split = max(3, int(original_clustering_min_samples * params.get("refinement_split_min_samples_factor", 0.9))) # from split_segment_by_recluster
    if len(current_segment_faces) < strict_min_samples_for_split * 2 : # Need at least 2x min_samples to form two clusters
         return [current_segment_faces.copy()]


    # --- "High-Confidence Keep" for large, reasonably planar segments (USING BADNESS SCORES) ---
    is_kept_by_dominant_logic = False
    min_faces_for_dominant_check = params.get("refinement_dominant_keep_min_faces", 500) 
    
    if len(current_segment_faces) >= min_faces_for_dominant_check:
        # Calculate badness scores using specific "dominant keep" thresholds for *judging* these scores
        # The score calculation functions themselves use general params for normalization.
        # Here, we compare the returned scores against "dominant keep" *decision* thresholds.

        dom_pca_badness = calculate_pca_badness_score(tri_mesh, current_segment_faces, params)
        dom_planar_badness = calculate_planar_badness_score(tri_mesh, current_segment_faces, params)

        # Decision thresholds for keeping dominant segments (higher means more tolerant)
        dominant_pca_decision_thresh = params.get("refinement_dominant_keep_pca_badness_thresh", 0.35) # e.g. 1-0.65
        dominant_planar_decision_thresh = params.get("refinement_dominant_keep_planar_badness_thresh", 2.5) 

        if dom_pca_badness <= dominant_pca_decision_thresh and \
           dom_planar_badness <= dominant_planar_decision_thresh:
            is_kept_by_dominant_logic = True
            # print(f"    Segment ({len(current_segment_faces)} faces) KEPT by dominant logic. PCA Badness: {dom_pca_badness:.2f}, Planar Badness: {dom_planar_badness:.2f}")
            return [current_segment_faces.copy()]

    # --- Standard Splitting Decision based on Badness Scores ---
    # (Only if not kept by dominant logic)
    
    # Calculate badness scores for standard check (functions are the same, decision thresholds differ)
    pca_badness = calculate_pca_badness_score(tri_mesh, current_segment_faces, params)
    planar_badness = calculate_planar_badness_score(tri_mesh, current_segment_faces, params)
    
    # Fetch decision thresholds for splitting
    pca_split_decision_thresh = params.get("refinement_pca_split_badness_thresh", 0.30) # e.g. 1-0.70 (stricter than dominant)
    planar_split_decision_thresh = params.get("refinement_planar_split_badness_thresh", 2.0) # (stricter than dominant)

    # Check if roughness homogeneity check is enabled
    split_due_to_roughness = False
    if params.get("refinement_check_roughness_homogeneity", False):
        # Assuming is_segment_roughness_homogeneous returns True if homogeneous (good), False if not (bad)
        if not is_segment_roughness_homogeneous(tri_mesh, current_segment_faces, params):
            split_due_to_roughness = True
            # print(f"    Segment ({len(current_segment_faces)}) fails roughness homogeneity. Will attempt split.")

    should_split_due_to_geometry = False
    if pca_badness > pca_split_decision_thresh:
        should_split_due_to_geometry = True
        # print(f"    Segment ({len(current_segment_faces)}) fails PCA badness for split: {pca_badness:.2f} > {pca_split_decision_thresh:.2f}")
    if planar_badness > planar_split_decision_thresh:
        should_split_due_to_geometry = True
        # print(f"    Segment ({len(current_segment_faces)}) fails Planar badness for split: {planar_badness:.2f} > {planar_split_decision_thresh:.2f}")
    
    if not should_split_due_to_geometry and not split_due_to_roughness:
        # print(f"    Segment ({len(current_segment_faces)}) deemed coherent enough by badness scores. Keeping.")
        return [current_segment_faces.copy()]
    else:
        # print(f"    Segment ({len(current_segment_faces)}) will be split. PCA Badness: {pca_badness:.2f}, Planar Badness: {planar_badness:.2f}")
        sub_segments = split_segment_by_recluster(tri_mesh, current_segment_faces, params, 
                                                  original_clustering_eps, original_clustering_min_samples)
        
        if len(sub_segments) == 1 and len(sub_segments[0]) == len(current_segment_faces):
            return [current_segment_faces.copy()] # Splitting didn't change anything

        all_re_refined_sub_segments = []
        for sub_seg_faces in sub_segments:
            all_re_refined_sub_segments.extend(
                refine_segment(tri_mesh, sub_seg_faces, params, original_clustering_eps, original_clustering_min_samples, recursion_depth + 1)
            )
        return all_re_refined_sub_segments if all_re_refined_sub_segments else [current_segment_faces.copy()]

    
def calculate_cluster_curvature(mesh, face_cluster):
    """
    Alternative roughness measure using curvature estimation
    """
    if len(face_cluster) == 0:
        return 0.0
        
    # Get face normals for the cluster
    cluster_normals = mesh.face_normals[face_cluster]
    
    # Calculate the average normal direction
    avg_normal = np.mean(cluster_normals, axis=0)
    if np.linalg.norm(avg_normal) > 1e-10:
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
    
    # Calculate deviation from average normal (curvature approximation)
    deviations = np.array([
        np.arccos(np.clip(np.dot(n, avg_normal), -1.0, 1.0))
        for n in cluster_normals
    ])
    
    # We can use different statistics here:
    # - Standard deviation (variation in surface orientation)
    # - Mean (average curvature)
    # - Max (maximum curvature)
    curvature = np.std(deviations)
    return curvature

def identify_fracture_surface_by_normals(tri_mesh_fragment, params):
    """
    Identifies fracture surface candidates using normal-based clustering and roughness metrics.
    Args:
        tri_mesh_fragment (trimesh.Trimesh): The input fragment.
        params (dict): Configuration parameters.
    Returns:
        np.ndarray: Boolean mask of faces identified as fracture candidates.
    """
    # Get parameters or use defaults
    normal_cluster_eps = params.get('normal_cluster_eps', 0.15)  # DBSCAN eps for normal clustering
    normal_cluster_min_samples = params.get('normal_cluster_min_samples', 5)  # Min faces per cluster
    roughness_threshold = params.get('roughness_threshold', 0.2)  # Threshold for fracture surface
    
    # Ensure normals are calculated
    if not hasattr(tri_mesh_fragment, 'face_normals') or tri_mesh_fragment.face_normals is None:
        tri_mesh_fragment.face_normals = None  # Reset to force recalculation
        tri_mesh_fragment.generate_face_normals()
    
    print(f"    Segmenter: Clustering faces by normal similarity...")
    face_clusters = cluster_faces_by_normal(
        tri_mesh_fragment, 
        eps=normal_cluster_eps, 
        min_samples=normal_cluster_min_samples
    )
    print(f"    Segmenter: Found {len(face_clusters)} face clusters.")
    
    # Calculate roughness for each cluster and identify fracture surfaces
    face_is_fracture_candidate = np.zeros(len(tri_mesh_fragment.faces), dtype=bool)
    
    print(f"    Segmenter: Calculating roughness for each cluster...")
    cluster_roughness = []
    
    # Use simpler curvature calculation instead of face roughness
    for cluster_idx, face_cluster in enumerate(face_clusters):
        # Use curvature calculation which is simpler and more robust
        roughness = calculate_cluster_curvature(tri_mesh_fragment, face_cluster)
        cluster_roughness.append(roughness)
        print(f"    Segmenter: Cluster {cluster_idx+1} has {len(face_cluster)} faces with roughness {roughness:.4f}")
        
        # Mark faces as fracture candidates if roughness exceeds threshold
        if roughness > roughness_threshold:
            face_is_fracture_candidate[face_cluster] = True
    
    num_candidate_faces = np.sum(face_is_fracture_candidate)
    print(f"    Segmenter: Identified {num_candidate_faces} fracture candidate faces based on roughness.")
    
    # If no faces meet the roughness threshold, use the cluster with highest roughness
    if num_candidate_faces == 0 and face_clusters:
        max_roughness_idx = np.argmax(cluster_roughness)
        face_is_fracture_candidate[face_clusters[max_roughness_idx]] = True
        num_candidate_faces = len(face_clusters[max_roughness_idx])
        print(f"    Segmenter: No clusters exceeded roughness threshold. Using cluster with highest roughness ({cluster_roughness[max_roughness_idx]:.4f}) with {num_candidate_faces} faces.")
        
    # Additionally, we can still consider boundary edges as in the original method
    # This combines both approaches for better robustness
    if params.get('use_boundary_edge_detection', True):
        # Original boundary edge detection
        boundary_faces = identify_fracture_candidate_faces_by_boundary(
            tri_mesh_fragment, 
            params.get("min_boundary_edges_for_fracture_face", 1)
        )
        
        # Combine both approaches
        combined_candidates = np.logical_or(face_is_fracture_candidate, boundary_faces)
        added_faces = np.sum(combined_candidates) - num_candidate_faces
        if added_faces > 0:
            print(f"    Segmenter: Added {added_faces} faces from boundary edge detection.")
            face_is_fracture_candidate = combined_candidates
    
    return face_is_fracture_candidate

def identify_fracture_candidate_faces_by_boundary(tri_mesh_fragment, min_boundary_edges_for_fracture_face=1):
    """
    Original method: Identifies faces that are likely part of a fracture surface based on boundary edges.
    Renamed from identify_fracture_candidate_faces to distinguish it from the new approach.
    """
    if not tri_mesh_fragment.is_watertight:
        boundary_edges_unique = tri_mesh_fragment.edges[trimesh.grouping.group_rows(tri_mesh_fragment.edges_sorted, require_count=1)]
    else:
        print(f"    Segmenter: Mesh {tri_mesh_fragment.metadata.get('name', 'Unnamed')} is watertight. Assuming no open fracture surfaces.")
        return np.zeros(len(tri_mesh_fragment.faces), dtype=bool)

    if len(boundary_edges_unique) == 0:
        print(f"    Segmenter: No boundary edges found for {tri_mesh_fragment.metadata.get('name', 'Unnamed')}. Might be watertight or an issue.")
        return np.zeros(len(tri_mesh_fragment.faces), dtype=bool)

    boundary_edge_set = set()
    for edge in boundary_edges_unique:
        boundary_edge_set.add(tuple(sorted(edge)))

    face_is_fracture_candidate = np.zeros(len(tri_mesh_fragment.faces), dtype=bool)

    for face_idx, face_vertices in enumerate(tri_mesh_fragment.faces):
        boundary_edge_count = 0
        face_edges = [
            tuple(sorted((face_vertices[0], face_vertices[1]))),
            tuple(sorted((face_vertices[1], face_vertices[2]))),
            tuple(sorted((face_vertices[2], face_vertices[0])))
        ]
        for edge in face_edges:
            if edge in boundary_edge_set:
                boundary_edge_count += 1
        
        if boundary_edge_count >= min_boundary_edges_for_fracture_face:
            face_is_fracture_candidate[face_idx] = True
            
    num_candidate_faces = np.sum(face_is_fracture_candidate)
    if num_candidate_faces == 0 :
        print(f"    Segmenter: No fracture candidate faces found by boundary method for {tri_mesh_fragment.metadata.get('name', 'Unnamed')}.")
    else:
        print(f"    Segmenter: Boundary method identified {num_candidate_faces} fracture candidate faces for {tri_mesh_fragment.metadata.get('name', 'Unnamed')}.")
        
    return face_is_fracture_candidate

# Keep the original function name but update it to use our new approach
def identify_fracture_candidate_faces(tri_mesh_fragment, params=None):
    """
    Main function to identify fracture surface candidates.
    Calls either normal-based or boundary-based method based on params.
    """
    if params is None:
        params = {"min_boundary_edges_for_fracture_face": 1}
    
    # Use normal-based approach if enabled
    if params.get('use_normal_based_segmentation', True):
        return identify_fracture_surface_by_normals(tri_mesh_fragment, params)
    else:
        # Fall back to original boundary edge method
        return identify_fracture_candidate_faces_by_boundary(
            tri_mesh_fragment, 
            params.get("min_boundary_edges_for_fracture_face", 1)
        )

def visualize_segmentation_clusters(o3d_mesh, clusters, tri_mesh, fragment_name="Unnamed"):
    """
    Visualizes all clusters with different colors to help identify distinct faces.
    """
    vis_geometries = []
    
    # Original mesh as wireframe for reference
    original_mesh_vis = copy.deepcopy(o3d_mesh)
    original_mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
    original_mesh_vis.compute_vertex_normals()
    
    # Add wireframe of original for better visibility
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(original_mesh_vis)
    edges.paint_uniform_color([0.5, 0.5, 0.5])  # Darker gray for edges
    vis_geometries.append(edges)
    
    # Create a mesh for each cluster with a different color
    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
    ]
    
    # Filter to show only larger clusters (to avoid visual clutter)
    min_cluster_size = 100  # Only show clusters with at least this many faces
    
    for i, cluster in enumerate(clusters):
        if len(cluster) < min_cluster_size:
            continue
            
        color = colors[i % len(colors)]
        cluster_mesh = o3d.geometry.TriangleMesh()
        cluster_mesh.vertices = o3d_mesh.vertices
        cluster_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces[cluster])
        cluster_mesh.remove_unreferenced_vertices()
        cluster_mesh.compute_vertex_normals()
        cluster_mesh.paint_uniform_color(color)
        vis_geometries.append(cluster_mesh)
    
    return vis_geometries

def extract_fracture_surface_mesh(o3d_mesh_fragment, fragment_name="Unnamed", params=None):
    params = params or {}
    if not o3d_mesh_fragment.has_triangles() or not o3d_mesh_fragment.has_vertices():
        print(f"    Segmenter: Input mesh {fragment_name} has no triangles/vertices.")
        return None
    try:
        tri_mesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh_fragment.vertices),
                                   faces=np.asarray(o3d_mesh_fragment.triangles),
                                   vertex_normals=np.asarray(o3d_mesh_fragment.vertex_normals) if o3d_mesh_fragment.has_vertex_normals() else None,
                                   process=False) # process=False to avoid auto-fixing, do it manually if needed
        tri_mesh.metadata['name'] = fragment_name
        # Ensure face normals are present and correct count
        if tri_mesh.faces is None or len(tri_mesh.faces) == 0:
             print(f"    Segmenter: Mesh {fragment_name} has no faces after trimesh conversion.")
             return None
        if tri_mesh.face_normals is None or len(tri_mesh.face_normals) != len(tri_mesh.faces):
            tri_mesh.refresh() # This calls process() internally, recalculates normals etc.
            if tri_mesh.face_normals is None or len(tri_mesh.face_normals) != len(tri_mesh.faces):
                 print(f"    Segmenter: Critical error, face normal count mismatch for {fragment_name} after refresh.")
                 return None

    except Exception as e:
        print(f"    Segmenter: Error converting/processing O3D mesh {fragment_name} to Trimesh: {e}")
        return None
    
    total_faces = len(tri_mesh.faces)

    initial_eps = params.get('normal_cluster_eps', 0.1)
    initial_min_samples = params.get('normal_cluster_min_samples', 10)

    print(f"    Segmenter [{fragment_name}]: Initial clustering ({total_faces} faces) with eps={initial_eps:.2f}, min_samples={initial_min_samples}...")
    initial_clusters_list, initial_labels_for_all_faces = cluster_faces_by_normal(
        tri_mesh, initial_eps, initial_min_samples, face_subset_indices=None
    )
    print(f"    Segmenter [{fragment_name}]: Found {len(initial_clusters_list)} initial clusters.")

    final_segments_for_selection = []
    enable_refinement = params.get("segment_refinement_enabled", True)
    refinement_abs_thresh = params.get("refinement_min_face_count_absolute", 200)
    refinement_perc_thresh = params.get("refinement_min_face_percentage", 0.10)
    
    # Handle noise cluster (label -1 from DBSCAN) explicitly if refinement is enabled
    noise_cluster_orig_indices = None
    temp_initial_clusters_list = [] # To hold non-noise clusters for the main loop

    if -1 in initial_labels_for_all_faces:
        noise_mask = (initial_labels_for_all_faces == -1)
        # initial_labels_for_all_faces corresponds to faces 0..N-1
        noise_cluster_orig_indices = np.where(noise_mask)[0]
        if len(noise_cluster_orig_indices) > 0:
            print(f"    Segmenter [{fragment_name}]: Initial noise cluster has {len(noise_cluster_orig_indices)} faces.")
            if enable_refinement:
                print(f"      Refining noise cluster for {fragment_name}...")
                refined_noise_subs = refine_segment(tri_mesh, noise_cluster_orig_indices, params, initial_eps, initial_min_samples)
                final_segments_for_selection.extend(sub for sub in refined_noise_subs if len(sub) > 0)
            else: # Not refining noise, add it as one segment if large enough
                if len(noise_cluster_orig_indices) >= initial_min_samples:
                     final_segments_for_selection.append(noise_cluster_orig_indices)
    
    # Process non-noise initial clusters
    for i, cluster_orig_indices in enumerate(initial_clusters_list):
        if len(cluster_orig_indices) == 0: continue
        # Check if this cluster corresponds to the noise label
        # We need the label of the first face to determine if it was the noise cluster
        # initial_labels_for_all_faces are labels for faces 0..N-1
        # cluster_orig_indices contains original face indices.
        first_face_label = initial_labels_for_all_faces[cluster_orig_indices[0]]
        
        if first_face_label == -1: # This was the noise cluster, already handled if refinement enabled
            if not enable_refinement and noise_cluster_orig_indices is not None: # It was handled above even if not refined
                pass # Already added if not refined and large enough
            continue # Skip further processing of the raw noise cluster

        segment_size = len(cluster_orig_indices)
        
        refine_this_segment = False
        if enable_refinement:
            if segment_size > refinement_abs_thresh or \
               (total_faces > 0 and (segment_size / total_faces) > refinement_perc_thresh):
                refine_this_segment = True
        
        if refine_this_segment:
            # print(f"      Refining cluster {i+1} ({segment_size} faces) for {fragment_name}...")
            refined_subs = refine_segment(tri_mesh, cluster_orig_indices, params, initial_eps, initial_min_samples)
            final_segments_for_selection.extend(sub for sub in refined_subs if len(sub) > 0)
        else:
            final_segments_for_selection.append(cluster_orig_indices) # Keep as is

    # Filter out very small final segments
    min_final_segment_size = max(3, params.get("refinement_min_final_segment_size", initial_min_samples // 2))
    final_segments_for_selection = [seg for seg in final_segments_for_selection if len(seg) >= min_final_segment_size]
    # Sort by size for consistent display
    final_segments_for_selection.sort(key=len, reverse=True)

    print(f"    Segmenter [{fragment_name}]: Final processing resulted in {len(final_segments_for_selection)} segments for user selection.")
    # for i_s, seg_s in enumerate(final_segments_for_selection): print(f"      Final Segment {i_s+1}: {len(seg_s)} faces")


    # --- User Interaction Part --- (largely same, operates on final_segments_for_selection)
    face_is_fracture_candidate = np.zeros(total_faces, dtype=bool)
    if params.get('visualize_segmentation', False) and final_segments_for_selection:
        print(f"    Segmenter [{fragment_name}]: Visualizing {len(final_segments_for_selection)} refined segments for selection...")
        vis_geometries = visualize_segmentation_clusters(o3d_mesh_fragment, final_segments_for_selection, tri_mesh, fragment_name)
        if vis_geometries:
             o3d.visualization.draw_geometries(vis_geometries, window_name=f"Refined Segments: {fragment_name} - Select Fracture Surfaces")
        else:
            print(f"    Segmenter [{fragment_name}]: No geometries to visualize for cluster selection.")

        print("\n=== Fracture Surface Selection (on Refined Segments) ===")
        for i_prompt, seg_prompt in enumerate(final_segments_for_selection):
            print(f"  Segment {i_prompt + 1}: {len(seg_prompt)} faces")
        cluster_selection_str = input(f"Enter segment numbers (1 to {len(final_segments_for_selection)}) for '{fragment_name}' (comma separated, or 'all', or 'none'): ")
        
        selected_indices_from_user = []
        if cluster_selection_str.lower() == 'all':
            selected_indices_from_user = list(range(len(final_segments_for_selection)))
        elif cluster_selection_str.lower() == 'none' or not cluster_selection_str.strip():
            selected_indices_from_user = []
        else:
            try:
                selected_indices_from_user = [int(x.strip()) - 1 for x in cluster_selection_str.split(',') if x.strip()]
            except ValueError:
                print("    Invalid input. Defaulting to 'none'.")
        
        for user_idx in selected_indices_from_user:
            if 0 <= user_idx < len(final_segments_for_selection):
                face_is_fracture_candidate[final_segments_for_selection[user_idx]] = True
        print(f"    User selected {np.sum(face_is_fracture_candidate)} faces as fracture surface for {fragment_name}.")

    else: # No visualization or no segments to select from -> automatic fallback
        print(f"    Segmenter [{fragment_name}]: No interactive selection. Using automatic boundary method for fracture faces.")
        face_is_fracture_candidate = identify_fracture_candidate_faces_by_boundary(
            tri_mesh, params.get("min_boundary_edges_for_fracture_face",1)
        )
        if not np.any(face_is_fracture_candidate) and final_segments_for_selection: # If boundary fails, maybe take largest non-planar segment?
            print(f"    Segmenter [{fragment_name}]: Boundary method found no faces. (Placeholder for smarter auto fallback).")


    if not np.any(face_is_fracture_candidate):
        print(f"    Segmenter [{fragment_name}]: No fracture faces identified. Returning None.")
        return None

    fracture_faces = tri_mesh.faces[face_is_fracture_candidate]
    # ... (rest of creating o3d_mesh from fracture_faces is same) ...
    fracture_surface_o3d = o3d.geometry.TriangleMesh()
    fracture_surface_o3d.vertices = o3d_mesh_fragment.vertices
    fracture_surface_o3d.triangles = o3d.utility.Vector3iVector(fracture_faces)
    fracture_surface_o3d.remove_unreferenced_vertices()
    fracture_surface_o3d.remove_degenerate_triangles()

    if not fracture_surface_o3d.has_triangles():
        print(f"    Segmenter [{fragment_name}]: Extracted fracture surface has no triangles after cleaning.")
        return None
        
    fracture_surface_o3d.compute_vertex_normals()
    print(f"    Segmenter [{fragment_name}]: Extracted fracture surface with {len(fracture_surface_o3d.vertices)}V, {len(fracture_surface_o3d.triangles)}T.")
    return fracture_surface_o3d

def visualize_segmentation(o3d_mesh, fracture_surface, fragment_name="Unnamed"):
    """
    Creates a visualization of the original mesh and the extracted fracture surface.
    Returns a list of Open3D geometries ready for visualization.
    """
    vis_geometries = []
    
    # Original mesh - make it semi-transparent gray
    original_mesh_vis = copy.deepcopy(o3d_mesh)
    original_mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
    original_mesh_vis.compute_vertex_normals()
    
    # Create a material for semi-transparency
    # Note: Direct material manipulation isn't available in pure Open3D Python API
    # We'll use a different approach - create wireframe to show structure
    
    vis_geometries.append(original_mesh_vis)
    
    # Add wireframe of original for better visibility
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(original_mesh_vis)
    edges.paint_uniform_color([0.5, 0.5, 0.5])  # Darker gray for edges
    vis_geometries.append(edges)
    
    # Fracture surface - bright color
    if fracture_surface and fracture_surface.has_triangles():
        fracture_surface_vis = copy.deepcopy(fracture_surface)
        fracture_surface_vis.paint_uniform_color([1.0, 0.0, 0.0])  # Red for fracture
        fracture_surface_vis.compute_vertex_normals()
        vis_geometries.append(fracture_surface_vis)
    
    return vis_geometries

if __name__ == '__main__':
    # Test code for normal-based segmentation
    # Create a simple example mesh that has both flat and rough areas
    test_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    
    # Add some noise to one face to make it "rough"
    vertices = np.asarray(test_mesh.vertices)
    triangles = np.asarray(test_mesh.triangles)
    
    # Find vertices on the top face (z=1)
    top_vertices_mask = np.isclose(vertices[:, 2], 1.0)
    top_vertices_indices = np.where(top_vertices_mask)[0]
    
    # Add random noise to these vertices
    noise_scale = 0.05
    for idx in top_vertices_indices:
        vertices[idx, 2] += np.random.uniform(-noise_scale, noise_scale)
    
    # Update the mesh
    test_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    test_mesh.compute_vertex_normals()
    
    # Convert to trimesh for segmentation
    tri_mesh = trimesh.Trimesh(vertices=np.asarray(test_mesh.vertices),
                              faces=np.asarray(test_mesh.triangles))
    tri_mesh.metadata['name'] = "TestCube"
    
    # Set up parameters
    test_params = {
        "normal_cluster_eps": 0.2,
        "normal_cluster_min_samples": 3,
        "roughness_threshold": 0.1,
        "use_normal_based_segmentation": True,
        "use_boundary_edge_detection": False
    }
    
    # Identify fracture surface
    fracture_face_mask = identify_fracture_candidate_faces(tri_mesh, test_params)
    
    # Create fracture surface mesh
    fracture_faces = tri_mesh.faces[fracture_face_mask]
    fracture_surface = o3d.geometry.TriangleMesh()
    fracture_surface.vertices = test_mesh.vertices
    fracture_surface.triangles = o3d.utility.Vector3iVector(fracture_faces)
    fracture_surface.remove_unreferenced_vertices()
    fracture_surface.compute_vertex_normals()
    
    # Visualize
    vis_geometries = visualize_segmentation(test_mesh, fracture_surface, "TestCube")
    o3d.visualization.draw_geometries(vis_geometries, window_name="Normal-based Segmentation Test")