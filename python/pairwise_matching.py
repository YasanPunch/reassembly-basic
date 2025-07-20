import open3d as o3d
import numpy as np
import random
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import copy
from collections import deque
from scipy.spatial import cKDTree

"""
3D Fragment Reassembly - Pairwise Matching Algorithm

This module implements a complete pairwise matching algorithm for 3D fragment reassembly
based on the following steps:

1. Boundary Curve Similarity (Λ<0.3) for fast filtering:
   - Smooth curves with Gaussian filter (w=6, σ=2)
   - Compute similarity via curvature/torsion (Eq. 3)

2. For candidate pairs:
   - Extract concave-convex patches (ε_K=ε_H=0.005)
   - Represent patches by μ, σ, S, A (Eq. 4-7)

3. Apply modified ICP:
   - Match points only on similar patches (Eq. 10)
   - Reject outliers via distance/normal/curvature thresholds

4. Validate if overlap area ≥ 20% (Eq. 11)

The algorithm provides a GUI interface for:
- Loading and processing point clouds
- Extracting boundary curves and patches
- Running pairwise matching with adjustable parameters
- Visualizing results and saving matched fragments

Usage:
    python pairwise-matching.py
"""

def voxel_downsample(point_cloud, voxel_size=2.0):
    return point_cloud.voxel_down_sample(voxel_size=voxel_size)

def region_growing(point_cloud, k_neighbors=30, normal_threshold=0.95, min_cluster_size=10):
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k_neighbors)
    )
    point_cloud.orient_normals_to_align_with_direction()

    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    n_points = len(points)
    print(f"Number of points: {n_points}")

    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    unvisited = set(range(n_points))
    clusters = []
    visited = [False] * n_points

    while unvisited:
        seed_index = unvisited.pop()
        if visited[seed_index]:
            continue
        current_cluster = [seed_index]
        visited[seed_index] = True
        unvisited_queue = [seed_index]

        while unvisited_queue:
            growing_index = unvisited_queue.pop(0)
            seed_point = points[growing_index]
            seed_normal = normals[growing_index]
            [k, neighbor_indices, _] = pcd_tree.search_radius_vector_3d(seed_point, radius=20)

            for neighbor_index in neighbor_indices:
                if not visited[neighbor_index]:
                    neighbor_normal = normals[neighbor_index]
                    similarity = np.dot(seed_normal, neighbor_normal)
                    if similarity > normal_threshold:
                        visited[neighbor_index] = True
                        unvisited.discard(neighbor_index)
                        current_cluster.append(neighbor_index)
                        unvisited_queue.append(neighbor_index)

        if len(current_cluster) >= min_cluster_size:
            clusters.append(current_cluster)

    print(f"Found {len(clusters)} clusters.")
    return clusters

def visualize_clusters(point_cloud, clusters):
    print(f"Visualizing {len(clusters)} clusters...")
    points = np.asarray(point_cloud.points)
    n_points = len(points)
    cluster_colors = [[random.random(), random.random(), random.random()] for _ in range(len(clusters))]
    colors = [[0, 0, 0]] * n_points
    for i, cluster_indices in enumerate(clusters):
        for point_index in cluster_indices:
            colors[point_index] = cluster_colors[i]

    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])

# ------------- ✅ NEW: Boundary Curve Extraction ----------------

def extract_pointcloud_boundaries(point_cloud, clusters, curvature_threshold=0.01, neighbor_radius=4):

    print("Extracting point cloud-based fracture boundaries with continuity...")
    all_linesets = []

    for cluster_indices in clusters:
        cluster_pcd = point_cloud.select_by_index(cluster_indices)
        if len(cluster_pcd.points) < 50:
            continue

        cluster_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
        )

        # Compute curvature
        points = np.asarray(cluster_pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(cluster_pcd)
        boundary_points = []

        for i in range(len(points)):
            [_, idx, _] = kdtree.search_radius_vector_3d(cluster_pcd.points[i], neighbor_radius)
            if len(idx) < 5:
                continue
            neighbors = np.asarray(cluster_pcd.points)[idx, :]
            cov = np.cov(neighbors.T)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.sort(eigvals)
            curvature = eigvals[0] / np.sum(eigvals)
            if curvature > curvature_threshold:
                boundary_points.append(cluster_pcd.points[i])

        if len(boundary_points) > 1:
            # Build an ordered line set including all points
            boundary_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(boundary_points))
            points_arr = np.asarray(boundary_pcd.points)
            visited = np.zeros(len(points_arr), dtype=bool)
            kdtree = o3d.geometry.KDTreeFlann(boundary_pcd)

            ordered_lines = []

            while not np.all(visited):
                # Start from an unvisited point
                unvisited_indices = np.where(visited == False)[0]
                current_idx = unvisited_indices[0]
                visited[current_idx] = True
                chain = [current_idx]

                for _ in range(len(points_arr) - 1):
                    [_, idxs, _] = kdtree.search_knn_vector_3d(points_arr[current_idx], 10)
                    found = False
                    for next_idx in idxs[1:]:  # Skip self
                        if not visited[next_idx]:
                            visited[next_idx] = True
                            ordered_lines.append([current_idx, next_idx])
                            current_idx = next_idx
                            chain.append(current_idx)
                            found = True
                            break
                    if not found:
                        break  # Start new segment

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points_arr),
                lines=o3d.utility.Vector2iVector(ordered_lines)
            )
            line_set.paint_uniform_color([1, 0, 0])  # Red lines
            all_linesets.append(line_set)

    return all_linesets



def extract_concave_convex_patches_with_labels(point_cloud, K_thresh=0.0001, H_thresh=0.0001, neighbor_radius=5, min_neighbors=6, min_cluster_size=20):
    print("Extracting and clustering concave and convex patches...")
    print(f"Using parameters: min_neighbors={min_neighbors}, min_cluster_size={min_cluster_size}")

    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )

    points = np.asarray(point_cloud.points)
    n_points = len(points)
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    labels = np.full(n_points, fill_value=-1)  # -1 = unclassified

    # 1. Curvature-based classification (first pass: collect H and K)
    all_H = []
    all_K = []
    H_per_point = np.zeros(n_points)
    K_per_point = np.zeros(n_points)
    for i in range(n_points):
        [_, idx, _] = kdtree.search_radius_vector_3d(point_cloud.points[i], neighbor_radius)
        if len(idx) < min_neighbors:
            all_H.append(0)
            all_K.append(0)
            continue
        neighbors = points[idx]
        cov = np.cov(neighbors.T)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)[::-1]
        k1, k2 = eigvals[0], eigvals[1]

        H = (k1 + k2) / 2
        K = k1 * k2
        all_H.append(H)
        all_K.append(K)
        H_per_point[i] = H
        K_per_point[i] = K

    mean_H = np.mean([h for h in all_H if h != 0])
    print(f'H min: {np.min(all_H)}, max: {np.max(all_H)}, mean: {mean_H}')
    print(f'K min: {np.min(all_K)}, max: {np.max(all_K)}, mean: {np.mean(all_K)}')

    # 2. Classification: below mean H = concave, above mean H = convex (if K > K_thresh)
    for i in range(n_points):
        if K_per_point[i] > K_thresh:
            if H_per_point[i] > mean_H:
                labels[i] = 1  # convex
            elif H_per_point[i] < mean_H:
                labels[i] = 0  # concave

    # 3. Patch clustering using region growing
    def cluster_type(target_type):
        clustered = []
        visited = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            if labels[i] != target_type or visited[i]:
                continue
            cluster = []
            queue = [i]
            visited[i] = True

            while queue:
                current = queue.pop(0)
                cluster.append(current)
                [_, neighbors, _] = kdtree.search_radius_vector_3d(point_cloud.points[current], neighbor_radius)
                for ni in neighbors:
                    if not visited[ni] and labels[ni] == target_type:
                        visited[ni] = True
                        queue.append(ni)

            if len(cluster) >= min_cluster_size:  # minimum patch size
                clustered.append(cluster)

        return clustered

    concave_clusters = cluster_type(0)
    convex_clusters = cluster_type(1)

    # 4. Assign distinct colors using Open3D's color utilities
    def get_distinct_colors(n):
        colors = []
        for i in range(n):
            # Generate distinct colors using HSV color space
            hue = i / n
            saturation = 0.8
            value = 0.9
            # Convert HSV to RGB
            h = hue * 6
            i = int(h)
            f = h - i
            p = value * (1 - saturation)
            q = value * (1 - saturation * f)
            t = value * (1 - saturation * (1 - f))
            
            if i == 0:
                r, g, b = value, t, p
            elif i == 1:
                r, g, b = q, value, p
            elif i == 2:
                r, g, b = p, value, t
            elif i == 3:
                r, g, b = p, q, value
            elif i == 4:
                r, g, b = t, p, value
            else:
                r, g, b = value, p, q
                
            colors.append([r, g, b])
        return colors

    colors = np.full((n_points, 3), fill_value=0.8)  # Gray background for unclassified

    concave_colors = get_distinct_colors(len(concave_clusters))
    convex_colors = get_distinct_colors(len(convex_clusters))

    patch_types = []  # 0 for concave, 1 for convex
    patch_indices = []

    for i, cluster in enumerate(concave_clusters):
        color = concave_colors[i]
        for idx in cluster:
            colors[idx] = color
        patch_types.append(0)
        patch_indices.append(cluster)

    for i, cluster in enumerate(convex_clusters):
        color = convex_colors[i]
        for idx in cluster:
            colors[idx] = color
        patch_types.append(1)
        patch_indices.append(cluster)

    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    print(f"Detected {len(concave_clusters)} concave patches and {len(convex_clusters)} convex patches.")
    return point_cloud, patch_types, patch_indices




def visualize_boundaries(point_cloud, line_sets):
    print("Visualizing boundary curves...")

    # Set all point cloud vertices to yellow
    n_points = np.asarray(point_cloud.points).shape[0]
    point_cloud.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]] * n_points)  # Yellow

    # Set all boundary line sets to black
    for line_set in line_sets:
        line_set.paint_uniform_color([0.0, 0.0, 0.0])  # Black

    # Visualize
    o3d.visualization.draw_geometries([point_cloud] + line_sets)

# ------------- ✅ NEW: Pairwise Matching Algorithm ----------------

class CurvePoint:
    def __init__(self, position, normal=None, curvature=None, torsion=None):
        self.position = np.array(position)
        self.normal = np.array(normal) if normal is not None else np.array([0, 0, 0])
        self.curvature = curvature if curvature is not None else 0.0
        self.torsion = torsion if torsion is not None else 0.0

def gaussian_smooth(points, w=6):
    """Gaussian Smoothing (Eq. 2)"""
    σ = w / 3
    kernel = [np.exp(-(j**2)/(2*σ**2)) for j in range(-w, w+1)]
    kernel = np.array(kernel)
    kernel /= np.sum(kernel)  # Normalize
    
    smoothed_points = []
    for i in range(w, len(points)-w):
        smoothed_point = np.zeros(3)
        for j in range(-w, w+1):
            smoothed_point += points[i+j] * kernel[j+w]
        smoothed_points.append(smoothed_point)
    
    return np.array(smoothed_points)

def compute_curvature_and_torsion(points, normals):
    """Compute curvature and torsion for curve points"""
    if len(points) < 3:
        return [0.0] * len(points), [0.0] * len(points)
    
    curvatures = []
    torsions = []
    
    for i in range(1, len(points)-1):
        # Compute curvature using three points
        p_prev = points[i-1]
        p_curr = points[i]
        p_next = points[i+1]
        
        # Tangent vectors
        t1 = p_curr - p_prev
        t2 = p_next - p_curr
        
        # Normalize
        t1_norm = np.linalg.norm(t1)
        t2_norm = np.linalg.norm(t2)
        
        if t1_norm > 0 and t2_norm > 0:
            t1 = t1 / t1_norm
            t2 = t2 / t2_norm
            
            # Curvature as change in tangent direction
            curvature = np.linalg.norm(t2 - t1) / 2.0
            
            # Torsion (simplified - using normal variation)
            if i < len(normals):
                normal_prev = normals[i-1] if i-1 < len(normals) else normals[i]
                normal_curr = normals[i]
                normal_next = normals[i+1] if i+1 < len(normals) else normals[i]
                
                torsion = np.linalg.norm(normal_next - normal_prev) / 2.0
            else:
                torsion = 0.0
        else:
            curvature = 0.0
            torsion = 0.0
        
        curvatures.append(curvature)
        torsions.append(torsion)
    
    # Pad with zeros for first and last points
    curvatures = [0.0] + curvatures + [0.0]
    torsions = [0.0] + torsions + [0.0]
    
    return curvatures, torsions

def curve_similarity(C1_points, C2_points, C1_curvatures, C2_curvatures, C1_torsions, C2_torsions, ε_Λ=0.3):
    """Curve Similarity (Eq. 3)"""
    Λ = np.zeros((len(C1_points), len(C2_points)))
    
    for i in range(1, len(C1_points)-1):
        for j in range(1, len(C2_points)-1):
            dist = 0
            for q in [-1, 0, 1]:  # 3-point neighborhood
                if i+q < len(C1_curvatures) and j+q < len(C2_curvatures):
                    κ_diff = C1_curvatures[i+q] - C2_curvatures[j+q]
                    τ_diff = C1_torsions[i+q] - C2_torsions[j+q]
                    dist += np.sqrt(κ_diff**2 + τ_diff**2)
            Λ[i,j] = dist / 3  # Mean Euclidean distance
    
    return Λ < ε_Λ  # Similarity mask

def patch_descriptor(patch_points, patch_normals):
    """Patch Descriptor (Eq. 4-7)"""
    if len(patch_points) == 0:
        return {"μ": 0, "σ": 0, "S": 0, "A": 0}
    
    # Compute mean curvature for each point (simplified)
    H = []
    for i, point in enumerate(patch_points):
        if i < len(patch_normals):
            # Simplified mean curvature computation
            # In practice, you'd compute this from the surface geometry
            H.append(np.linalg.norm(patch_normals[i]))
        else:
            H.append(0.0)
    
    # Curvature stats (Eq. 5-6)
    μ = np.mean(H)
    σ = np.std(H)
    
    # PCA-based features (Eq. 7)
    centroid = np.mean(patch_points, axis=0)
    cov = np.cov((patch_points - centroid).T)
    λ, _ = np.linalg.eigh(cov)  # Eigenvalues λ₀ ≥ λ₁ ≥ λ₂
    λ_sorted = np.sort(λ)[::-1]  # Descending order
    
    # Handle zero eigenvalues
    λ_sorted = np.maximum(λ_sorted, 1e-10)
    
    S = np.sqrt(np.sum(λ_sorted))       # Size signature (Eq. 7)
    A = np.sqrt(np.abs(λ_sorted[1]/λ_sorted[2]))  # Anisotropy (Eq. 7)
    
    return {"μ": μ, "σ": σ, "S": S, "A": A}

def patches_similar(R1, R2, ε_μ=0.5, ε_σ=0.4, ε_a=0.3, ε_s=0.3):
    """Patch Similarity (Eq. 10)"""
    # Handle division by zero
    if R1["S"] + R2["S"] == 0:
        SS = 0
    else:
        SS = np.abs(R1["S"] - R2["S"]) / (R1["S"] + R2["S"])  # Size similarity (Eq. 8)
    
    if R1["A"] + R2["A"] == 0:
        AS = 0
    else:
        AS = np.abs(R1["A"] - R2["A"]) / (R1["A"] + R2["A"])  # Anisotropy similarity (Eq. 9)
    
    return (
        np.abs(R1["μ"] - R2["μ"]) <= ε_μ and
        np.abs(R1["σ"] - R2["σ"]) <= ε_σ and
        AS <= ε_a and
        SS <= ε_s
    )

def patches_similar_with_normals(R1, R2, patch1_normals, patch2_normals, ε_μ=0.5, ε_σ=0.4, ε_a=0.3, ε_s=0.3):
    # First check geometric similarity
    if not patches_similar(R1, R2, ε_μ, ε_σ, ε_a, ε_s):
        return False
    
    # Then check normal directions are roughly opposite (complementary surfaces)
    mean_normal1 = np.mean(patch1_normals, axis=0)
    mean_normal2 = np.mean(patch2_normals, axis=0)
    
    # Normalize
    mean_normal1 = mean_normal1 / np.linalg.norm(mean_normal1)
    mean_normal2 = mean_normal2 / np.linalg.norm(mean_normal2)
    
    # Check if normals point in opposite directions (dot product ≈ -1)
    dot_product = np.dot(mean_normal1, mean_normal2)
    return dot_product < -0.5  # Allow some tolerance

def is_valid_point_pair(x_pos, y_pos, x_normal, y_normal, x_curvature, y_curvature, ε_d, ε_n, ε_c=0.3):
    """Modified ICP Constraints (Sec. IV-B)"""
    dist_ok = np.linalg.norm(x_pos - y_pos) < ε_d
    normal_ok = np.linalg.norm(x_normal - y_normal) < ε_n
    curvature_ok = np.abs(x_curvature - y_curvature) < ε_c
    return dist_ok and normal_ok and curvature_ok

def find_closest_point(query_point, point_cloud, kdtree):
    """Find closest point using KD-tree"""
    [_, idx, _] = kdtree.search_knn_vector_3d(query_point, 1)
    return idx[0]

def overlap_ratio(S1_points, S2_points, S1_normals, S2_normals, S1_curvatures, S2_curvatures, 
                 ε_dis=0.01, ε_θ=0.01, ε_curv=0.03):
    """Overlap Validation (Eq. 11)"""
    if len(S1_points) == 0:
        return False
    
    # Build KD-tree for S2
    S2_pcd = o3d.geometry.PointCloud()
    S2_pcd.points = o3d.utility.Vector3dVector(S2_points)
    kdtree = o3d.geometry.KDTreeFlann(S2_pcd)
    
    overlap_count = 0
    for i, x_pos in enumerate(S1_points):
        y_idx = find_closest_point(x_pos, S2_pcd, kdtree)
        y_pos = S2_points[y_idx]
        y_normal = S2_normals[y_idx] if y_idx < len(S2_normals) else np.array([0, 0, 0])
        y_curvature = S2_curvatures[y_idx] if y_idx < len(S2_curvatures) else 0.0
        
        x_normal = S1_normals[i] if i < len(S1_normals) else np.array([0, 0, 0])
        x_curvature = S1_curvatures[i] if i < len(S1_curvatures) else 0.0
        
        if (np.linalg.norm(x_pos - y_pos) <= ε_dis and
            np.linalg.norm(x_normal - y_normal) <= ε_θ and
            np.abs(x_curvature - y_curvature) <= ε_curv):
            overlap_count += 1
    
    return overlap_count / len(S1_points) >= 0.2  # Min 20% overlap

def extract_boundary_curves_from_linesets(line_sets, point_cloud):
    """Extract boundary curves from line sets"""
    curves = []
    
    for line_set in line_sets:
        points = np.asarray(line_set.points)
        lines = np.asarray(line_set.lines)
        
        if len(points) == 0 or len(lines) == 0:
            continue
        
        # Build curve from line segments
        curve_points = []
        curve_normals = []
        
        # Get normals from original point cloud
        original_points = np.asarray(point_cloud.points)
        original_normals = np.asarray(point_cloud.normals) if point_cloud.has_normals() else np.zeros_like(original_points)
        
        # Start with first line
        if len(lines) > 0:
            start_idx = lines[0][0]
            end_idx = lines[0][1]
            curve_points.append(points[start_idx])
            curve_points.append(points[end_idx])
            
            # Find corresponding normals in original point cloud
            start_normal = find_closest_normal(points[start_idx], original_points, original_normals)
            end_normal = find_closest_normal(points[end_idx], original_points, original_normals)
            curve_normals.append(start_normal)
            curve_normals.append(end_normal)
        
        # Continue building curve by connecting line segments
        for line in lines[1:]:
            line_start = points[line[0]]
            line_end = points[line[1]]
            
            # Check if this line connects to the current curve
            if np.allclose(line_start, curve_points[-1], atol=1e-6):
                curve_points.append(line_end)
                end_normal = find_closest_normal(line_end, original_points, original_normals)
                curve_normals.append(end_normal)
            elif np.allclose(line_end, curve_points[-1], atol=1e-6):
                curve_points.append(line_start)
                start_normal = find_closest_normal(line_start, original_points, original_normals)
                curve_normals.append(start_normal)
        
        if len(curve_points) > 2:
            curves.append((np.array(curve_points), np.array(curve_normals)))
    
    return curves

def find_closest_normal(point, original_points, original_normals):
    """Find the normal of the closest point in the original point cloud"""
    distances = np.linalg.norm(original_points - point, axis=1)
    closest_idx = np.argmin(distances)
    return original_normals[closest_idx]

def extract_patches_with_types_from_labels(point_cloud, patch_types, patch_indices):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals) if point_cloud.has_normals() else np.zeros_like(points)
    patches = []
    types = []
    for t, idxs in zip(patch_types, patch_indices):
        patch_points = points[idxs]
        patch_normals = normals[idxs]
        if len(patch_points) > 10:
            patches.append((patch_points, patch_normals))
            types.append(t)
    return patches, types

def compute_initial_transform(patch1_points, patch2_points):
    # Compute centroids
    centroid1 = np.mean(patch1_points, axis=0)
    centroid2 = np.mean(patch2_points, axis=0)
    
    # Align centroids
    init_translation = centroid2 - centroid1
    init_transform = np.eye(4)
    init_transform[:3, 3] = init_translation
    return init_transform

def enhanced_icp(pcd1, pcd2, init_transform, max_distance=0.05):
    # Use point-to-plane ICP with normal information
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, 
        max_correspondence_distance=max_distance,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    return icp_result

def validate_match(pcd1, pcd2, transform, min_overlap=0.2, distance_threshold=0.05):
    # Transform pcd1
    pcd1_transformed = copy.deepcopy(pcd1)
    pcd1_transformed.transform(transform)
    
    # Compute overlap using distance threshold
    distances = np.asarray(pcd1_transformed.compute_point_cloud_distance(pcd2))
    overlap_points = np.sum(distances < distance_threshold)
    overlap_ratio = overlap_points / len(pcd1.points)
    
    return overlap_ratio > min_overlap

def pairwise_matching_algorithm(point_cloud1, point_cloud2, line_sets1, line_sets2, 
                               patch_cloud1, patch_cloud2, patch_types1, patch_indices1, patch_types2, patch_indices2, ε_Λ=0.3, ε_K=0.005, ε_H=0.005, icp_distance=0.01):
    """Main pairwise matching algorithm with concave-convex patch matching"""
    print("Starting pairwise matching algorithm...")
    
    # Step 1: Extract and smooth boundary curves
    print("Extracting boundary curves...")
    curves1 = extract_boundary_curves_from_linesets(line_sets1, point_cloud1)
    curves2 = extract_boundary_curves_from_linesets(line_sets2, point_cloud2)
    
    print(f"Found {len(curves1)} curves in object 1, {len(curves2)} curves in object 2")
    
    # Step 2: Boundary curve similarity filtering
    candidate_pairs = []
    
    for i, (curve1_points, curve1_normals) in enumerate(curves1):
        # Smooth curve 1
        if len(curve1_points) > 12:  # Need enough points for smoothing
            smoothed_curve1 = gaussian_smooth(curve1_points, w=6)
            curve1_curvatures, curve1_torsions = compute_curvature_and_torsion(smoothed_curve1, curve1_normals)
        else:
            smoothed_curve1 = curve1_points
            curve1_curvatures, curve1_torsions = compute_curvature_and_torsion(curve1_points, curve1_normals)
        
        for j, (curve2_points, curve2_normals) in enumerate(curves2):
            # Smooth curve 2
            if len(curve2_points) > 12:
                smoothed_curve2 = gaussian_smooth(curve2_points, w=6)
                curve2_curvatures, curve2_torsions = compute_curvature_and_torsion(smoothed_curve2, curve2_normals)
            else:
                smoothed_curve2 = curve2_points
                curve2_curvatures, curve2_torsions = compute_curvature_and_torsion(curve2_points, curve2_normals)
            
            # Check similarity
            similarity_mask = curve_similarity(smoothed_curve1, smoothed_curve2, 
                                            curve1_curvatures, curve2_curvatures,
                                            curve1_torsions, curve2_torsions, ε_Λ)
            
            if np.any(similarity_mask):
                candidate_pairs.append((i, j, similarity_mask))
                print(f"Found candidate pair: curve {i} from object 1, curve {j} from object 2")
    
    print(f"Found {len(candidate_pairs)} candidate curve pairs")
    
    # Step 3: Extract patches and compute descriptors
    print("Extracting patches...")
    patches1, types1 = extract_patches_with_types_from_labels(point_cloud1, patch_types1, patch_indices1)
    patches2, types2 = extract_patches_with_types_from_labels(point_cloud2, patch_types2, patch_indices2)
    
    print(f"Found {len(patches1)} patches in object 1, {len(patches2)} patches in object 2")
    
    # Compute patch descriptors
    descriptors1 = []
    for patch_points, patch_normals in patches1:
        desc = patch_descriptor(patch_points, patch_normals)
        descriptors1.append(desc)
    
    descriptors2 = []
    for patch_points, patch_normals in patches2:
        desc = patch_descriptor(patch_points, patch_normals)
        descriptors2.append(desc)
    
    # Step 4: Find concave-convex matching patches (0: concave, 1: convex)
    matching_patches = []
    for i, (desc1, t1) in enumerate(zip(descriptors1, types1)):
        for j, (desc2, t2) in enumerate(zip(descriptors2, types2)):
            # Only match concave to convex
            if (t1 == 0 and t2 == 1) or (t1 == 1 and t2 == 0):
                patch1_points, patch1_normals = patches1[i]
                patch2_points, patch2_normals = patches2[j]
                if patches_similar_with_normals(desc1, desc2, patch1_normals, patch2_normals):
                    matching_patches.append((i, j))
                    print(f"Found concave-convex matching patches: patch {i} (type {t1}) from object 1, patch {j} (type {t2}) from object 2")
    
    print(f"Found {len(matching_patches)} concave-convex patch pairs")
    
    # Step 5: Modified ICP with patch constraints
    best_transformation = None
    best_overlap = 0.0
    
    for curve_pair in candidate_pairs:
        curve1_idx, curve2_idx, similarity_mask = curve_pair
        
        # Get corresponding patches
        curve1_points = curves1[curve1_idx][0]
        curve2_points = curves2[curve2_idx][0]
        
        # Find matching patches near these curves
        for patch_pair in matching_patches:
            patch1_idx, patch2_idx = patch_pair
            patch1_points, patch1_normals = patches1[patch1_idx]
            patch2_points, patch2_normals = patches2[patch2_idx]
            # Dummy curvature arrays (replace with real if available)
            patch1_curv = np.zeros(len(patch1_points))
            patch2_curv = np.zeros(len(patch2_points))
            try:
                # Create point clouds for ICP
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(patch1_points)
                pcd1.normals = o3d.utility.Vector3dVector(patch1_normals)
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(patch2_points)
                pcd2.normals = o3d.utility.Vector3dVector(patch2_normals)
                # Apply ICP
                icp_result = enhanced_icp(pcd1, pcd2, compute_initial_transform(patch1_points, patch2_points))
                # Transform patch1 points and normals
                patch1_points_aligned = (icp_result.transformation[:3,:3] @ patch1_points.T).T + icp_result.transformation[:3,3]
                patch1_normals_aligned = (icp_result.transformation[:3,:3] @ patch1_normals.T).T
                # Compute overlap ratio
                overlap = compute_overlap_ratio(patch1_points_aligned, patch2_points, patch1_normals_aligned, patch2_normals, patch1_curv, patch2_curv)
                print(f"Patch pair {patch_pair}: ICP fitness={icp_result.fitness:.3f}, overlap={overlap:.3f}")
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_transformation = icp_result.transformation
                    print(f"New best match found with overlap: {best_overlap:.3f}")
            except Exception as e:
                print(f"ICP/overlap failed for patch pair {patch_pair}: {e}")
                continue
    
    return best_transformation, best_overlap

# ----------------------------------------------------------------

class ReassemblyGUI:
    def __init__(self):
        # Point clouds for both objects
        self.point_clouds = [None, None]  # [object1, object2]
        self.original_point_clouds = [None, None]
        self.clusters = [None, None]
        self.line_sets = [None, None]
        self.patch_colored_clouds = [None, None]
        self.patch_types = [None, None]
        self.patch_indices = [None, None]
        self.matched_result = None  # For storing the pairwise matched result
        
        # Default parameters
        self.voxel_size = 1.0
        self.k_neighbors = 20
        self.normal_threshold = 0.90
        self.min_cluster_size = 20
        self.K_thresh = 0.001
        self.H_thresh = 0.001
        self.neighbor_radius = 5
        self.min_neighbors = 8
        self.min_patch_size = 25
        self.min_patch_size_percent = 5.0  # Default 5%
        
        # Pairwise matching parameters
        self.curve_similarity_threshold = 0.5
        self.K_patch_threshold = 0.01
        self.H_patch_threshold = 0.01
        self.icp_distance_threshold = 0.05
        self.overlap_threshold = 0.05

        # Initialize the application
        gui.Application.instance.initialize()
        
        # Create main window
        self.window = gui.Application.instance.create_window("Pairwise Matching GUI", 1920, 1080)
        
        # Create scenes for both objects and matched result
        self.scenes = [[], [], []]  # [object1_scenes, object2_scenes, matched_scenes]
        self.scene_labels = [[], [], []]
        scene_names = ["Boundary Curves", "Patches"]
        matched_names = ["Matched Result"]
        
        # Create scenes for both objects
        for obj_idx in range(2):
            for i in range(2):  # Only 2 views per object now
                # Create scene widget
                scene = gui.SceneWidget()
                scene.scene = rendering.Open3DScene(self.window.renderer)
                self.scenes[obj_idx].append(scene)
                
                # Create label for scene
                label = gui.Label(f"Object {obj_idx + 1} - {scene_names[i]}")
                label.text_color = gui.Color(1.0, 1.0, 1.0)
                self.scene_labels[obj_idx].append(label)
                
                # Add to window
                self.window.add_child(label)
                self.window.add_child(scene)
        
        # Create scene for matched result
        for i in range(1):  # One view for matched result
            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(self.window.renderer)
            self.scenes[2].append(scene)
            
            label = gui.Label(matched_names[i])
            label.text_color = gui.Color(1.0, 1.0, 1.0)
            self.scene_labels[2].append(label)
            
            self.window.add_child(label)
            self.window.add_child(scene)

        # Create control panel
        self.panel = gui.Vert(0, gui.Margins(0.25, 0.25, 0.25, 0.25))
        
        # Add file loading buttons for both objects
        for obj_idx in range(2):
            load_button = gui.Button(f"Load Object {obj_idx + 1}")
            load_button.set_on_clicked(lambda idx=obj_idx: self.load_point_cloud(idx))
            self.panel.add_child(load_button)

        # Add parameter controls
        self.panel.add_child(gui.Label("Parameters"))
        
        # Voxel size slider
        voxel_layout = gui.Horiz()
        voxel_layout.add_child(gui.Label("Voxel Size:"))
        self.voxel_slider = gui.Slider(gui.Slider.DOUBLE)
        self.voxel_slider.set_limits(0.1, 5.0)
        self.voxel_slider.double_value = self.voxel_size
        voxel_layout.add_child(self.voxel_slider)
        self.panel.add_child(voxel_layout)

        # K neighbors slider
        k_layout = gui.Horiz()
        k_layout.add_child(gui.Label("K Neighbors:"))
        self.k_slider = gui.Slider(gui.Slider.INT)
        self.k_slider.set_limits(5, 50)
        self.k_slider.int_value = self.k_neighbors
        k_layout.add_child(self.k_slider)
        self.panel.add_child(k_layout)

        # Normal threshold slider
        normal_layout = gui.Horiz()
        normal_layout.add_child(gui.Label("Normal Threshold:"))
        self.normal_slider = gui.Slider(gui.Slider.DOUBLE)
        self.normal_slider.set_limits(0.5, 1.0)
        self.normal_slider.double_value = self.normal_threshold
        normal_layout.add_child(self.normal_slider)
        self.panel.add_child(normal_layout)

        # Min cluster size slider
        cluster_layout = gui.Horiz()
        cluster_layout.add_child(gui.Label("Min Cluster Size:"))
        self.cluster_slider = gui.Slider(gui.Slider.INT)
        self.cluster_slider.set_limits(10, 200)
        self.cluster_slider.int_value = self.min_cluster_size
        cluster_layout.add_child(self.cluster_slider)
        self.panel.add_child(cluster_layout)

        # K threshold slider
        k_thresh_layout = gui.Horiz()
        k_thresh_layout.add_child(gui.Label("K Threshold:"))
        self.k_thresh_slider = gui.Slider(gui.Slider.DOUBLE)
        self.k_thresh_slider.set_limits(0.0001, 0.01)
        self.k_thresh_slider.double_value = self.K_thresh
        k_thresh_layout.add_child(self.k_thresh_slider)
        self.panel.add_child(k_thresh_layout)

        # H threshold slider
        h_thresh_layout = gui.Horiz()
        h_thresh_layout.add_child(gui.Label("H Threshold:"))
        self.h_thresh_slider = gui.Slider(gui.Slider.DOUBLE)
        self.h_thresh_slider.set_limits(0.0001, 0.01)
        self.h_thresh_slider.double_value = self.H_thresh
        h_thresh_layout.add_child(self.h_thresh_slider)
        self.panel.add_child(h_thresh_layout)

        # Add separator
        self.panel.add_child(gui.Label("Pairwise Matching Parameters"))
        
        # Curve similarity threshold slider
        curve_sim_layout = gui.Horiz()
        curve_sim_layout.add_child(gui.Label("Curve Similarity:"))
        self.curve_sim_slider = gui.Slider(gui.Slider.DOUBLE)
        self.curve_sim_slider.set_limits(0.1, 0.5)
        self.curve_sim_slider.double_value = self.curve_similarity_threshold
        curve_sim_layout.add_child(self.curve_sim_slider)
        self.panel.add_child(curve_sim_layout)

        # K patch threshold slider
        k_patch_layout = gui.Horiz()
        k_patch_layout.add_child(gui.Label("K Patch Threshold:"))
        self.k_patch_slider = gui.Slider(gui.Slider.DOUBLE)
        self.k_patch_slider.set_limits(0.001, 0.01)
        self.k_patch_slider.double_value = self.K_patch_threshold
        k_patch_layout.add_child(self.k_patch_slider)
        self.panel.add_child(k_patch_layout)

        # H patch threshold slider
        h_patch_layout = gui.Horiz()
        h_patch_layout.add_child(gui.Label("H Patch Threshold:"))
        self.h_patch_slider = gui.Slider(gui.Slider.DOUBLE)
        self.h_patch_slider.set_limits(0.001, 0.01)
        self.h_patch_slider.double_value = self.H_patch_threshold
        h_patch_layout.add_child(self.h_patch_slider)
        self.panel.add_child(h_patch_layout)

        # ICP distance threshold slider
        icp_dist_layout = gui.Horiz()
        icp_dist_layout.add_child(gui.Label("ICP Distance:"))
        self.icp_dist_slider = gui.Slider(gui.Slider.DOUBLE)
        self.icp_dist_slider.set_limits(0.001, 0.05)
        self.icp_dist_slider.double_value = self.icp_distance_threshold
        icp_dist_layout.add_child(self.icp_dist_slider)
        self.panel.add_child(icp_dist_layout)

        # Overlap threshold slider
        overlap_layout = gui.Horiz()
        overlap_layout.add_child(gui.Label("Overlap Threshold:"))
        self.overlap_slider = gui.Slider(gui.Slider.DOUBLE)
        self.overlap_slider.set_limits(0.05, 0.5)
        self.overlap_slider.double_value = self.overlap_threshold
        overlap_layout.add_child(self.overlap_slider)
        self.panel.add_child(overlap_layout)

        # Min patch size percentage slider
        min_patch_layout = gui.Horiz()
        min_patch_layout.add_child(gui.Label("Min Fracture Patch Size (%):"))
        self.min_patch_slider = gui.Slider(gui.Slider.DOUBLE)
        self.min_patch_slider.set_limits(1.0, 20.0)
        self.min_patch_slider.double_value = self.min_patch_size_percent
        self.min_patch_slider.set_on_value_changed(self.set_min_patch_size_percent)
        min_patch_layout.add_child(self.min_patch_slider)
        self.panel.add_child(min_patch_layout)

        # Progress label
        self.progress_label = gui.Label("Ready")
        self.progress_label.text_color = gui.Color(0.0, 1.0, 0.0)  # Green
        self.panel.add_child(self.progress_label)

        # Process button
        process_button = gui.Button("Process Both Objects")
        process_button.set_on_clicked(self.process_both_objects)
        self.panel.add_child(process_button)

        # Pairwise match button
        match_button = gui.Button("Pairwise Match")
        match_button.set_on_clicked(self.pairwise_match)
        self.panel.add_child(match_button)

        # Save result button
        save_button = gui.Button("Save Matched Result")
        save_button.set_on_clicked(self.save_matched_result)
        self.panel.add_child(save_button)

        # Add surface extraction button for both objects
        for obj_idx in range(2):
            extract_button = gui.Button(f"Extract Surfaces {obj_idx + 1}")
            extract_button.set_on_clicked(lambda idx=obj_idx: self.extract_surfaces(idx))
            self.panel.add_child(extract_button)

        # Add navigation buttons for pipeline steps
        nav_layout = gui.Horiz()
        self.prev_button = gui.Button("Previous Step")
        self.prev_button.set_on_clicked(self.prev_step)
        nav_layout.add_child(self.prev_button)
        self.next_button = gui.Button("Next Step")
        self.next_button.set_on_clicked(self.next_step)
        nav_layout.add_child(self.next_button)
        self.panel.add_child(nav_layout)

        # Add panel to window
        self.window.add_child(self.panel)

        # Set up layout
        self.window.set_on_layout(self._on_layout)

        self.pipeline_step = 0
        self.pipeline_steps = [
            "Original Mesh",
            "Smoothed Mesh",
            "Curvedness & Segmentation",
            "Boundary Extraction",
            "Patch Extraction",
            "Coarse Alignment",
            "Fine Alignment (ICP)",
            "Reassembly Result"
        ]

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        em = layout_context.theme.font_size
        width = 17 * em

        # Calculate scene dimensions
        scene_width = (r.get_right() - width) / 3  # Divide into 3 sections
        scene_height = r.height / 2  # Divide height by 2 for each view

        # Position scenes for both objects
        for obj_idx in range(2):
            for i in range(2):  # Only 2 views per object
                x = r.x + obj_idx * scene_width
                y = r.y + i * scene_height
                
                # Position label
                self.scene_labels[obj_idx][i].frame = gui.Rect(x, y, scene_width, em)
                
                # Position scene
                self.scenes[obj_idx][i].frame = gui.Rect(x, y + em, scene_width, scene_height - em)
        
        # Position matched result scene
        x = r.x + 2 * scene_width
        y = r.y
        self.scene_labels[2][0].frame = gui.Rect(x, y, scene_width, em)
        self.scenes[2][0].frame = gui.Rect(x, y + em, scene_width, r.height - em)
        
        # Position the control panel
        self.panel.frame = gui.Rect(r.get_right() - width, r.y, width, r.height)

    def load_point_cloud(self, obj_idx):
        dialog = gui.FileDialog(gui.FileDialog.OPEN, f"Choose point cloud file for Object {obj_idx + 1}", self.window.theme)
        dialog.add_filter(".ply", "Point cloud files (.ply)")
        dialog.add_filter("", "All files")
        
        dialog.set_on_cancel(self._on_file_dialog_cancel)
        dialog.set_on_done(lambda filename: self._on_load_dialog_done(filename, obj_idx))
        self.window.show_dialog(dialog)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename, obj_idx):
        try:
            self.window.close_dialog()
            print(f"\nLoading point cloud for Object {obj_idx + 1} from: {filename}")
            self.original_point_clouds[obj_idx] = o3d.io.read_point_cloud(filename)
            if not self.original_point_clouds[obj_idx].is_empty():
                print(f"Successfully loaded point cloud with {len(self.original_point_clouds[obj_idx].points)} points")
                self.point_clouds[obj_idx] = self.original_point_clouds[obj_idx]
                self.process_single_object(obj_idx)
            else:
                print("Error: Loaded point cloud is empty")
        except Exception as e:
            print(f"Error loading point cloud: {str(e)}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())

    def process_single_object(self, obj_idx):
        try:
            if self.point_clouds[obj_idx] is None:
                print(f"Error: No point cloud loaded for Object {obj_idx + 1}")
                return

            print(f"\n=== Processing Object {obj_idx + 1} ===")
            
            # Update parameters from sliders
            self.voxel_size = self.voxel_slider.double_value
            self.k_neighbors = self.k_slider.int_value
            self.normal_threshold = self.normal_slider.double_value
            self.min_cluster_size = self.cluster_slider.int_value
            self.K_thresh = self.k_thresh_slider.double_value
            self.H_thresh = self.h_thresh_slider.double_value

            # Reset to original point cloud if available
            if self.original_point_clouds[obj_idx] is not None:
                print("Resetting to original point cloud...")
                self.point_clouds[obj_idx] = self.original_point_clouds[obj_idx]

            # Process point cloud
            print("\nPerforming voxel downsampling...")
            self.point_clouds[obj_idx] = voxel_downsample(self.point_clouds[obj_idx], self.voxel_size)

            print("\nPerforming region growing...")
            self.clusters[obj_idx] = region_growing(
                self.point_clouds[obj_idx],
                k_neighbors=self.k_neighbors,
                normal_threshold=self.normal_threshold,
                min_cluster_size=self.min_cluster_size
            )

            print("\nExtracting boundaries...")
            self.line_sets[obj_idx] = extract_pointcloud_boundaries(self.point_clouds[obj_idx], self.clusters[obj_idx])

            # Show boundary curves
            boundary_cloud = o3d.geometry.PointCloud(self.point_clouds[obj_idx])
            boundary_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.point_size = 3.0
            self.update_scene(obj_idx, 0, boundary_cloud, material)
            for i, line_set in enumerate(self.line_sets[obj_idx]):
                line_material = rendering.MaterialRecord()
                line_material.shader = "unlitLine"
                line_material.line_width = 2.0
                self.scenes[obj_idx][0].scene.add_geometry(f"line_set_{i}", line_set, line_material)
            self.scenes[obj_idx][0].force_redraw()

            print("\nExtracting concave/convex patches...")
            self.patch_colored_clouds[obj_idx], self.patch_types[obj_idx], self.patch_indices[obj_idx] = extract_concave_convex_patches_with_labels(
                self.point_clouds[obj_idx],
                K_thresh=self.K_thresh,
                H_thresh=self.H_thresh,
                neighbor_radius=self.neighbor_radius,
                min_neighbors=self.min_neighbors,
                min_cluster_size=self.min_patch_size
            )

            # Show patches
            if not self.patch_colored_clouds[obj_idx].has_normals():
                self.patch_colored_clouds[obj_idx].orient_normals_to_align_with_direction()
            self.update_scene(obj_idx, 1, self.patch_colored_clouds[obj_idx])

            print(f"=== Processing Complete for Object {obj_idx + 1} ===\n")

        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())

    def process_both_objects(self):
        for obj_idx in range(2):
            self.process_single_object(obj_idx)

    def pairwise_match(self):
        """Implement the complete pairwise matching algorithm"""
        try:
            # Check if both objects are loaded and processed
            if (self.point_clouds[0] is None or self.point_clouds[1] is None or
                self.line_sets[0] is None or self.line_sets[1] is None or
                self.patch_colored_clouds[0] is None or self.patch_colored_clouds[1] is None or
                self.patch_types[0] is None or self.patch_types[1] is None or
                self.patch_indices[0] is None or self.patch_indices[1] is None):
                print("Error: Both objects must be loaded and processed before pairwise matching")
                return
            
            print("\n=== Starting Pairwise Matching ===")
            
            # Update progress
            self.progress_label.text = "Running pairwise matching..."
            self.progress_label.text_color = gui.Color(1.0, 1.0, 0.0)  # Yellow
            
            # Update parameters from sliders
            self.curve_similarity_threshold = self.curve_sim_slider.double_value
            self.K_patch_threshold = self.k_patch_slider.double_value
            self.H_patch_threshold = self.h_patch_slider.double_value
            self.icp_distance_threshold = self.icp_dist_slider.double_value
            self.overlap_threshold = self.overlap_slider.double_value
            
            print(f"Using parameters: ε_Λ={self.curve_similarity_threshold}, ε_K={self.K_patch_threshold}, ε_H={self.H_patch_threshold}")
            
            # Run the pairwise matching algorithm
            transformation, overlap_score = pairwise_matching_algorithm(
                self.point_clouds[0], self.point_clouds[1],
                self.line_sets[0], self.line_sets[1],
                self.patch_colored_clouds[0], self.patch_colored_clouds[1],
                self.patch_types[0], self.patch_indices[0], self.patch_types[1], self.patch_indices[1],
                ε_Λ=self.curve_similarity_threshold, ε_K=self.K_patch_threshold, ε_H=self.H_patch_threshold, icp_distance=self.icp_distance_threshold
            )
            
            if transformation is not None and overlap_score > self.overlap_threshold:
                print(f"\n✅ Match found! Overlap score: {overlap_score:.3f}")
                print("Transformation matrix:")
                print(transformation)
                
                # Update progress
                self.progress_label.text = f"Match found! Score: {overlap_score:.3f}"
                self.progress_label.text_color = gui.Color(0.0, 1.0, 0.0)  # Green
                
                # Apply transformation to second object
                transformed_cloud = o3d.geometry.PointCloud(self.point_clouds[1])
                transformed_cloud.transform(transformation)
                
                # Create combined visualization
                combined_cloud = o3d.geometry.PointCloud()
                points1 = np.asarray(self.point_clouds[0].points)
                points2 = np.asarray(transformed_cloud.points)
                
                # Color the first object blue and second object red
                colors1 = np.array([[0, 0, 1]] * len(points1))  # Blue
                colors2 = np.array([[1, 0, 0]] * len(points2))  # Red
                
                combined_points = np.vstack((points1, points2))
                combined_colors = np.vstack((colors1, colors2))
                
                combined_cloud.points = o3d.utility.Vector3dVector(combined_points)
                combined_cloud.colors = o3d.utility.Vector3dVector(combined_colors)
                
                # Store the matched result
                self.matched_result = {
                    'transformation': transformation,
                    'overlap_score': overlap_score,
                    'combined_cloud': combined_cloud,
                    'transformed_cloud': transformed_cloud
                }
                
                # Update the matched result scene
                self.update_scene(2, 0, combined_cloud)
                
                print("✅ Pairwise matching completed successfully!")
                
            else:
                print(f"\n❌ No valid match found. Best overlap score: {overlap_score:.3f}")
                print("Try adjusting parameters or using different objects.")
                
                # Update progress
                self.progress_label.text = f"No match found. Score: {overlap_score:.3f}"
                self.progress_label.text_color = gui.Color(1.0, 0.0, 0.0)  # Red
                
                # Show original objects side by side
                combined_cloud = o3d.geometry.PointCloud()
                points1 = np.asarray(self.point_clouds[0].points)
                points2 = np.asarray(self.point_clouds[1].points)
                
                # Offset the second point cloud to the right
                points2[:, 0] += np.max(points1[:, 0]) + 10
                
                # Color the objects
                colors1 = np.array([[0, 0, 1]] * len(points1))  # Blue
                colors2 = np.array([[1, 0, 0]] * len(points2))  # Red
                
                combined_points = np.vstack((points1, points2))
                combined_colors = np.vstack((colors1, colors2))
                
                combined_cloud.points = o3d.utility.Vector3dVector(combined_points)
                combined_cloud.colors = o3d.utility.Vector3dVector(combined_colors)
                
                self.update_scene(2, 0, combined_cloud)
                
        except Exception as e:
            print(f"\n❌ Error during pairwise matching: {str(e)}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
            
            # Update progress
            self.progress_label.text = f"Error: {str(e)[:30]}..."
            self.progress_label.text_color = gui.Color(1.0, 0.0, 0.0)  # Red
            
            # Show error visualization
            if self.point_clouds[0] is not None and self.point_clouds[1] is not None:
                combined_cloud = o3d.geometry.PointCloud()
                points1 = np.asarray(self.point_clouds[0].points)
                points2 = np.asarray(self.point_clouds[1].points)
                
                # Offset the second point cloud to the right
                points2[:, 0] += np.max(points1[:, 0]) + 10
                
                combined_points = np.vstack((points1, points2))
                combined_cloud.points = o3d.utility.Vector3dVector(combined_points)
                combined_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for error
                
                self.update_scene(2, 0, combined_cloud)

    def save_matched_result(self):
        """Save the matched result to a file"""
        try:
            if self.matched_result is None:
                print("❌ No matched result to save. Run pairwise matching first.")
                return
            
            dialog = gui.FileDialog(gui.FileDialog.SAVE, "Save matched result", self.window.theme)
            dialog.add_filter(".ply", "Point cloud files (.ply)")
            dialog.add_filter("", "All files")
            
            dialog.set_on_cancel(self._on_file_dialog_cancel)
            dialog.set_on_done(self._on_save_dialog_done)
            self.window.show_dialog(dialog)
            
        except Exception as e:
            print(f"❌ Error saving matched result: {str(e)}")

    def _on_save_dialog_done(self, filename):
        try:
            self.window.close_dialog()
            
            if self.matched_result is not None:
                # Save the combined point cloud
                o3d.io.write_point_cloud(filename, self.matched_result['combined_cloud'])
                print(f"✅ Matched result saved to: {filename}")
                
                # Also save the transformation matrix
                transform_filename = filename.replace('.ply', '_transformation.txt')
                np.savetxt(transform_filename, self.matched_result['transformation'])
                print(f"✅ Transformation matrix saved to: {transform_filename}")
                
                # Save metadata
                metadata_filename = filename.replace('.ply', '_metadata.txt')
                with open(metadata_filename, 'w') as f:
                    f.write(f"Overlap Score: {self.matched_result['overlap_score']:.6f}\n")
                    f.write(f"Transformation Matrix:\n")
                    f.write(str(self.matched_result['transformation']))
                print(f"✅ Metadata saved to: {metadata_filename}")
            else:
                print("❌ No matched result to save")
                
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")

    def update_scene(self, obj_idx, scene_idx, geometry, material=None):
        if material is None:
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.point_size = 3.0

        # Clear existing geometry
        self.scenes[obj_idx][scene_idx].scene.clear_geometry()
        
        # Add new geometry
        self.scenes[obj_idx][scene_idx].scene.add_geometry("geometry", geometry, material)
        
        # Set up camera
        bounds = geometry.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = bounds.get_extent()
        radius = np.linalg.norm(extent) * 0.5
        
        # Set up camera with a good view of the geometry
        self.scenes[obj_idx][scene_idx].setup_camera(60, bounds, center)
        self.scenes[obj_idx][scene_idx].look_at(center, center + [0, 0, radius], [0, 1, 0])
        
        # Force redraw of the scene widget
        self.scenes[obj_idx][scene_idx].force_redraw()

    def extract_surfaces(self, obj_idx):
        import open3d as o3d
        import numpy as np
        from collections import deque
        if self.point_clouds[obj_idx] is None:
            print(f"No point cloud loaded for Object {obj_idx + 1}")
            return
        pcd = self.point_clouds[obj_idx]
        if not isinstance(pcd, o3d.geometry.PointCloud):
            print("Input is not a point cloud.")
            return
        # Estimate mesh from point cloud (Poisson)
        pcd.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        n_verts = len(vertices)
        # Build adjacency list for 2-ring neighborhood
        adjacency = [[] for _ in range(n_verts)]
        for tri in faces:
            for i in range(3):
                a, b = tri[i], tri[(i+1)%3]
                if b not in adjacency[a]: adjacency[a].append(b)
                if a not in adjacency[b]: adjacency[b].append(a)
        def get_k_ring(v, k=2):
            visited = set([v])
            frontier = set([v])
            for _ in range(k):
                next_frontier = set()
                for u in frontier:
                    for n in adjacency[u]:
                        if n not in visited:
                            next_frontier.add(n)
                            visited.add(n)
                frontier = next_frontier
            visited.remove(v)
            return list(visited)
        # For each vertex, fit quadratic to 2-ring neighborhood and compute principal curvatures
        curvedness = np.zeros(n_verts)
        k1s = np.zeros(n_verts)
        k2s = np.zeros(n_verts)
        for vi in range(n_verts):
            nbrs = get_k_ring(vi, k=2)
            if len(nbrs) < 6:
                continue
            pts = vertices[nbrs] - vertices[vi]
            X = np.c_[pts[:,0]**2, pts[:,1]**2, pts[:,0]*pts[:,1], pts[:,0], pts[:,1], np.ones(len(pts))]
            y = pts[:,2]
            try:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            except:
                continue
            a, b, c, d, e, f = coef
            H = a + b
            K = a*b - (c**2)/4
            k1 = H + np.sqrt(max(0, H**2 - 4*K))/2
            k2 = H - np.sqrt(max(0, H**2 - 4*K))/2
            k1s[vi] = k1
            k2s[vi] = k2
            curvedness[vi] = np.sqrt((k1**2 + k2**2)/2)
        # Compute confidence for each vertex
        confidence = np.zeros(n_verts)
        for vi in range(n_verts):
            nbrs = get_k_ring(vi, k=2)
            nbrs_all = nbrs + [vi]
            cvals = curvedness[nbrs_all]
            mNp = np.mean(cvals)
            sNp = np.std(cvals) if np.std(cvals) > 1e-6 else 1.0
            confidence[vi] = np.abs(curvedness[vi] - mNp) / sNp
        # Select keypoints (maxima and minima with confidence > 1.0)
        keypoints = []
        for vi in range(n_verts):
            nbrs = get_k_ring(vi, k=2)
            nbrs_all = nbrs + [vi]
            cvals = curvedness[nbrs_all]
            if confidence[vi] > 1.0:
                if curvedness[vi] == np.max(cvals) or curvedness[vi] == np.min(cvals):
                    keypoints.append(vi)
        # Segment mesh into patches using region growing on curvedness
        patch_labels = -np.ones(n_verts, dtype=int)
        patch_id = 0
        threshold = np.median(curvedness)
        for vi in range(n_verts):
            if patch_labels[vi] != -1:
                continue
            queue = deque([vi])
            patch_labels[vi] = patch_id
            while queue:
                vj = queue.popleft()
                for nk in adjacency[vj]:
                    if patch_labels[nk] == -1 and abs(curvedness[nk] - curvedness[vj]) < 0.1*threshold:
                        patch_labels[nk] = patch_id
                        queue.append(nk)
            patch_id += 1
        # --- NEW: Use boundary curves and roughness to identify fracture surfaces ---
        # 1. Extract boundary curves from the mesh vertices
        # Convert mesh to point cloud for boundary extraction
        mesh_pcd = o3d.geometry.PointCloud()
        mesh_pcd.points = o3d.utility.Vector3dVector(vertices)
        mesh_pcd.normals = o3d.utility.Vector3dVector(mesh.vertex_normals)
        # Use region growing to get clusters (simulate as one cluster for mesh)
        clusters = [list(range(n_verts))]
        line_sets = extract_pointcloud_boundaries(mesh_pcd, clusters)
        # Collect all boundary points
        boundary_points = np.concatenate([np.asarray(ls.points) for ls in line_sets]) if line_sets else np.zeros((0,3))
        # Estimate mesh resolution
        if len(faces) > 0:
            edge_lengths = [np.linalg.norm(vertices[tri[i]] - vertices[tri[(i+1)%3]]) for tri in faces for i in range(3)]
            mesh_res = np.median(edge_lengths)
        else:
            mesh_res = 1.0
        # 2. For each patch, compute roughness and check boundary adjacency
        patch_roughness = []
        patch_is_fracture = []
        for pid in range(patch_id):
            patch_verts = np.where(patch_labels == pid)[0]
            if len(patch_verts) == 0:
                patch_roughness.append(0)
                patch_is_fracture.append(False)
                continue
            roughness = np.std(curvedness[patch_verts])
            patch_roughness.append(roughness)
            # Check if any patch vertex is within 2x mesh_res of a boundary point
            is_adjacent = False
            if len(boundary_points) > 0:
                dists = np.linalg.norm(vertices[patch_verts][:,None] - boundary_points[None,:,:], axis=2)
                min_dists = np.min(dists, axis=1)
                if np.any(min_dists < 2 * mesh_res):
                    is_adjacent = True
            patch_is_fracture.append(is_adjacent)
        # 3. Select patches with top 25% roughness and boundary adjacency
        roughness_thresh = np.percentile(patch_roughness, 75)
        fracture_patches = [pid for pid in range(patch_id) if patch_roughness[pid] >= roughness_thresh and patch_is_fracture[pid]]
        # Define distinct colors for fracture surfaces
        fracture_colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 0.0, 1.0],  # Blue
            [0.0, 1.0, 0.0],  # Green
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
        ]
        # Assign colors to vertices
        vert_colors = np.full((n_verts, 3), 0.7)  # Gray for non-fracture areas
        for i, pid in enumerate(fracture_patches):
            patch_verts = np.where(patch_labels == pid)[0]
            color = fracture_colors[i % len(fracture_colors)]
            vert_colors[patch_verts] = color
        mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
        # Show segmented mesh in the boundary curve view
        self.update_scene(obj_idx, 0, mesh)
        print(f"Extracted {len(fracture_patches)} fracture surfaces (using roughness + boundary). Keypoints: {len(keypoints)}")
        # Optionally, store or return curvedness, keypoints, patch_labels for further use

    def set_min_patch_size_percent(self, value):
        self.min_patch_size_percent = value

    def prev_step(self):
        if self.pipeline_step > 0:
            self.pipeline_step -= 1
            self.visualize_pipeline_step()

    def next_step(self):
        if self.pipeline_step < len(self.pipeline_steps) - 1:
            self.pipeline_step += 1
            self.visualize_pipeline_step()

    def visualize_pipeline_step(self):
        # For each object, update the scene to show the current pipeline step
        for obj_idx in range(2):
            if self.point_clouds[obj_idx] is None:
                continue
            # Step 0: Original Mesh
            if self.pipeline_step == 0:
                self.update_scene(obj_idx, 0, self.point_clouds[obj_idx])
            # Step 1: Smoothed Mesh
            elif self.pipeline_step == 1:
                smoothed = self.laplacian_smooth(self.point_clouds[obj_idx])
                self.update_scene(obj_idx, 0, smoothed)
            # Step 2: Curvedness & Segmentation
            elif self.pipeline_step == 2:
                mesh, curvedness, patch_labels = self.segment_surface(self.point_clouds[obj_idx])
                self.update_scene(obj_idx, 0, mesh)
            # Step 3: Boundary Extraction
            elif self.pipeline_step == 3:
                mesh, curvedness, patch_labels = self.segment_surface(self.point_clouds[obj_idx])
                line_sets = self.extract_boundaries(mesh)
                self.update_scene(obj_idx, 0, mesh)
                for i, line_set in enumerate(line_sets):
                    self.scenes[obj_idx][0].scene.add_geometry(f"line_set_{i}", line_set, rendering.MaterialRecord())
            # Step 4: Patch Extraction
            elif self.pipeline_step == 4:
                mesh, curvedness, patch_labels = self.segment_surface(self.point_clouds[obj_idx])
                patch_mesh = self.visualize_patches(mesh, patch_labels)
                self.update_scene(obj_idx, 0, patch_mesh)
            # Step 5: Coarse Alignment
            # (Show both objects, aligned by boundary curve matching)
            # Step 6: Fine Alignment (ICP)
            # (Show both objects, aligned by patch-based ICP)
            # Step 7: Reassembly Result
            # (Show final reassembly)
        # Optionally, update labels or status to indicate current step

    def laplacian_smooth(self, pcd):
        import open3d as o3d
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
        mesh.compute_vertex_normals()
        return mesh

    def segment_surface(self, pcd):
        import open3d as o3d
        import numpy as np
        from collections import deque
        # Estimate mesh from point cloud (Poisson)
        pcd.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        n_verts = len(vertices)
        # Build adjacency list for 2-ring neighborhood
        adjacency = [[] for _ in range(n_verts)]
        for tri in faces:
            for i in range(3):
                a, b = tri[i], tri[(i+1)%3]
                if b not in adjacency[a]: adjacency[a].append(b)
                if a not in adjacency[b]: adjacency[b].append(a)
        def get_k_ring(v, k=2):
            visited = set([v])
            frontier = set([v])
            for _ in range(k):
                next_frontier = set()
                for u in frontier:
                    for n in adjacency[u]:
                        if n not in visited:
                            next_frontier.add(n)
                            visited.add(n)
                frontier = next_frontier
            visited.remove(v)
            return list(visited)
        # For each vertex, fit quadratic to 2-ring neighborhood and compute principal curvatures
        curvedness = np.zeros(n_verts)
        for vi in range(n_verts):
            nbrs = get_k_ring(vi, k=2)
            if len(nbrs) < 6:
                continue
            pts = vertices[nbrs] - vertices[vi]
            X = np.c_[pts[:,0]**2, pts[:,1]**2, pts[:,0]*pts[:,1], pts[:,0], pts[:,1], np.ones(len(pts))]
            y = pts[:,2]
            try:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            except:
                continue
            a, b, c, d, e, f = coef
            H = a + b
            K = a*b - (c**2)/4
            k1 = H + np.sqrt(max(0, H**2 - 4*K))/2
            k2 = H - np.sqrt(max(0, H**2 - 4*K))/2
            curvedness[vi] = np.sqrt((k1**2 + k2**2)/2)
        # Segment mesh into patches using region growing on curvedness
        patch_labels = -np.ones(n_verts, dtype=int)
        patch_id = 0
        threshold = np.median(curvedness)
        for vi in range(n_verts):
            if patch_labels[vi] != -1:
                continue
            queue = deque([vi])
            patch_labels[vi] = patch_id
            while queue:
                vj = queue.popleft()
                for nk in adjacency[vj]:
                    if patch_labels[nk] == -1 and abs(curvedness[nk] - curvedness[vj]) < 0.1*threshold:
                        patch_labels[nk] = patch_id
                        queue.append(nk)
            patch_id += 1
        return mesh, curvedness, patch_labels

    def extract_boundaries(self, mesh):
        import open3d as o3d
        import numpy as np
        # Find boundary edges using Open3D
        mesh.compute_vertex_normals()
        edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=True))
        if len(edges) == 0:
            return []
        # Collect boundary points
        boundary_points = set(edges.flatten())
        boundary_points = list(boundary_points)
        boundary_coords = np.asarray(mesh.vertices)[boundary_points]
        # Try to order boundary points (simple nearest-neighbor ordering)
        if len(boundary_coords) < 2:
            return []
        ordered = [boundary_points[0]]
        used = set(ordered)
        for _ in range(1, len(boundary_points)):
            last = ordered[-1]
            candidates = [i for i in boundary_points if i not in used]
            if not candidates:
                break
            dists = [np.linalg.norm(mesh.vertices[last] - mesh.vertices[c]) for c in candidates]
            next_idx = candidates[np.argmin(dists)]
            ordered.append(next_idx)
            used.add(next_idx)
        # Create a LineSet for the ordered boundary
        points = np.asarray(mesh.vertices)[ordered]
        lines = [[i, i+1] for i in range(len(points)-1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.paint_uniform_color([1, 0, 0])
        return [line_set]

    def visualize_patches(self, mesh, patch_labels):
        import numpy as np
        n_verts = len(np.asarray(mesh.vertices))
        n_patches = np.max(patch_labels) + 1
        # Generate distinct colors for each patch
        colors = np.zeros((n_verts, 3))
        palette = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
        ]
        for pid in range(n_patches):
            color = palette[pid % len(palette)]
            colors[patch_labels == pid] = color
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

def compute_patch_descriptors(mesh, curvedness, patch_labels):
    import numpy as np
    vertices = np.asarray(mesh.vertices)
    n_patches = np.max(patch_labels) + 1
    descriptors = []
    for pid in range(n_patches):
        idxs = np.where(patch_labels == pid)[0]
        if len(idxs) == 0:
            descriptors.append({"mean": 0, "std": 0, "size": 0, "anisotropy": 0})
            continue
        patch_curvedness = curvedness[idxs]
        patch_points = vertices[idxs]
        mu = np.mean(patch_curvedness)
        sigma = np.std(patch_curvedness)
        # PCA for size and anisotropy
        centroid = np.mean(patch_points, axis=0)
        cov = np.cov((patch_points - centroid).T)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)[::-1]
        eigvals = np.maximum(eigvals, 1e-10)
        size = np.sqrt(np.sum(eigvals))
        anisotropy = np.sqrt(np.abs(eigvals[1]/eigvals[2])) if eigvals[2] > 0 else 0
        descriptors.append({"mean": mu, "std": sigma, "size": size, "anisotropy": anisotropy})
    return descriptors

def match_patches(descriptors1, types1, descriptors2, types2, thresholds=None):
    # thresholds: dict with keys 'mean', 'std', 'size', 'anisotropy'
    if thresholds is None:
        thresholds = {'mean': 0.5, 'std': 0.4, 'size': 0.3, 'anisotropy': 0.3}
    matches = []
    for i, (desc1, t1) in enumerate(zip(descriptors1, types1)):
        for j, (desc2, t2) in enumerate(zip(descriptors2, types2)):
            # Only match concave to convex and vice versa
            if (t1 == 0 and t2 == 1) or (t1 == 1 and t2 == 0):
                mean_ok = abs(desc1['mean'] - desc2['mean']) <= thresholds['mean']
                std_ok = abs(desc1['std'] - desc2['std']) <= thresholds['std']
                size_ok = abs(desc1['size'] - desc2['size']) / (desc1['size'] + desc2['size'] + 1e-10) <= thresholds['size']
                aniso_ok = abs(desc1['anisotropy'] - desc2['anisotropy']) / (desc1['anisotropy'] + desc2['anisotropy'] + 1e-10) <= thresholds['anisotropy']
                if mean_ok and std_ok and size_ok and aniso_ok:
                    matches.append((i, j))
    return matches

def compare_boundary_curves(curve1, curve2, window=3, threshold=0.3, min_overlap_ratio=0.125):
    import numpy as np
    # curve1, curve2: Nx3 and Mx3 arrays of ordered boundary points
    m, n = len(curve1), len(curve2)
    # Build similarity matrix
    sim_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dist = 0
            for q in range(-window, window+1):
                i_q = (i+q) % m
                j_q = (j+q) % n
                dist += np.linalg.norm(curve1[i_q] - curve2[j_q])
            sim_matrix[i, j] = dist / (2*window+1)
    # Find similar segments (diagonal runs with sim < threshold)
    matches = []
    min_overlap = int(min(m, n) * min_overlap_ratio)
    for i in range(m):
        for j in range(n):
            # Check for diagonal run
            run = []
            for k in range(min(m, n)):
                ii = (i + k) % m
                jj = (j + k) % n
                if sim_matrix[ii, jj] < threshold:
                    run.append((ii, jj))
                else:
                    if len(run) >= min_overlap:
                        matches.append(run)
                    run = []
            if len(run) >= min_overlap:
                matches.append(run)
    # Return the best matching segment(s) (longest runs)
    matches = sorted(matches, key=len, reverse=True)
    return matches  # Each match is a list of (i, j) index pairs

def compute_coarse_alignment(src_points, tgt_points):
    import numpy as np
    assert src_points.shape == tgt_points.shape
    centroid_src = np.mean(src_points, axis=0)
    centroid_tgt = np.mean(tgt_points, axis=0)
    src_centered = src_points - centroid_src
    tgt_centered = tgt_points - centroid_tgt
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_tgt - R @ centroid_src
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def fine_align_icp(pcd1, pcd2, init_transform, normals1, normals2, curv1, curv2, dist_thresh=0.01, normal_thresh=0.1, curv_thresh=0.3, max_iter=50):
    import open3d as o3d
    import numpy as np
    # pcd1, pcd2: Open3D PointClouds (patch points)
    # normals1, normals2: Nx3 arrays
    # curv1, curv2: N arrays
    # Outlier rejection: distance, normal, curvature
    def correspondence_checker(source, target, trans):
        src_pts = np.asarray(source.points)
        tgt_pts = np.asarray(target.points)
        src_n = normals1
        tgt_n = normals2
        src_c = curv1
        tgt_c = curv2
        src_pts_t = (trans[:3,:3] @ src_pts.T).T + trans[:3,3]
        matches = []
        for i, pt in enumerate(src_pts_t):
            dists = np.linalg.norm(tgt_pts - pt, axis=1)
            j = np.argmin(dists)
            if dists[j] > dist_thresh:
                continue
            if np.linalg.norm(src_n[i] - tgt_n[j]) > normal_thresh:
                continue
            if abs(src_c[i] - tgt_c[j]) > curv_thresh:
                continue
            matches.append((i, j))
        return matches
    # Run ICP with custom correspondence checker
    reg = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, dist_thresh, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    # Optionally, filter correspondences after ICP
    matches = correspondence_checker(pcd1, pcd2, reg.transformation)
    return reg, matches

def compute_overlap_ratio(points1, points2, normals1, normals2, curv1, curv2, dist_thresh=0.01, normal_thresh=0.1, curv_thresh=0.3):
    import numpy as np
    from scipy.spatial import cKDTree
    tree2 = cKDTree(points2)
    overlap_count = 0
    for i, pt in enumerate(points1):
        d, j = tree2.query(pt)
        if d > dist_thresh:
            continue
        if np.linalg.norm(normals1[i] - normals2[j]) > normal_thresh:
            continue
        if abs(curv1[i] - curv2[j]) > curv_thresh:
            continue
        overlap_count += 1
    return overlap_count / len(points1) if len(points1) > 0 else 0.0

if __name__ == "__main__":
    gui_app = ReassemblyGUI()
    gui.Application.instance.run()

