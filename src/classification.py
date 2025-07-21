import os
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
import argparse
from sklearn.neighbors import NearestNeighbors
from utils.test_utils import load_3d_models_from_folder, visualize_models

from segmentation import region_growing_segmentation, AdaptiveFractureDetector


def calculate_local_bending_energy(mesh, k=10):
    """
    Calculate the Local Bending Energy for each vertex in a mesh.
    
    The Local Bending Energy e_k(p) is defined as:
    e_k(p) = (1/k) * sum_{i=1}^k [||n_p - n_qi||^2 / ||p - qi||^2]
    
    where:
    - p is the vertex point
    - qi are the k-nearest neighbors of p
    - n_p is the normal at point p
    - n_qi is the normal at neighbor qi
    - ||n_p - n_qi||^2 is the squared distance between normals
    - ||p - qi||^2 is the squared distance between points
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        k (int): Number of nearest neighbors to consider (default: 10)
    
    Returns:
        numpy.ndarray: Array of bending energy values for each vertex
    """
    # Ensure mesh has vertices and normals
    if not mesh.has_vertices():
        raise ValueError("Mesh must have vertices")
    
    # Get vertices and normals
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    # If normals are not computed, compute them
    if len(normals) == 0 or np.all(normals == 0):
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
    
    n_vertices = len(vertices)
    bending_energies = np.zeros(n_vertices)
    
    # Find k-nearest neighbors for all vertices
    # We use k+1 because the first neighbor will be the point itself
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(vertices)
    distances, indices = nbrs.kneighbors(vertices)
    
    # Calculate bending energy for each vertex
    for i in range(n_vertices):
        # Get the current point and its normal
        p = vertices[i]
        n_p = normals[i]
        
        # Get k-nearest neighbors (excluding the point itself)
        neighbor_indices = indices[i][1:]  # Skip first neighbor (point itself)
        neighbor_distances = distances[i][1:]  # Skip first distance (0)
        
        # Calculate bending energy for this vertex
        energy_sum = 0.0
        
        for j, neighbor_idx in enumerate(neighbor_indices):
            # Get neighbor point and normal
            q_i = vertices[neighbor_idx]
            n_qi = normals[neighbor_idx]
            
            # Calculate squared distance between normals
            normal_diff = n_p - n_qi
            normal_distance_squared = np.dot(normal_diff, normal_diff)
            
            # Calculate squared distance between points
            point_diff = p - q_i
            point_distance_squared = np.dot(point_diff, point_diff)
            
            # Avoid division by zero
            if point_distance_squared > 1e-10:
                energy_sum += normal_distance_squared / point_distance_squared
            else:
                # If points are very close, use a small epsilon
                energy_sum += normal_distance_squared / 1e-10
        
        # Average over k neighbors
        bending_energies[i] = energy_sum / k
    
    return bending_energies


def visualize_bending_energy(mesh, bending_energies, window_name="Bending Energy Visualization"):
    """
    Visualize the bending energy values on the mesh using color coding.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        bending_energies (numpy.ndarray): Bending energy values for each vertex
        window_name (str): Name of the visualization window
    """
    # Use percentile-based normalization to handle extreme outliers
    percentile_95 = np.percentile(bending_energies, 95)
    percentile_99 = np.percentile(bending_energies, 99)
    
    # Create a more robust normalization that handles outliers
    if np.max(bending_energies) > np.min(bending_energies):
        # Use 95th percentile as the upper bound for better visualization
        upper_bound = min(percentile_95, np.max(bending_energies))
        normalized_energies = np.clip(bending_energies / upper_bound, 0, 1)
    else:
        normalized_energies = np.zeros_like(bending_energies)
    
    # Create color map (red for high energy, blue for low energy)
    colors = np.zeros((len(normalized_energies), 3))
    colors[:, 0] = normalized_energies  # Red channel (high energy)
    colors[:, 2] = 1.0 - normalized_energies  # Blue channel (low energy)
    
    # Apply colors to mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Visualize
    print(f"\nVisualizing bending energy...")
    print(f"Min bending energy: {np.min(bending_energies):.6f}")
    print(f"Max bending energy: {np.max(bending_energies):.6f}")
    print(f"Mean bending energy: {np.mean(bending_energies):.6f}")
    print(f"95th percentile: {percentile_95:.6f}")
    print(f"99th percentile: {percentile_99:.6f}")
    print(f"Upper bound used for normalization: {min(percentile_95, np.max(bending_energies)):.6f}")
    print("Color coding: Red = High curvature, Blue = Low curvature")
    print("Note: Using 95th percentile normalization to handle extreme outliers")
    
    o3d.visualization.draw_geometries(
        [mesh, coordinate_frame],
        window_name=window_name,
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    )


def analyze_mesh_curvature(mesh, k=10):
    """
    Analyze mesh curvature using Local Bending Energy.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        k (int): Number of nearest neighbors for bending energy calculation
    
    Returns:
        dict: Dictionary containing curvature analysis results
    """
    print(f"Analyzing mesh curvature with k={k} nearest neighbors...")
    
    # Calculate bending energy
    bending_energies = calculate_local_bending_energy(mesh, k)
    
    # Compute statistics
    stats = {
        'min_energy': np.min(bending_energies),
        'max_energy': np.max(bending_energies),
        'mean_energy': np.mean(bending_energies),
        'std_energy': np.std(bending_energies),
        'median_energy': np.median(bending_energies),
        'bending_energies': bending_energies
    }
    
    print(f"Curvature Analysis Results:")
    print(f"  Min bending energy: {stats['min_energy']:.6f}")
    print(f"  Max bending energy: {stats['max_energy']:.6f}")
    print(f"  Mean bending energy: {stats['mean_energy']:.6f}")
    print(f"  Std bending energy: {stats['std_energy']:.6f}")
    print(f"  Median bending energy: {stats['median_energy']:.6f}")
    
    return stats


def segment_mesh_and_analyze_curvature(mesh, k=10, segmentation_params=None):
    """
    Segment the mesh and analyze curvature for each region separately.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        k (int): Number of nearest neighbors for bending energy calculation
        segmentation_params (dict): Parameters for segmentation
    
    Returns:
        dict: Dictionary containing segmentation and curvature analysis results
    """
    print(f"Segmenting mesh and analyzing curvature for each region...")
    
    # Convert Open3D mesh to trimesh for segmentation
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Use default segmentation parameters if none provided
    if segmentation_params is None:
        segmentation_params = {
            'angle_threshold': 30.0,
            'curvature_threshold': 0.1,
            'min_region_size': 50,
            'max_region_size': 5000
        }
    
    print(f"Segmentation parameters: {segmentation_params}")
    
    # Perform segmentation
    try:
        segments = region_growing_segmentation(tri_mesh, segmentation_params)
        print(f"Segmentation completed: {len(segments)} regions found")
    except Exception as e:
        print(f"Segmentation failed: {e}")
        print("Falling back to single region analysis...")
        return analyze_mesh_curvature(mesh, k)
    
    # Analyze each region
    region_results = []
    all_bending_energies = []
    
    for i, segment_faces in enumerate(segments):
        print(f"\n--- Analyzing Region {i+1}/{len(segments)} ---")
        print(f"Region size: {len(segment_faces)} faces")
        
        # Create a submesh for this region
        region_faces = tri_mesh.faces[segment_faces]
        
        # Get unique vertices used by these faces
        unique_vertex_indices = np.unique(region_faces.flatten())
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
        
        # Create new vertices and remapped faces
        region_vertices = tri_mesh.vertices[unique_vertex_indices]
        region_faces_remapped = np.array([[vertex_map[vertex_idx] for vertex_idx in face] for face in region_faces])
        
        # Create trimesh object for this region
        region_mesh = trimesh.Trimesh(vertices=region_vertices, faces=region_faces_remapped)
        
        # Convert back to Open3D for curvature analysis
        region_o3d_mesh = o3d.geometry.TriangleMesh()
        region_o3d_mesh.vertices = o3d.utility.Vector3dVector(region_mesh.vertices)
        region_o3d_mesh.triangles = o3d.utility.Vector3iVector(region_mesh.faces)
        region_o3d_mesh.compute_vertex_normals()
        
        # Analyze curvature for this region
        region_stats = analyze_mesh_curvature(region_o3d_mesh, k)
        
        # Store results
        region_result = {
            'region_id': i,
            'num_faces': len(segment_faces),
            'num_vertices': len(region_mesh.vertices),
            'stats': region_stats,
            'segment_faces': segment_faces,
            'region_mesh': region_o3d_mesh
        }
        region_results.append(region_result)
        
        # Collect all bending energies for overall statistics
        all_bending_energies.extend(region_stats['bending_energies'])
    
    # Compute overall statistics
    overall_stats = {
        'num_regions': len(segments),
        'total_vertices': len(vertices),
        'total_faces': len(faces),
        'overall_min_energy': np.min(all_bending_energies),
        'overall_max_energy': np.max(all_bending_energies),
        'overall_mean_energy': np.mean(all_bending_energies),
        'overall_std_energy': np.std(all_bending_energies),
        'overall_median_energy': np.median(all_bending_energies),
        'region_results': region_results
    }
    
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total regions: {overall_stats['num_regions']}")
    print(f"Overall min bending energy: {overall_stats['overall_min_energy']:.6f}")
    print(f"Overall max bending energy: {overall_stats['overall_max_energy']:.6f}")
    print(f"Overall mean bending energy: {overall_stats['overall_mean_energy']:.6f}")
    print(f"Overall std bending energy: {overall_stats['overall_std_energy']:.6f}")
    print(f"Overall median bending energy: {overall_stats['overall_median_energy']:.6f}")
    
    return overall_stats


def visualize_segmented_curvature(mesh, segmentation_results, window_name="Segmented Curvature Analysis", region_offset=0.1):
    """
    Visualize curvature analysis for each segmented region.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Original mesh
        segmentation_results (dict): Results from segment_mesh_and_analyze_curvature
        window_name (str): Name of the visualization window
        region_offset (float): Percentage of region to offset inward to avoid edge artifacts
    """
    region_results = segmentation_results['region_results']
    
    print(f"\nVisualizing {len(region_results)} segmented regions...")
    
    # Get the original mesh vertices and faces
    original_vertices = np.asarray(mesh.vertices)
    original_faces = np.asarray(mesh.triangles)
    
    for i, region_result in enumerate(region_results):
        print(f"\n--- Visualizing Region {i+1}/{len(region_results)} ---")
        print(f"Region {i+1} statistics:")
        print(f"  Faces: {region_result['num_faces']}")
        print(f"  Vertices: {region_result['num_vertices']}")
        print(f"  Min energy: {region_result['stats']['min_energy']:.6f}")
        print(f"  Max energy: {region_result['stats']['max_energy']:.6f}")
        print(f"  Mean energy: {region_result['stats']['mean_energy']:.6f}")
        
        # Create a copy of the original mesh for visualization
        viz_mesh = o3d.geometry.TriangleMesh()
        viz_mesh.vertices = o3d.utility.Vector3dVector(original_vertices)
        viz_mesh.triangles = o3d.utility.Vector3iVector(original_faces)
        viz_mesh.compute_vertex_normals()
        
        # Initialize all vertices to grey
        num_vertices = len(original_vertices)
        colors = np.full((num_vertices, 3), 0.5)  # Grey color
        
        # Get the segment faces for this region
        segment_faces = region_result['segment_faces']
        
        # Get the region mesh vertices and their bending energies
        region_vertices = np.asarray(region_result['region_mesh'].vertices)
        region_bending_energies = region_result['stats']['bending_energies']
        
        # Create a proper mapping from region vertices to original vertices
        region_vertex_indices = []
        for face_idx in segment_faces:
            face = original_faces[face_idx]
            region_vertex_indices.extend(face)
        region_vertex_indices = list(set(region_vertex_indices))  # Remove duplicates
        region_vertices_set = set(region_vertex_indices)
        
        # Create a proper mapping from original vertex indices to region vertex indices
        original_to_region_mapping = {}
        
        # For each face in the region, map its vertices
        for face_idx in segment_faces:
            face = original_faces[face_idx]
            for vertex_idx in face:
                if vertex_idx in region_vertices_set:
                    # Get the original vertex position
                    original_vertex_pos = original_vertices[vertex_idx]
                    
                    # Find the closest vertex in the region mesh
                    distances = np.linalg.norm(region_vertices - original_vertex_pos, axis=1)
                    closest_region_vertex_idx = np.argmin(distances)
                    
                    # Only use this mapping if the vertices are very close (same vertex)
                    if distances[closest_region_vertex_idx] < 1e-6:  # Small threshold for numerical precision
                        original_to_region_mapping[vertex_idx] = closest_region_vertex_idx
        
        # Create bending energy array for the full mesh (grey for non-region vertices)
        full_mesh_bending_energies = np.zeros(num_vertices)
        
        # Apply region bending energies to the mapped vertices
        for vertex_idx in region_vertices_set:
            if vertex_idx < num_vertices and vertex_idx in original_to_region_mapping:
                region_vertex_idx = original_to_region_mapping[vertex_idx]
                if region_vertex_idx < len(region_bending_energies):
                    full_mesh_bending_energies[vertex_idx] = region_bending_energies[region_vertex_idx]
        
        # Detect and analyze curvature patches
        print(f"\n--- Analyzing Curvature Patches for Region {i+1} ---")
        patch_info = detect_curvature_patches(
            viz_mesh, 
            full_mesh_bending_energies, 
            region_vertices_set,
            offset_percentage=region_offset
        )
        
        # Calculate region area and normalize patch statistics
        region_area = calculate_region_area(viz_mesh, region_vertices_set)
        patch_info = normalize_patch_count_by_area(patch_info, region_area)
        
        # Get the offset region for visualization
        offset_region_set = None
        if region_offset > 0:
            offset_region_set = offset_region_boundaries(viz_mesh, region_vertices_set, region_offset)
        
        # Print patch statistics
        print(f"Patch Analysis Results:")
        print(f"  Region area: {region_area:.6f} square units")
        print(f"  High curvature threshold: {patch_info['high_curvature_threshold']:.6f}")
        print(f"  Number of patches: {patch_info['num_patches']}")
        print(f"  Normalized patch count: {patch_info['normalized_patch_count']:.6f} patches per unit area")
        print(f"  Total high curvature vertices: {patch_info['total_high_curvature_vertices']}")
        print(f"  Normalized high curvature vertices: {patch_info['normalized_high_curvature_vertices']:.6f} per unit area")
        if patch_info['num_patches'] > 0:
            print(f"  Average patch size: {patch_info['avg_patch_size']:.1f} vertices")
            print(f"  Normalized average patch size: {patch_info['normalized_avg_patch_size']:.6f} per unit area")
            print(f"  Largest patch: {patch_info['max_patch_size']} vertices")
            print(f"  Normalized largest patch: {patch_info['normalized_max_patch_size']:.6f} per unit area")
            print(f"  Smallest patch: {patch_info['min_patch_size']} vertices")
            print(f"  Normalized smallest patch: {patch_info['normalized_min_patch_size']:.6f} per unit area")
        
        # Store patch information in region results
        region_result['patch_info'] = patch_info
        
        # Use the existing visualize_bending_energy function with the full mesh
        # But we need to modify it slightly to handle the grey regions
        visualize_bending_energy_with_grey_regions(
            viz_mesh, 
            full_mesh_bending_energies, 
            region_vertices_set,
            f"{window_name} - Region {i+1}",
            offset_region_set
        )
        
        # Visualize patches if any exist
        if patch_info['num_patches'] > 0:
            print(f"\nVisualizing patches for Region {i+1}...")
            visualize_curvature_patches(
                viz_mesh,
                full_mesh_bending_energies,
                region_vertices_set,
                patch_info,
                f"{window_name} - Patches - Region {i+1}",
                offset_region_set
            )
    
    # Print comparison summary across all regions
    print_region_comparison_summary(region_results)


def visualize_bending_energy_with_grey_regions(mesh, bending_energies, region_vertices_set, window_name="Bending Energy Visualization", offset_region_set=None):
    """
    Visualize the bending energy values on the mesh using color coding, with non-region vertices in grey.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        bending_energies (numpy.ndarray): Bending energy values for each vertex (0 for non-region vertices)
        region_vertices_set (set): Set of vertex indices that belong to the current region
        window_name (str): Name of the visualization window
        offset_region_set (set): Set of vertex indices for the offset region (if None, no offset is shown)
    """
    # Use percentile-based normalization to handle extreme outliers (only for non-zero values)
    non_zero_energies = bending_energies[bending_energies > 0]
    
    if len(non_zero_energies) > 0:
        percentile_95 = np.percentile(non_zero_energies, 95)
        
        # Create a more robust normalization that handles outliers
        if np.max(non_zero_energies) > np.min(non_zero_energies):
            # Use 95th percentile as the upper bound for better visualization
            upper_bound = min(percentile_95, np.max(non_zero_energies))
            normalized_energies = np.clip(bending_energies / upper_bound, 0, 1)
        else:
            normalized_energies = np.zeros_like(bending_energies)
    else:
        normalized_energies = np.zeros_like(bending_energies)
        percentile_95 = 0
    
    # Create color map (red for high energy, blue for low energy, grey for non-region, black for offset)
    colors = np.zeros((len(normalized_energies), 3))
    
    for i, (energy, normalized_energy) in enumerate(zip(bending_energies, normalized_energies)):
        if i in region_vertices_set and energy > 0:
            if offset_region_set is not None and i not in offset_region_set:
                # Vertex in original region but not in offset region (boundary area) - color black
                colors[i, :] = 0.0  # Black color
            else:
                # Region vertex: color based on energy
                colors[i, 0] = normalized_energy  # Red channel (high energy)
                colors[i, 2] = 1.0 - normalized_energy  # Blue channel (low energy)
        else:
            # Non-region vertex: grey
            colors[i, :] = 0.5  # Grey color
    
    # Apply colors to mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Visualize
    print(f"\nVisualizing bending energy...")
    if len(non_zero_energies) > 0:
        print(f"Min bending energy: {np.min(non_zero_energies):.6f}")
        print(f"Max bending energy: {np.max(non_zero_energies):.6f}")
        print(f"Mean bending energy: {np.mean(non_zero_energies):.6f}")
        print(f"95th percentile: {percentile_95:.6f}")
        print(f"Upper bound used for normalization: {min(percentile_95, np.max(non_zero_energies)):.6f}")
    else:
        print("No bending energy data for this region")
    
    if offset_region_set is not None:
        print("Color coding: Red = High curvature, Blue = Low curvature, Black = Excluded boundary, Grey = Other regions")
    else:
        print("Color coding: Red = High curvature, Blue = Low curvature, Grey = Other regions")
    print("Note: Using 95th percentile normalization to handle extreme outliers")
    
    o3d.visualization.draw_geometries(
        [mesh, coordinate_frame],
        window_name=window_name,
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    )


def offset_region_boundaries(mesh, region_vertices_set, offset_percentage=0.1):
    """
    Offset region boundaries inward to avoid edge artifacts.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        region_vertices_set (set): Set of vertex indices that belong to the current region
        offset_percentage (float): Percentage of region to offset inward (default: 0.1 = 10%)
    
    Returns:
        set: Set of vertex indices for the offset region (smaller than original)
    """
    if len(region_vertices_set) == 0:
        return set()
    
    # Build adjacency graph for the mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Create vertex adjacency dictionary
    vertex_adjacency = {}
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            if v1 not in vertex_adjacency:
                vertex_adjacency[v1] = set()
            if v2 not in vertex_adjacency:
                vertex_adjacency[v2] = set()
            vertex_adjacency[v1].add(v2)
            vertex_adjacency[v2].add(v1)
    
    # Find boundary vertices (vertices that have neighbors outside the region)
    boundary_vertices = set()
    for vertex_idx in region_vertices_set:
        if vertex_idx in vertex_adjacency:
            for neighbor in vertex_adjacency[vertex_idx]:
                if neighbor not in region_vertices_set:
                    boundary_vertices.add(vertex_idx)
                    break
    
    # Calculate how many layers to remove
    total_vertices = len(region_vertices_set)
    vertices_to_remove = int(total_vertices * offset_percentage)
    
    if vertices_to_remove == 0:
        return region_vertices_set
    
    # Remove boundary vertices in layers
    offset_region = region_vertices_set.copy()
    removed_count = 0
    
    while removed_count < vertices_to_remove and len(offset_region) > 0:
        # Find current boundary vertices
        current_boundary = set()
        for vertex_idx in offset_region:
            if vertex_idx in vertex_adjacency:
                for neighbor in vertex_adjacency[vertex_idx]:
                    if neighbor not in offset_region:
                        current_boundary.add(vertex_idx)
                        break
        
        if len(current_boundary) == 0:
            break  # No more boundary vertices to remove
        
        # Remove boundary vertices (up to the target number)
        vertices_to_remove_this_layer = min(len(current_boundary), vertices_to_remove - removed_count)
        vertices_to_remove_list = list(current_boundary)[:vertices_to_remove_this_layer]
        
        for vertex_idx in vertices_to_remove_list:
            offset_region.remove(vertex_idx)
            removed_count += 1
    
    print(f"Region offset: Removed {removed_count} boundary vertices ({removed_count/len(region_vertices_set)*100:.1f}% of region)")
    print(f"Original region size: {len(region_vertices_set)} vertices")
    print(f"Offset region size: {len(offset_region)} vertices")
    
    return offset_region


def detect_curvature_patches(mesh, bending_energies, region_vertices_set, percentile_threshold=95, offset_percentage=0.1):
    """
    Detect and count patches of high curvature within a region.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        bending_energies (numpy.ndarray): Bending energy values for each vertex
        region_vertices_set (set): Set of vertex indices that belong to the current region
        percentile_threshold (float): Percentile threshold for high curvature (default: 95)
        offset_percentage (float): Percentage of region to offset inward to avoid edge artifacts (default: 0.1)
    
    Returns:
        dict: Dictionary containing patch information
    """
    # Offset region boundaries to avoid edge artifacts
    if offset_percentage > 0:
        print(f"Offsetting region boundaries by {offset_percentage*100:.1f}% to avoid edge artifacts...")
        analysis_region = offset_region_boundaries(mesh, region_vertices_set, offset_percentage)
    else:
        analysis_region = region_vertices_set
    
    # Get non-zero energies for the offset region
    region_energies = [bending_energies[i] for i in analysis_region if bending_energies[i] > 0]
    
    if len(region_energies) == 0:
        return {
            'num_patches': 0,
            'patches': [],
            'high_curvature_threshold': 0,
            'high_curvature_vertices': set(),
            'analysis_region_size': len(analysis_region),
            'original_region_size': len(region_vertices_set)
        }
    
    # Use the same normalization as the visualization function
    # This ensures consistency between visualization and patch detection
    non_zero_energies = [e for e in region_energies if e > 0]
    
    if len(non_zero_energies) > 0:
        # Use 95th percentile as the upper bound (same as visualization)
        percentile_95 = np.percentile(non_zero_energies, 95)
        upper_bound = min(percentile_95, np.max(non_zero_energies))
        
        # Normalize energies the same way as visualization
        normalized_energies = np.clip(np.array(region_energies) / upper_bound, 0, 1)
        
        # Find vertices that would appear red in visualization
        # Red vertices have normalized_energy > 0.5 (more red than blue)
        red_threshold = 0.5
        high_curvature_vertices = set()
        
        for i, vertex_idx in enumerate(analysis_region):
            if bending_energies[vertex_idx] > 0:
                # Find the corresponding normalized energy
                energy_idx = region_energies.index(bending_energies[vertex_idx])
                if energy_idx < len(normalized_energies):
                    normalized_energy = normalized_energies[energy_idx]
                    if normalized_energy > red_threshold:
                        high_curvature_vertices.add(vertex_idx)
    else:
        high_curvature_vertices = set()
        upper_bound = 0
    
    if len(high_curvature_vertices) == 0:
        return {
            'num_patches': 0,
            'patches': [],
            'high_curvature_threshold': upper_bound,
            'high_curvature_vertices': high_curvature_vertices,
            'analysis_region_size': len(analysis_region),
            'original_region_size': len(region_vertices_set)
        }
    
    # Build adjacency graph for the mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Create vertex adjacency dictionary
    vertex_adjacency = {}
    for face in faces:
        for i in range(3):
            v1, v2, v3 = face[i], face[(i+1)%3], face[(i+2)%3]
            if v1 not in vertex_adjacency:
                vertex_adjacency[v1] = set()
            if v2 not in vertex_adjacency:
                vertex_adjacency[v2] = set()
            vertex_adjacency[v1].add(v2)
            vertex_adjacency[v2].add(v1)
    
    # Find connected components (patches) among high curvature vertices
    patches = []
    visited = set()
    
    for start_vertex in high_curvature_vertices:
        if start_vertex in visited:
            continue
        
        # BFS to find connected component
        patch = set()
        queue = [start_vertex]
        visited.add(start_vertex)
        
        while queue:
            current_vertex = queue.pop(0)
            patch.add(current_vertex)
            
            # Check neighbors
            if current_vertex in vertex_adjacency:
                for neighbor in vertex_adjacency[current_vertex]:
                    if (neighbor in high_curvature_vertices and 
                        neighbor not in visited):
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        if len(patch) > 0:
            patches.append(patch)
    
    # Calculate patch statistics
    patch_sizes = [len(patch) for patch in patches]
    
    return {
        'num_patches': len(patches),
        'patches': patches,
        'patch_sizes': patch_sizes,
        'high_curvature_threshold': upper_bound,
        'high_curvature_vertices': high_curvature_vertices,
        'total_high_curvature_vertices': len(high_curvature_vertices),
        'avg_patch_size': np.mean(patch_sizes) if patch_sizes else 0,
        'max_patch_size': np.max(patch_sizes) if patch_sizes else 0,
        'min_patch_size': np.min(patch_sizes) if patch_sizes else 0,
        'red_threshold_used': 0.5,
        'analysis_region_size': len(analysis_region),
        'original_region_size': len(region_vertices_set),
        'offset_percentage': offset_percentage
    }


def visualize_curvature_patches(mesh, bending_energies, region_vertices_set, patch_info, window_name="Curvature Patches", offset_region_set=None):
    """
    Visualize curvature patches with different colors for each patch.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        bending_energies (numpy.ndarray): Bending energy values for each vertex
        region_vertices_set (set): Set of vertex indices that belong to the current region
        patch_info (dict): Patch information from detect_curvature_patches
        window_name (str): Name of the visualization window
        offset_region_set (set): Set of vertex indices for the offset region (if None, no offset is shown)
    """
    # Use percentile-based normalization to handle extreme outliers (only for non-zero values)
    non_zero_energies = bending_energies[bending_energies > 0]
    
    if len(non_zero_energies) > 0:
        percentile_95 = np.percentile(non_zero_energies, 95)
        
        # Create a more robust normalization that handles outliers
        if np.max(non_zero_energies) > np.min(non_zero_energies):
            # Use 95th percentile as the upper bound for better visualization
            upper_bound = min(percentile_95, np.max(non_zero_energies))
            normalized_energies = np.clip(bending_energies / upper_bound, 0, 1)
        else:
            normalized_energies = np.zeros_like(bending_energies)
    else:
        normalized_energies = np.zeros_like(bending_energies)
        percentile_95 = 0
    
    # Create color map
    colors = np.zeros((len(normalized_energies), 3))
    
    # Generate distinct colors for patches
    num_patches = patch_info['num_patches']
    if num_patches > 0:
        # Use a color palette for patches
        patch_colors = []
        for i in range(num_patches):
            # Generate distinct colors using HSV
            hue = i / num_patches
            saturation = 0.8
            value = 0.9
            
            # Convert HSV to RGB
            h = hue * 6
            c = value * saturation
            x = c * (1 - abs(h % 2 - 1))
            m = value - c
            
            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            patch_colors.append([r + m, g + m, b + m])
    
    # Apply colors
    for i, (energy, normalized_energy) in enumerate(zip(bending_energies, normalized_energies)):
        if i in region_vertices_set and energy > 0:
            if offset_region_set is not None and i not in offset_region_set:
                # Vertex in original region but not in offset region (boundary area) - color black
                colors[i, :] = 0.0  # Black color
            else:
                # Check if this vertex belongs to a patch
                vertex_in_patch = False
                for patch_idx, patch in enumerate(patch_info['patches']):
                    if i in patch:
                        colors[i] = patch_colors[patch_idx]
                        vertex_in_patch = True
                        break
                
                if not vertex_in_patch:
                    # Region vertex but not in a patch: use standard blue-red coloring
                    colors[i, 0] = normalized_energy  # Red channel (high energy)
                    colors[i, 2] = 1.0 - normalized_energy  # Blue channel (low energy)
        else:
            # Non-region vertex: grey
            colors[i, :] = 0.5  # Grey color
    
    # Apply colors to mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Visualize
    print(f"\nVisualizing curvature patches...")
    print(f"High curvature threshold: {patch_info['high_curvature_threshold']:.6f}")
    print(f"Number of patches: {patch_info['num_patches']}")
    print(f"Total high curvature vertices: {patch_info['total_high_curvature_vertices']}")
    if patch_info['num_patches'] > 0:
        print(f"Average patch size: {patch_info['avg_patch_size']:.1f}")
        print(f"Largest patch: {patch_info['max_patch_size']} vertices")
        print(f"Smallest patch: {patch_info['min_patch_size']} vertices")
    
    if offset_region_set is not None:
        print("Color coding: Different colors = Different patches, Blue-Red = Standard curvature, Black = Excluded boundary, Grey = Other regions")
    else:
        print("Color coding: Different colors = Different patches, Blue-Red = Standard curvature, Grey = Other regions")
    
    o3d.visualization.draw_geometries(
        [mesh, coordinate_frame],
        window_name=window_name,
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    )


def calculate_region_area(mesh, region_vertices_set):
    """
    Calculate the surface area of a region.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        region_vertices_set (set): Set of vertex indices that belong to the current region
    
    Returns:
        float: Total surface area of the region
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Find faces that belong to this region (faces with all vertices in the region)
    region_faces = []
    for face_idx, face in enumerate(faces):
        if all(vertex_idx in region_vertices_set for vertex_idx in face):
            region_faces.append(face_idx)
    
    # Calculate area of each face in the region
    total_area = 0.0
    for face_idx in region_faces:
        face = faces[face_idx]
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Calculate face area using cross product
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross_product = np.cross(edge1, edge2)
        face_area = 0.5 * np.linalg.norm(cross_product)
        total_area += face_area
    
    return total_area


def normalize_patch_count_by_area(patch_info, region_area):
    """
    Normalize patch count and statistics by region area.
    
    Args:
        patch_info (dict): Patch information from detect_curvature_patches
        region_area (float): Surface area of the region
    
    Returns:
        dict: Patch information with area-normalized metrics
    """
    if region_area <= 0:
        return patch_info
    
    # Calculate normalized metrics
    normalized_patch_count = patch_info['num_patches'] / region_area
    normalized_high_curvature_vertices = patch_info['total_high_curvature_vertices'] / region_area
    
    # Normalize patch sizes by area
    normalized_patch_sizes = [patch_size / region_area for patch_size in patch_info['patch_sizes']]
    
    # Calculate normalized statistics
    normalized_avg_patch_size = np.mean(normalized_patch_sizes) if normalized_patch_sizes else 0
    normalized_max_patch_size = np.max(normalized_patch_sizes) if normalized_patch_sizes else 0
    normalized_min_patch_size = np.min(normalized_patch_sizes) if normalized_patch_sizes else 0
    
    # Add normalized metrics to the patch info
    patch_info.update({
        'region_area': region_area,
        'normalized_patch_count': normalized_patch_count,
        'normalized_high_curvature_vertices': normalized_high_curvature_vertices,
        'normalized_patch_sizes': normalized_patch_sizes,
        'normalized_avg_patch_size': normalized_avg_patch_size,
        'normalized_max_patch_size': normalized_max_patch_size,
        'normalized_min_patch_size': normalized_min_patch_size
    })
    
    return patch_info


def print_region_comparison_summary(region_results):
    """
    Print a summary comparison of normalized patch counts across all regions.
    
    Args:
        region_results (list): List of region results from segmentation analysis
    """
    print(f"\n{'='*80}")
    print(f"REGION COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Collect normalized metrics from all regions
    comparison_data = []
    for i, region_result in enumerate(region_results):
        if 'patch_info' in region_result:
            patch_info = region_result['patch_info']
            comparison_data.append({
                'region_id': i + 1,
                'area': patch_info.get('region_area', 0),
                'num_patches': patch_info.get('num_patches', 0),
                'normalized_patch_count': patch_info.get('normalized_patch_count', 0),
                'normalized_high_curvature_vertices': patch_info.get('normalized_high_curvature_vertices', 0),
                'avg_patch_size': patch_info.get('avg_patch_size', 0),
                'normalized_avg_patch_size': patch_info.get('normalized_avg_patch_size', 0)
            })
    
    if not comparison_data:
        print("No patch data available for comparison.")
        return
    
    # Sort by normalized patch count (descending)
    comparison_data.sort(key=lambda x: x['normalized_patch_count'], reverse=True)
    
    print(f"{'Region':<8} {'Area':<12} {'Patches':<10} {'Norm. Patches':<15} {'Norm. Vertices':<15} {'Avg Size':<10}")
    print(f"{'-'*8} {'-'*12} {'-'*10} {'-'*15} {'-'*15} {'-'*10}")
    
    for data in comparison_data:
        print(f"{data['region_id']:<8} {data['area']:<12.4f} {data['num_patches']:<10} "
              f"{data['normalized_patch_count']:<15.6f} {data['normalized_high_curvature_vertices']:<15.6f} "
              f"{data['avg_patch_size']:<10.1f}")
    
    # Calculate overall statistics
    total_patches = sum(data['num_patches'] for data in comparison_data)
    total_area = sum(data['area'] for data in comparison_data)
    avg_normalized_patch_count = np.mean([data['normalized_patch_count'] for data in comparison_data])
    std_normalized_patch_count = np.std([data['normalized_patch_count'] for data in comparison_data])
    
    print(f"\nOverall Statistics:")
    print(f"  Total patches across all regions: {total_patches}")
    print(f"  Total area across all regions: {total_area:.4f} square units")
    print(f"  Average normalized patch count: {avg_normalized_patch_count:.6f} patches per unit area")
    print(f"  Standard deviation of normalized patch count: {std_normalized_patch_count:.6f}")
    
    # Identify regions with highest and lowest patch density
    if comparison_data:
        highest_density = comparison_data[0]
        lowest_density = comparison_data[-1]
        
        print(f"\nRegion with highest patch density: Region {highest_density['region_id']}")
        print(f"  Normalized patch count: {highest_density['normalized_patch_count']:.6f} patches per unit area")
        print(f"  Area: {highest_density['area']:.4f} square units")
        print(f"  Total patches: {highest_density['num_patches']}")
        
        print(f"\nRegion with lowest patch density: Region {lowest_density['region_id']}")
        print(f"  Normalized patch count: {lowest_density['normalized_patch_count']:.6f} patches per unit area")
        print(f"  Area: {lowest_density['area']:.4f} square units")
        print(f"  Total patches: {lowest_density['num_patches']}")
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Load and visualize 3D models from a folder")
    parser.add_argument(
        "--folder", 
        type=str, 
        default="data/input_fragments",
        help="Path to folder containing 3D models (default: data/input_fragments)"
    )
    parser.add_argument(
        "--no-trimesh-fallback", 
        action="store_true",
        help="Disable trimesh fallback loading"
    )
    parser.add_argument(
        "--window-name", 
        type=str, 
        default="3D Models Viewer",
        help="Name of the visualization window"
    )
    parser.add_argument(
        "--all-together", 
        action="store_true",
        help="Visualize all models together instead of one by one"
    )
    parser.add_argument(
        "--no-curvature-analysis",
        action="store_true",
        help="Disable curvature analysis and use regular visualization instead"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=50,
        help="Number of nearest neighbors for curvature analysis (default: 10)"
    )
    parser.add_argument(
        "--segment-first",
        action="store_true",
        help="Segment the mesh before analyzing curvature for each region separately"
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=30.0,
        help="Angle threshold for segmentation (default: 30.0 degrees)"
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=0.1,
        help="Curvature threshold for segmentation (default: 0.1)"
    )
    parser.add_argument(
        "--min-region-size",
        type=int,
        default=50,
        help="Minimum region size for segmentation (default: 50 faces)"
    )
    parser.add_argument(
        "--max-region-size",
        type=int,
        default=5000,
        help="Maximum region size for segmentation (default: 5000 faces)"
    )
    parser.add_argument(
        "--region-offset",
        type=float,
        default=0.2,
        help="Percentage of region to offset inward to avoid edge artifacts (default: 0.1 = 10%)"
    )
    
    args = parser.parse_args()
    
    print("3D Model Loader and Visualizer")
    print("=" * 40)
    print(f"Loading models from: {args.folder}")
    
    # Load models
    geometries = load_3d_models_from_folder(
        args.folder, 
        use_trimesh_fallback=not args.no_trimesh_fallback
    )
    
    if geometries:
        print(f"\nSuccessfully loaded {len(geometries)} models")
        
        if not args.no_curvature_analysis:
            # Analyze curvature for each mesh (default behavior)
            for i, geometry in enumerate(geometries):
                if isinstance(geometry, o3d.geometry.TriangleMesh):
                    print(f"\n{'='*50}")
                    print(f"Analyzing curvature for model {i+1}/{len(geometries)}")
                    print(f"{'='*50}")
                    
                    if args.segment_first:
                        # Segment first, then analyze each region
                        print("Using segmented curvature analysis...")
                        
                        # Prepare segmentation parameters
                        segmentation_params = {
                            'angle_threshold': args.angle_threshold,
                            'curvature_threshold': args.curvature_threshold,
                            'min_region_size': args.min_region_size,
                            'max_region_size': args.max_region_size
                        }
                        
                        # Perform segmented analysis
                        segmentation_results = segment_mesh_and_analyze_curvature(
                            geometry, 
                            args.k_neighbors, 
                            segmentation_params
                        )
                        
                        # Visualize each region
                        visualize_segmented_curvature(
                            geometry,
                            segmentation_results,
                            f"Segmented Curvature - Model {i+1}",
                            args.region_offset
                        )
                    else:
                        # Regular single-region analysis
                        print("Using single-region curvature analysis...")
                        
                        # Analyze curvature
                        stats = analyze_mesh_curvature(geometry, args.k_neighbors)
                        
                        # Visualize bending energy
                        visualize_bending_energy(
                            geometry, 
                            stats['bending_energies'],
                            f"Bending Energy - Model {i+1}"
                        )
                    
                    # Ask user if they want to continue to next model
                    if i < len(geometries) - 1:
                        response = input(f"\nPress Enter to analyze next model, or 'q' to quit: ").strip().lower()
                        if response == 'q':
                            print("Analysis stopped by user.")
                            break
                else:
                    print(f"Model {i+1} is not a triangle mesh, skipping curvature analysis.")
        else:
            # Regular visualization (when --no-curvature-analysis is used)
            visualize_models(geometries, args.window_name, visualize_one_by_one=not args.all_together)
    else:
        print("No models were loaded successfully.")


if __name__ == "__main__":
    main()
