"""
Patch Detection Module

This module provides functions for detecting and analyzing patches of high curvature or roughness.
"""

import numpy as np
import open3d as o3d
from mesh_utils import offset_region_boundaries, calculate_region_area


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


def detect_roughness_patches(mesh, roughness_characteristics, region_vertices_set, percentile_threshold=95, offset_percentage=0.1):
    """
    Detect and count patches of high roughness within a region.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        roughness_characteristics (numpy.ndarray): Surface roughness characteristic values for each vertex
        region_vertices_set (set): Set of vertex indices that belong to the current region
        percentile_threshold (float): Percentile threshold for high roughness (default: 95)
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
    
    # Get non-zero roughness for the offset region
    region_roughness = [roughness_characteristics[i] for i in analysis_region if roughness_characteristics[i] > 0]
    
    if len(region_roughness) == 0:
        return {
            'num_patches': 0,
            'patches': [],
            'high_roughness_threshold': 0,
            'high_roughness_vertices': set(),
            'analysis_region_size': len(analysis_region),
            'original_region_size': len(region_vertices_set)
        }
    
    # Use the same normalization as the visualization function
    # This ensures consistency between visualization and patch detection
    non_zero_roughness = [r for r in region_roughness if r > 0]
    
    if len(non_zero_roughness) > 0:
        # Use 95th percentile as the upper bound (same as visualization)
        percentile_95 = np.percentile(non_zero_roughness, 95)
        upper_bound = min(percentile_95, np.max(non_zero_roughness))
        
        # Normalize roughness the same way as visualization
        normalized_roughness = np.clip(np.array(region_roughness) / upper_bound, 0, 1)
        
        # Find vertices that would appear red in visualization
        # Red vertices have normalized_roughness > 0.5 (more red than blue)
        red_threshold = 0.5
        high_roughness_vertices = set()
        
        for i, vertex_idx in enumerate(analysis_region):
            if roughness_characteristics[vertex_idx] > 0:
                # Find the corresponding normalized roughness
                roughness_idx = region_roughness.index(roughness_characteristics[vertex_idx])
                if roughness_idx < len(normalized_roughness):
                    normalized_roughness_value = normalized_roughness[roughness_idx]
                    if normalized_roughness_value > red_threshold:
                        high_roughness_vertices.add(vertex_idx)
    else:
        high_roughness_vertices = set()
        upper_bound = 0
    
    if len(high_roughness_vertices) == 0:
        return {
            'num_patches': 0,
            'patches': [],
            'high_roughness_threshold': upper_bound,
            'high_roughness_vertices': high_roughness_vertices,
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
    
    # Find connected components (patches) among high roughness vertices
    patches = []
    visited = set()
    
    for start_vertex in high_roughness_vertices:
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
                    if (neighbor in high_roughness_vertices and 
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
        'high_roughness_threshold': upper_bound,
        'high_roughness_vertices': high_roughness_vertices,
        'total_high_roughness_vertices': len(high_roughness_vertices),
        'avg_patch_size': np.mean(patch_sizes) if patch_sizes else 0,
        'max_patch_size': np.max(patch_sizes) if patch_sizes else 0,
        'min_patch_size': np.min(patch_sizes) if patch_sizes else 0,
        'red_threshold_used': 0.5,
        'analysis_region_size': len(analysis_region),
        'original_region_size': len(region_vertices_set),
        'offset_percentage': offset_percentage
    }


def normalize_patch_count_by_area(patch_info, region_area):
    """
    Normalize patch count and statistics by region area.
    
    Args:
        patch_info (dict): Patch information from detect_curvature_patches or detect_roughness_patches
        region_area (float): Surface area of the region
    
    Returns:
        dict: Patch information with area-normalized metrics
    """
    if region_area <= 0:
        return patch_info
    
    # Determine if this is curvature or roughness patch info
    is_roughness = 'total_high_roughness_vertices' in patch_info
    
    # Calculate normalized metrics
    normalized_patch_count = patch_info['num_patches'] / region_area
    
    if is_roughness:
        normalized_high_roughness_vertices = patch_info['total_high_roughness_vertices'] / region_area
    else:
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
        'normalized_patch_sizes': normalized_patch_sizes,
        'normalized_avg_patch_size': normalized_avg_patch_size,
        'normalized_max_patch_size': normalized_max_patch_size,
        'normalized_min_patch_size': normalized_min_patch_size
    })
    
    # Add the appropriate normalized vertex count
    if is_roughness:
        patch_info['normalized_high_roughness_vertices'] = normalized_high_roughness_vertices
    else:
        patch_info['normalized_high_curvature_vertices'] = normalized_high_curvature_vertices
    
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
            
            # Determine if this is curvature or roughness patch info
            is_roughness = 'normalized_high_roughness_vertices' in patch_info
            
            comparison_data.append({
                'region_id': i + 1,
                'area': patch_info.get('region_area', 0),
                'num_patches': patch_info.get('num_patches', 0),
                'normalized_patch_count': patch_info.get('normalized_patch_count', 0),
                'normalized_high_vertices': patch_info.get('normalized_high_roughness_vertices' if is_roughness else 'normalized_high_curvature_vertices', 0),
                'avg_patch_size': patch_info.get('avg_patch_size', 0),
                'normalized_avg_patch_size': patch_info.get('normalized_avg_patch_size', 0),
                'is_roughness': is_roughness
            })
    
    if not comparison_data:
        print("No patch data available for comparison.")
        return
    
    # Sort by normalized patch count (descending)
    comparison_data.sort(key=lambda x: x['normalized_patch_count'], reverse=True)
    
    # Determine the type of analysis based on the first region
    analysis_type = "Roughness" if comparison_data[0]['is_roughness'] else "Curvature"
    
    print(f"Analysis Type: {analysis_type}")
    print(f"{'Region':<8} {'Area':<12} {'Patches':<10} {'Norm. Patches':<15} {'Norm. Vertices':<15} {'Avg Size':<10}")
    print(f"{'-'*8} {'-'*12} {'-'*10} {'-'*15} {'-'*15} {'-'*10}")
    
    for data in comparison_data:
        print(f"{data['region_id']:<8} {data['area']:<12.4f} {data['num_patches']:<10} "
              f"{data['normalized_patch_count']:<15.6f} {data['normalized_high_vertices']:<15.6f} "
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