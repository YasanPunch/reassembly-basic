"""
Mesh Utilities Module

This module provides utility functions for mesh processing and analysis.
"""

import numpy as np
import open3d as o3d
import trimesh
from segmentation import region_growing_segmentation


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
    from .curvature_analysis import analyze_mesh_curvature
    
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


def segment_mesh_and_analyze_roughness(mesh, k=10, r=None, segmentation_params=None):
    """
    Segment the mesh and analyze roughness for each region separately.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        k (int): Number of nearest neighbors for local bending energy calculation
        r (float): Kernel radius for neighborhood definition. If None, will be auto-calculated
        segmentation_params (dict): Parameters for segmentation
    
    Returns:
        dict: Dictionary containing segmentation and roughness analysis results
    """
    from roughness_analysis import analyze_mesh_roughness
    
    print(f"Segmenting mesh and analyzing roughness for each region...")
    
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
        return analyze_mesh_roughness(mesh, k, r)
    
    # Analyze each region
    region_results = []
    all_roughness_characteristics = []
    
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
        
        # Convert back to Open3D for roughness analysis
        region_o3d_mesh = o3d.geometry.TriangleMesh()
        region_o3d_mesh.vertices = o3d.utility.Vector3dVector(region_mesh.vertices)
        region_o3d_mesh.triangles = o3d.utility.Vector3iVector(region_mesh.faces)
        region_o3d_mesh.compute_vertex_normals()
        
        # Analyze roughness for this region
        region_stats = analyze_mesh_roughness(region_o3d_mesh, k, r)
        
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
        
        # Collect all roughness characteristics for overall statistics
        all_roughness_characteristics.extend(region_stats['roughness_characteristics'])
    
    # Compute overall statistics
    overall_stats = {
        'num_regions': len(segments),
        'total_vertices': len(vertices),
        'total_faces': len(faces),
        'overall_min_roughness': np.min(all_roughness_characteristics),
        'overall_max_roughness': np.max(all_roughness_characteristics),
        'overall_mean_roughness': np.mean(all_roughness_characteristics),
        'overall_std_roughness': np.std(all_roughness_characteristics),
        'overall_median_roughness': np.median(all_roughness_characteristics),
        'region_results': region_results
    }
    
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total regions: {overall_stats['num_regions']}")
    print(f"Overall min roughness: {overall_stats['overall_min_roughness']:.6f}")
    print(f"Overall max roughness: {overall_stats['overall_max_roughness']:.6f}")
    print(f"Overall mean roughness: {overall_stats['overall_mean_roughness']:.6f}")
    print(f"Overall std roughness: {overall_stats['overall_std_roughness']:.6f}")
    print(f"Overall median roughness: {overall_stats['overall_median_roughness']:.6f}")
    
    return overall_stats 