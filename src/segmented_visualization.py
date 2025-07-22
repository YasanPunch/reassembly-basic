"""
Segmented Visualization Module

This module provides functions for visualizing segmented analysis results.
"""

import numpy as np
import open3d as o3d
from patch_detection import detect_curvature_patches, detect_roughness_patches, normalize_patch_count_by_area, print_region_comparison_summary
from mesh_utils import calculate_region_area, offset_region_boundaries
from visualization import (
    visualize_bending_energy_with_grey_regions, 
    visualize_roughness_with_grey_regions,
    visualize_curvature_patches,
    visualize_roughness_patches
)


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


def visualize_segmented_roughness(mesh, segmentation_results, window_name="Segmented Roughness Analysis", region_offset=0.1):
    """
    Visualize roughness analysis for each segmented region.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Original mesh
        segmentation_results (dict): Results from segment_mesh_and_analyze_roughness
        window_name (str): Name of the visualization window
        region_offset (float): Percentage of region to offset inward to avoid edge artifacts
    """
    region_results = segmentation_results['region_results']
    
    print(f"\nVisualizing {len(region_results)} segmented regions (roughness analysis)...")
    
    # Get the original mesh vertices and faces
    original_vertices = np.asarray(mesh.vertices)
    original_faces = np.asarray(mesh.triangles)
    
    for i, region_result in enumerate(region_results):
        print(f"\n--- Visualizing Region {i+1}/{len(region_results)} ---")
        print(f"Region {i+1} statistics:")
        print(f"  Faces: {region_result['num_faces']}")
        print(f"  Vertices: {region_result['num_vertices']}")
        print(f"  Min roughness: {region_result['stats']['min_roughness']:.6f}")
        print(f"  Max roughness: {region_result['stats']['max_roughness']:.6f}")
        print(f"  Mean roughness: {region_result['stats']['mean_roughness']:.6f}")
        
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
        
        # Get the region mesh vertices and their roughness characteristics
        region_vertices = np.asarray(region_result['region_mesh'].vertices)
        region_roughness_characteristics = region_result['stats']['roughness_characteristics']
        
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
        
        # Create roughness characteristic array for the full mesh (grey for non-region vertices)
        full_mesh_roughness_characteristics = np.zeros(num_vertices)
        
        # Apply region roughness characteristics to the mapped vertices
        for vertex_idx in region_vertices_set:
            if vertex_idx < num_vertices and vertex_idx in original_to_region_mapping:
                region_vertex_idx = original_to_region_mapping[vertex_idx]
                if region_vertex_idx < len(region_roughness_characteristics):
                    full_mesh_roughness_characteristics[vertex_idx] = region_roughness_characteristics[region_vertex_idx]
        
        # Detect and analyze roughness patches
        print(f"\n--- Analyzing Roughness Patches for Region {i+1} ---")
        patch_info = detect_roughness_patches(
            viz_mesh, 
            full_mesh_roughness_characteristics, 
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
        print(f"Roughness Patch Analysis Results:")
        print(f"  Region area: {region_area:.6f} square units")
        print(f"  High roughness threshold: {patch_info['high_roughness_threshold']:.6f}")
        print(f"  Number of patches: {patch_info['num_patches']}")
        print(f"  Normalized patch count: {patch_info['normalized_patch_count']:.6f} patches per unit area")
        print(f"  Total high roughness vertices: {patch_info['total_high_roughness_vertices']}")
        print(f"  Normalized high roughness vertices: {patch_info['normalized_high_roughness_vertices']:.6f} per unit area")
        if patch_info['num_patches'] > 0:
            print(f"  Average patch size: {patch_info['avg_patch_size']:.1f} vertices")
            print(f"  Normalized average patch size: {patch_info['normalized_avg_patch_size']:.6f} per unit area")
            print(f"  Largest patch: {patch_info['max_patch_size']} vertices")
            print(f"  Normalized largest patch: {patch_info['normalized_max_patch_size']:.6f} per unit area")
            print(f"  Smallest patch: {patch_info['min_patch_size']} vertices")
            print(f"  Normalized smallest patch: {patch_info['normalized_min_patch_size']:.6f} per unit area")
        
        # Store patch information in region results
        region_result['patch_info'] = patch_info
        
        # Use the existing visualize_roughness_characteristic function with the full mesh
        # But we need to modify it slightly to handle the grey regions
        visualize_roughness_with_grey_regions(
            viz_mesh, 
            full_mesh_roughness_characteristics, 
            region_vertices_set,
            f"{window_name} - Region {i+1}",
            offset_region_set
        )
        
        # Visualize patches if any exist
        if patch_info['num_patches'] > 0:
            print(f"\nVisualizing roughness patches for Region {i+1}...")
            visualize_roughness_patches(
                viz_mesh,
                full_mesh_roughness_characteristics,
                region_vertices_set,
                patch_info,
                f"{window_name} - Patches - Region {i+1}",
                offset_region_set
            )
    
    # Print comparison summary across all regions
    print_region_comparison_summary(region_results) 