"""
Visualization Module

This module provides functions for visualizing curvature and roughness analysis results.
"""

import numpy as np
import open3d as o3d


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


def visualize_roughness_characteristic(mesh, roughness_characteristics, window_name="Surface Roughness Visualization"):
    """
    Visualize the surface roughness characteristic values on the mesh using color coding.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        roughness_characteristics (numpy.ndarray): Surface roughness characteristic values for each vertex
        window_name (str): Name of the visualization window
    """
    # Use percentile-based normalization to handle extreme outliers
    percentile_95 = np.percentile(roughness_characteristics, 95)
    percentile_99 = np.percentile(roughness_characteristics, 99)
    
    # Create a more robust normalization that handles outliers
    if np.max(roughness_characteristics) > np.min(roughness_characteristics):
        # Use 95th percentile as the upper bound for better visualization
        upper_bound = min(percentile_95, np.max(roughness_characteristics))
        normalized_roughness = np.clip(roughness_characteristics / upper_bound, 0, 1)
    else:
        normalized_roughness = np.zeros_like(roughness_characteristics)
    
    # Create color map (red for high roughness, blue for low roughness)
    colors = np.zeros((len(normalized_roughness), 3))
    colors[:, 0] = normalized_roughness  # Red channel (high roughness)
    colors[:, 2] = 1.0 - normalized_roughness  # Blue channel (low roughness)
    
    # Apply colors to mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Visualize
    print(f"\nVisualizing surface roughness characteristic...")
    print(f"Min roughness: {np.min(roughness_characteristics):.6f}")
    print(f"Max roughness: {np.max(roughness_characteristics):.6f}")
    print(f"Mean roughness: {np.mean(roughness_characteristics):.6f}")
    print(f"95th percentile: {percentile_95:.6f}")
    print(f"99th percentile: {percentile_99:.6f}")
    print(f"Upper bound used for normalization: {min(percentile_95, np.max(roughness_characteristics)):.6f}")
    print("Color coding: Red = High roughness, Blue = Low roughness")
    print("Note: Using 95th percentile normalization to handle extreme outliers")
    
    o3d.visualization.draw_geometries(
        [mesh, coordinate_frame],
        window_name=window_name,
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    )


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


def visualize_roughness_with_grey_regions(mesh, roughness_characteristics, region_vertices_set, window_name="Surface Roughness Visualization", offset_region_set=None):
    """
    Visualize the surface roughness characteristic values on the mesh using color coding, with non-region vertices in grey.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        roughness_characteristics (numpy.ndarray): Surface roughness characteristic values for each vertex (0 for non-region vertices)
        region_vertices_set (set): Set of vertex indices that belong to the current region
        window_name (str): Name of the visualization window
        offset_region_set (set): Set of vertex indices for the offset region (if None, no offset is shown)
    """
    # Use percentile-based normalization to handle extreme outliers (only for non-zero values)
    non_zero_roughness = roughness_characteristics[roughness_characteristics > 0]
    
    if len(non_zero_roughness) > 0:
        percentile_95 = np.percentile(non_zero_roughness, 95)
        
        # Create a more robust normalization that handles outliers
        if np.max(non_zero_roughness) > np.min(non_zero_roughness):
            # Use 95th percentile as the upper bound for better visualization
            upper_bound = min(percentile_95, np.max(non_zero_roughness))
            normalized_roughness = np.clip(roughness_characteristics / upper_bound, 0, 1)
        else:
            normalized_roughness = np.zeros_like(roughness_characteristics)
    else:
        normalized_roughness = np.zeros_like(roughness_characteristics)
        percentile_95 = 0
    
    # Create color map (red for high roughness, blue for low roughness, grey for non-region, black for offset)
    colors = np.zeros((len(normalized_roughness), 3))
    
    for i, (roughness, normalized_roughness_value) in enumerate(zip(roughness_characteristics, normalized_roughness)):
        if i in region_vertices_set and roughness > 0:
            if offset_region_set is not None and i not in offset_region_set:
                # Vertex in original region but not in offset region (boundary area) - color black
                colors[i, :] = 0.0  # Black color
            else:
                # Region vertex: color based on roughness
                colors[i, 0] = normalized_roughness_value  # Red channel (high roughness)
                colors[i, 2] = 1.0 - normalized_roughness_value  # Blue channel (low roughness)
        else:
            # Non-region vertex: grey
            colors[i, :] = 0.5  # Grey color
    
    # Apply colors to mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Visualize
    print(f"\nVisualizing surface roughness characteristic...")
    if len(non_zero_roughness) > 0:
        print(f"Min roughness: {np.min(non_zero_roughness):.6f}")
        print(f"Max roughness: {np.max(non_zero_roughness):.6f}")
        print(f"Mean roughness: {np.mean(non_zero_roughness):.6f}")
        print(f"95th percentile: {percentile_95:.6f}")
        print(f"Upper bound used for normalization: {min(percentile_95, np.max(non_zero_roughness)):.6f}")
    else:
        print("No roughness data for this region")
    
    if offset_region_set is not None:
        print("Color coding: Red = High roughness, Blue = Low roughness, Black = Excluded boundary, Grey = Other regions")
    else:
        print("Color coding: Red = High roughness, Blue = Low roughness, Grey = Other regions")
    print("Note: Using 95th percentile normalization to handle extreme outliers")
    
    o3d.visualization.draw_geometries(
        [mesh, coordinate_frame],
        window_name=window_name,
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    )


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


def visualize_roughness_patches(mesh, roughness_characteristics, region_vertices_set, patch_info, window_name="Roughness Patches", offset_region_set=None):
    """
    Visualize roughness patches with different colors for each patch.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        roughness_characteristics (numpy.ndarray): Surface roughness characteristic values for each vertex
        region_vertices_set (set): Set of vertex indices that belong to the current region
        patch_info (dict): Patch information from detect_roughness_patches
        window_name (str): Name of the visualization window
        offset_region_set (set): Set of vertex indices for the offset region (if None, no offset is shown)
    """
    # Use percentile-based normalization to handle extreme outliers (only for non-zero values)
    non_zero_roughness = roughness_characteristics[roughness_characteristics > 0]
    
    if len(non_zero_roughness) > 0:
        percentile_95 = np.percentile(non_zero_roughness, 95)
        
        # Create a more robust normalization that handles outliers
        if np.max(non_zero_roughness) > np.min(non_zero_roughness):
            # Use 95th percentile as the upper bound for better visualization
            upper_bound = min(percentile_95, np.max(non_zero_roughness))
            normalized_roughness = np.clip(roughness_characteristics / upper_bound, 0, 1)
        else:
            normalized_roughness = np.zeros_like(roughness_characteristics)
    else:
        normalized_roughness = np.zeros_like(roughness_characteristics)
        percentile_95 = 0
    
    # Create color map
    colors = np.zeros((len(normalized_roughness), 3))
    
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
    for i, (roughness, normalized_roughness_value) in enumerate(zip(roughness_characteristics, normalized_roughness)):
        if i in region_vertices_set and roughness > 0:
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
                    colors[i, 0] = normalized_roughness_value  # Red channel (high roughness)
                    colors[i, 2] = 1.0 - normalized_roughness_value  # Blue channel (low roughness)
        else:
            # Non-region vertex: grey
            colors[i, :] = 0.5  # Grey color
    
    # Apply colors to mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Visualize
    print(f"\nVisualizing roughness patches...")
    print(f"High roughness threshold: {patch_info['high_roughness_threshold']:.6f}")
    print(f"Number of patches: {patch_info['num_patches']}")
    print(f"Total high roughness vertices: {patch_info['total_high_roughness_vertices']}")
    if patch_info['num_patches'] > 0:
        print(f"Average patch size: {patch_info['avg_patch_size']:.1f}")
        print(f"Largest patch: {patch_info['max_patch_size']} vertices")
        print(f"Smallest patch: {patch_info['min_patch_size']} vertices")
    
    if offset_region_set is not None:
        print("Color coding: Different colors = Different patches, Blue-Red = Standard roughness, Black = Excluded boundary, Grey = Other regions")
    else:
        print("Color coding: Different colors = Different patches, Blue-Red = Standard roughness, Grey = Other regions")
    
    o3d.visualization.draw_geometries(
        [mesh, coordinate_frame],
        window_name=window_name,
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    ) 