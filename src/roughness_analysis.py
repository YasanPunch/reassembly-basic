"""
Roughness Analysis Module

This module provides functions for analyzing mesh roughness using Surface Roughness Characteristic.
"""

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from curvature_analysis import calculate_local_bending_energy


def calculate_surface_roughness_characteristic(mesh, k=10, r=None):
    """
    Calculate the Surface Roughness Characteristic for each vertex in a mesh.
    
    The Surface Roughness Characteristic ē_k,r(p) is defined as:
    ē_k,r(p) = (1/|N_r(p)|) * sum_{q∈N_r(p)} e_k(q)
    
    where:
    - N_r(p) is the local neighborhood of p with radius r
    - |N_r(p)| is the number of points within the neighborhood N_r(p)
    - e_k(q) is the local bending energy at point q
    - The neighborhood N_r(p) = B_r(p) ∩ Φ, where B_r(p) is a ball of radius r centered at p,
      and Φ is the discrete set of measurement points from the 3D scan
    
    This averaging process smooths out noise and provides a more stable, global measure 
    of roughness around each point.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        k (int): Number of nearest neighbors for local bending energy calculation (default: 10)
        r (float): Kernel radius for neighborhood definition. If None, will be auto-calculated
                  based on mesh density (default: None)
    
    Returns:
        numpy.ndarray: Array of surface roughness characteristic values for each vertex
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
    
    # Auto-calculate radius if not provided
    if r is None:
        # Calculate average edge length as a reasonable default radius
        faces = np.asarray(mesh.triangles)
        edge_lengths = []
        
        for face in faces:
            v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge_lengths.extend([
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v3 - v2),
                np.linalg.norm(v1 - v3)
            ])
        
        avg_edge_length = np.mean(edge_lengths)
        r = avg_edge_length * 3  # Use 3x average edge length as default radius
        print(f"Auto-calculated radius: {r:.6f} (3x average edge length)")
    
    # First, calculate local bending energies for all vertices
    print(f"Calculating local bending energies with k={k}...")
    local_bending_energies = calculate_local_bending_energy(mesh, k)
    
    # Build a spatial index for efficient radius-based neighbor search
    print(f"Building spatial index for radius-based neighbor search (r={r:.6f})...")
    nbrs = NearestNeighbors(algorithm='ball_tree').fit(vertices)
    
    # Find all neighbors within radius r for each vertex
    print("Finding neighbors within radius...")
    roughness_characteristics = np.zeros(n_vertices)
    
    for i in range(n_vertices):
        # Get the current point
        p = vertices[i]
        
        # Find all neighbors within radius r
        # We use radius_neighbors to get all points within distance r
        distances, indices = nbrs.radius_neighbors([p], radius=r)
        
        # Get the neighborhood (excluding the point itself)
        neighborhood_indices = indices[0]
        neighborhood_distances = distances[0]
        
        # Filter out the point itself (distance = 0)
        valid_neighbors = []
        for idx, dist in zip(neighborhood_indices, neighborhood_distances):
            if dist > 1e-10:  # Exclude the point itself
                valid_neighbors.append(idx)
        
        # Calculate the surface roughness characteristic
        if len(valid_neighbors) > 0:
            # Sum the local bending energies of all neighbors
            neighbor_energies = [local_bending_energies[j] for j in valid_neighbors]
            roughness_characteristics[i] = np.mean(neighbor_energies)
        else:
            # If no neighbors found, use the local bending energy of the point itself
            roughness_characteristics[i] = local_bending_energies[i]
    
    print(f"Surface roughness characteristic calculation completed.")
    print(f"  Neighborhood radius: {r:.6f}")
    print(f"  Local bending energy k: {k}")
    print(f"  Min roughness: {np.min(roughness_characteristics):.6f}")
    print(f"  Max roughness: {np.max(roughness_characteristics):.6f}")
    print(f"  Mean roughness: {np.mean(roughness_characteristics):.6f}")
    
    return roughness_characteristics


def analyze_mesh_roughness(mesh, k=10, r=None):
    """
    Analyze mesh roughness using Surface Roughness Characteristic.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh
        k (int): Number of nearest neighbors for local bending energy calculation
        r (float): Kernel radius for neighborhood definition. If None, will be auto-calculated
    
    Returns:
        dict: Dictionary containing roughness analysis results
    """
    print(f"Analyzing mesh roughness with k={k} nearest neighbors...")
    
    # Calculate surface roughness characteristic
    roughness_characteristics = calculate_surface_roughness_characteristic(mesh, k, r)
    
    # Compute statistics
    stats = {
        'min_roughness': np.min(roughness_characteristics),
        'max_roughness': np.max(roughness_characteristics),
        'mean_roughness': np.mean(roughness_characteristics),
        'std_roughness': np.std(roughness_characteristics),
        'median_roughness': np.median(roughness_characteristics),
        'roughness_characteristics': roughness_characteristics
    }
    
    print(f"Roughness Analysis Results:")
    print(f"  Min roughness: {stats['min_roughness']:.6f}")
    print(f"  Max roughness: {stats['max_roughness']:.6f}")
    print(f"  Mean roughness: {stats['mean_roughness']:.6f}")
    print(f"  Std roughness: {stats['std_roughness']:.6f}")
    print(f"  Median roughness: {stats['median_roughness']:.6f}")
    
    return stats 