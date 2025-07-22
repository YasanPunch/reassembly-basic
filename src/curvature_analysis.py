"""
Curvature Analysis Module

This module provides functions for analyzing mesh curvature using Local Bending Energy.
"""

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


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