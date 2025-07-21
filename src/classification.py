import os
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
import argparse
from sklearn.neighbors import NearestNeighbors
from utils.test_utils import load_3d_models_from_folder, visualize_models


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


def main():
    parser = argparse.ArgumentParser(description="Load and visualize 3D models from a folder")
    parser.add_argument(
        "--folder", 
        type=str, 
        default="../data/input_fragments",
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
