import open3d as o3d
import trimesh
import numpy as np
from sklearn.cluster import DBSCAN
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

def cluster_faces_by_normal(mesh, eps=0.1, min_samples=10):
    """
    Cluster faces by normal similarity using DBSCAN.
    Args:
        mesh (trimesh.Trimesh): Input mesh
        eps (float): DBSCAN epsilon parameter - max distance between samples
        min_samples (int): DBSCAN min_samples parameter - min cluster size
    Returns:
        list: List of arrays containing face indices for each cluster
    """
    # Get face normals
    face_normals = mesh.face_normals
    
    # Normalize normals (should be normalized already, but just to be safe)
    norms = np.linalg.norm(face_normals, axis=1)
    mask = norms > 1e-10  # Avoid division by zero
    normalized_normals = np.zeros_like(face_normals)
    normalized_normals[mask] = face_normals[mask] / norms[mask, np.newaxis]
    
    # Cluster the normals using DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(normalized_normals)
    labels = db.labels_
    
    # Group faces by their cluster label
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    clusters = []
    
    for i in range(n_clusters):
        # Get indices of faces belonging to cluster i
        cluster_indices = np.where(labels == i)[0]
        clusters.append(cluster_indices)
    
    # Add noise points as a separate cluster if they exist
    if -1 in labels:
        noise_indices = np.where(labels == -1)[0]
        if len(noise_indices) > 0:
            clusters.append(noise_indices)
    
    return clusters

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
    """
    Extracts fracture surface after user verification of clusters.
    """
    params = params or {}

    if not o3d_mesh_fragment.has_triangles() or not o3d_mesh_fragment.has_vertices():
        print(f"    Segmenter: Input mesh {fragment_name} has no triangles/vertices.")
        return None

    try:
        # Convert to Trimesh for robust operations
        tri_mesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh_fragment.vertices),
                                   faces=np.asarray(o3d_mesh_fragment.triangles),
                                   vertex_normals=np.asarray(o3d_mesh_fragment.vertex_normals) if o3d_mesh_fragment.has_vertex_normals() else None)
        tri_mesh.metadata['name'] = fragment_name
    except Exception as e:
        print(f"    Segmenter: Error converting O3D mesh {fragment_name} to Trimesh: {e}")
        return None

    # Ensure normals are calculated
    if not hasattr(tri_mesh, 'face_normals') or tri_mesh.face_normals is None:
        tri_mesh.generate_face_normals()
    
    # Cluster faces by normal
    normal_cluster_eps = params.get('normal_cluster_eps', 0.05)
    normal_cluster_min_samples = params.get('normal_cluster_min_samples', 10)
    
    print(f"    Segmenter: Clustering faces by normal similarity...")
    clusters = cluster_faces_by_normal(
        tri_mesh, 
        eps=normal_cluster_eps, 
        min_samples=normal_cluster_min_samples
    )
    
    print(f"    Segmenter: Found {len(clusters)} face clusters.")
    
    # Log cluster sizes
    cluster_sizes = [len(c) for c in clusters]
    sorted_indices = np.argsort(cluster_sizes)[::-1]  # Sort by size, largest first
    for i, idx in enumerate(sorted_indices[:10]):  # Show top 10 largest clusters
        print(f"    Segmenter: Cluster #{idx+1} has {cluster_sizes[idx]} faces")
    
    # Visualize all clusters with different colors
    if params.get('visualize_segmentation', False):
        vis_geometries = visualize_segmentation_clusters(o3d_mesh_fragment, clusters, tri_mesh, fragment_name)
        o3d.visualization.draw_geometries(vis_geometries, 
                                        window_name=f"Cluster Visualization: {fragment_name}")
        
        # Ask user to select which clusters are fracture surfaces
        print("\n=== Fracture Surface Selection ===")
        print("Please enter cluster numbers that represent fracture surfaces.")
        print("For example, enter '1,3,5' to select clusters 1, 3, and 5.")
        cluster_selection = input("Enter cluster numbers (comma separated) or 'all' for all clusters: ")
        
        if cluster_selection.lower() == 'all':
            selected_clusters = list(range(len(clusters)))
        else:
            try:
                # Parse user input (adjust for 0-based indexing)
                selected_clusters = [int(x.strip())-1 for x in cluster_selection.split(',') if x.strip()]
            except ValueError:
                print("    Invalid input. Using all clusters.")
                selected_clusters = list(range(len(clusters)))
                
        # Create mask for selected clusters
        face_is_fracture_candidate = np.zeros(len(tri_mesh.faces), dtype=bool)
        for idx in selected_clusters:
            if 0 <= idx < len(clusters):
                face_is_fracture_candidate[clusters[idx]] = True
            
        print(f"    Selected {np.sum(face_is_fracture_candidate)} faces as fracture surface.")
    else:
        # Fall back to automatic selection based on roughness
        face_is_fracture_candidate = identify_fracture_candidate_faces(tri_mesh, params)
    
    # Create fracture surface mesh
    if not np.any(face_is_fracture_candidate):
        return None  # No fracture faces identified

    fracture_faces = tri_mesh.faces[face_is_fracture_candidate]
    
    fracture_surface_o3d = o3d.geometry.TriangleMesh()
    fracture_surface_o3d.vertices = o3d_mesh_fragment.vertices
    fracture_surface_o3d.triangles = o3d.utility.Vector3iVector(fracture_faces)
    
    fracture_surface_o3d.remove_unreferenced_vertices()
    fracture_surface_o3d.remove_degenerate_triangles()
    
    if not fracture_surface_o3d.has_triangles():
        print(f"    Segmenter: Extracted fracture surface for {fragment_name} has no triangles after cleaning.")
        return None
        
    fracture_surface_o3d.compute_vertex_normals()
    print(f"    Segmenter: Extracted fracture surface for {fragment_name} with {len(fracture_surface_o3d.vertices)} vertices and {len(fracture_surface_o3d.triangles)} triangles.")
    return fracture_surface_o3d

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