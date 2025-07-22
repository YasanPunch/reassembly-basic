import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
from collections import defaultdict

def estimate_normals(pcd, search_param):
    """Estimates normals for a point cloud."""
    pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_consistent_tangent_plane(k=15) # Optional: Try to orient normals consistently
    return pcd

def get_mesh_boundary_vertices(o3d_mesh):
    """
    Identifies boundary vertices of an Open3D mesh using Trimesh.
    Args:
        o3d_mesh (o3d.geometry.TriangleMesh): The input Open3D mesh.
    Returns:
        o3d.geometry.PointCloud: A point cloud of boundary vertices, or None.
    """
    if not o3d_mesh.has_triangles():
        print("Warning: Mesh has no triangles, cannot find boundary vertices.")
        return None
        
    try:
        # Convert Open3D mesh to Trimesh mesh
        tri_mesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices),
                                   faces=np.asarray(o3d_mesh.triangles))
        
        # Trimesh's `outline` function can find boundary edges.
        # `vertices_unique` on these edges gives boundary vertices.
        boundary_edges = tri_mesh.outline()
        if boundary_edges is None or len(boundary_edges.entities) == 0:
            print("No boundary edges found by trimesh.outline(). Mesh might be watertight or complex.")
             # Fallback: try edges_unique on non-manifold edges if outline fails
            unique_edges = tri_mesh.edges[trimesh.grouping.group_rows(tri_mesh.edges_sorted, require_count=1)]
            if len(unique_edges) == 0:
                print("No unique edges found either. Assuming no simple boundary.")
                return None # No boundary vertices or mesh is watertight
            boundary_vertex_indices = np.unique(unique_edges.flatten())

        else:
            boundary_vertex_indices = np.unique(boundary_edges.vertices_sequence.reshape(-1))

        if len(boundary_vertex_indices) == 0:
            print("No boundary vertices identified.")
            return None

        boundary_points = np.asarray(o3d_mesh.vertices)[boundary_vertex_indices]
        
        boundary_pcd = o3d.geometry.PointCloud()
        boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
        return boundary_pcd
        
    except Exception as e:
        print(f"Error in get_mesh_boundary_vertices: {e}")
        # This can happen if the mesh is non-manifold in a way trimesh struggles with
        # As a fallback, consider all vertices if it's an open mesh, or specific criteria
        return None


def compute_fpfh_features(pcd, voxel_size, radius_normal, radius_feature):
    """
    Computes FPFH features for a point cloud.
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Voxel size for downsampling (used to determine radii).
        radius_normal (float): Radius for normal estimation.
        radius_feature (float): Radius for FPFH feature computation.
    Returns:
        o3d.pipelines.registration.Feature: FPFH features.
    """
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh


def get_bounding_box_dimensions(mesh):
    """Returns the dimensions (length, width, height) of the mesh's AABB."""
    aabb = mesh.get_axis_aligned_bounding_box()
    return aabb.get_extent()


def get_adjacent_faces(mesh, fracture_face_indices):
    """
    Given a mesh (Open3D or Trimesh) and a set of face indices (e.g., fracture surface),
    return the set of face indices that are adjacent (share an edge) but not in the original set.
    Args:
        mesh: o3d.geometry.TriangleMesh or trimesh.Trimesh
        fracture_face_indices: list or set of face indices (integers)
    Returns:
        list of adjacent face indices (integers)
    """
    # Convert to Trimesh if needed
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                               faces=np.asarray(mesh.triangles))
    fracture_face_indices = set(fracture_face_indices)
    adjacent_faces = set()
    # Ensure face adjacency is computed
    if not hasattr(mesh, 'face_adjacency') or mesh.face_adjacency is None:
        mesh.face_adjacency = trimesh.graph.face_adjacency(mesh.faces)
    for f1, f2 in mesh.face_adjacency:
        if f1 in fracture_face_indices and f2 not in fracture_face_indices:
            adjacent_faces.add(f2)
        elif f2 in fracture_face_indices and f1 not in fracture_face_indices:
            adjacent_faces.add(f1)
    return list(adjacent_faces)


def compute_adjacent_face_normal_similarity(mesh_src, adj_faces_src, mesh_tgt, adj_faces_tgt):
    """
    Compute the similarity between adjacent faces of two meshes based on their normals.
    Args:
        mesh_src: o3d.geometry.TriangleMesh (source, already transformed)
        adj_faces_src: list of face indices (source adjacent faces)
        mesh_tgt: o3d.geometry.TriangleMesh (target)
        adj_faces_tgt: list of face indices (target adjacent faces)
    Returns:
        float: similarity score in [0, 1] (1 = perfect match, 0 = worst)
    """
    # Get centroids and normals for source and target adjacent faces
    src_faces = np.asarray(mesh_src.triangles)[adj_faces_src]
    src_verts = np.asarray(mesh_src.vertices)
    src_normals = np.asarray(mesh_src.triangle_normals)[adj_faces_src]
    src_centroids = np.mean(src_verts[src_faces], axis=1)

    tgt_faces = np.asarray(mesh_tgt.triangles)[adj_faces_tgt]
    tgt_verts = np.asarray(mesh_tgt.vertices)
    tgt_normals = np.asarray(mesh_tgt.triangle_normals)[adj_faces_tgt]
    tgt_centroids = np.mean(tgt_verts[tgt_faces], axis=1)

    # For each source adjacent face, find the closest target adjacent face
    if len(tgt_centroids) == 0 or len(src_centroids) == 0:
        return 0.0
    tree = cKDTree(tgt_centroids)
    dists, idxs = tree.query(src_centroids)

    # Compute dot products of normals
    dot_products = np.einsum('ij,ij->i', src_normals, tgt_normals[idxs])
    # Normalize to [0, 1]
    similarity = np.clip((dot_products + 1) / 2, 0, 1)
    return float(np.mean(similarity))


def compute_face_curvatures(mesh, face_indices):
    """
    Estimate curvature for each face as the angle between the face normal and the average normal of its neighbors.
    Returns a numpy array of curvature values (in radians).
    """
    tris = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.triangle_normals)
    curvatures = np.zeros(len(face_indices))
    # Build adjacency
    face_neighbors = defaultdict(set)
    for i, tri1 in enumerate(tris):
        for j, tri2 in enumerate(tris):
            if i != j and len(set(tri1) & set(tri2)) >= 2:
                face_neighbors[i].add(j)
    for idx, face_idx in enumerate(face_indices):
        neighbors = list(face_neighbors[face_idx])
        if not neighbors:
            curvatures[idx] = 0.0
            continue
        avg_neighbor_normal = np.mean(normals[neighbors], axis=0)
        avg_neighbor_normal /= np.linalg.norm(avg_neighbor_normal) + 1e-8
        dot = np.clip(np.dot(normals[face_idx], avg_neighbor_normal), -1, 1)
        curvatures[idx] = np.arccos(dot)
    return curvatures

def compute_bumpiness_similarity(mesh_src, adj_faces_src, mesh_tgt, adj_faces_tgt):
    """
    Compare bumpiness (roughness) of two sets of adjacent faces.
    Returns a similarity score in [0, 1] (1 = perfect match).
    """
    src_normals = np.asarray(mesh_src.triangle_normals)[adj_faces_src]
    tgt_normals = np.asarray(mesh_tgt.triangle_normals)[adj_faces_tgt]
    src_bump = np.std(src_normals, axis=0).mean()
    tgt_bump = np.std(tgt_normals, axis=0).mean()
    denom = max(src_bump, tgt_bump, 1e-6)
    sim = 1.0 - abs(src_bump - tgt_bump) / denom
    return np.clip(sim, 0, 1)

def compute_curvature_similarity(mesh_src, adj_faces_src, mesh_tgt, adj_faces_tgt):
    """
    Compare curvature distributions of two sets of adjacent faces.
    Returns a similarity score in [0, 1] (1 = perfect match).
    """
    src_curv = compute_face_curvatures(mesh_src, adj_faces_src)
    tgt_curv = compute_face_curvatures(mesh_tgt, adj_faces_tgt)
    if len(src_curv) == 0 or len(tgt_curv) == 0:
        return 0.0
    src_mean = np.mean(src_curv)
    tgt_mean = np.mean(tgt_curv)
    denom = max(src_mean, tgt_mean, 1e-6)
    sim = 1.0 - abs(src_mean - tgt_mean) / denom
    return np.clip(sim, 0, 1)


if __name__ == '__main__':
    # Create a sample mesh (e.g., a plane with a hole)
    vertices = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0], # Outer square
        [0.25,0.25,0], [0.75,0.25,0], [0.75,0.75,0], [0.25,0.75,0] # Inner square (hole)
    ])
    triangles = np.array([
        [0,1,5], [0,5,4], # Bottom part of hole boundary
        [1,2,6], [1,6,5], # Right part
        [2,3,7], [2,7,6], # Top part
        [3,0,4], [3,4,7]  # Left part
    ])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    print("Original mesh vertices:", len(mesh.vertices))
    
    boundary_pcd = get_mesh_boundary_vertices(mesh)
    if boundary_pcd:
        print("Boundary vertices found:", len(boundary_pcd.points))
        # o3d.visualization.draw_geometries([mesh, boundary_pcd.paint_uniform_color([1,0,0])])
    else:
        print("No boundary vertices found for the sample mesh.")

    # Test FPFH (needs a denser point cloud)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    sphere_pcd = sphere.sample_points_poisson_disk(number_of_points=500)
    
    voxel_s = 0.05
    radius_n = voxel_s * 2
    radius_f = voxel_s * 5
    
    fpfh_features = compute_fpfh_features(sphere_pcd, voxel_s, radius_n, radius_f)
    print("FPFH feature dimension:", fpfh_features.num())
    print("FPFH data shape:", fpfh_features.data.shape)

    # Test BBox
    print("Sphere BBox dimensions:", get_bounding_box_dimensions(sphere))