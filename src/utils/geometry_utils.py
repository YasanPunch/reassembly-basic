import numpy as np
import open3d as o3d
import trimesh

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