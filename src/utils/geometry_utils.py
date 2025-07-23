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


def boolean_intersection_penetration_test(mesh1_o3d, mesh1_name, mesh2_o3d, mesh2_name, params, viz_collector=None, min_volume_override=None):
    """Boolean intersection test for penetration detection between two meshes.
    Returns (is_valid, intersection_ratio, intersection_mesh).
    Logs to viz_collector if provided.
    """
    try:
        # Convert Open3D meshes to Trimesh
        mesh1_tri = trimesh.Trimesh(
            vertices=np.asarray(mesh1_o3d.vertices),
            faces=np.asarray(mesh1_o3d.triangles)
        )
        mesh2_tri = trimesh.Trimesh(
            vertices=np.asarray(mesh2_o3d.vertices),
            faces=np.asarray(mesh2_o3d.triangles)
        )
        # Ensure meshes are watertight
        if not mesh1_tri.is_watertight and len(mesh1_tri.faces) > 0:
            mesh1_tri.fill_holes()
        if not mesh2_tri.is_watertight and len(mesh2_tri.faces) > 0:
            mesh2_tri.fill_holes()
        # Check if meshes are valid
        if len(mesh1_tri.faces) == 0 or len(mesh2_tri.faces) == 0:
            if viz_collector is not None:
                viz_collector.append({
                    'step': 'penetration_test_skipped_empty_mesh', 'type': 'event',
                    'mesh1_name': mesh1_name, 'mesh2_name': mesh2_name
                })
            return True, 0.0, None
        # Calculate volumes
        vol1 = mesh1_tri.volume
        vol2 = mesh2_tri.volume
        if vol1 <= 0 or vol2 <= 0:
            if viz_collector is not None:
                viz_collector.append({
                    'step': 'penetration_test_skipped_zero_volume', 'type': 'event',
                    'mesh1_name': mesh1_name, 'mesh2_name': mesh2_name
                })
            return True, 0.0, None
        # Perform boolean intersection
        try:
            intersection_mesh = trimesh.boolean.intersection([mesh1_tri, mesh2_tri])
            if intersection_mesh is None or len(intersection_mesh.faces) == 0:
                if viz_collector is not None:
                    viz_collector.append({
                        'step': 'penetration_test_no_intersection', 'type': 'event',
                        'mesh1_name': mesh1_name, 'mesh2_name': mesh2_name
                    })
                return True, 0.0, None
            # Calculate intersection volume and ratio
            intersection_volume = intersection_mesh.volume
            if min_volume_override is not None:
                total_volume = min_volume_override
            else:
                total_volume = min(vol1, vol2)  # Use smaller volume for ratio calculation
            intersection_ratio = (intersection_volume / total_volume) if total_volume > 0 else 0.0
            # Get penetration threshold from params (default to 0.1 = 10%)
            penetration_threshold = params.get("boolean_penetration_threshold", 0.1)
            # Log result
            if viz_collector is not None:
                viz_collector.append({
                    'step': 'penetration_test_result', 'type': 'event',
                    'mesh1_name': mesh1_name, 'mesh2_name': mesh2_name,
                    'intersection_volume': intersection_volume,
                    'vol1': vol1, 'vol2': vol2,
                    'intersection_ratio': intersection_ratio,
                    'penetration_threshold': penetration_threshold,
                    'min_volume_used': total_volume
                })
            # Check if penetration ratio is acceptable
            if intersection_ratio <= penetration_threshold:
                return True, intersection_ratio, intersection_mesh
            else:
                return False, intersection_ratio, intersection_mesh
        except Exception as bool_error:
            if viz_collector is not None:
                viz_collector.append({
                    'step': 'penetration_test_error', 'type': 'event',
                    'mesh1_name': mesh1_name, 'mesh2_name': mesh2_name,
                    'error_message': str(bool_error)
                })
            return None, 0.0, None
    except Exception as e:
        if viz_collector is not None:
            viz_collector.append({
                'step': 'penetration_test_error', 'type': 'event',
                'mesh1_name': mesh1_name, 'mesh2_name': mesh2_name,
                'error_message': str(e)
            })
        return None, 0.0, None
