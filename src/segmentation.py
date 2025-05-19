import open3d as o3d
import trimesh
import numpy as np

def identify_fracture_candidate_faces(tri_mesh_fragment, min_boundary_edges_for_fracture_face=1):
    """
    Identifies faces that are likely part of a fracture surface.
    A simple heuristic: faces with at least N boundary edges.

    Args:
        tri_mesh_fragment (trimesh.Trimesh): The input fragment as a Trimesh object.
        min_boundary_edges_for_fracture_face (int): Minimum number of boundary edges a face
                                                    must have to be considered a fracture candidate.

    Returns:
        np.ndarray: Boolean mask of faces identified as fracture candidates.
    """
    if not tri_mesh_fragment.is_watertight:
        # Identify all edges that are unique (appear in only one face's edge list)
        # These are boundary edges.
        boundary_edges_unique = tri_mesh_fragment.edges[trimesh.grouping.group_rows(tri_mesh_fragment.edges_sorted, require_count=1)]
    else:
        # Watertight meshes technically have no boundary edges by this definition.
        # For a fragment of an originally watertight object, this shouldn't happen unless it's a fully enclosed piece.
        # In such a case, all surfaces are "original" or it's not a fragment of a larger piece.
        # For simplicity, if it's watertight, we assume no "fracture" surfaces by this heuristic.
        print(f"    Segmenter: Mesh {tri_mesh_fragment.metadata.get('name', 'Unnamed')} is watertight. Assuming no open fracture surfaces.")
        return np.zeros(len(tri_mesh_fragment.faces), dtype=bool)


    if len(boundary_edges_unique) == 0:
        print(f"    Segmenter: No boundary edges found for {tri_mesh_fragment.metadata.get('name', 'Unnamed')}. Might be watertight or an issue.")
        return np.zeros(len(tri_mesh_fragment.faces), dtype=bool)

    # Create a set of boundary edge vertex pairs for quick lookup
    boundary_edge_set = set()
    for edge in boundary_edges_unique:
        # Ensure consistent ordering for the set (e.g., smaller index first)
        boundary_edge_set.add(tuple(sorted(edge)))

    face_is_fracture_candidate = np.zeros(len(tri_mesh_fragment.faces), dtype=bool)

    for face_idx, face_vertices in enumerate(tri_mesh_fragment.faces):
        boundary_edge_count = 0
        # Edges of the current face: (v0,v1), (v1,v2), (v2,v0)
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
        print(f"    Segmenter: No fracture candidate faces found for {tri_mesh_fragment.metadata.get('name', 'Unnamed')} using min_boundary_edges={min_boundary_edges_for_fracture_face}.")
    else:
        print(f"    Segmenter: Identified {num_candidate_faces} fracture candidate faces for {tri_mesh_fragment.metadata.get('name', 'Unnamed')}.")
        
    return face_is_fracture_candidate


def extract_fracture_surface_mesh(o3d_mesh_fragment, fragment_name="Unnamed", params=None):
    """
    Extracts a new mesh composed only of identified fracture surface candidate faces.

    Args:
        o3d_mesh_fragment (o3d.geometry.TriangleMesh): The input Open3D mesh fragment.
        fragment_name (str): Name of the fragment for logging.
        params (dict): Configuration parameters, e.g., for min_boundary_edges.

    Returns:
        o3d.geometry.TriangleMesh or None: A new Open3D mesh of the fracture surface,
                                           or None if no fracture surface is found.
    """
    params = params or {}
    min_boundary_edges = params.get("min_boundary_edges_for_fracture_face", 1)

    if not o3d_mesh_fragment.has_triangles() or not o3d_mesh_fragment.has_vertices():
        print(f"    Segmenter: Input mesh {fragment_name} has no triangles/vertices.")
        return None

    try:
        # Convert to Trimesh for robust operations
        tri_mesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh_fragment.vertices),
                                   faces=np.asarray(o3d_mesh_fragment.triangles),
                                   vertex_normals=np.asarray(o3d_mesh_fragment.vertex_normals) if o3d_mesh_fragment.has_vertex_normals() else None)
        tri_mesh.metadata['name'] = fragment_name # For logging within identify_fracture_candidate_faces
    except Exception as e:
        print(f"    Segmenter: Error converting O3D mesh {fragment_name} to Trimesh: {e}")
        return None

    fracture_face_mask = identify_fracture_candidate_faces(tri_mesh, min_boundary_edges)

    if not np.any(fracture_face_mask):
        return None # No fracture faces identified

    # Create a new mesh from the selected faces
    fracture_faces = tri_mesh.faces[fracture_face_mask]
    
    # We need to use all original vertices, then Open3D will clean up unused ones
    # when creating the mesh from these specific faces.
    fracture_surface_o3d = o3d.geometry.TriangleMesh()
    fracture_surface_o3d.vertices = o3d_mesh_fragment.vertices # Use original vertices
    fracture_surface_o3d.triangles = o3d.utility.Vector3iVector(fracture_faces)
    
    # Clean the mesh: remove unreferenced vertices and degenerate triangles
    fracture_surface_o3d.remove_unreferenced_vertices()
    fracture_surface_o3d.remove_degenerate_triangles()
    
    if not fracture_surface_o3d.has_triangles():
        print(f"    Segmenter: Extracted fracture surface for {fragment_name} has no triangles after cleaning.")
        return None
        
    fracture_surface_o3d.compute_vertex_normals() # Recompute normals for the new surface
    print(f"    Segmenter: Extracted fracture surface for {fragment_name} with {len(fracture_surface_o3d.vertices)} vertices and {len(fracture_surface_o3d.triangles)} triangles.")
    return fracture_surface_o3d


if __name__ == '__main__':
    # Test segmentation
    # Create a cube with one face missing to simulate a fracture surface
    source_mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    
    # To make it interesting, let's manually create a "broken" piece
    # A cube missing its +X face (faces with x=1)
    verts = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ])
    faces = np.array([
        [0,3,2], [0,2,1], # -X face at x=0 - NO, this is bottom face z=0
        # Let's use Open3D's box and remove faces
        # Box vertices:
        # (0,0,0), (1,0,0), (0,1,0), (1,1,0)
        # (0,0,1), (1,0,1), (0,1,1), (1,1,1)
        # Faces:
        # (0,2,1), (1,2,3) # z=0  (Bottom)
        # (4,5,7), (4,7,6) # z=1  (Top)
        # (0,1,5), (0,5,4) # y=0  (Front)
        # (2,6,7), (2,7,3) # y=1  (Back)
        # (0,4,6), (0,6,2) # x=0  (Left)
        # (1,7,5), (1,3,7) # x=1  (Right) - These are the ones to remove to make it a fracture surface

    ])
    # For simplicity, let's use a pre-made OBJ that is a fragment
    # Or use the dummy parts from assembly.py
    
    # Using a part of a sphere as a fragment
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    # Crop the sphere to get a fragment
    bbox_to_crop = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1,-1,-0.1), max_bound=(1,1,1))
    fragment_mesh = sphere.crop(bbox_to_crop)
    fragment_mesh.compute_vertex_normals() # Ensure normals

    print(f"Original fragment: {len(fragment_mesh.vertices)} verts, {len(fragment_mesh.triangles)} tris")
    # o3d.visualization.draw_geometries([fragment_mesh], window_name="Original Fragment")

    test_params = {"min_boundary_edges_for_fracture_face": 1}
    fracture_surface = extract_fracture_surface_mesh(fragment_mesh, "TestSphereFragment", test_params)

    if fracture_surface and fracture_surface.has_triangles():
        print(f"Fracture surface: {len(fracture_surface.vertices)} verts, {len(fracture_surface.triangles)} tris")
        # Color the fracture surface for visualization
        fracture_surface.paint_uniform_color([1,0,0]) # Red
        # o3d.visualization.draw_geometries([fragment_mesh.paint_uniform_color([0.7,0.7,0.7]), fracture_surface],
                                        #   window_name="Fragment with Fracture Surface Highlighted")
        
        # Sample points from this fracture surface
        num_points_to_sample = 2000
        if len(fracture_surface.vertices) > 10 : # Ensure enough geometry to sample
            pcd_from_fracture = fracture_surface.sample_points_poisson_disk(num_points_to_sample)
            if pcd_from_fracture.has_points():
                print(f"Sampled {len(pcd_from_fracture.points)} from fracture surface.")
                # o3d.visualization.draw_geometries([pcd_from_fracture], window_name="PCD from Fracture Surface")
            else:
                print("Failed to sample points from fracture surface.")
        else:
            print("Fracture surface too small to sample points effectively.")

    else:
        print("No fracture surface extracted or it was empty.")