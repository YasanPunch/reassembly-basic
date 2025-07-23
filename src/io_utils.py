import open3d as o3d
import trimesh
import os
import json

def load_fragment(file_path, file_type=None):
    """
    Loads a single 3D model fragment.
    Converts Trimesh object to Open3D PointCloud for consistency in processing.
    Args:
        file_path (str): Path to the model file.
        file_type (str, optional): 'obj', 'stl', 'ply'. Inferred if None.
    Returns:
        o3d.geometry.TriangleMesh: Loaded Open3D mesh, or None if loading fails.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        # Trimesh is often more robust for loading various formats
        mesh_trimesh = trimesh.load_mesh(file_path, file_type=file_type)

        # Ensure it's a TriangleMesh
        if isinstance(mesh_trimesh, trimesh.Scene):
            # If scene, combine all geometries into one mesh
            # This is a simplification; complex scenes might need more sophisticated handling
            mesh_trimesh = trimesh.util.concatenate(
                tuple(g for g in mesh_trimesh.geometry.values() if isinstance(g, trimesh.Trimesh))
            )
        if not isinstance(mesh_trimesh, trimesh.Trimesh):
            print(f"Warning: Loaded object from {file_path} is not a Trimesh instance. Type: {type(mesh_trimesh)}")
            # Attempt to convert if it's point cloud like
            if hasattr(mesh_trimesh, 'vertices'):
                mesh_trimesh = trimesh.Trimesh(vertices=mesh_trimesh.vertices) # No faces
            else:
                return None

        # Convert trimesh to Open3D TriangleMesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.compute_triangle_normals()

        print(f"Successfully loaded: {file_path}")
        return o3d_mesh
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_fragments_from_directory(directory_path):
    """
    Loads all supported 3D model files from a directory.
    Args:
        directory_path (str): Path to the directory containing fragments.
    Returns:
        list: A list of o3d.geometry.TriangleMesh objects.
    """
    fragments = []
    supported_extensions = ['.obj', '.stl', '.ply', '.off', '.gltf', '.glb'] # Trimesh supports many
    print(f"Loading fragments from: {directory_path}")
    for filename in sorted(os.listdir(directory_path)): # Sorted for consistent order
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_extensions):
            fragment = load_fragment(file_path)
            if fragment:
                # Check if mesh is empty
                if not fragment.has_vertices() or not fragment.has_triangles():
                    print(f"Warning: Fragment {filename} is empty or has no triangles. Skipping.")
                    continue
                fragments.append(
                    {
                        "mesh": fragment,
                        "name": filename,
                        "original_index": len(fragments),
                        "path": file_path,
                    }
                )
        else:
            print(f"Skipping non-supported file or directory: {filename}")
    print(f"Loaded {len(fragments)} fragments.")
    return fragments


def load_fragments_from_paths(file_paths):
    """
    Loads all supported 3D model files from a list of file paths.
    Args:
        file_paths (list): List of file paths to load fragments from.
    Returns:
        list: A list of dictionaries containing fragment data:
              [{"mesh": o3d.geometry.TriangleMesh, "name": str, "original_index": int, "path": str}, ...]
    """
    fragments = []
    supported_extensions = [
        ".obj",
        ".stl",
        ".ply",
        ".off",
        ".gltf",
        ".glb",
    ]  # Trimesh supports many
    print(f"Loading fragments from {len(file_paths)} file paths")

    for i, file_path in enumerate(file_paths):
        if not os.path.isfile(file_path):
            print(f"Warning: File path does not exist: {file_path}. Skipping.")
            continue

        filename = os.path.basename(file_path)
        if not any(filename.lower().endswith(ext) for ext in supported_extensions):
            print(f"Warning: Unsupported file format: {filename}. Skipping.")
            continue

        fragment = load_fragment(file_path)
        if fragment:
            # Check if mesh is empty
            if not fragment.has_vertices() or not fragment.has_triangles():
                print(
                    f"Warning: Fragment {filename} is empty or has no triangles. Skipping."
                )
                continue
            fragments.append(
                {
                    "mesh": fragment,
                    "name": filename,
                    "original_index": len(fragments),
                    "path": file_path,
                }
            )
        else:
            print(f"Warning: Failed to load fragment: {filename}")

    print(f"Loaded {len(fragments)} fragments.")
    return fragments


def save_mesh(mesh, file_path):
    """
    Saves an Open3D mesh to a file.
    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh to save.
        file_path (str): Path to save the model file.
        file_type (str, optional): 'obj', 'stl', 'ply'. Inferred if None.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # For OBJ, Open3D might not write textures/materials properly.
        # If materials are critical, consider using trimesh for saving OBJ.
        # For simplicity here, we use Open3D's writer.
        success = o3d.io.write_triangle_mesh(file_path, mesh, write_ascii=True)
        if success:
            print(f"Successfully saved mesh to {file_path}")
        else:
            print(f"Error: Failed to save mesh to {file_path} using Open3D.")
            # Fallback or alternative saving if needed
            # mesh_trimesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
            # mesh_trimesh.export(file_path)
            # print(f"Successfully saved mesh to {file_path} using Trimesh as fallback.")

    except Exception as e:
        print(f"Error saving mesh to {file_path}: {e}")


def combine_meshes(mesh_list, transformations=None):
    """
    Combines a list of Open3D meshes into a single mesh.
    Applies transformations if provided.
    Args:
        mesh_list (list of o3d.geometry.TriangleMesh): List of meshes to combine.
        transformations (list of np.ndarray, optional): List of 4x4 transformation matrices,
                                                         one for each mesh.
    Returns:
        o3d.geometry.TriangleMesh: The combined mesh.
    """
    combined_mesh = o3d.geometry.TriangleMesh()
    if transformations is not None and len(mesh_list) != len(transformations):
        raise ValueError("Number of meshes and transformations must match.")

    for i, mesh_part in enumerate(mesh_list):
        temp_mesh = o3d.geometry.TriangleMesh() # Create a copy to transform
        temp_mesh.vertices = mesh_part.vertices
        temp_mesh.triangles = mesh_part.triangles
        temp_mesh.vertex_colors = mesh_part.vertex_colors # Preserve colors if any
        temp_mesh.vertex_normals = mesh_part.vertex_normals

        if transformations:
            temp_mesh.transform(transformations[i])

        combined_mesh += temp_mesh # Open3D supports += for mesh union

    return combined_mesh


def load_parameters(config_file=None):
    """Load parameters from a JSON configuration file.

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict: Loaded parameters from the config file.

    Raises:
        FileNotFoundError: If the config file is not found.
        json.JSONDecodeError: If the JSON file cannot be decoded.
    """
    print("\n[1. Loading Parameters]")
    try:
        with open(config_file, "r") as f:
            params = json.load(f)
        print(f"  Parameters loaded from: {config_file}")
        return params

    except FileNotFoundError:
        print(f"Error: Config file not found at {config_file}. Exiting.")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_file}. Exiting.")
        raise
