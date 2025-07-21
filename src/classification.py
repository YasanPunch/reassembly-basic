import os
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
import argparse


def load_mesh_with_open3d(file_path):
    """Load a mesh using Open3D."""
    try:
        # Try to load as mesh first
        mesh = o3d.io.read_triangle_mesh(str(file_path))
        if not mesh.has_vertices():
            # If no vertices, try as point cloud
            mesh = o3d.io.read_point_cloud(str(file_path))
        return mesh
    except Exception as e:
        print(f"Open3D failed to load {file_path}: {e}")
        return None


def load_mesh_with_trimesh(file_path):
    """Load a mesh using Trimesh and convert to Open3D format."""
    try:
        # Load with trimesh
        mesh = trimesh.load(str(file_path))
        
        if isinstance(mesh, trimesh.Trimesh):
            # Convert trimesh mesh to open3d mesh
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            # Add normals if available
            if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
                o3d_mesh.triangle_normals = o3d.utility.Vector3dVector(mesh.face_normals)
            
            return o3d_mesh
            
        elif isinstance(mesh, trimesh.PointCloud):
            # Convert point cloud
            points = np.array(mesh.points)
            o3d_pc = o3d.geometry.PointCloud()
            o3d_pc.points = o3d.utility.Vector3dVector(points)
            
            # Add colors if available
            if hasattr(mesh, 'colors') and mesh.colors is not None:
                o3d_pc.colors = o3d.utility.Vector3dVector(mesh.colors)
                
            return o3d_pc
            
    except Exception as e:
        print(f"Trimesh failed to load {file_path}: {e}")
        return None


def load_3d_models_from_folder(folder_path, use_trimesh_fallback=True):
    """
    Load all 3D models from a folder.
    
    Args:
        folder_path (str): Path to the folder containing 3D models
        use_trimesh_fallback (bool): Whether to use trimesh as fallback if Open3D fails
    
    Returns:
        list: List of loaded Open3D geometries
    """
    folder_path = Path(folder_path)
    geometries = []
    
    # Supported file extensions
    supported_extensions = {'.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx', '.3ds', '.dae'}
    
    if not folder_path.exists():
        print(f"Folder {folder_path} does not exist!")
        return geometries
    
    # Find all 3D model files
    model_files = []
    for ext in supported_extensions:
        model_files.extend(folder_path.glob(f"*{ext}"))
        model_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    if not model_files:
        print(f"No 3D model files found in {folder_path}")
        return geometries
    
    print(f"Found {len(model_files)} 3D model files:")
    for file_path in model_files:
        print(f"  - {file_path.name}")
    
    # Load each model
    for file_path in model_files:
        print(f"\nLoading {file_path.name}...")
        
        # Try Open3D first
        geometry = load_mesh_with_open3d(file_path)
        
        # If Open3D fails and trimesh fallback is enabled, try trimesh
        if geometry is None and use_trimesh_fallback:
            print(f"  Trying trimesh fallback for {file_path.name}...")
            geometry = load_mesh_with_trimesh(file_path)
        
        if geometry is not None:
            # Add some visual properties
            if isinstance(geometry, o3d.geometry.TriangleMesh):
                # Compute normals for better visualization
                geometry.compute_vertex_normals()
                geometry.compute_triangle_normals()
                
                # Add a random color to distinguish different meshes
                color = np.random.uniform(0.3, 1.0, 3)
                geometry.paint_uniform_color(color)
            
            geometries.append(geometry)
            print(f"  Successfully loaded {file_path.name}")
        else:
            print(f"  Failed to load {file_path.name}")
    
    return geometries


def visualize_models(geometries, window_name="3D Models Viewer"):
    """
    Visualize the loaded 3D models using Open3D's draw_geometries.
    
    Args:
        geometries (list): List of Open3D geometries to visualize
        window_name (str): Name of the visualization window
    """
    if not geometries:
        print("No geometries to visualize!")
        return
    
    print(f"\nVisualizing {len(geometries)} models...")
    print("Controls:")
    print("  - Mouse: Rotate, zoom, pan")
    print("  - Shift + Mouse: Pan")
    print("  - Ctrl + Mouse: Zoom")
    print("  - Press 'Q' to exit")
    
    # Create a coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(coordinate_frame)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    )


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
        visualize_models(geometries, args.window_name)
    else:
        print("No models were loaded successfully.")


if __name__ == "__main__":
    main()
