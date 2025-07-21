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
                
                # Print geometry stats
                vertices = len(geometry.vertices)
                faces = len(geometry.triangles)
                print(f"  Successfully loaded {file_path.name}")
                print(f"    Vertices: {vertices:,}")
                print(f"    Faces: {faces:,}")
            elif isinstance(geometry, o3d.geometry.PointCloud):
                points = len(geometry.points)
                print(f"  Successfully loaded {file_path.name}")
                print(f"    Points: {points:,}")
            else:
                print(f"  Successfully loaded {file_path.name}")
                print(f"    Geometry type: {type(geometry).__name__}")
            
            geometries.append(geometry)
        else:
            print(f"  Failed to load {file_path.name}")
    
    return geometries


def visualize_models(geometries, window_name="3D Models Viewer", visualize_one_by_one=True):
    """
    Visualize the loaded 3D models using Open3D's draw_geometries.
    
    Args:
        geometries (list): List of Open3D geometries to visualize
        window_name (str): Name of the visualization window
        visualize_one_by_one (bool): If True, visualize models one by one; if False, visualize all together
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
    
    if visualize_one_by_one:
        # Visualize each model individually
        for i, geometry in enumerate(geometries):
            print(f"\nVisualizing model {i+1}/{len(geometries)}...")
            
            # Print geometry information
            if isinstance(geometry, o3d.geometry.TriangleMesh):
                vertices = len(geometry.vertices)
                faces = len(geometry.triangles)
                print(f"  Vertices: {vertices:,}")
                print(f"  Faces: {faces:,}")
            elif isinstance(geometry, o3d.geometry.PointCloud):
                points = len(geometry.points)
                print(f"  Points: {points:,}")
            else:
                print(f"  Geometry type: {type(geometry).__name__}")
            
            # Create a coordinate frame for reference
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            # Visualize single model
            o3d.visualization.draw_geometries(
                [geometry, coordinate_frame],
                window_name=f"{window_name} - Model {i+1}",
                width=1200,
                height=800,
                point_show_normal=False,
                mesh_show_back_face=True
            )
            
            # Ask user if they want to continue to next model
            if i < len(geometries) - 1:
                response = input(f"\nPress Enter to view next model, or 'q' to quit: ").strip().lower()
                if response == 'q':
                    print("Visualization stopped by user.")
                    break
    else:
        # Visualize all models together (original behavior)
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
