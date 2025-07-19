#!/usr/bin/env python3
"""
Simple example showing how to classify faces into fracture vs original surfaces
using a single adaptive method without comparing multiple approaches.
"""

import open3d as o3d
import numpy as np
import json
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from segmentation import (
    classify_fracture_vs_original_faces,
    visualize_face_classification,
    extract_fracture_surfaces_simple
)
import trimesh


def create_test_mesh():
    """Create a test mesh with mixed surface properties."""
    print("Creating test mesh...")
    
    # Create a cube
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    vertices = np.asarray(mesh.vertices)
    
    # Make some faces rough (simulate fractures)
    np.random.seed(42)
    
    # Top face - very rough (fracture)
    top_vertices = np.where(vertices[:, 2] > 1.9)[0]
    vertices[top_vertices, 2] += np.random.normal(0, 0.2, len(top_vertices))
    
    # Front face - moderately rough (potential fracture)
    front_vertices = np.where(vertices[:, 1] > 1.9)[0]
    vertices[front_vertices, 1] += np.random.normal(0, 0.1, len(front_vertices))
    
    # Keep other faces smooth (original surfaces)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    
    return mesh


def load_parameters():
    """Load parameters from config file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'simple_classification_params.json')
    
    try:
        with open(config_path, 'r') as f:
            params = json.load(f)
        print(f"Loaded parameters from {config_path}")
        return params
    except FileNotFoundError:
        print(f"Config file not found. Using default parameters.")
        return {
            'roughness_threshold_percentile': 75,
            'curvature_threshold_percentile': 75,
            'boundary_complexity_threshold_percentile': 75,
            'symmetry_threshold_percentile': 25,
            'planarity_threshold_percentile': 25,
            'final_classification_threshold': 0.5,
            'visualize_classification': True,
            'fracture_detection_weights': {
                'curvature': 0.3,
                'roughness': 0.3,
                'boundary_complexity': 0.2,
                'symmetry': 0.1,
                'planarity': 0.1
            }
        }


def main():
    """Main function demonstrating simple face classification."""
    print("=" * 60)
    print("SIMPLE FACE CLASSIFICATION DEMO")
    print("=" * 60)
    print()
    print("This demo shows:")
    print("1. Single-method classification of faces")
    print("2. Adaptive thresholds based on object properties")
    print("3. Direct fracture vs original surface classification")
    print()
    
    # Load parameters
    params = load_parameters()
    
    # Create test mesh
    o3d_mesh = create_test_mesh()
    
    # Convert to trimesh for processing
    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False
    )
    
    print(f"Processing mesh with {len(tri_mesh.faces)} faces...")
    
    # Classify faces using single method
    print("\n" + "=" * 40)
    print("CLASSIFYING FACES")
    print("=" * 40)
    
    classification_result = classify_fracture_vs_original_faces(tri_mesh, params)
    
    # Show results
    fracture_faces = classification_result['fracture_faces']
    original_faces = classification_result['original_faces']
    face_scores = classification_result['face_scores']
    thresholds = classification_result['thresholds']
    
    print(f"\nClassification Summary:")
    print(f"  Total faces: {len(tri_mesh.faces)}")
    print(f"  Fracture faces: {np.sum(fracture_faces)} ({np.sum(fracture_faces)/len(tri_mesh.faces)*100:.1f}%)")
    print(f"  Original faces: {np.sum(original_faces)} ({np.sum(original_faces)/len(tri_mesh.faces)*100:.1f}%)")
    print(f"  Average fracture score: {np.mean(face_scores):.3f}")
    print(f"  Max fracture score: {np.max(face_scores):.3f}")
    
    # Show some example face classifications
    print(f"\nExample Face Classifications:")
    for i in range(min(10, len(tri_mesh.faces))):
        face_type = "FRACTURE" if fracture_faces[i] else "ORIGINAL"
        score = face_scores[i]
        print(f"  Face {i:2d}: {face_type:8s} (score: {score:.3f})")
    
    # Visualize results
    if params.get('visualize_classification', True):
        print(f"\nVisualizing classification results...")
        visualize_face_classification(o3d_mesh, classification_result, "TestMesh")
    
    # Extract fracture surfaces
    print("\n" + "=" * 40)
    print("EXTRACTING FRACTURE SURFACES")
    print("=" * 40)
    
    fracture_surfaces = extract_fracture_surfaces_simple(o3d_mesh, "TestMesh", params)
    
    if fracture_surfaces:
        print(f"Extracted {len(fracture_surfaces)} fracture surface(s):")
        for i, surface in enumerate(fracture_surfaces):
            print(f"  Surface {i+1}: {len(surface.vertices)} vertices, {len(surface.triangles)} triangles")
        
        # Visualize fracture surfaces
        print(f"\nVisualizing extracted fracture surfaces...")
        for i, surface in enumerate(fracture_surfaces):
            surface.paint_uniform_color([1.0, 0.0, 0.0])  # Red
            surface.compute_vertex_normals()
        
        # Show original mesh in gray
        original_vis = o3d_mesh
        original_vis.paint_uniform_color([0.7, 0.7, 0.7])
        original_vis.compute_vertex_normals()
        
        # Create wireframe
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(original_vis)
        wireframe.paint_uniform_color([0.5, 0.5, 0.5])
        
        vis_objects = [original_vis, wireframe] + fracture_surfaces
        o3d.visualization.draw_geometries(vis_objects, window_name="Extracted Fracture Surfaces")
    
    else:
        print("No fracture surfaces detected.")
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("✓ Single-method classification (no method comparison)")
    print("✓ Adaptive thresholds based on object properties")
    print("✓ Direct fracture vs original surface classification")
    print("✓ Geometric property analysis (roughness, curvature, etc.)")
    print("✓ Visual feedback showing classification results")
    print("✓ Fracture surface extraction")


def classify_your_own_mesh(mesh_path):
    """
    Classify faces in your own mesh file.
    
    Args:
        mesh_path: path to your mesh file (.ply, .obj, .stl, etc.)
    """
    print(f"Loading mesh from {mesh_path}...")
    
    try:
        # Load mesh
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        if not o3d_mesh.has_triangles():
            print("Error: Mesh has no triangles")
            return
        
        # Convert to trimesh
        tri_mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
            process=False
        )
        
        # Load parameters
        params = load_parameters()
        
        # Classify faces
        print("Classifying faces...")
        classification_result = classify_fracture_vs_original_faces(tri_mesh, params)
        
        # Show results
        fracture_count = np.sum(classification_result['fracture_faces'])
        total_count = len(tri_mesh.faces)
        
        print(f"Classification Results for {mesh_path}:")
        print(f"  Fracture faces: {fracture_count}/{total_count} ({fracture_count/total_count*100:.1f}%)")
        
        # Visualize
        if params.get('visualize_classification', True):
            visualize_face_classification(o3d_mesh, classification_result, os.path.basename(mesh_path))
        
    except Exception as e:
        print(f"Error processing mesh: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Classify user-provided mesh
        mesh_path = sys.argv[1]
        classify_your_own_mesh(mesh_path)
    else:
        # Run demo
        main()
        
        print("\nTo classify your own mesh, run:")
        print("python simple_classification_example.py path/to/your/mesh.ply") 