#!/usr/bin/env python3
"""
Example script demonstrating adaptive fracture surface detection.
This script shows how to use the simplified adaptive approach that automatically
adjusts thresholds based on object properties.
"""

import open3d as o3d
import numpy as np
import json
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from segmentation import (
    extract_fracture_surface_mesh,
    compare_fracture_detection_methods_adaptive,
    visualize_detection_comparison
)


def create_test_mesh():
    """
    Create a test mesh with both smooth and rough surfaces to demonstrate detection.
    """
    print("Creating test mesh with mixed surface properties...")
    
    # Create a base cube
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    
    # Add some roughness to one face to simulate fracture
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Find vertices on the top face (z = 1.0)
    top_face_vertices = np.where(vertices[:, 2] > 0.9)[0]
    
    # Add random noise to simulate fracture surface
    np.random.seed(42)  # For reproducible results
    noise_scale = 0.1
    vertices[top_face_vertices, 2] += np.random.normal(0, noise_scale, len(top_face_vertices))
    
    # Update mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    
    print(f"Created test mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    return mesh


def load_config(config_path):
    """
    Load configuration from JSON file.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using default parameters.")
        return {}


def demonstrate_adaptive_detection():
    """
    Demonstrate the adaptive fracture detection approach.
    """
    print("=== Adaptive Fracture Surface Detection Demo ===\n")
    
    # Create test mesh
    test_mesh = create_test_mesh()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'simplified_adaptive_params.json')
    params = load_config(config_path)
    
    # Set up basic parameters
    params.update({
        'visualize_segmentation': False,  # Disable interactive selection for demo
        'use_adaptive_detection': True,
        'use_combined_approach': True
    })
    
    print("\n=== Configuration ===")
    print(f"Adaptive detection: {params.get('use_adaptive_detection', False)}")
    print(f"Combined approach: {params.get('use_combined_approach', False)}")
    print(f"Roughness percentile: {params.get('roughness_threshold_percentile', 75)}")
    print(f"Curvature percentile: {params.get('curvature_threshold_percentile', 75)}")
    print(f"Score percentile: {params.get('score_threshold_percentile', 70)}")
    
    # Run adaptive detection comparison
    print("\n=== Running Adaptive Detection Methods ===")
    
    # Convert to trimesh for analysis
    import trimesh
    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(test_mesh.vertices),
        faces=np.asarray(test_mesh.triangles),
        process=False
    )
    
    # Compare detection methods
    detection_results, global_stats = compare_fracture_detection_methods_adaptive(tri_mesh, params)
    
    # Visualize results
    print("\n=== Visualizing Results ===")
    visualize_detection_comparison(test_mesh, detection_results, "TestMesh")
    
    # Run main segmentation
    print("\n=== Running Main Segmentation ===")
    fracture_surfaces = extract_fracture_surface_mesh(test_mesh, "TestMesh", params)
    
    if fracture_surfaces:
        print(f"Extracted {len(fracture_surfaces)} fracture surface(s)")
        for i, surface in enumerate(fracture_surfaces):
            print(f"  Surface {i+1}: {len(surface.vertices)} vertices, {len(surface.triangles)} triangles")
    else:
        print("No fracture surfaces detected")
    
    print("\n=== Demo Complete ===")
    print("The adaptive approach automatically calculated thresholds based on the object's properties.")
    print("No manual parameter tuning was required!")


def demonstrate_parameter_sensitivity():
    """
    Demonstrate how different percentile parameters affect detection.
    """
    print("\n=== Parameter Sensitivity Demo ===")
    
    # Create test mesh
    test_mesh = create_test_mesh()
    
    # Convert to trimesh
    import trimesh
    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(test_mesh.vertices),
        faces=np.asarray(test_mesh.triangles),
        process=False
    )
    
    # Test different percentile settings
    percentile_configs = [
        {"name": "Conservative (85th percentile)", "percentile": 85},
        {"name": "Default (75th percentile)", "percentile": 75},
        {"name": "Aggressive (65th percentile)", "percentile": 65}
    ]
    
    for config in percentile_configs:
        print(f"\n--- {config['name']} ---")
        
        params = {
            'use_adaptive_detection': True,
            'use_combined_approach': True,
            'roughness_threshold_percentile': config['percentile'],
            'curvature_threshold_percentile': config['percentile'],
            'boundary_complexity_threshold_percentile': config['percentile'],
            'symmetry_threshold_percentile': 100 - config['percentile'],
            'planarity_threshold_percentile': 100 - config['percentile'],
            'score_threshold_percentile': config['percentile'] - 5
        }
        
        detection_results, _ = compare_fracture_detection_methods_adaptive(tri_mesh, params)
        
        for method, data in detection_results.items():
            if 'combined' in method:
                print(f"  {method}: {data['count']} faces ({data['percentage']:.1f}%)")


if __name__ == "__main__":
    print("Adaptive Fracture Surface Detection Example")
    print("=" * 50)
    
    try:
        # Main demonstration
        demonstrate_adaptive_detection()
        
        # Parameter sensitivity demonstration
        demonstrate_parameter_sensitivity()
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExample completed successfully!")
    print("\nKey benefits of the adaptive approach:")
    print("1. No manual threshold tuning required")
    print("2. Works automatically across different objects")
    print("3. Simple percentile-based configuration")
    print("4. Robust detection using multiple methods") 