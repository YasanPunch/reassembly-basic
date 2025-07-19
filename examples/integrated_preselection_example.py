#!/usr/bin/env python3
"""
Example demonstrating the integrated automatic pre-selection system.

This shows how:
1. Automatic classification runs first and pre-selects regions
2. Interactive visualization shows pre-selected regions in gold
3. User can modify the selection if needed
4. Normal flow continues with final selection
"""

import open3d as o3d
import numpy as np
import json
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from segmentation import extract_fracture_surface_mesh
import trimesh


def create_mixed_surface_mesh():
    """Create a test mesh with both smooth and rough surfaces."""
    print("Creating test mesh with mixed surface properties...")
    
    # Create a cube
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    vertices = np.asarray(mesh.vertices)
    
    # Make some faces rough (simulate fractures)
    np.random.seed(42)
    
    # Top face - very rough (should be auto-selected)
    top_vertices = np.where(vertices[:, 2] > 1.9)[0]
    vertices[top_vertices, 2] += np.random.normal(0, 0.25, len(top_vertices))
    
    # Front face - moderately rough (should be auto-selected)
    front_vertices = np.where(vertices[:, 1] > 1.9)[0]
    vertices[front_vertices, 1] += np.random.normal(0, 0.15, len(front_vertices))
    
    # Right face - slightly rough (might be auto-selected)
    right_vertices = np.where(vertices[:, 0] > 1.9)[0]
    vertices[right_vertices, 0] += np.random.normal(0, 0.08, len(right_vertices))
    
    # Keep other faces smooth (should NOT be auto-selected)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    
    print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    print("Expected auto-selections: Top face (very rough), Front face (moderately rough)")
    print("Possible auto-selection: Right face (slightly rough)")
    print("Expected non-selections: Bottom, Back, Left faces (smooth)")
    
    return mesh


def load_parameters():
    """Load parameters for integrated pre-selection."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'integrated_preselection_params.json')
    
    try:
        with open(config_path, 'r') as f:
            params = json.load(f)
        print(f"Loaded parameters from {config_path}")
        return params
    except FileNotFoundError:
        print("Config file not found. Using default parameters.")
        return {
            'visualize_segmentation': True,
            'use_automatic_preselection': True,
            'preselection_fracture_ratio_threshold': 0.3,
            'final_classification_threshold': 0.5,
            'roughness_threshold_percentile': 75,
            'curvature_threshold_percentile': 75,
            'boundary_complexity_threshold_percentile': 75,
            'symmetry_threshold_percentile': 25,
            'planarity_threshold_percentile': 25,
            'fracture_detection_weights': {
                'curvature': 0.3,
                'roughness': 0.3,
                'boundary_complexity': 0.2,
                'symmetry': 0.1,
                'planarity': 0.1
            }
        }


def main():
    """Main function demonstrating integrated pre-selection."""
    print("=" * 80)
    print("INTEGRATED AUTOMATIC PRE-SELECTION DEMO")
    print("=" * 80)
    print()
    print("This demo shows the integrated system where:")
    print("1. Automatic classification runs first")
    print("2. Regions are PRE-SELECTED based on fracture probability")
    print("3. Interactive visualization shows pre-selected regions in GOLD")
    print("4. You can modify the selection by toggling regions")
    print("5. Normal processing continues with final selection")
    print()
    
    # Load parameters
    params = load_parameters()
    
    # Create test mesh
    test_mesh = create_mixed_surface_mesh()
    
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATED FRACTURE SURFACE EXTRACTION")
    print("=" * 60)
    print()
    print("The system will now:")
    print("1. Run region growing segmentation")
    print("2. Automatically classify faces as fracture vs original")
    print("3. Pre-select regions with high fracture probability")
    print("4. Show interactive visualization with pre-selected regions")
    print()
    print("In the visualization:")
    print("  🟡 GOLD regions = Auto-selected by algorithm")
    print("  ⚫ BLACK regions = User-selected (if you toggle any)")
    print("  🌈 COLORED regions = Not selected")
    print()
    print("Controls:")
    print("  1-9, 0: Toggle region selection")
    print("  N/P: Navigate pages")
    print("  S: Confirm selection and continue")
    print("  Q: Quit without selection")
    print()
    
    input("Press Enter to start the integrated system...")
    
    # Run the integrated system
    fracture_surfaces = extract_fracture_surface_mesh(test_mesh, "IntegratedDemo", params)
    
    if fracture_surfaces:
        if isinstance(fracture_surfaces, list):
            print(f"\n✅ Successfully extracted {len(fracture_surfaces)} fracture surface(s)!")
            total_vertices = sum(len(surf.vertices) for surf in fracture_surfaces)
            total_triangles = sum(len(surf.triangles) for surf in fracture_surfaces)
            print(f"   Total: {total_vertices} vertices, {total_triangles} triangles")
        else:
            print(f"\n✅ Successfully extracted fracture surface!")
            print(f"   {len(fracture_surfaces.vertices)} vertices, {len(fracture_surfaces.triangles)} triangles")
            fracture_surfaces = [fracture_surfaces]  # Make it a list for visualization
        
        # Visualize final results
        print(f"\n📊 Visualizing final fracture surfaces...")
        
        # Color fracture surfaces
        for i, surface in enumerate(fracture_surfaces):
            color = [1.0, 0.2, 0.2] if i == 0 else [0.8, 0.2, 0.2]  # Different shades of red
            surface.paint_uniform_color(color)
            surface.compute_vertex_normals()
        
        # Show original mesh in gray
        original_vis = test_mesh
        original_vis.paint_uniform_color([0.7, 0.7, 0.7])
        original_vis.compute_vertex_normals()
        
        # Create wireframe
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(original_vis)
        wireframe.paint_uniform_color([0.5, 0.5, 0.5])
        
        # Show everything together
        vis_objects = [original_vis, wireframe] + fracture_surfaces
        o3d.visualization.draw_geometries(vis_objects, window_name="Final Fracture Surfaces")
        
    else:
        print(f"\n❌ No fracture surfaces were selected or extracted.")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("What happened:")
    print("✓ Automatic classification analyzed geometric properties")
    print("✓ Regions were pre-selected based on fracture probability") 
    print("✓ Interactive visualization showed pre-selected regions")
    print("✓ You could modify the selection if needed")
    print("✓ Normal processing continued with final selection")
    print()
    print("Key Benefits:")
    print("• No manual analysis needed - algorithm does the work")
    print("• User retains control - can override automatic selection")
    print("• Seamless integration - fits into existing workflow")
    print("• Adaptive thresholds - works across different object types")


def test_different_thresholds():
    """Test how different threshold settings affect pre-selection."""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT THRESHOLD SETTINGS")
    print("=" * 60)
    
    test_mesh = create_mixed_surface_mesh()
    
    # Test different threshold settings
    threshold_configs = [
        {"name": "Conservative (85th percentile)", "roughness": 85, "curvature": 85},
        {"name": "Default (75th percentile)", "roughness": 75, "curvature": 75},
        {"name": "Aggressive (65th percentile)", "roughness": 65, "curvature": 65}
    ]
    
    for config in threshold_configs:
        print(f"\n--- {config['name']} ---")
        
        params = {
            'use_automatic_preselection': True,
            'visualize_segmentation': False,  # Skip visualization for this test
            'preselection_fracture_ratio_threshold': 0.3,
            'roughness_threshold_percentile': config['roughness'],
            'curvature_threshold_percentile': config['curvature'],
            'boundary_complexity_threshold_percentile': config['roughness'],
            'symmetry_threshold_percentile': 100 - config['roughness'],
            'planarity_threshold_percentile': 100 - config['roughness']
        }
        
        try:
            # Just run the extraction to see pre-selection results
            fracture_surfaces = extract_fracture_surface_mesh(test_mesh, f"Test_{config['name']}", params)
            
            if fracture_surfaces:
                if isinstance(fracture_surfaces, list):
                    total_faces = sum(len(surf.triangles) for surf in fracture_surfaces)
                else:
                    total_faces = len(fracture_surfaces.triangles)
                print(f"  Result: {total_faces} faces selected as fracture surfaces")
            else:
                print(f"  Result: No fracture surfaces detected")
                
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    try:
        # Main demo
        main()
        
        # Test different thresholds
        response = input("\nWould you like to test different threshold settings? (y/n): ")
        if response.lower() == 'y':
            test_different_thresholds()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nDemo finished!") 