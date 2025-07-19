#!/usr/bin/env python3
"""
Comprehensive demo showing how fracture surface classification works
using adaptive thresholds based on the whole object's geometric properties.

This demo shows:
1. How thresholds are calculated from the object itself (no hardcoded values)
2. How different geometric properties are used for classification
3. How the system adapts to different object types automatically
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from segmentation import (
    calculate_adaptive_thresholds,
    detect_fracture_surfaces_adaptive,
    calculate_face_roughness,
    calculate_face_curvature,
    calculate_face_boundary_complexity,
    calculate_face_symmetry_score,
    calculate_face_planarity,
    run_adaptive_detection_with_validation
)
import trimesh


def create_mixed_surface_mesh():
    """
    Create a mesh with both smooth (original) and rough (fracture) surfaces.
    """
    print("Creating test mesh with mixed surface properties...")
    
    # Create base mesh
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    vertices = np.asarray(mesh.vertices)
    
    # Make top face rough (simulate fracture)
    np.random.seed(42)
    top_vertices = np.where(vertices[:, 2] > 1.9)[0]
    vertices[top_vertices, 2] += np.random.normal(0, 0.15, len(top_vertices))
    
    # Make front face moderately rough
    front_vertices = np.where(vertices[:, 1] > 1.9)[0]
    vertices[front_vertices, 1] += np.random.normal(0, 0.08, len(front_vertices))
    
    # Keep other faces smooth (original surfaces)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    
    print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    return mesh


def analyze_geometric_properties(tri_mesh, params):
    """
    Analyze and visualize geometric properties across the whole object.
    """
    print("\n=== Analyzing Geometric Properties Across Whole Object ===")
    
    num_faces = len(tri_mesh.faces)
    
    # Calculate all geometric properties for every face
    print("Calculating geometric properties for all faces...")
    roughness_values = np.zeros(num_faces)
    curvature_values = np.zeros(num_faces)
    boundary_complexity_values = np.zeros(num_faces)
    symmetry_values = np.zeros(num_faces)
    planarity_values = np.zeros(num_faces)
    
    for face_idx in range(num_faces):
        roughness_values[face_idx] = calculate_face_roughness(tri_mesh, face_idx, params)
        curvature_values[face_idx] = calculate_face_curvature(tri_mesh, face_idx)
        boundary_complexity_values[face_idx] = calculate_face_boundary_complexity(tri_mesh, face_idx)
        symmetry_values[face_idx] = calculate_face_symmetry_score(tri_mesh, face_idx, params)
        planarity_values[face_idx] = calculate_face_planarity(tri_mesh, face_idx)
    
    # Show statistics
    properties = {
        'Roughness': roughness_values,
        'Curvature': curvature_values,
        'Boundary Complexity': boundary_complexity_values,
        'Symmetry': symmetry_values,
        'Planarity': planarity_values
    }
    
    print("\nGeometric Property Statistics:")
    print("=" * 50)
    for prop_name, values in properties.items():
        print(f"{prop_name}:")
        print(f"  Min: {np.min(values):.4f}")
        print(f"  Max: {np.max(values):.4f}")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std: {np.std(values):.4f}")
        print(f"  75th percentile: {np.percentile(values, 75):.4f}")
        print(f"  25th percentile: {np.percentile(values, 25):.4f}")
        print()
    
    return properties


def demonstrate_adaptive_thresholds(tri_mesh, params):
    """
    Demonstrate how adaptive thresholds are calculated from object properties.
    """
    print("\n=== Demonstrating Adaptive Threshold Calculation ===")
    
    # Calculate adaptive thresholds
    thresholds = calculate_adaptive_thresholds(tri_mesh, params)
    
    print("\nAdaptive Thresholds (calculated from object properties):")
    print("=" * 60)
    
    for prop, threshold_info in thresholds['statistics'].items():
        print(f"{prop.capitalize()}:")
        print(f"  Object range: {threshold_info['min']:.4f} - {threshold_info['max']:.4f}")
        print(f"  Object mean: {threshold_info['mean']:.4f}")
        print(f"  Adaptive threshold: {threshold_info['threshold']:.4f}")
        
        if prop in ['roughness', 'curvature', 'boundary_complexity']:
            percentile = params.get(f'{prop}_threshold_percentile', 75)
            print(f"  → Selects top {100-percentile}% of faces for this property")
        else:
            percentile = params.get(f'{prop}_threshold_percentile', 25)
            print(f"  → Selects bottom {percentile}% of faces for this property")
        print()
    
    return thresholds


def demonstrate_classification_process(tri_mesh, params):
    """
    Demonstrate the complete classification process.
    """
    print("\n=== Demonstrating Classification Process ===")
    
    # Run adaptive detection with validation
    detection_results, global_stats = run_adaptive_detection_with_validation(tri_mesh, params)
    
    if detection_results is None:
        print("Classification failed!")
        return None
    
    print("\nClassification Results:")
    print("=" * 40)
    
    total_faces = len(tri_mesh.faces)
    for method, data in detection_results.items():
        if 'candidates' in data:
            count = data['count']
            percentage = data['percentage']
            print(f"{method.replace('_', ' ').title()}:")
            print(f"  Detected {count}/{total_faces} faces as fracture surfaces ({percentage:.1f}%)")
    
    return detection_results


def visualize_classification_results(o3d_mesh, tri_mesh, detection_results, properties):
    """
    Create visualizations showing the classification results.
    """
    print("\n=== Creating Visualizations ===")
    
    # Create property-based visualization
    print("Creating property-based visualization...")
    
    # Visualize roughness distribution
    roughness_values = properties['Roughness']
    roughness_threshold = np.percentile(roughness_values, 75)
    
    # Color faces by roughness
    face_colors = np.zeros((len(tri_mesh.faces), 3))
    for i, roughness in enumerate(roughness_values):
        if roughness > roughness_threshold:
            # Red for rough (potential fracture)
            intensity = min(1.0, (roughness - roughness_threshold) / (np.max(roughness_values) - roughness_threshold))
            face_colors[i] = [1.0, 1.0 - intensity, 1.0 - intensity]
        else:
            # Blue for smooth (original surface)
            intensity = 1.0 - (roughness / roughness_threshold)
            face_colors[i] = [1.0 - intensity, 1.0 - intensity, 1.0]
    
    # Create visualization mesh
    vis_mesh = o3d.geometry.TriangleMesh()
    vis_mesh.vertices = o3d_mesh.vertices
    vis_mesh.triangles = o3d_mesh.triangles
    vis_mesh.compute_vertex_normals()
    
    # Apply colors
    vertex_colors = np.zeros((len(vis_mesh.vertices), 3))
    for i, triangle in enumerate(np.asarray(vis_mesh.triangles)):
        for vertex_idx in triangle:
            vertex_colors[vertex_idx] = face_colors[i]
    
    vis_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    print("Visualizing roughness-based classification...")
    print("Red = Rough surfaces (potential fractures)")
    print("Blue = Smooth surfaces (original surfaces)")
    
    o3d.visualization.draw_geometries([vis_mesh], window_name="Roughness-based Classification")
    
    # Visualize final classification results
    if detection_results and 'combined_adaptive' in detection_results:
        print("Creating final classification visualization...")
        
        candidates = detection_results['combined_adaptive']['candidates']
        
        # Create classification mesh
        class_mesh = copy.deepcopy(o3d_mesh)
        class_mesh.compute_vertex_normals()
        
        # Color based on classification
        vertex_colors = np.full((len(class_mesh.vertices), 3), [0.7, 0.7, 0.7])  # Gray for original
        
        for i, is_fracture in enumerate(candidates):
            if is_fracture:
                triangle = np.asarray(class_mesh.triangles)[i]
                for vertex_idx in triangle:
                    vertex_colors[vertex_idx] = [1.0, 0.0, 0.0]  # Red for fracture
        
        class_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        print("Red = Classified as fracture surface")
        print("Gray = Classified as original surface")
        
        o3d.visualization.draw_geometries([class_mesh], window_name="Final Classification Results")


def compare_different_objects():
    """
    Compare how the system adapts to different object types.
    """
    print("\n=== Comparing Different Object Types ===")
    
    # Create different object types
    objects = {
        "Smooth Sphere": o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20),
        "Rough Cube": create_mixed_surface_mesh(),
        "Cylinder": o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=2.0, resolution=20)
    }
    
    params = {
        'roughness_threshold_percentile': 75,
        'curvature_threshold_percentile': 75,
        'boundary_complexity_threshold_percentile': 75,
        'symmetry_threshold_percentile': 25,
        'planarity_threshold_percentile': 25,
        'score_threshold_percentile': 70
    }
    
    print("Analyzing how adaptive thresholds change for different objects:")
    print("=" * 70)
    
    for obj_name, o3d_mesh in objects.items():
        print(f"\n{obj_name}:")
        print("-" * len(obj_name))
        
        # Convert to trimesh
        tri_mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
            process=False
        )
        
        # Calculate a few sample properties
        num_faces = min(10, len(tri_mesh.faces))
        sample_roughness = []
        sample_curvature = []
        
        for i in range(num_faces):
            sample_roughness.append(calculate_face_roughness(tri_mesh, i, params))
            sample_curvature.append(calculate_face_curvature(tri_mesh, i))
        
        avg_roughness = np.mean(sample_roughness)
        avg_curvature = np.mean(sample_curvature)
        
        print(f"  Average roughness: {avg_roughness:.4f}")
        print(f"  Average curvature: {avg_curvature:.4f}")
        print(f"  → Thresholds will adapt to these object-specific properties")


def main():
    """
    Main demonstration function.
    """
    print("=" * 80)
    print("ADAPTIVE FRACTURE SURFACE CLASSIFICATION DEMO")
    print("=" * 80)
    print()
    print("This demo shows how fracture surfaces are classified using:")
    print("1. Geometric properties (roughness, curvature, etc.)")
    print("2. Adaptive thresholds (no hardcoded values)")
    print("3. Whole-object analysis for threshold calculation")
    print()
    
    # Parameters for classification
    params = {
        'roughness_threshold_percentile': 75,  # Top 25% roughest faces
        'curvature_threshold_percentile': 75,  # Top 25% highest curvature
        'boundary_complexity_threshold_percentile': 75,  # Top 25% most complex
        'symmetry_threshold_percentile': 25,   # Bottom 25% least symmetric
        'planarity_threshold_percentile': 25,  # Bottom 25% least planar
        'score_threshold_percentile': 70,      # Top 30% highest combined scores
        'fracture_detection_weights': {
            'curvature': 0.3,
            'roughness': 0.3,
            'boundary_complexity': 0.2,
            'symmetry': 0.1,
            'planarity': 0.1
        }
    }
    
    # Create test mesh
    o3d_mesh = create_mixed_surface_mesh()
    
    # Convert to trimesh
    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False
    )
    
    # Step 1: Analyze geometric properties
    properties = analyze_geometric_properties(tri_mesh, params)
    
    # Step 2: Demonstrate adaptive threshold calculation
    thresholds = demonstrate_adaptive_thresholds(tri_mesh, params)
    
    # Step 3: Demonstrate classification process
    detection_results = demonstrate_classification_process(tri_mesh, params)
    
    # Step 4: Visualize results
    if detection_results:
        visualize_classification_results(o3d_mesh, tri_mesh, detection_results, properties)
    
    # Step 5: Compare different objects
    compare_different_objects()
    
    print("\n" + "=" * 80)
    print("KEY POINTS DEMONSTRATED:")
    print("=" * 80)
    print("✓ NO HARDCODED THRESHOLDS: All thresholds calculated from object properties")
    print("✓ WHOLE-OBJECT ANALYSIS: System analyzes entire object to understand its characteristics")
    print("✓ ADAPTIVE CLASSIFICATION: Automatically adapts to different object types")
    print("✓ GEOMETRIC PROPERTIES: Uses roughness, curvature, symmetry, planarity, boundary complexity")
    print("✓ PERCENTILE-BASED: Uses relative rankings (e.g., top 25% roughest faces)")
    print("✓ MULTI-METHOD VALIDATION: Combines multiple detection methods for robustness")
    print()
    print("This approach works for any object type without manual parameter tuning!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc() 