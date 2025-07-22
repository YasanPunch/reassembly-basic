"""
3D Model Classification and Analysis

This module provides a command-line interface for analyzing 3D models using curvature and roughness analysis.
"""

import os
import numpy as np
import open3d as o3d
import argparse
from utils.test_utils import load_3d_models_from_folder, visualize_models

# Import from the new modular structure
from curvature_analysis import analyze_mesh_curvature
from roughness_analysis import analyze_mesh_roughness
from mesh_utils import segment_mesh_and_analyze_curvature, segment_mesh_and_analyze_roughness
from visualization import visualize_bending_energy, visualize_roughness_characteristic
from segmented_visualization import visualize_segmented_curvature, visualize_segmented_roughness
from supervised_learning import run_supervised_fracture_detection


def main():
    parser = argparse.ArgumentParser(description="Load and visualize 3D models from a folder")
    parser.add_argument(
        "--folder", 
        type=str, 
        default="data/input_fragments",
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
    parser.add_argument(
        "--all-together", 
        action="store_true",
        help="Visualize all models together instead of one by one"
    )
    parser.add_argument(
        "--no-curvature-analysis",
        action="store_true",
        help="Disable curvature analysis and use regular visualization instead"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=100,
        help="Number of nearest neighbors for curvature analysis"
    )
    parser.add_argument(
        "--segment-first",
        action="store_true",
        help="Segment the mesh before analyzing curvature for each region separately"
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=30.0,
        help="Angle threshold for segmentation (default: 30.0 degrees)"
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=0.1,
        help="Curvature threshold for segmentation (default: 0.1)"
    )
    parser.add_argument(
        "--min-region-size",
        type=int,
        default=50,
        help="Minimum region size for segmentation (default: 50 faces)"
    )
    parser.add_argument(
        "--max-region-size",
        type=int,
        default=5000,
        help="Maximum region size for segmentation (default: 5000 faces)"
    )
    parser.add_argument(
        "--region-offset",
        type=float,
        default=0.2,
        help="Percentage of region to offset inward to avoid edge artifacts (default: 0.1 = 10%)"
    )
    parser.add_argument(
        "--use-roughness-analysis",
        action="store_true",
        help="Use surface roughness characteristic analysis instead of local bending energy"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Kernel radius for roughness analysis. If not specified, will be auto-calculated (default: None)"
    )
    parser.add_argument(
        "--supervised-learning",
        action="store_true",
        help="Use supervised learning to optimize parameters and classify fracture surfaces"
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
        
        if not args.no_curvature_analysis:
            # Analyze curvature for each mesh (default behavior)
            for i, geometry in enumerate(geometries):
                if isinstance(geometry, o3d.geometry.TriangleMesh):
                    print(f"\n{'='*50}")
                    print(f"Analyzing {'roughness' if args.use_roughness_analysis else 'curvature'} for model {i+1}/{len(geometries)}")
                    print(f"{'='*50}")
                    
                    if args.segment_first:
                        # Segment first, then analyze each region
                        print(f"Using segmented {'roughness' if args.use_roughness_analysis else 'curvature'} analysis...")
                        
                        # Prepare segmentation parameters
                        segmentation_params = {
                            'angle_threshold': args.angle_threshold,
                            'curvature_threshold': args.curvature_threshold,
                            'min_region_size': args.min_region_size,
                            'max_region_size': args.max_region_size
                        }
                        
                        # Perform segmented analysis
                        if args.use_roughness_analysis:
                            segmentation_results = segment_mesh_and_analyze_roughness(
                                geometry, 
                                args.k_neighbors, 
                                args.radius,
                                segmentation_params
                            )
                            
                            # Visualize each region
                            if args.supervised_learning:
                                run_supervised_fracture_detection(geometry, segmentation_results)
                            else:
                                visualize_segmented_roughness(
                                    geometry,
                                    segmentation_results,
                                    f"Segmented Roughness - Model {i+1}",
                                    args.region_offset
                                )
                        else:
                            segmentation_results = segment_mesh_and_analyze_curvature(
                                geometry, 
                                args.k_neighbors, 
                                segmentation_params
                            )
                            
                            # Visualize each region
                            if args.supervised_learning:
                                run_supervised_fracture_detection(geometry, segmentation_results)
                            else:
                                visualize_segmented_curvature(
                                    geometry,
                                    segmentation_results,
                                    f"Segmented Curvature - Model {i+1}",
                                    args.region_offset
                                )
                    else:
                        # Regular single-region analysis
                        print(f"Using single-region {'roughness' if args.use_roughness_analysis else 'curvature'} analysis...")
                        
                        # Analyze curvature or roughness
                        if args.use_roughness_analysis:
                            stats = analyze_mesh_roughness(geometry, args.k_neighbors, args.radius)
                            
                            # Visualize roughness characteristic
                            visualize_roughness_characteristic(
                                geometry, 
                                stats['roughness_characteristics'],
                                f"Surface Roughness - Model {i+1}"
                            )
                        else:
                            stats = analyze_mesh_curvature(geometry, args.k_neighbors)
                            
                            # Visualize bending energy
                            visualize_bending_energy(
                                geometry, 
                                stats['bending_energies'],
                                f"Bending Energy - Model {i+1}"
                            )
                else:
                    print(f"Model {i+1} is not a triangle mesh, skipping curvature analysis.")
        else:
            # Regular visualization (when --no-curvature-analysis is used)
            visualize_models(geometries, args.window_name, visualize_one_by_one=not args.all_together)
    else:
        print("No models were loaded successfully.")


if __name__ == "__main__":
    main()
