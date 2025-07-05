#!/usr/bin/env python3
"""
Debug script to test the matching process and identify issues.
"""

import os
import sys
import json
import numpy as np
import open3d as o3d
import copy

# Add src to path
sys.path.append('src')

import src.io_utils
import src.preprocessing
import src.feature_extraction
import src.matching
import src.alignment

def debug_matching_process():
    """Debug the matching process step by step."""
    
    # Load parameters
    config_file = "config/reconstruction_params.json"
    with open(config_file, 'r') as f:
        params = json.load(f)
    
    # Fix missing parameter
    if "min_ransac_fitness_threshold" not in params:
        params["min_ransac_fitness_threshold"] = params.get("min_ransac_fitness", 0.1)
    
    # Adjust parameters for debugging
    params["voxel_downsample_size"] = 1.0  # Reduce from 3.0
    params["ransac_iterations"] = 100000    # Reduce from 4000000
    params["min_match_score"] = 0.1         # Lower threshold for debugging
    params["min_enhanced_score"] = 0.1      # Lower threshold for debugging
    params["min_quality_score"] = 0.05      # Lower threshold for debugging
    
    print("=== DEBUGGING MATCHING PROCESS ===")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    
    # Load fragments
    input_dir = "data/input_fragments"
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return
    
    fragments_data_raw = src.io_utils.load_fragments_from_directory(input_dir)
    if not fragments_data_raw:
        print("No fragments loaded.")
        return
    
    print(f"\nLoaded {len(fragments_data_raw)} fragments:")
    for i, frag in enumerate(fragments_data_raw):
        mesh = frag['mesh']
        print(f"  {i}: {frag['name']} - {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Process fragments
    processed_fragments = []
    for i, frag_info in enumerate(fragments_data_raw):
        print(f"\nProcessing fragment {i+1}/{len(fragments_data_raw)}: {frag_info['name']}")
        
        # Preprocess
        pcd_for_features, fracture_surface_mesh = src.preprocessing.preprocess_fragment(
            frag_info, params, viz_collector=None
        )
        
        if pcd_for_features is None or not pcd_for_features.has_points():
            print(f"  Warning: Preprocessing failed - no point cloud generated")
            continue
        
        print(f"  Point cloud: {len(pcd_for_features.points)} points")
        
        # Extract features
        features, _ = src.feature_extraction.extract_features_from_pcd(pcd_for_features, params)
        
        if features is None or features.num() == 0:
            print(f"  Warning: Feature extraction failed - no FPFH features")
            continue
        
        print(f"  FPFH features: {features.num()} features")
        
        processed_fragments.append({
            'name': frag_info['name'],
            'original_index': frag_info['original_index'],
            'original_mesh': frag_info['mesh'],
            'pcd_for_features': pcd_for_features,
            'features': features
        })
    
    print(f"\nSuccessfully processed {len(processed_fragments)} fragments")
    
    if len(processed_fragments) < 2:
        print("Not enough processed fragments for matching.")
        return
    
    # Test pairwise matching
    print(f"\n=== TESTING PAIRWISE MATCHING ===")
    
    # Test individual pairs
    for i in range(len(processed_fragments)):
        for j in range(i+1, len(processed_fragments)):
            print(f"\nTesting pair: {processed_fragments[i]['name']} <-> {processed_fragments[j]['name']}")
            
            frag_i = processed_fragments[i]
            frag_j = processed_fragments[j]
            
            # Test alignment directly
            source_pcd = frag_i['pcd_for_features']
            target_pcd = frag_j['pcd_for_features']
            source_fpfh = frag_i['features']
            target_fpfh = frag_j['features']
            
            print(f"  Source PCD: {len(source_pcd.points)} points")
            print(f"  Target PCD: {len(target_pcd.points)} points")
            print(f"  Source FPFH: {source_fpfh.num()} features")
            print(f"  Target FPFH: {target_fpfh.num()} features")
            
            # Test alignment
            transformation, fitness, rmse = src.alignment.align_fragments_pcd(
                source_pcd, target_pcd, source_fpfh, target_fpfh, params
            )
            
            print(f"  Alignment result: transformation={'Found' if transformation is not None else 'None'}, "
                  f"fitness={fitness:.4f}, rmse={rmse:.4f}")
            
            if transformation is not None:
                print(f"  ✓ Match found!")
            else:
                print(f"  ✗ No match")
    
    # Test full matching function
    print(f"\n=== TESTING FULL MATCHING FUNCTION ===")
    matches = src.matching.find_pairwise_matches(processed_fragments, params)
    
    print(f"Found {len(matches)} matches:")
    for i, match in enumerate(matches):
        print(f"  {i+1}: {match['source_name']} -> {match['target_name']} "
              f"(score: {match['score']:.3f}, fitness: {match['fitness']:.3f}, rmse: {match['rmse']:.3f})")

if __name__ == "__main__":
    debug_matching_process() 