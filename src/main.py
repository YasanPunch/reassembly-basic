import os
import json
import argparse
import open3d as o3d
print("o3d-version: ", o3d.__version__) # Keep for debugging
import time
import numpy as np
import copy # For deep copying geometries for visualization log

# For checking module paths (can be removed in final version)
#import importlib.util
#import sys
# print("--- Python Path (sys.path) ---")
# for p_path in sys.path:
#     print(p_path)
# print("--------------------------------")
# spec_preprocessing = importlib.util.find_spec("src.preprocessing")
# if spec_preprocessing:
#     print(f"DEBUG: Python found 'src.preprocessing' at: {spec_preprocessing.origin}")
# else:
#     print("DEBUG: Python could NOT find 'src.preprocessing' via importlib.util.find_spec")
# ---

import src.io_utils
import src.preprocessing
import src.segmentation # Though preprocessing calls segmentation
import src.feature_extraction
import src.matching
import src.assembly
import src.utils.visualization_utils as viz_utils # Changed import style

def main(args):
    print("DEBUG: main(args) function entered.")
    print("--- 3D Model Fragment Reconstructor (Advanced) ---")
    start_time = time.time()

    visualization_log = [] # Initialize the log for visualization steps

    # 1. Load Parameters
    print("\n[1. Loading Parameters]")
    try:
        with open(args.config_file, 'r') as f:
            params = json.load(f)
        print(f"  Parameters loaded from: {args.config_file}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config_file}. Exiting.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.config_file}. Exiting.")
        return

    # 2. Load Fragments
    print("\n[2. Loading Fragments]")
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}. Exiting.")
        return
    
    # fragments_data_raw will be list of {'mesh': o3d_mesh, 'name': str, 'original_index': int}
    fragments_data_raw = src.io_utils.load_fragments_from_directory(args.input_dir)
    if not fragments_data_raw:
        print("No fragments loaded. Exiting.")
        return
    print(f"  Loaded {len(fragments_data_raw)} fragments.")
    for i, frag_info in enumerate(fragments_data_raw):
        # Log initial fragment geometry as arrays
        mesh_geom = frag_info['mesh']
        visualization_log.append({
            'step': 'initial_fragment',
            'name': frag_info['name'],
            'original_index': frag_info['original_index'],
            'type': 'mesh', # Indicate geometry type
            'vertices': np.asarray(mesh_geom.vertices),
            'triangles': np.asarray(mesh_geom.triangles),
            'vertex_colors': np.asarray(mesh_geom.vertex_colors) if mesh_geom.has_vertex_colors() else None,
            'vertex_normals': np.asarray(mesh_geom.vertex_normals) if mesh_geom.has_vertex_normals() else None,
        })


    # This will store more processed data:
    # {'name', 'original_index', 'original_mesh', 
    #  'fracture_surface_mesh' (optional), 'pcd_for_features', 'features'}
    processed_fragments_pipeline_data = []

    # 3. Preprocessing, Segmentation & Feature Extraction (per fragment)
    print("\n[3. Preprocessing, Segmentation, and Feature Extraction]")
    for i, frag_info_raw in enumerate(fragments_data_raw):
        print(f"  Processing fragment: {frag_info_raw['name']} ({i+1}/{len(fragments_data_raw)})")
        
        # Preprocessing now includes segmentation and returns (pcd_for_features, fracture_surface_mesh)
        # It also appends to visualization_log internally
        pcd_for_features, fracture_surface_mesh = src.preprocessing.preprocess_fragment(
            frag_info_raw, params, viz_collector=visualization_log
        )
        
        if pcd_for_features is None or not pcd_for_features.has_points():
            print(f"    Warning: Preprocessing resulted in empty point cloud for features for {frag_info_raw['name']}. Skipping.")
            # We could still add original_mesh to pipeline_data if we want to try assembling it later without features
            processed_fragments_pipeline_data.append({
                'name': frag_info_raw['name'],
                'original_index': frag_info_raw['original_index'],
                'original_mesh': frag_info_raw['mesh'],
                'fracture_surface_mesh': fracture_surface_mesh, # Could be None
                'pcd_for_features': None,
                'features': None
            })
            continue

        features, _ = src.feature_extraction.extract_features_from_pcd(pcd_for_features, params)
        
        if features is None or features.num() == 0 :
             print(f"    Warning: Feature extraction failed or yielded empty features for {frag_info_raw['name']}. Skipping.")
             processed_fragments_pipeline_data.append({
                'name': frag_info_raw['name'],
                'original_index': frag_info_raw['original_index'],
                'original_mesh': frag_info_raw['mesh'],
                'fracture_surface_mesh': fracture_surface_mesh,
                'pcd_for_features': pcd_for_features, # Store the pcd even if features are None
                'features': None
            })
             continue

        processed_fragments_pipeline_data.append({
            'name': frag_info_raw['name'],
            'original_index': frag_info_raw['original_index'],
            'original_mesh': frag_info_raw['mesh'], # Keep original for final assembly
            'fracture_surface_mesh': fracture_surface_mesh, # For visualization/debug
            'pcd_for_features': pcd_for_features,  # PCD used for FPFH (from fracture surface)
            'features': features # FPFH features
        })
    
    # Filter out fragments that failed feature extraction (essential for matching)
    valid_fragments_data = [fd for fd in processed_fragments_pipeline_data if fd.get('features') is not None and fd['features'].num() > 0]
    if len(valid_fragments_data) < len(processed_fragments_pipeline_data):
        print(f"  Warning: {len(processed_fragments_pipeline_data) - len(valid_fragments_data)} fragments had no valid features and were excluded from matching.")
    
    if len(valid_fragments_data) < 2: # Need at least 2 fragments for pairwise matching
        print("Not enough valid fragments with features for pairwise matching. Exiting or saving unaligned.")
        # Save unaligned original meshes if any loaded
        if fragments_data_raw:
            os.makedirs(args.output_dir, exist_ok=True)
            all_original_meshes = [fd['mesh'] for fd in fragments_data_raw]
            combined_unaligned = src.io_utils.combine_meshes(all_original_meshes)
            output_path = os.path.join(args.output_dir, "reconstructed_model_unaligned_originals.obj")
            src.io_utils.save_mesh(combined_unaligned, output_path)
            print(f"  Saved all original unaligned fragments to {output_path}")
        if args.visualize_steps_file: # Save log even on early exit
            viz_utils.save_visualization_log(visualization_log, args.visualize_steps_file)
        return

    # 4. Pairwise Matching
    print("\n[4. Finding Pairwise Matches]")
    # pairwise_matches will be list of dicts. Indices refer to `valid_fragments_data`
    pairwise_matches = src.matching.find_pairwise_matches(valid_fragments_data, params) 
    
    # Log pairwise matching attempts and results for visualization
    for idx_match, match in enumerate(pairwise_matches): # Added idx_match for unique naming if needed
        source_data = valid_fragments_data[match['source_idx']]
        target_data = valid_fragments_data[match['target_idx']]
        
        source_pcd_ff = source_data['pcd_for_features']
        target_pcd_ff = target_data['pcd_for_features']

        visualization_log.append({
            'step': 'pairwise_match_success',
            'match_index': idx_match, # For replayer to potentially focus on specific matches
            'source_name': source_data['name'],
            'target_name': target_data['name'],
            # Store PCD for features data
            'source_pcd_type': 'pointcloud',
            'source_pcd_points': np.asarray(source_pcd_ff.points),
            'source_pcd_colors': np.asarray(source_pcd_ff.colors) if source_pcd_ff.has_colors() else None,
            'source_pcd_normals': np.asarray(source_pcd_ff.normals) if source_pcd_ff.has_normals() else None,
            'target_pcd_type': 'pointcloud',
            'target_pcd_points': np.asarray(target_pcd_ff.points),
            'target_pcd_colors': np.asarray(target_pcd_ff.colors) if target_pcd_ff.has_colors() else None,
            'target_pcd_normals': np.asarray(target_pcd_ff.normals) if target_pcd_ff.has_normals() else None,
            # Store original mesh data for context
            'source_original_mesh_type': 'mesh',
            'source_original_mesh_verts': np.asarray(source_data['original_mesh'].vertices),
            'source_original_mesh_tris': np.asarray(source_data['original_mesh'].triangles),
            'target_original_mesh_type': 'mesh',
            'target_original_mesh_verts': np.asarray(target_data['original_mesh'].vertices),
            'target_original_mesh_tris': np.asarray(target_data['original_mesh'].triangles),
            'transformation': match['transformation'], # This is already a NumPy array
            'score': match['score'],
            'rmse': match['rmse']
        })

    if not pairwise_matches:
        print("  No suitable pairwise matches found above threshold. Attempting to save fragments separately.")
        os.makedirs(args.output_dir, exist_ok=True)
        all_original_meshes = [fd['original_mesh'] for fd in valid_fragments_data] # Use original meshes
        combined_unaligned = src.io_utils.combine_meshes(all_original_meshes)
        output_path = os.path.join(args.output_dir, "reconstructed_model_no_matches.obj")
        src.io_utils.save_mesh(combined_unaligned, output_path)
        print(f"  Saved unaligned valid fragments to {output_path}")
        # Process visualization log even if assembly fails
        if args.visualize_steps_file:
            viz_utils.save_visualization_log(visualization_log, args.visualize_steps_file)
        return
    print(f"  Found {len(pairwise_matches)} potential pairwise matches above threshold.")
    
    # Direct visualization of top N pairwise matches (runtime)
    if args.num_viz_pairwise > 0 and pairwise_matches:
        print(f"  Visualizing top {min(args.num_viz_pairwise, len(pairwise_matches))} pairwise matches (using PCDs for features)...")
        sorted_matches_for_viz = sorted(pairwise_matches, key=lambda x: x['score'], reverse=True)
        for i_viz, match_viz in enumerate(sorted_matches_for_viz):
            if i_viz >= args.num_viz_pairwise:
                break
            s_data = valid_fragments_data[match_viz['source_idx']]
            t_data = valid_fragments_data[match_viz['target_idx']]
            viz_utils.draw_registration_result(
                s_data['pcd_for_features'], t_data['pcd_for_features'], match_viz['transformation'],
                window_name=f"Runtime Pairwise Match {i_viz+1}: {s_data['name']} to {t_data['name']}"
            )

    # 5. Global Assembly
    print("\n[5. Performing Global Assembly]")
    # The Assembler needs the 'original_mesh' from valid_fragments_data for the final assembly
    # It will also use the visualization_log to record its steps.
    assembler = src.assembly.Assembler(valid_fragments_data, pairwise_matches, params, visualization_log)
    reconstructed_model = assembler.greedy_assembly() # This method should now log its choices to visualization_log

    # Log final assembly result (if any)
    if reconstructed_model and reconstructed_model.has_vertices():
        visualization_log.append({
            'step': 'final_assembly_result',
            'type': 'mesh',
            'vertices': np.asarray(reconstructed_model.vertices),
            'triangles': np.asarray(reconstructed_model.triangles),
            'vertex_colors': np.asarray(reconstructed_model.vertex_colors) if reconstructed_model.has_vertex_colors() else None,
            'vertex_normals': np.asarray(reconstructed_model.vertex_normals) if reconstructed_model.has_vertex_normals() else None,
            'placed_fragments_info': [ # Store names and final transforms
                {'name': valid_fragments_data[i]['name'], 
                 'original_index': valid_fragments_data[i]['original_index'], # For coloring in replay
                 'transform': assembler.fragment_transforms[i]}
                for i, placed in enumerate(assembler.is_fragment_placed) if placed
            ]
        })

    # 6. Output
    print("\n[6. Saving Reconstructed Model]")
    if reconstructed_model and reconstructed_model.has_vertices():
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "reconstructed_model.obj")
        src.io_utils.save_mesh(reconstructed_model, output_path)
        print(f"  Reconstructed model saved to: {output_path}")

        if args.visualize_final: # Separate flag for just the final model
            print("  Visualizing final assembly...")
            o3d.visualization.draw_geometries([reconstructed_model], window_name="Final Reconstructed Model")
    else:
        print("  Assembly failed or resulted in an empty model. No output saved.")
        
    # 7. Process/Save Visualization Log
    if args.visualize_steps_file:
        viz_utils.save_visualization_log(visualization_log, args.visualize_steps_file)
        print(f"  Visualization log saved to {args.visualize_steps_file}.")
        print(f"  You can use a separate script/notebook to replay these steps.")
    elif args.visualize_interactively: # New flag for interactive viz (if implemented in viz_utils)
        print("\n[7. Launching Interactive Visualization of Steps]")
        viz_utils.interactive_step_visualization(visualization_log, valid_fragments_data)


    total_time = time.time() - start_time
    print(f"\n--- Reconstruction Finished in {total_time:.2f} seconds ---")


if __name__ == "__main__":
    print("DEBUG: __main__ block entered.")

    parser = argparse.ArgumentParser(description="3D Model Fragment Reconstructor - Advanced")
    parser.add_argument("--input_dir", type=str, default="data/input_fragments",
                        help="Directory containing input fragment files.")
    parser.add_argument("--output_dir", type=str, default="data/output_assembly",
                        help="Directory to save the reconstructed model.")
    parser.add_argument("--config_file", type=str, default="config/reconstruction_params.json",
                        help="Path to the JSON configuration file.")
    parser.add_argument("--visualize_final", action="store_true",
                        help="Enable Open3D visualization of the final assembled model.")
    parser.add_argument("--visualize_steps_file", type=str, default=None, # e.g., "data/viz_log.pkl"
                        help="File path to save the visualization log for offline analysis.")
    parser.add_argument("--visualize_interactively", action="store_true",
                        help="Enable interactive step-by-step visualization (if implemented).")
    # Example for visualizing a few pairwise matches from main (simpler than full interactive)
    parser.add_argument("--num_viz_pairwise", type=int, default=0,
                        help="Number of top pairwise matches to visualize directly during runtime (0 for none).")


    # Ensure default config exists if not specified
    default_config_path = parser.get_default("config_file")
    if not os.path.exists(default_config_path):
        print(f"Warning: Default config file '{default_config_path}' not found. Creating a default one.")
        config_dir = os.path.dirname(default_config_path)
        if config_dir and not os.path.exists(config_dir):
             os.makedirs(config_dir, exist_ok=True)
        
        # Ensure this default config matches the new parameters we need
        dummy_cfg_content = {
            "voxel_downsample_size": 7.0,
            "normal_estimation_radius": 14.0,
            "normal_estimation_max_nn": 30,
            "fpfh_feature_radius": 35.0,
            "fpfh_feature_max_nn": 100,
            "ransac_distance_threshold_factor": 1.5,
            "ransac_edge_length_factor": 0.9,
            "ransac_iterations": 1000000, # Reduced for faster testing initially
            "ransac_n_points": 4,
            "ransac_confidence": 0.999,
            "icp_max_correspondence_distance_factor": 2.0,
            "icp_relative_fitness": 1e-6,
            "icp_relative_rmse": 1e-6,
            "icp_max_iteration": 50,
            "min_match_score": 0.3,
            # Segmentation params
            "min_boundary_edges_for_fracture_face": 1,
            "fracture_surface_dense_sample_points": 10000, # Sample densely first
            "add_preprocessing_noise": True,
            "preprocessing_noise_factor": 0.01,
            "orient_normals_k": 15,
            # Overlap check params
            "max_assembly_overlap_factor_aabb": 0.9, # For the coarse AABB check
            "overlap_check_sample_points": 300,
            "overlap_penetration_allowance_ratio": 0.15,
            "overlap_penetration_depth_factor": 0.25
        }
        with open(default_config_path, 'w') as f:
            json.dump(dummy_cfg_content, f, indent=4)
        print(f"Created default '{default_config_path}'. Please review it for your models.")

    parsed_args = parser.parse_args()

    # Direct pairwise visualization if requested
    if parsed_args.num_viz_pairwise > 0:
        print(f"DEBUG: Will attempt to visualize top {parsed_args.num_viz_pairwise} pairwise matches if found.")
        # This visualization will happen *after* pairwise matching but *before* assembly
        # For this to work cleanly, we'd need to run parts of main logic here, or make main more modular.
        # For now, let's put a placeholder and note that full interactive viz is better.
        # We can add a specific call after the pairwise_matches generation inside main().

    main(parsed_args)