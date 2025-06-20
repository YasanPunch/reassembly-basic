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

# Import both legacy and enhanced modules
import src.alignment
from src.alignment import align_fragments_papaioannou
from src.matching import find_pairwise_matches_enhanced, ConstraintManager
from src.assembly import ConstrainedAssembler

# Import enhanced modules
try:
    ENHANCED_MODULES_AVAILABLE = True
    print("Enhanced Papaioannou modules loaded successfully")
except ImportError as e:
    print(f"Warning: Enhanced modules not available: {e}")
    print("Falling back to legacy modules")
    ENHANCED_MODULES_AVAILABLE = False

def create_enhanced_parameter_set(base_params):
    enhanced_params = base_params.copy()
    enhanced_params.update({
        'use_papaioannou_method': True,
        'papaioannou_resolution': 128,
        'max_rotation_angle': np.pi,
        'max_translation_factor': 0.15,
        'max_acceptable_error': 0.3,
        'esa_max_iter': 1500,
        'esa_n_fail': 4,
        'esa_n_attempted': 80,
        'esa_n_accepted': 8,
        'esa_adaptive_cooling': True,
        'enable_material_constraints': False,
        'enable_fracture_direction_constraints': True,
        'fracture_angle_tolerance': np.pi/18,
        'material_axis_tolerance': np.pi/36,
        'material_bias_weight': 0.2,
        'overlap_bias_weight': 0.15,
        'constraint_satisfaction_bonus': 0.25,
        'papaioannou_method_bonus': 0.1,
        'size_compatibility_bonus': 0.05,
        'enable_global_optimization': True,
        'validate_constraints': True,
        'attempt_constraint_correction': False,
        'enable_volumetric_analysis': True,
        'surface_overlap_threshold': 0.75,
        'max_volumetric_overlap': 0.4,
        'overlap_sample_density': 500,
        'overlap_penetration_tolerance': 0.05,
        'matching_max_workers': 4,
        'min_confidence_score': 0.2,
        'max_matches_per_fragment': 8,
        'auto_detect_constraints': True,
        'encourage_surface_overlap': True,
        'fallback_to_legacy': True,
        'legacy_weight': 0.3
    })
    return enhanced_params

def setup_constraint_manager(fragments_data, params):
    if not ENHANCED_MODULES_AVAILABLE:
        return None
    constraint_manager = ConstraintManager()
    if params.get('auto_detect_constraints', True):
        for frag_data in fragments_data:
            frag_name = frag_data['name']
            if 'fracture_surface_mesh' in frag_data and frag_data['fracture_surface_mesh'] is not None:
                fracture_mesh = frag_data['fracture_surface_mesh']
                if fracture_mesh.has_triangles():
                    fracture_mesh.compute_triangle_normals()
                    triangle_normals = np.asarray(fracture_mesh.triangle_normals)
                    if len(triangle_normals) > 0:
                        avg_normal = np.mean(triangle_normals, axis=0)
                        avg_normal = avg_normal / np.linalg.norm(avg_normal)
                        constraint_manager.add_fracture_direction_constraint(
                            frag_name, [avg_normal, -avg_normal], tolerance=params.get('fracture_angle_tolerance', np.pi/18)
                        )
                        print(f"  Auto-detected fracture direction for {frag_name}: {avg_normal}")
            if 'material_axis' in frag_data:
                constraint_manager.add_material_axis_constraint(
                    frag_name,
                    frag_data['material_axis'],
                    tolerance=params.get('material_axis_tolerance', np.pi/36)
                )
                print(f"  Added material axis constraint for {frag_name}")
    return constraint_manager

def run_enhanced_pipeline(args, params, interactive=False):
    print("\n=== ENHANCED 3D RECONSTRUCTION PIPELINE (Papaioannou Method) ===")
    start_time = time.time()
    visualization_log = []
    print("\n[1. Loading Fragments]")
    fragments_data_raw = src.io_utils.load_fragments_from_directory(args.input_dir)
    if not fragments_data_raw:
        print("No fragments loaded. Exiting.")
        return
    print(f"  Loaded {len(fragments_data_raw)} fragments.")
    for i, frag_info in enumerate(fragments_data_raw):
        mesh_geom = frag_info['mesh']
        visualization_log.append({
            'step': 'initial_fragment',
            'name': frag_info['name'],
            'original_index': frag_info['original_index'],
            'type': 'mesh',
            'vertices': np.asarray(mesh_geom.vertices),
            'triangles': np.asarray(mesh_geom.triangles),
            'vertex_colors': np.asarray(mesh_geom.vertex_colors) if mesh_geom.has_vertex_colors() else None,
            'vertex_normals': np.asarray(mesh_geom.vertex_normals) if mesh_geom.has_vertex_normals() else None,
        })
    print("\n[2. Enhanced Preprocessing and Segmentation]")
    processed_fragments_data = []
    for i, frag_info_raw in enumerate(fragments_data_raw):
        print(f"  Processing fragment: {frag_info_raw['name']} ({i+1}/{len(fragments_data_raw)})")
        pcd_for_features, fracture_surface_mesh = src.preprocessing.preprocess_fragment(
            frag_info_raw, params, viz_collector=visualization_log
        )
        if pcd_for_features is None or not pcd_for_features.has_points():
            print(f"    Warning: Preprocessing failed for {frag_info_raw['name']}. Skipping.")
            continue
        features = None
        if params.get('extract_features_for_legacy', True):
            features, _ = src.feature_extraction.extract_features_from_pcd(pcd_for_features, params)
            processed_fragments_data.append({
                'name': frag_info_raw['name'],
                'original_index': frag_info_raw['original_index'],
                'original_mesh': frag_info_raw['mesh'],
                'fracture_surface_mesh': fracture_surface_mesh,
                'pcd_for_features': pcd_for_features,
                'features': features
            })
    if len(processed_fragments_data) < 2:
        print("Not enough valid fragments for matching. Exiting.")
        return
    print(f"  Successfully processed {len(processed_fragments_data)} fragments.")
    print("\n[3. Setting Up Constraints]")
    constraint_manager = setup_constraint_manager(processed_fragments_data, params)
    if constraint_manager:
        print("  Constraint manager initialized with auto-detected constraints.")
    else:
        print("  No constraints configured.")
    print("\n[4. Enhanced Pairwise Matching (Papaioannou Method)]")
    pairwise_matches = find_pairwise_matches_enhanced(
        processed_fragments_data, params, constraint_manager
    )
    if not pairwise_matches:
        print("  No suitable matches found. Creating composite of unconnected fragments.")
        save_unconnected_fragments(processed_fragments_data, args.output_dir, visualization_log)
        if args.visualize_steps_file:
            viz_utils.save_visualization_log(visualization_log, args.visualize_steps_file)
        return
    print(f"  Found {len(pairwise_matches)} potential matches.")
    log_pairwise_matches(pairwise_matches, processed_fragments_data, visualization_log)
    if interactive:
        interactive_reassembly_loop_enhanced(processed_fragments_data, params, constraint_manager)
        return
    print("\n[5. Enhanced Constrained Assembly]")
    assembler = ConstrainedAssembler(
        processed_fragments_data, pairwise_matches, params, 
        visualization_log, constraint_manager
    )
    reconstructed_model = assembler.assemble_with_constraints()
    print("\n[6. Saving Results]")
    save_results(reconstructed_model, args, params, visualization_log, start_time)
    if args.visualize_final and reconstructed_model:
        print("\n[7. Final Visualization]")
        o3d.visualization.draw_geometries([reconstructed_model], window_name="Enhanced Assembly Result")

def interactive_reassembly_loop_enhanced(fragments_data, params, constraint_manager=None, top_k=5):
    import src.matching
    import src.utils.visualization_utils as viz_utils
    import copy
    import open3d as o3d
    current_fragments = copy.deepcopy(fragments_data)
    step = 1
    while len(current_fragments) > 1:
        print(f"\n[Interactive Reassembly Step {step}] {len(current_fragments)} fragments/composites remaining.")
        matches = find_pairwise_matches_enhanced(current_fragments, params, constraint_manager)
        if not matches:
            print("No more valid matches found. Stopping.")
            break
        matches = sorted(matches, key=lambda x: x['score'], reverse=True)
        print(f"Top {min(top_k, len(matches))} matches:")
        for i, match in enumerate(matches[:top_k]):
            print(f"  [{i+1}] {current_fragments[match['source_idx']]['name']} -> {current_fragments[match['target_idx']]['name']} (Score: {match['score']:.3f}, RMSE: {match['rmse']:.3f})")
            s_data = current_fragments[match['source_idx']]
            t_data = current_fragments[match['target_idx']]
            source_geom = copy.deepcopy(s_data['original_mesh'])
            target_geom = copy.deepcopy(t_data['original_mesh'])
            if not source_geom.has_vertex_normals(): source_geom.compute_vertex_normals()
            if not target_geom.has_vertex_normals(): target_geom.compute_vertex_normals()
            viz_utils.draw_registration_result(
                source_geom, target_geom, match['transformation'],
                window_name=f"Step {step} - Match {i+1}: {s_data['name']} to {t_data['name']} (Score: {match['score']:.3f})"
            )
        selection = input(f"Select a match to merge [1-{min(top_k, len(matches))}] (or 'q' to quit): ").strip()
        if selection.lower() == 'q':
            print("User aborted interactive reassembly.")
            break
        try:
            idx = int(selection) - 1
            if not (0 <= idx < min(top_k, len(matches))):
                print("Invalid selection. Try again.")
                continue
        except ValueError:
            print("Invalid input. Try again.")
            continue
        chosen = matches[idx]
        s_idx, t_idx = chosen['source_idx'], chosen['target_idx']
        s_data = current_fragments[s_idx]
        t_data = current_fragments[t_idx]
        merged_mesh = copy.deepcopy(s_data['original_mesh'])
        merged_mesh.transform(chosen['transformation'])
        composite_mesh = o3d.geometry.TriangleMesh()
        composite_mesh += merged_mesh
        composite_mesh += t_data['original_mesh']
        composite_mesh.compute_vertex_normals()
        composite = {
            'name': f"Composite({s_data['name']}+{t_data['name']})",
            'original_index': -1,
            'original_mesh': composite_mesh,
            'fracture_surface_mesh': None,
            'pcd_for_features': None,
            'features': None
        }
        new_fragments = []
        for i, frag in enumerate(current_fragments):
            if i not in (s_idx, t_idx):
                new_fragments.append(frag)
        new_fragments.append(composite)
        current_fragments = new_fragments
        step += 1
    if len(current_fragments) == 1:
        print("\n[Interactive Reassembly Complete] Final composite:")
        final_mesh = current_fragments[0]['original_mesh']
        o3d.visualization.draw_geometries([final_mesh], window_name="Final Interactive Assembly")
    else:
        print("Interactive reassembly did not complete.")

def log_pairwise_matches(pairwise_matches, fragments_data, visualization_log):
    for idx_match, match in enumerate(pairwise_matches[:10]):
        source_data = fragments_data[match['source_idx']]
        target_data = fragments_data[match['target_idx']]
        log_entry = {
            'step': 'pairwise_match_enhanced',
            'match_index': idx_match,
            'source_name': source_data['name'],
            'target_name': target_data['name'],
            'transformation': match['transformation'],
            'score': match['score'],
            'confidence': match.get('confidence', match['score']),
            'method': match.get('method', 'unknown'),
            'constraints_used': match.get('constraints_used', False)
        }
        visualization_log.append(log_entry)

def save_unconnected_fragments(fragments_data, output_dir, visualization_log):
    os.makedirs(output_dir, exist_ok=True)
    all_meshes = [fd['original_mesh'] for fd in fragments_data]
    combined_unconnected = src.io_utils.combine_meshes(all_meshes)
    output_path = os.path.join(output_dir, "unconnected_fragments.obj")
    src.io_utils.save_mesh(combined_unconnected, output_path)
    print(f"  Saved unconnected fragments to: {output_path}")

def save_results(reconstructed_model, args, params, visualization_log, start_time):
    os.makedirs(args.output_dir, exist_ok=True)
    total_time = time.time() - start_time
    if reconstructed_model and reconstructed_model.has_vertices():
        output_path = os.path.join(args.output_dir, "reconstructed_model_enhanced.obj")
        src.io_utils.save_mesh(reconstructed_model, output_path)
        print(f"  Enhanced reconstruction saved to: {output_path}")
        metadata = {
            'method': 'Papaioannou Enhanced' if params.get('use_papaioannou_method') else 'Legacy',
            'total_time_seconds': total_time,
            'num_vertices': len(reconstructed_model.vertices),
            'num_triangles': len(reconstructed_model.triangles),
            'parameters_used': {k: v for k, v in params.items() if isinstance(v, (int, float, bool, str))},
            'enhanced_features_used': ENHANCED_MODULES_AVAILABLE
        }
        metadata_path = os.path.join(args.output_dir, "reconstruction_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        if visualization_log is not None:
            visualization_log.append({
                    'step': 'final_assembly_result_enhanced',
                'type': 'mesh',
                'vertices': np.asarray(reconstructed_model.vertices),
                'triangles': np.asarray(reconstructed_model.triangles),
                    'total_processing_time': total_time,
                    'method_used': 'Papaioannou Enhanced' if params.get('use_papaioannou_method') else 'Legacy'
                })
    else:
        print("  Reconstruction failed - no output saved.")
    if args.visualize_steps_file and visualization_log:
        viz_utils.save_visualization_log(visualization_log, args.visualize_steps_file)
        print(f"  Visualization log saved to: {args.visualize_steps_file}")
    print(f"\n--- Total Processing Time: {total_time:.2f} seconds ---")

def main(args):
    print("DEBUG: Enhanced main(args) function entered.")
    print("--- 3D Model Fragment Reconstructor (Enhanced with Papaioannou Method) ---")
    print("\n[Loading Configuration]")
    try:
        with open(args.config_file, 'r') as f:
            base_params = json.load(f)
        print(f"  Base parameters loaded from: {args.config_file}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config: {e}. Using defaults.")
        base_params = {}
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}. Exiting.")
        return
    method = getattr(args, 'method', 'enhanced').lower() if hasattr(args, 'method') else 'enhanced'
    if getattr(args, 'interactive_reassembly', False):
        enhanced_params = create_enhanced_parameter_set(base_params)
        run_enhanced_pipeline(args, enhanced_params, interactive=True)
        return
    if method == 'enhanced' and ENHANCED_MODULES_AVAILABLE:
        enhanced_params = create_enhanced_parameter_set(base_params)
        run_enhanced_pipeline(args, enhanced_params)
    else:
        print("Enhanced modules not available or unknown method. Exiting.")
        return

if __name__ == "__main__":
    print("DEBUG: Enhanced __main__ block entered.")
    parser = argparse.ArgumentParser(description="3D Model Fragment Reconstructor - Enhanced with Papaioannou Method")
    parser.add_argument("--input_dir", type=str, default="data/input_fragments",
                        help="Directory containing input fragment files.")
    parser.add_argument("--output_dir", type=str, default="data/output_assembly_enhanced",
                        help="Directory to save the reconstructed model.")
    parser.add_argument("--config_file", type=str, default="config/reconstruction_params.json",
                        help="Path to the JSON configuration file.")
    parser.add_argument("--method", type=str, choices=['enhanced', 'legacy', 'hybrid'], default='enhanced',
                        help="Reconstruction method: 'enhanced' (Papaioannou), 'legacy' (FPFH+RANSAC+ICP), or 'hybrid'")
    parser.add_argument("--visualize_final", action="store_true",
                        help="Enable Open3D visualization of the final assembled model.")
    parser.add_argument("--visualize_steps_file", type=str, default=None,
                        help="File path to save the visualization log (e.g., 'data/enhanced_viz_log.pkl')")
    parser.add_argument("--visualize_interactively", action="store_true",
                        help="Enable interactive step-by-step visualization.")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Z-buffer resolution for Papaioannou method")
    parser.add_argument("--enable_constraints", action="store_true",
                        help="Enable automatic constraint detection and application")
    parser.add_argument("--max_iterations", type=int, default=1500,
                        help="Maximum iterations for Enhanced Simulated Annealing")
    parser.add_argument("--num_viz_pairwise", type=int, default=0,
                        help="Number of top pairwise matches to visualize (legacy compatibility)")
    parser.add_argument('--interactive_reassembly', action='store_true', help='Run interactive reassembly loop with enhanced methods')
    default_config_path = "config/reconstruction_params.json"
    if not os.path.exists(default_config_path):
        print(f"Creating default enhanced config at '{default_config_path}'")
        config_dir = os.path.dirname(default_config_path)
        if config_dir and not os.path.exists(config_dir):
             os.makedirs(config_dir, exist_ok=True)
        enhanced_default_config = {
            "// --- Enhanced Papaioannou Method Parameters ---": None,
            "use_papaioannou_method": True,
            "papaioannou_resolution": 128,
            "max_rotation_angle_rad": 3.14159,
            "max_translation_factor": 0.15,
            "max_acceptable_error": 0.3,
            "esa_max_iter": 1500,
            "enable_global_optimization": True,
            "enable_constraint_detection": True,
            "surface_overlap_threshold": 0.75,
            "// --- Legacy FPFH Parameters (for fallback) ---": None,
            "voxel_downsample_size": 3.0,
            "normal_estimation_radius": 14.0,
            "normal_estimation_max_nn": 30,
            "fpfh_feature_radius": 35.0,
            "fpfh_feature_max_nn": 100,
            "ransac_distance_threshold_factor": 1.5,
            "ransac_iterations": 1000000,
            "icp_max_correspondence_distance_factor": 2.0,
            "min_match_score": 0.3,
            "// --- Segmentation Parameters ---": None,
            "max_angle_deviation_deg": 30.0,
            "min_region_area_percentage": 2.0,
            "visualize_segmentation": False,
            "fracture_surface_dense_sample_points": 10000,
            "add_preprocessing_noise": True,
            "preprocessing_noise_factor": 0.01,
            "// --- Assembly Parameters ---": None,
            "max_assembly_overlap_factor_aabb": 0.8,
            "overlap_check_sample_points": 500,
            "overlap_penetration_allowance_ratio": 0.15,
            "overlap_penetration_depth_factor": 0.25
        }
        with open(default_config_path, 'w') as f:
            json.dump(enhanced_default_config, f, indent=4)
        print(f"Created enhanced default config. Review and modify as needed.")
    parsed_args = parser.parse_args()
    main(parsed_args)