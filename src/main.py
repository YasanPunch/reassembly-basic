import os
import json
import argparse
import open3d as o3d
import time
import numpy as np
import copy
import random

np.random.seed(42)
random.seed(42)

import src.io_utils
import src.preprocessing
import src.matching
import src.assembly


def main(args):
    print("DEBUG: main(args) function entered.")
    print("--- 3D Model Fragment Reconstructor (Advanced) ---")
    start_time = time.time()

    # 1. Load Parameters
    print("\n[1. Loading Parameters]")
    try:
        with open(args.config_file, "r") as f:
            params = json.load(f)
        print(f"  Parameters loaded from: {args.config_file}")

        # Override snapping parameter if command-line argument is provided
        if args.disable_snapping:
            params["enable_post_processing_snapping"] = False
            print("  Post-processing snapping disabled via command-line argument")

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
    # This will store more processed data:
    # {'name', 'original_index', 'original_mesh',
    #  'fracture_surface_mesh' (optional), 'pcd_for_features', 'features'}
    processed_fragments_pipeline_data = []

    # 3. Preprocessing, Segmentation & Feature Extraction (per fragment)
    print("\n[3. Preprocessing, Segmentation, and Feature Extraction]")
    for i, frag_info_raw in enumerate(fragments_data_raw):
        print(
            f"  Processing fragment: {frag_info_raw['name']} ({i+1}/{len(fragments_data_raw)})"
        )

        # Preprocessing now returns lists: (pcds_for_features_list, features_list, fracture_surfaces)
        pcds_for_features_list, features_list, fracture_surfaces = (
            src.preprocessing.preprocess_fragment(frag_info_raw, params)
        )

        # If no valid surfaces, store empty lists and continue
        if not pcds_for_features_list or all(
            pcd is None or not pcd.has_points() for pcd in pcds_for_features_list
        ):
            print(
                f"    Warning: Preprocessing resulted in no valid point clouds for features for {frag_info_raw['name']}. Skipping."
            )
            processed_fragments_pipeline_data.append(
                {
                    "name": frag_info_raw["name"],
                    "original_index": frag_info_raw["original_index"],
                    "original_mesh": frag_info_raw["mesh"],
                    "fracture_surfaces": fracture_surfaces,
                    "pcds_for_features": [],
                    "features_list": [],
                }
            )
            continue

        # Store lists for each fragment
        processed_fragments_pipeline_data.append(
            {
                "name": frag_info_raw["name"],
                "original_index": frag_info_raw["original_index"],
                "original_mesh": frag_info_raw["mesh"],
                "fracture_surfaces": fracture_surfaces,
                "pcds_for_features": pcds_for_features_list,
                "features_list": features_list,
            }
        )

    # Filter out fragments that failed feature extraction (essential for matching)
    valid_fragments_data = [
        fd
        for fd in processed_fragments_pipeline_data
        if fd.get("features_list")
        and any(f is not None and f.num() > 0 for f in fd["features_list"])
    ]
    if len(valid_fragments_data) < len(processed_fragments_pipeline_data):
        print(
            f"  Warning: {len(processed_fragments_pipeline_data) - len(valid_fragments_data)} fragments had no valid features and were excluded from matching."
        )

    if len(valid_fragments_data) < 2:  # Need at least 2 fragments for pairwise matching
        print(
            "Not enough valid fragments with features for pairwise matching. Exiting or saving unaligned."
        )
        # Save unaligned original meshes if any loaded
        if fragments_data_raw:
            os.makedirs(args.output_dir, exist_ok=True)
            all_original_meshes = [fd["mesh"] for fd in fragments_data_raw]
            combined_unaligned = src.io_utils.combine_meshes(all_original_meshes)
            output_path = os.path.join(
                args.output_dir, "reconstructed_model_unaligned_originals.obj"
            )
            src.io_utils.save_mesh(combined_unaligned, output_path)
            print(f"  Saved all original unaligned fragments to {output_path}")

        return

    # 4. Pairwise Matching
    print("\n[4. Finding Pairwise Matches]")
    # pairwise_matches will be list of dicts. Indices refer to `valid_fragments_data`
    pairwise_matches = src.matching.find_pairwise_matches(
        valid_fragments_data,
        params,
        debug=args.debug_pairwise_matching,
        top_n_per_pair=args.top_n_matches_per_pair,
    )

    if not pairwise_matches:
        print(
            "  No suitable pairwise matches found above threshold. Attempting to save fragments separately."
        )
        os.makedirs(args.output_dir, exist_ok=True)
        all_original_meshes = [
            fd["original_mesh"] for fd in valid_fragments_data
        ]  # Use original meshes
        combined_unaligned = src.io_utils.combine_meshes(all_original_meshes)
        output_path = os.path.join(
            args.output_dir, "reconstructed_model_no_matches.obj"
        )
        src.io_utils.save_mesh(combined_unaligned, output_path)
        print(f"  Saved unaligned valid fragments to {output_path}")
        return
    print(
        f"  Found {len(pairwise_matches)} potential pairwise matches above threshold."
    )

    # Direct visualization of all pairwise matches (runtime)
    if args.num_viz_pairwise > 0 and pairwise_matches:
        print(
            f"  Visualizing all {len(pairwise_matches)} pairwise matches (using original meshes)..."
        )
        sorted_matches_for_viz = sorted(
            pairwise_matches, key=lambda x: x["score"], reverse=True
        )
        for i_viz, match_viz in enumerate(sorted_matches_for_viz):
            s_data = valid_fragments_data[match_viz["source_idx"]]
            t_data = valid_fragments_data[match_viz["target_idx"]]

            # Use original meshes instead of pcd_for_features
            source_geom_to_viz = copy.deepcopy(s_data["original_mesh"])
            target_geom_to_viz = copy.deepcopy(t_data["original_mesh"])

            # Ensure they have normals for consistent display if not already computed
            if not source_geom_to_viz.has_vertex_normals():
                source_geom_to_viz.compute_vertex_normals()
            if not target_geom_to_viz.has_vertex_normals():
                target_geom_to_viz.compute_vertex_normals()

    # LIMIT DEBUG VISUALIZATION TO TOP N PAIRWISE MATCHES (CHANGE N HERE IF NEEDED)
    DEBUG_TOP_N_MATCHES = 5
    if args.debug_pairwise_matching and pairwise_matches:
        pairwise_matches = pairwise_matches[:DEBUG_TOP_N_MATCHES]

    # 5. Global Assembly
    print("\n[5. Performing Global Assembly]")
    # The Assembler needs the 'original_mesh' from valid_fragments_data for the final assembly
    assembler = src.assembly.Assembler(valid_fragments_data, pairwise_matches, params)
    reconstructed_model = assembler.greedy_assembly()

    if reconstructed_model and reconstructed_model.has_vertices():
        if args.visualize_final:
            print("  Visualizing final composite assembly...")
            o3d.visualization.draw_geometries(
                [reconstructed_model], window_name="Final Composite Assembly"
            )
    else:
        print("  Assembly failed or resulted in an empty model. No output saved.")

    total_time = time.time() - start_time
    print(f"\n--- Reconstruction Finished in {total_time:.2f} seconds ---")


if __name__ == "__main__":
    print("DEBUG: __main__ block entered.")

    parser = argparse.ArgumentParser(
        description="3D Model Fragment Reconstructor - Advanced"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/input_fragments",
        help="Directory containing input fragment files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output_assembly",
        help="Directory to save the reconstructed model.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/reconstruction_params.json",
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--visualize_final",
        action="store_true",
        help="Enable Open3D visualization of the final assembled model.",
    )
    parser.add_argument(
        "--num_viz_pairwise",
        type=int,
        default=0,
        help="Number of top pairwise matches to visualize directly during runtime (0 for none).",
    )
    parser.add_argument(
        "--visualize_segmentation",
        action="store_true",
        help="Enable visualization of segmentation results for each fragment.",
    )
    parser.add_argument(
        "--debug_pairwise_matching",
        action="store_true",
        help="Enable debug visualization for pairwise matching.",
    )
    parser.add_argument(
        "--top_n_matches_per_pair",
        type=int,
        default=3,
        help="Number of top matches to keep per fragment pair (default: 3)",
    )
    parser.add_argument(
        "--disable_snapping",
        action="store_true",
        help="Disable post-processing snapping step (useful when snapping messes up correctly aligned fragments)",
    )

    parsed_args = parser.parse_args()

    if parsed_args.num_viz_pairwise > 0:
        print(
            f"DEBUG: Will attempt to visualize top {parsed_args.num_viz_pairwise} pairwise matches if found."
        )

    main(parsed_args)
