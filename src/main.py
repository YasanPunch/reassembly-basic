import os
import json
import argparse
import open3d as o3d

print("o3d-version: ", o3d.__version__)
import time
import numpy as np

# checker for missing dependencies
import importlib.util
import sys

print("--- Python Path (sys.path) ---")
for p_path in sys.path:
    print(p_path)
print("--------------------------------")

spec_preprocessing = importlib.util.find_spec("preprocessing")
if spec_preprocessing:
    print(f"DEBUG: Python found 'preprocessing' at: {spec_preprocessing.origin}")
else:
    print(
        "DEBUG: Python could NOT find 'src.preprocessing' via importlib.util.find_spec"
    )
# checker end

import io_utils
import preprocessing
import feature_extraction
import matching
import assembly
import utils.visualization_utils  # Optional visualization


def main(args):
    print("DEBUG: main(args) function entered.")
    print("--- 3D Model Fragment Reconstructor ---")
    start_time = time.time()

    # 1. Load Parameters
    print("\n[1. Loading Parameters]")
    try:
        with open(args.config_file, "r") as f:
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

    fragments_data = io_utils.load_fragments_from_directory(args.input_dir)  # MODIFIED
    print(f"DEBUG main: Number of fragments loaded: {len(fragments_data)}")
    if not fragments_data:
        print("No fragments loaded. Exiting.")
        return
    print(f"  Loaded {len(fragments_data)} fragments.")

    processed_fragments_pipeline_data = []

    # 3. Preprocessing & Feature Extraction (per fragment)
    print("\n[3. Preprocessing and Feature Extraction]")
    for i, frag_info in enumerate(fragments_data):
        print(
            f"  Processing fragment: {frag_info['name']} ({i + 1}/{len(fragments_data)})"
        )

        pcd = preprocessing.preprocess_fragment(frag_info["mesh"], params)  # MODIFIED
        o3d.visualization.draw_geometries([pcd])
        if not pcd.has_points():
            print(
                f"    Warning: Preprocessing resulted in empty point cloud for {frag_info['name']}. Skipping."
            )
            # Add placeholder to maintain indexing if necessary, or handle missing data downstream
            processed_fragments_pipeline_data.append(
                {**frag_info, "pcd": None, "features": None, "pcd_for_features": None}
            )
            continue
        # print(f"    Preprocessed into {len(pcd.points)} points.")

        # Feature Extraction (e.g., FPFH)
        # Returns (o3d.pipelines.registration.Feature, o3d.geometry.PointCloud used for features)
        # features, pcd_for_features = extract_features_from_pcd(pcd, params) # Original
        features, pcd_for_features = feature_extraction.extract_features_from_pcd(
            pcd, params
        )  # MODIFIED
        if (
            features is None
            or pcd_for_features is None
            or not pcd_for_features.has_points()
            or features.num() == 0
        ):
            print(
                f"    Warning: Feature extraction failed or yielded empty features for {frag_info['name']}. Skipping."
            )
            processed_fragments_pipeline_data.append(
                {**frag_info, "pcd": pcd, "features": None, "pcd_for_features": None}
            )
            continue
        # print(f"    Extracted {features.num()} FPFH features.")

        processed_fragments_pipeline_data.append(
            {
                **frag_info,
                "pcd": pcd,
                "features": features,
                "pcd_for_features": pcd_for_features,
            }
        )

    valid_fragments_data = [
        fd for fd in processed_fragments_pipeline_data if fd["features"] is not None
    ]
    if len(valid_fragments_data) < len(processed_fragments_pipeline_data):
        print(
            f"  Warning: {len(processed_fragments_pipeline_data) - len(valid_fragments_data)} fragments failed processing and were excluded."
        )

    if (
        len(valid_fragments_data) < 1
    ):  # Need at least 1 fragment, ideally 2 for matching
        print("Not enough valid fragments processed for assembly. Exiting.")
        return
    if len(valid_fragments_data) == 1:
        print("Only one valid fragment. 'Assembly' will be this single fragment.")
        output_path = os.path.join(args.output_dir, "reconstructed_model.obj")
        io_utils.save_mesh(valid_fragments_data[0]["mesh"], output_path)  # MODIFIED
        total_time = time.time() - start_time
        print(f"\n--- Reconstruction Finished in {total_time:.2f} seconds ---")
        return

    # 4. Pairwise Matching
    print("\n[4. Finding Pairwise Matches]")
    # pairwise_matches is a list of dicts:
    # {'source_idx': int, 'target_idx': int, 'transformation': np.ndarray, 'score': float}
    # Indices here refer to the order in `valid_fragments_data`
    pairwise_matches = matching.find_pairwise_matches(
        valid_fragments_data, params
    )  # MODIFIED
    if not pairwise_matches:
        print(
            "  No suitable pairwise matches found. Attempting to save fragments separately or as a scene."
        )
        # As a fallback, combine original meshes without transformation (they'll be at origin)
        # Or save them to an 'unassembled' subfolder
        os.makedirs(args.output_dir, exist_ok=True)
        combined_unaligned = io_utils.combine_meshes(
            [fd["mesh"] for fd in valid_fragments_data]
        )  # MODIFIED
        output_path = os.path.join(args.output_dir, "reconstructed_model_unaligned.obj")
        io_utils.save_mesh(combined_unaligned, output_path)  # MODIFIED
        print(f"  Saved unaligned fragments to {output_path}")
        total_time = time.time() - start_time
        print(
            f"\n--- Reconstruction Finished (No Assembly) in {total_time:.2f} seconds ---"
        )
        return
    print(f"  Found {len(pairwise_matches)} potential pairwise matches.")

    # (Optional) Visualize some good pairwise matches
    if args.visualize and pairwise_matches:
        # visualize_steps('pairwise_matches', valid_fragments_data, pairwise_matches, num_to_show=min(3, len(pairwise_matches)))
        pass  # Visualization function needs to be more fleshed out for this

    # 5. Global Assembly
    print("\n[5. Performing Global Assembly]")
    # The Assembler will use the 'mesh' (original full-res) from valid_fragments_data for the final assembly
    # but uses the pairwise_matches (based on PCDs) to guide the assembly.
    assembler = assembly.Assembler(
        valid_fragments_data, pairwise_matches, params
    )  # MODIFIED
    reconstructed_model = assembler.greedy_assembly()

    # 6. Output
    print("\n[6. Saving Reconstructed Model]")
    if reconstructed_model and reconstructed_model.has_vertices():
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "reconstructed_model.obj")
        io_utils.save_mesh(reconstructed_model, output_path)  # MODIFIED
        print(f"  Reconstructed model saved to: {output_path}")

        if args.visualize:
            print("  Visualizing final assembly...")
            o3d.visualization.draw_geometries(
                [reconstructed_model], window_name="Final Reconstructed Model"
            )
    else:
        print("  Assembly failed or resulted in an empty model. No output saved.")
        # Could save the best attempt or individual transformed pieces if needed.

    total_time = time.time() - start_time
    print(f"\n--- Reconstruction Finished in {total_time:.2f} seconds ---")


if __name__ == "__main__":
    # DEBUG print to confirm entry
    print("DEBUG: __main__ block entered.")  # You can keep or remove this later

    parser = argparse.ArgumentParser(description="3D Model Fragment Reconstructor")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../data/input_fragments",
        help="Directory containing input fragment files (.obj, .stl, .ply).",
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
        "--visualize",
        action="store_true",
        help="Enable visualization of steps (requires Open3D GUI).",
    )

    # --- REMOVE DUMMY DATA AND CONFIG CREATION ---
    # # Create dummy data and config if they don't exist for a quick test run
    # if not os.path.exists("data/input_fragments"):
    #     os.makedirs("data/input_fragments", exist_ok=True)
    #     print("Created dummy 'data/input_fragments' directory.") # Or print warning if not found

    # Check if config file exists, if not, maybe create a default or warn
    if not os.path.exists(
        parser.get_default("config_file")
    ):  # Check default or provided
        # Option 1: Exit with error
        # print(f"Error: Config file '{parser.get_default('config_file')}' not found. Please create one or specify a valid path.")
        # sys.exit(1) # Requires import sys

        # Option 2: Create a default one (as before, but now it's more explicit)
        print(
            f"Warning: Config file '{parser.get_default('config_file')}' not found. Creating a default one."
        )
        if not os.path.exists(os.path.dirname(parser.get_default("config_file"))):
            os.makedirs(
                os.path.dirname(parser.get_default("config_file")), exist_ok=True
            )
        dummy_cfg_content = {
            "voxel_downsample_size": 10.0,  # Adjust this based on your Tombstone model scale
            "normal_estimation_radius": 14.0,
            "normal_estimation_max_nn": 30,
            "fpfh_feature_radius": 35.0,
            "fpfh_feature_max_nn": 100,
            "ransac_distance_threshold_factor": 1.5,
            "ransac_edge_length_factor": 0.9,
            "ransac_iterations": 4000000,
            "ransac_n_points": 4,
            "ransac_confidence": 0.909,
            "icp_max_correspondence_distance_factor": 2.0,
            "icp_relative_fitness": 1e-6,
            "icp_relative_rmse": 1e-6,
            "icp_max_iteration": 50,
            "min_match_score": 0.3,  # This might need tuning for your data
            "max_assembly_overlap_factor": 0.6,
        }
        with open(parser.get_default("config_file"), "w") as f:
            json.dump(dummy_cfg_content, f, indent=4)
        print(
            f"Created default '{parser.get_default('config_file')}'. Please review it for your models."
        )

    # --- DUMMY FRAGMENT CREATION REMOVED ---
    # # Example: Create two simple cube fragments for testing
    # # You would replace these with actual fragmented model pieces
    # cube_part1_obj = """ ... """
    # cube_part2_obj = """ ... """
    # # For the test, let's ensure part2 is slightly transformed away from part1
    # # We will place part1.obj and part2_transformed.obj in the input directory
    # if not os.path.exists("data/input_fragments/part1.obj"):
    #     # ... (removed)
    # if not os.path.exists("data/input_fragments/part2_tf.obj"):
    #     # ... (removed)
    # --- END OF REMOVALS ---

    parsed_args = parser.parse_args()
    main(parsed_args)
