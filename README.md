# 3D Model Fragment Reconstructor (Advanced)

This project provides an advanced application for analyzing fragments of a 3D model and attempting to reconstruct the original object. It leverages the Open3D library for point cloud processing, mesh operations, feature extraction (FPFH), and registration (RANSAC, ICP). A key feature is its sophisticated segmentation pipeline to identify fracture surfaces, coupled with detailed step logging for visualization and debugging.

## Features

*   Loads 3D model fragments from various file formats (OBJ, STL, PLY, etc.).
*   **Advanced Segmentation:**
    *   Identifies potential fracture surfaces on each fragment using normal-based clustering and geometric analysis (roughness, planarity).
    *   Supports **interactive visual selection** of fracture surfaces via the Open3D GUI if `visualize_segmentation` is enabled in parameters.
    *   Refines segments recursively based on connectivity, coherency, and size.
*   **Targeted Preprocessing:**
    *   Generates a point cloud (`pcd_for_features`) by densely sampling the identified fracture surface (or the whole mesh as a fallback).
    *   Downsamples this point cloud and estimates normals, preparing it for feature extraction.
*   Extracts FPFH (Fast Point Feature Histograms) features from the `pcd_for_features`.
*   Performs pairwise registration between fragments using RANSAC (global) and ICP (local refinement) based on feature matching on these specialized point clouds.
*   Implements a greedy assembly strategy to combine original fragment meshes based on the best pairwise alignments found.
*   Includes overlap checks (AABB and point-sampling based) to prevent gross interpenetration during assembly.
*   Saves the reconstructed model.
*   **Visualization & Debugging:**
    *   Logs detailed intermediate steps of the reconstruction pipeline (initial fragments, segmentation results, point clouds, pairwise matches, assembly steps) to a `.pkl` file.
    *   Includes a `replay_log.py` script to visualize these logged steps sequentially using Open3D.
    *   Option to visualize the final assembly and top pairwise matches at runtime.
*   Highly configurable parameters for all stages of the pipeline via a JSON configuration file.

## Project Structure
my-3d-reassembly/
├── data/
│ ├── input_fragments/ # Input fragment files (e.g., piece_01.obj)
│ └── output_assembly/ # Output reconstructed model(s)
├── src/ # Source code
│ ├── main.py # Main script to run the reconstruction
│ ├── io_utils.py # Loading/saving 3D models and meshes
│ ├── preprocessing.py # Fragment preparation (calls segmentation)
│ ├── segmentation.py # Fracture surface identification and refinement
│ ├── feature_extraction.py # FPFH feature extraction
│ ├── matching.py # Finding correspondences between fragments
│ ├── alignment.py # Aligning pairs of fragments (RANSAC, ICP)
│ ├── assembly.py # Global assembly strategy and overlap check
│ └── utils/
│ ├── geometry_utils.py # Geometric helper functions
│ └── visualization_utils.py # Visualization helpers, log saving/loading
├── config/
│ ├── reconstruction_params.json # Main algorithm parameters
│ ├── params_for_cube.json # Example parameters for a specific model type
│ └── param-info.txt # (Generated) Information about each parameter
├── docs/
│ └── design_notes.md # (Optional) Design notes
├── tests/ # Unit tests
├── replay_log.py # Script to replay visualization logs
├── .gitignore
├── requirements.txt # Python dependencies
└── README.md # This file

## Prerequisites

*   Python 3.8+
*   Libraries listed in `requirements.txt`. Ensure you have a C++ compiler for some dependencies.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd my-3d-reassembly
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Open3D installation can sometimes be tricky. If you encounter issues, refer to the [official Open3D installation guide](http://www.open3d.org/docs/latest/getting_started.html).*

## Usage

### 1. Running the Reconstruction

1.  **Place your 3D model fragments** (e.g., `.obj`, `.stl` files) into a directory, for example, `data/input_fragments/`.
2.  **Configure parameters:**
    *   Copy `config/reconstruction_params.json` or `config/params_for_cube.json` and modify it, or edit directly.
    *   Refer to `config/param-info.txt` for detailed explanations of each parameter.
    *   **Crucial parameters** to check first: `voxel_downsample_size` (relative to your model's scale), segmentation parameters (`normal_cluster_eps`, `normal_cluster_min_samples`, `roughness_threshold`, `refinement_min_final_segment_size`), and `min_match_score`.
    *   To enable interactive segmentation visualization and selection, set `"visualize_segmentation": true` in your JSON config.
3.  **Run the main application:**
    ```bash
    python src/main.py --input_dir path/to/your/fragments --output_dir path/to/your/output --config_file path/to/your/config.json
    ```
    Or using module execution (useful if you have Python path issues):
    ```bash
    python -m src.main --input_dir path/to/your/fragments --config_file path/to/your/config.json
    ```

    **Command-line arguments for `main.py`:**
    *   `--input_dir`: Directory containing input fragments (default: `data/input_fragments`).
    *   `--output_dir`: Directory where the reconstructed model will be saved (default: `data/output_assembly`).
    *   `--config_file`: Path to the parameters JSON file (default: `config/reconstruction_params.json`).
    *   `--visualize_final`: If present, enables Open3D visualization of the final assembled model at the end of the run.
    *   `--visualize_steps_file path/to/log.pkl`: If provided, saves a detailed log of visualization steps to the specified `.pkl` file. This file can then be replayed. Example: `data/my_run_log.pkl`.
    *   `--num_viz_pairwise N`: Visualize the top `N` pairwise alignments (using feature point clouds) directly during runtime (0 for none).
    *   `--visualize_interactively`: (Currently a placeholder) Intended for a fully interactive step-by-step visualization within a single window.

    **Interactive Segmentation:** If `"visualize_segmentation": true` is set in the config file, for each fragment, an Open3D window will appear.
    *   Use number keys (1-9, 0 for 10th) to select/deselect segments on the current page. Selected segments turn black.
    *   Use 'N'/'P' to navigate pages if there are more than 10 segments.
    *   Press 'S' to confirm your selection of fracture surfaces for the current fragment and proceed.
    *   Press 'Q' to skip selection for the current fragment (no fracture surfaces will be explicitly chosen by you).

4.  **Check the output:**
    The reconstructed model will be saved in the specified output directory (e.g., `data/output_assembly/reconstructed_model.obj`).
    If `--visualize_steps_file` was used, a `.pkl` log file will also be present.

### 2. Replaying a Visualization Log

If you generated a visualization log (e.g., `my_run_log.pkl`) using `main.py`:

1.  **Run the replay script:**
    ```bash
    python replay_log.py
    ```
    *   By default, it looks for `data/my_run_log.pkl`. You can modify the `log_file` variable in `replay_log.py` to point to a different file.
    *   It also tries to load fragment names from `data/input_fragments` to apply consistent coloring. Adjust `input_frags_dir` in the script if your fragments were elsewhere.
2.  **Navigate:** Press 'Q' (or close the window) in each Open3D window to proceed to the next logged step.

## How It Works (High-Level)

1.  **Load Fragments & Parameters:** Input 3D fragment meshes are loaded. Algorithm parameters are read from the JSON config.
2.  **Per-Fragment Processing (Preprocessing, Segmentation, Feature PCD Generation):**
    *   **Segmentation:** For each fragment, the system attempts to identify "fracture surfaces" – the broken areas that should mate with other fragments.
        *   Faces are initially clustered based on normal similarity (DBSCAN).
        *   These clusters are then refined recursively based on connectivity, geometric coherency (PCA of normals, planarity tests), and size. Badness scores are used to guide splitting.
        *   If `visualize_segmentation` is true in parameters, the user is prompted to interactively select the final fracture surface segments from the candidates.
        *   A `fracture_surface_mesh` is extracted. If segmentation fails or is skipped, the whole original mesh might be used as a fallback.
    *   **Point Cloud Generation for Features:** A dense point cloud (`pcd_for_features`) is sampled specifically from this identified `fracture_surface_mesh`.
    *   **Preprocessing `pcd_for_features`:** This point cloud is voxel-downsampled, and normals are estimated. Optional noise can be added.
3.  **Feature Extraction:** FPFH (Fast Point Feature Histogram) descriptors are computed for points in each `pcd_for_features`. These descriptors capture local geometric properties of the fracture surfaces.
4.  **Pairwise Matching & Alignment:**
    *   For each unique pair of fragments, their `pcd_for_features` and corresponding FPFH descriptors are used for matching.
    *   **RANSAC:** A RANSAC-based algorithm finds an initial coarse alignment (transformation matrix) based on FPFH correspondences.
    *   **ICP (Iterative Closest Point):** The coarse alignment is refined using point-to-plane ICP to achieve a precise fit between the two `pcd_for_features`.
    *   A `score` (fitness from ICP) and RMSE are assigned to each potential pairwise alignment. Matches below `min_match_score` are discarded.
5.  **Global Assembly:**
    *   A greedy approach is used:
        *   Start with a seed fragment (typically part of the best-scoring pairwise match).
        *   Iteratively find the unplaced fragment that best aligns (highest score from pairwise matches, acceptable overlap) to any part of the currently assembled structure.
        *   The transformation found from aligning `pcd_for_features` is applied to the *original full mesh* of the fragment.
        *   **Overlap Check:** Before adding a fragment, it's checked against already placed components to prevent significant interpenetration, first using AABB intersection volume, then by sampling points on one mesh and checking their signed distance to the other.
        *   Transform and add the original mesh of this fragment to the assembly.
        *   Repeat until no more fragments can be added or all are placed.
6.  **Output & Logging:**
    *   The final assembled model, composed of the original fragment meshes transformed into their globally aligned positions, is saved.
    *   If enabled, a detailed visualization log is saved, capturing geometries and transformations at various pipeline stages.

## Limitations

*   **Greedy Assembly:** The greedy approach may not find the globally optimal assembly and can get stuck in local minima. Error propagation is a risk.
*   **Sensitivity to Parameters:** Performance heavily depends on parameters in `reconstruction_params.json`, especially for segmentation and matching. These often need careful tuning for specific datasets (scale, fragment condition, material).
*   **Segmentation Quality:** The success of the feature-based matching heavily relies on accurately identifying and isolating the true fracture surfaces. Poor segmentation leads to poor features and matches.
*   **Requires Good Overlap & Features:** Assumes fragments have sufficient overlapping areas on their fracture surfaces with distinct geometric features for FPFH and ICP to work well.
*   **No Handling of Symmetric/Ambiguous Parts:** May struggle with highly symmetric objects or pieces that could fit in multiple valid ways.
*   **Scalability:** Combinatorial complexity of pairwise matching (N*(N-1)) can be slow for many fragments. RANSAC can also be computationally intensive.
*   **Mesh Quality:** Input meshes should ideally be reasonably clean. While segmentation attempts to isolate good surfaces, extreme noise or non-manifold geometry can cause issues.
*   **Overlap Check:** The current overlap check is heuristic and might not catch all interpenetrations perfectly or could be overly restrictive in some cases.
*   **Interactive Steps:** Interactive segmentation, while powerful, can be time-consuming for datasets with many fragments.

## Potential Future Enhancements

*   **Improved Global Assembly:**
    *   Graph-based assembly (e.g., finding a consistent cycle/path in a compatibility graph of pairwise alignments).
    *   Pose graph optimization to globally refine all transformations simultaneously after initial greedy assembly.
*   **More Robust Feature Descriptors:** Explore other local or global shape descriptors, possibly learning-based.
*   **Advanced Segmentation:**
    *   Automated selection of fracture surfaces using learned models or more advanced heuristics.
    *   Better handling of noisy input meshes during segmentation.
*   **Machine Learning Integration:**
    *   Learning feature matchers or pose predictors.
    *   Reinforcement learning for assembly sequencing.
*   **Improved Overlap/Collision Detection:** More sophisticated methods (e.g., using libraries like FCL - Flexible Collision Library).
*   **User Interaction Refinements:** Allow manual correction of pairwise alignments or interactive adjustment during assembly.
*   **Parallelization:** Speed up pairwise matching, feature extraction, and other computationally intensive steps.
*   **Support for Color/Texture:** Use visual information to aid matching and alignment if available on fragments.
