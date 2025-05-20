# 3D Model Fragment Reconstructor

This project is a basic application to analyze pieces (fragments) of a 3D model and attempt to reconstruct the original object. It uses algorithmic approaches, primarily leveraging the Open3D library for point cloud processing, feature extraction (FPFH), and registration (RANSAC, ICP).

## Features

*   Loads 3D model fragments from various file formats (OBJ, STL, PLY).
*   Preprocesses fragments by converting them to point clouds, downsampling, and estimating normals.
*   Extracts FPFH (Fast Point Feature Histograms) features from the point clouds.
*   Performs pairwise registration between fragments using RANSAC (global) and ICP (local refinement) based on feature matching.
*   Implements a greedy assembly strategy to combine fragments into a single model based on the best pairwise alignments.
*   Saves the reconstructed model.
*   Configurable parameters for different stages of the pipeline.

## Project Structure

```
3d_model_reconstructor/
├── data/
│   ├── input_fragments/          # Input fragment files (e.g., piece_01.obj)
│   └── output_assembly/          # Output reconstructed model(s)
├── src/                          # Source code
│   ├── main.py                   # Main script
│   ├── io_utils.py               # Loading/saving 3D models
│   ├── preprocessing.py          # Fragment preparation
│   ├── feature_extraction.py     # Geometric feature extraction
│   ├── matching.py               # Finding correspondences between fragments
│   ├── alignment.py              # Aligning pairs of fragments
│   ├── assembly.py               # Global assembly strategy
│   └── utils/
│       ├── geometry_utils.py
│       └── visualization_utils.py # Optional visualization helpers
├── config/
│   └── reconstruction_params.json # Algorithm parameters
├── notebooks/                    # (Optional) Jupyter notebooks for experiments
├── tests/                        # (Placeholder) Unit tests
├── .gitignore
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Prerequisites

*   Python 3.8+
*   Libraries listed in `requirements.txt`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd 3d_model_reconstructor
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

## Usage

1.  **Place your 3D model fragments** (e.g., `.obj`, `.stl` files) into the `data/input_fragments/` directory.

2.  **Adjust parameters (optional):**
    Modify `config/reconstruction_params.json` to fine-tune the algorithm parameters if needed. The defaults are set for general cases but might need adjustment based on the scale, complexity, and condition of your fragments. Key parameters include:
    *   `voxel_downsample_size`: Affects speed and detail. Smaller values mean more detail but slower processing.
    *   `min_match_score`: Threshold for accepting a pairwise alignment.
    *   RANSAC and ICP parameters.

3.  **Run the application:**
    ```bash
    python src/main.py
    or
    python -m src.main
    or #best for debugging
    python -m src.main --visualize_steps_file data/my_run_log.pkl
    ```
    You can also specify command-line arguments:
    ```bash
    python src/main.py --input_dir path/to/your/fragments --output_dir path/to/your/output --config_file path/to/your/config.json --visualize
    ```
    *   `--input_dir`: Directory containing input fragments.
    *   `--output_dir`: Directory where the reconstructed model will be saved.
    *   `--config_file`: Path to the parameters JSON file. (Usually reconstruction_params.json)
    *   `--visualize`: If present, enables Open3D visualizations of some steps (e.g., the final assembly).

4.  **Check the output:**
    The reconstructed model will be saved in the specified output directory (default: `data/output_assembly/reconstructed_model.obj`).

## How It Works (High-Level)

1.  **Load & Preprocess:** Fragments are loaded. Each mesh is converted to a point cloud, downsampled using a voxel grid, and normals are estimated.
2.  **Feature Extraction:** FPFH (Fast Point Feature Histogram) descriptors are computed for points in each processed point cloud. These descriptors capture local geometric properties.
3.  **Pairwise Matching & Alignment:**
    *   For each pair of fragments, FPFH features are matched.
    *   **RANSAC:** A RANSAC-based algorithm finds an initial coarse alignment based on feature correspondences.
    *   **ICP (Iterative Closest Point):** The coarse alignment is refined using ICP (Point-to-Plane variant) to achieve a precise fit.
    *   A `score` (fitness from ICP) is assigned to each potential pairwise alignment.
4.  **Global Assembly:**
    *   A greedy approach is used:
        *   Start with a seed fragment (e.g., part of the best-scoring pairwise match).
        *   Iteratively find the unplaced fragment that best aligns (highest score, acceptable overlap) to any part of the currently assembled structure.
        *   Transform and add this fragment to the assembly.
        *   Repeat until no more fragments can be added or all are placed.
    *   Overlap checks are performed to prevent gross interpenetration of parts.
5.  **Output:** The final assembled model, composed of the original fragment meshes transformed into their globally aligned positions, is saved.

## Limitations

*   **Greedy Assembly:** The greedy approach may not find the globally optimal assembly and can get stuck in local minima.
*   **Sensitivity to Parameters:** Performance heavily depends on the parameters in `reconstruction_params.json`. These often need tuning for specific datasets.
*   **Requires Good Overlap & Features:** Assumes fragments have sufficient overlapping areas with distinct geometric features for FPFH and ICP to work well.
*   **No Handling of Symmetric/Ambiguous Parts:** May struggle with highly symmetric objects or pieces that could fit in multiple ways.
*   **Error Propagation:** Errors in early pairwise alignments can propagate through the greedy assembly.
*   **Scalability:** Combinatorial complexity of pairwise matching grows with N*(N-1)/2. The RANSAC step can be slow for many features/points.
*   **Non-Manifold/Noisy Meshes:** Input meshes should ideally be clean and manifold. Robustness to noise or mesh defects is limited.
*   **Overlap Check:** The current overlap check is basic and might not catch all interpenetrations perfectly or might be too restrictive.

## Potential Future Enhancements

*   **Improved Global Assembly:**
    *   Graph-based assembly (e.g., finding a consistent cycle/path in a compatibility graph).
    *   Pose graph optimization to globally refine all transformations simultaneously.
*   **More Robust Feature Descriptors:** Explore other local or global shape descriptors.
*   **Semantic Information:** If part types or approximate initial poses are known, this could guide assembly.
*   **Machine Learning:**
    *   Learning feature matchers or pose predictors.
    *   Reinforcement learning for assembly sequencing.
*   **Improved Overlap/Collision Detection:** More sophisticated methods using signed distance fields or boolean operations.
*   **User Interaction:** Allow manual guidance or correction of alignments.
*   **Parallelization:** Speed up pairwise matching and other computationally intensive steps.

## Contributing

This is a basic example. Contributions for improvements, bug fixes, or new features are welcome! (Standard contribution guidelines would go here).