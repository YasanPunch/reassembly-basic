# 3D Model Classification and Analysis System

This system provides advanced 3D model analysis using curvature and roughness analysis, with optional supervised learning for fracture surface detection.

## Quick Start

### Basic 3D Model Analysis
```bash
# Load and visualize 3D models from a folder
python src/classification.py --folder data/input_fragments

# Analyze curvature for each model
python src/classification.py --folder data/input_fragments --k-neighbors 100

# Use roughness analysis instead of curvature
python src/classification.py --folder data/input_fragments --use-roughness-analysis --radius 5.0
```

### Segmentation-Based Analysis
```bash
# Segment mesh first, then analyze each region
python src/classification.py --segment-first --k-neighbors 100

# With custom segmentation parameters
python src/classification.py --segment-first \
    --angle-threshold 30.0 \
    --curvature-threshold 0.1 \
    --min-region-size 50 \
    --max-region-size 5000 \
    --region-offset 0.2
```

### Supervised Learning for Fracture Detection

#### Train a New Model
```bash
# Train supervised learning model on first 3D model
python src/classification.py --segment-first --use-roughness-analysis --supervised-learning
```

This will:
1. **Interactive labeling**: You'll label regions as fractured/non-fractured
2. **Parameter optimization**: System finds optimal k, r values
3. **Model training**: Random Forest classifier is trained
4. **Model saving**: Model is saved globally for reuse
5. **Classification**: All regions are classified
6. **Visualization**: Results are shown

#### Use Existing Model
```bash
# Use previously trained model (no training needed)
python src/classification.py --segment-first --use-roughness-analysis --supervised-learning --use-existing-model
```

#### Process Multiple Models Efficiently
```bash
# Train on first model, use for all subsequent models
python src/classification.py --segment-first --use-roughness-analysis --supervised-learning
```

The system automatically:
- **First model**: Trains new model
- **Subsequent models**: Uses existing trained model

#### Model Management
```bash
# Clear existing model from memory
python src/classification.py --clear-model

# Don't save model globally
python src/classification.py --segment-first --use-roughness-analysis --supervised-learning --no-save-model
```

## Command Line Arguments

### Basic Options
- `--folder`: Path to folder containing 3D models (default: `data/input_fragments`)
- `--no-trimesh-fallback`: Disable trimesh fallback loading
- `--window-name`: Name of visualization window
- `--all-together`: Visualize all models together instead of one by one
- `--no-curvature-analysis`: Disable curvature analysis, use regular visualization

### Analysis Parameters
- `--k-neighbors`: Number of nearest neighbors for curvature/roughness analysis (default: 100)
- `--use-roughness-analysis`: Use surface roughness characteristic instead of local bending energy
- `--radius`: Kernel radius for roughness analysis (auto-calculated if not specified)

### Segmentation Parameters
- `--segment-first`: Segment mesh before analyzing each region
- `--angle-threshold`: Angle threshold for segmentation (default: 30.0 degrees)
- `--curvature-threshold`: Curvature threshold for segmentation (default: 0.1)
- `--min-region-size`: Minimum region size for segmentation (default: 50 faces)
- `--max-region-size`: Maximum region size for segmentation (default: 5000 faces)
- `--region-offset`: Percentage of region to offset inward to avoid edge artifacts (default: 0.2 = 20%)

### Supervised Learning Options
- `--supervised-learning`: Use supervised learning for fracture detection
- `--use-existing-model`: Use existing trained model instead of training new one
- `--no-save-model`: Don't save trained model globally for reuse
- `--clear-model`: Clear any existing trained model from memory

## Analysis Methods

### 1. Local Bending Energy (e_k(p))
Measures local curvature at each point using k-nearest neighbors:
```
e_k(p) = (1/k) * Σ ||n_p - n_qi||² / ||p - qi||²
```

### 2. Surface Roughness Characteristic (ē_k,r(p))
Averages local bending energy over a radius r neighborhood:
```
ē_k,r(p) = (1/|N_r(p)|) * Σ e_k(q) for q ∈ N_r(p)
```

### 3. Supervised Learning Pipeline
1. **Interactive Selection**: Manually label regions as fractured/non-fractured
2. **Parameter Optimization**: Grid search over k and r values using cross-validation
3. **Feature Extraction**: Statistical features (mean, std, percentiles, skewness, kurtosis)
4. **Classification**: Random Forest classifier with optimized parameters
5. **Visualization**: Color-coded results (red=fractured, blue=non-fractured)

## Model Reuse System

### Benefits
- **Time Savings**: Train once, use many times
- **Consistency**: Same parameters across all models
- **Efficiency**: No repeated labeling or optimization

### How It Works
1. **First Model**: Full training process (labeling + optimization + training)
2. **Subsequent Models**: Direct classification using trained model
3. **Automatic Detection**: System detects when to train vs when to reuse

### Model Metadata
The system tracks:
- Training date and time
- Cross-validation accuracy
- Number of training samples
- Optimal k, r parameters
- Mesh properties (vertices, faces, regions)

## Visualization Features

### Color Coding
- **Curvature/Roughness**: Blue (low) to Red (high)
- **Classification**: Red (fractured), Blue (non-fractured)
- **Confidence**: Intensity based on classification probability
- **Comparison**: Green/Cyan (agreement), Yellow (disagreement), Purple (unlabeled)

### Interactive Controls
- **Mouse**: Rotate, zoom, pan
- **Shift + Mouse**: Pan
- **Ctrl + Mouse**: Zoom
- **Q**: Exit visualization
- **F/N/S/Q**: During labeling (F=fractured, N=non-fractured, S=skip, Q=quit)

## Example Workflows

### Workflow 1: Quick Analysis
```bash
# Basic curvature analysis
python src/classification.py --folder data/models
```

### Workflow 2: Detailed Segmentation
```bash
# Segment and analyze each region
python src/classification.py --segment-first --k-neighbors 50
```

### Workflow 3: Fracture Detection
```bash
# Train model on representative object
python src/classification.py --segment-first --use-roughness-analysis --supervised-learning

# Apply to similar objects
python src/classification.py --segment-first --use-roughness-analysis --supervised-learning --use-existing-model
```

### Workflow 4: Batch Processing
```bash
# Process multiple models efficiently
python src/classification.py --segment-first --use-roughness-analysis --supervised-learning
# Automatically trains on first model, uses for all others
```

## Technical Details

### Supported File Formats
- OBJ, PLY, STL, and other formats supported by Open3D and Trimesh

### Dependencies
- Open3D (≥0.17.0)
- Trimesh (≥3.0)
- NumPy, SciPy
- Scikit-learn (for supervised learning)
- Matplotlib (for visualization)

### Performance Considerations
- **Memory**: Models are stored in memory only (not persistent across sessions)
- **Speed**: Model reuse provides significant speedup for multiple models
- **Accuracy**: Cross-validation ensures robust parameter selection

## Troubleshooting

### Common Issues
1. **"Model not fitted" error**: Clear model with `--clear-model` and retrain
2. **Poor classification**: Try different k, r values or relabel training data
3. **Memory issues**: Process fewer models at once or clear model between runs

### Debug Options
- Use `--clear-model` to reset model state
- Check model metadata for training information
- Verify model is fitted before reuse

---

## Original Reassembly System

USE THIS COMMAND TO EXECUTE -> `python -m src.main --visualize_steps_file data/log_auto_test.pkl --num_viz_pairwise 3 --debug_pairwise_matching`

--num_viz_pairwise shows the Number of top pairwise matches to visualize directly during runtime (0 for none).

--debug_pairwise_matching displays outputs before and after RANSAC and ICP for every pairwise matching completed. 

--top_n_matches_per_pair decides the number of top-scoring matches kept for each pair of fragments during pairwise matching step. This is 3 by default. 

--visualize steps file is not necessary. It should log debug steps into a file. Not tested.

SAME COMMAND WITHOUT DEBUG -> `python -m src.main --visualize_steps_file data/log_auto_test.pkl --num_viz_pairwise 3 --top_n_matches_per_pair 3`

NOTE

Overlap check in global reassembly is too strict currently. It is turned off in configuration parameters. 
Debug visualization can also be toggled on/off from config params. 

