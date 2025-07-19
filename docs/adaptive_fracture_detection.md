# Adaptive Fracture Surface Detection

This document describes the simplified adaptive approach for detecting fracture surfaces that automatically adjusts thresholds based on the object's properties.

## Overview

Instead of using fixed thresholds that may not work well across different objects, the adaptive approach analyzes the object's actual geometric properties and uses relative thresholds (percentiles) to identify fracture surfaces.

## Key Improvements

### 1. **Adaptive Thresholds**
- **Before**: Fixed thresholds like `roughness > 0.2`
- **After**: Relative thresholds like `roughness > 75th percentile`

### 2. **Simplified Configuration**
- **Before**: Multiple complex parameters for each detection method
- **After**: Simple percentile-based parameters that work across different objects

### 3. **Object-Aware Detection**
- Automatically adapts to the object's scale, complexity, and geometric characteristics
- No need to tune parameters for each new object

## How It Works

### Step 1: Property Analysis
The system calculates geometric properties for every face in the mesh:
- **Roughness**: Surface roughness based on normal variation
- **Curvature**: Local curvature using neighborhood analysis
- **Boundary Complexity**: Complexity of face boundaries
- **Symmetry**: How symmetric each face is
- **Planarity**: How well faces fit a plane

### Step 2: Adaptive Threshold Calculation
For each property, the system calculates adaptive thresholds:
```python
# Example: Roughness threshold
roughness_values = [calculate_face_roughness(mesh, face) for face in all_faces]
roughness_threshold = np.percentile(roughness_values, 75)  # Top 25% roughest
```

### Step 3: Relative Scoring
Faces are scored relative to the calculated thresholds:
```python
# Normalize scores above threshold to [0,1]
if face_roughness > roughness_threshold:
    score = (face_roughness - threshold) / (max_roughness - threshold)
else:
    score = 0
```

### Step 4: Combined Detection
Multiple detection methods are combined for robust results:
- **Adaptive Geometric Analysis**: Uses all geometric properties
- **Adaptive Statistical Analysis**: Analyzes region statistics
- **Adaptive Curvature Detection**: Simple curvature-based detection

## Configuration Parameters

### Simplified Parameters

```json
{
    "use_adaptive_detection": true,
    "use_combined_approach": true,
    
    "roughness_threshold_percentile": 75,
    "curvature_threshold_percentile": 75,
    "boundary_complexity_threshold_percentile": 75,
    "symmetry_threshold_percentile": 25,
    "planarity_threshold_percentile": 25,
    "score_threshold_percentile": 70,
    
    "fracture_detection_weights": {
        "curvature": 0.3,
        "roughness": 0.3,
        "boundary_complexity": 0.2,
        "symmetry": 0.1,
        "planarity": 0.1
    },
    
    "combined_min_agreement": 2
}
```

### Parameter Explanation

| Parameter | Description | Default | Meaning |
|-----------|-------------|---------|---------|
| `roughness_threshold_percentile` | Percentile for roughness threshold | 75 | Top 25% roughest faces |
| `curvature_threshold_percentile` | Percentile for curvature threshold | 75 | Top 25% highest curvature |
| `boundary_complexity_threshold_percentile` | Percentile for boundary complexity | 75 | Top 25% most complex boundaries |
| `symmetry_threshold_percentile` | Percentile for symmetry threshold | 25 | Bottom 25% least symmetric |
| `planarity_threshold_percentile` | Percentile for planarity threshold | 25 | Bottom 25% least planar |
| `score_threshold_percentile` | Percentile for final score threshold | 70 | Top 30% highest scores |
| `combined_min_agreement` | Minimum methods that must agree | 2 | At least 2 methods must detect |

## Usage Examples

### Basic Usage

```python
from src.segmentation import extract_fracture_surface_mesh

# Simple adaptive configuration
params = {
    'use_adaptive_detection': True,
    'use_combined_approach': True
}

# Extract fracture surfaces (thresholds calculated automatically)
fracture_surfaces = extract_fracture_surface_mesh(mesh, "Fragment1", params)
```

### Custom Percentiles

```python
# For more aggressive detection (select more faces)
params = {
    'use_adaptive_detection': True,
    'roughness_threshold_percentile': 60,  # Top 40% instead of 25%
    'curvature_threshold_percentile': 60,
    'score_threshold_percentile': 60
}

# For more conservative detection (select fewer faces)
params = {
    'use_adaptive_detection': True,
    'roughness_threshold_percentile': 85,  # Top 15% instead of 25%
    'curvature_threshold_percentile': 85,
    'score_threshold_percentile': 80
}
```

### Custom Weights

```python
# Emphasize roughness and curvature more
params = {
    'use_adaptive_detection': True,
    'fracture_detection_weights': {
        'curvature': 0.4,
        'roughness': 0.4,
        'boundary_complexity': 0.1,
        'symmetry': 0.05,
        'planarity': 0.05
    }
}
```

## Comparison with Fixed Thresholds

### Example: Roughness Detection

**Fixed Threshold Approach:**
```python
# This might work for one object but fail for another
if face_roughness > 0.2:  # Fixed threshold
    mark_as_fracture()
```

**Adaptive Approach:**
```python
# Automatically adapts to any object
roughness_threshold = np.percentile(all_roughness_values, 75)
if face_roughness > roughness_threshold:  # Adaptive threshold
    mark_as_fracture()
```

### Benefits of Adaptive Approach

1. **No Parameter Tuning**: Works automatically across different objects
2. **Scale Invariant**: Adapts to objects of different sizes
3. **Quality Robust**: Works with meshes of different quality levels
4. **Intuitive**: Percentiles are easier to understand than absolute values

## Performance Considerations

### Computational Cost
- **Property Calculation**: O(n) where n is number of faces
- **Threshold Calculation**: O(n log n) for percentile calculation
- **Overall**: Still very fast for typical meshes

### Memory Usage
- Stores property values for all faces
- Minimal additional memory overhead

## Best Practices

### 1. **Start with Defaults**
The default percentiles (75th for most properties) work well for most objects.

### 2. **Adjust Based on Object Type**
- **Smooth Objects**: Use higher percentiles (80-90) for more selective detection
- **Rough Objects**: Use lower percentiles (60-70) for more inclusive detection

### 3. **Use Combined Approach**
Always enable `use_combined_approach` for more robust results.

### 4. **Monitor Results**
Check the console output to see how many faces are detected by each method.

## Troubleshooting

### Too Many False Positives
- Increase percentiles (e.g., 75 → 85)
- Increase `combined_min_agreement` (e.g., 2 → 3)

### Missing Fracture Surfaces
- Decrease percentiles (e.g., 75 → 65)
- Decrease `combined_min_agreement` (e.g., 2 → 1)

### Slow Performance
- Reduce `elevation_map_resolution` if using bumpiness detection
- Use fewer detection methods

## Migration from Fixed Thresholds

### Old Configuration
```json
{
    "use_bumpiness_detection": true,
    "bumpiness_threshold": 0.2,
    "use_advanced_detection": true,
    "fracture_detection_threshold": 0.5
}
```

### New Configuration
```json
{
    "use_adaptive_detection": true,
    "use_combined_approach": true,
    "roughness_threshold_percentile": 75,
    "score_threshold_percentile": 70
}
```

The adaptive approach is backward compatible - you can still use the old parameters if needed, but the new approach is recommended for better results across different objects. 