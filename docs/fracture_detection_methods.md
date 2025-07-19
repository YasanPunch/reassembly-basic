# Fracture Surface Detection Methods

This document describes various methods for distinguishing between fractured surfaces and original surfaces in 3D mesh fragments.

## Overview

Fracture surfaces typically exhibit different geometric properties compared to original surfaces. This implementation provides multiple detection methods that can be used individually or in combination to identify fracture surfaces.

## Detection Methods

### 1. Region Growing + Bumpiness Detection

**Principle**: Fracture surfaces are typically rougher and more irregular than original surfaces.

**Implementation**: 
- Uses region growing to group faces with similar normals
- Calculates surface bumpiness using elevation maps and Laplacian operators
- Selects regions with high bumpiness scores

**Parameters**:
- `use_bumpiness_detection`: Enable/disable this method
- `bumpiness_threshold`: Threshold for bumpiness scores (default: 0.2)
- `elevation_map_resolution`: Resolution for elevation map calculation (default: 64)

**Pros**: 
- Based on well-established roughness metrics
- Works well for clearly rough fracture surfaces

**Cons**: 
- May miss smooth fracture surfaces
- Computationally expensive for large meshes

### 2. Advanced Geometric Analysis

**Principle**: Combines multiple geometric properties to create a comprehensive fracture score.

**Properties Analyzed**:
- **Curvature**: Fracture surfaces often have higher local curvature
- **Roughness**: Surface roughness based on normal variation
- **Boundary Complexity**: Fracture surfaces often have more complex boundaries
- **Symmetry**: Original surfaces are typically more symmetric
- **Planarity**: Original surfaces are often more planar

**Implementation**:
- Calculates each property for every face
- Normalizes scores to [0,1] range
- Combines scores using weighted average
- Applies threshold to identify fracture candidates

**Parameters**:
- `use_advanced_detection`: Enable/disable this method
- `advanced_detection_ratio_threshold`: Minimum ratio of faces in a region that must be detected (default: 0.3)
- `fracture_detection_threshold`: Combined score threshold (default: 0.5)
- `fracture_detection_weights`: Weights for each property

**Pros**: 
- Comprehensive analysis
- Configurable weights
- Good balance of accuracy and performance

**Cons**: 
- Requires parameter tuning
- May be sensitive to mesh quality

### 3. Statistical Analysis

**Principle**: Analyzes statistical properties of surface regions to determine fracture likelihood.

**Statistics Analyzed**:
- **Normal Variation**: Standard deviation of normal angles
- **Area Distribution**: Coefficient of variation of face areas
- **Surface Roughness**: Based on normal variation
- **Region Size**: Smaller regions are more likely to be fractures

**Implementation**:
- Groups faces into regions using region growing
- Calculates statistical properties for each region
- Uses rule-based classification with confidence scoring

**Parameters**:
- `use_statistical_analysis`: Enable/disable this method
- `statistical_confidence_threshold`: Minimum confidence for classification (default: 0.6)

**Pros**: 
- Robust statistical approach
- Provides confidence scores
- Good for irregular fracture patterns

**Cons**: 
- Requires sufficient region size
- May miss small fracture features

### 4. Simple Curvature-based Detection

**Principle**: Fracture surfaces typically have higher local curvature than original surfaces.

**Implementation**:
- Calculates curvature for each face using neighborhood analysis
- Uses percentile-based thresholding
- Selects faces above the threshold

**Parameters**:
- `curvature_threshold_percentile`: Percentile threshold (default: 75)

**Pros**: 
- Simple and fast
- Intuitive concept
- Good baseline method

**Cons**: 
- May miss low-curvature fractures
- Sensitive to mesh noise

### 5. Combined Approach

**Principle**: Combines results from multiple detection methods for improved accuracy.

**Implementation**:
- Runs all enabled detection methods
- Requires agreement from multiple methods
- Configurable minimum agreement threshold

**Parameters**:
- `use_combined_detection`: Enable/disable this method
- `combined_min_agreement`: Minimum number of methods that must agree (default: 2)

**Pros**: 
- Reduces false positives
- More robust detection
- Configurable consensus requirements

**Cons**: 
- May miss fractures detected by only one method
- Requires multiple methods to be enabled

## Confidence Scoring

The `get_fracture_surface_confidence()` function provides a confidence score (0-1) for any set of faces:

**Indicators Used**:
- Normal variation
- Surface roughness
- Area irregularity
- Size factor
- Boundary complexity

**Usage**:
```python
confidence, indicators = get_fracture_surface_confidence(tri_mesh, face_indices, params)
```

## Configuration

### Basic Configuration

```json
{
    "use_bumpiness_detection": true,
    "use_advanced_detection": true,
    "use_statistical_analysis": true,
    "use_combined_detection": true
}
```

### Advanced Configuration

```json
{
    "fracture_detection_weights": {
        "curvature": 0.3,
        "roughness": 0.3,
        "boundary_complexity": 0.2,
        "symmetry": -0.1,
        "planarity": -0.1
    },
    "confidence_weights": {
        "normal_variation": 0.25,
        "roughness": 0.25,
        "area_irregularity": 0.2,
        "size_factor": 0.15,
        "boundary_complexity": 0.15
    }
}
```

## Usage Examples

### Basic Usage

```python
from src.segmentation import extract_fracture_surface_mesh

# Basic parameters
params = {
    'use_bumpiness_detection': True,
    'bumpiness_threshold': 0.2
}

# Extract fracture surfaces
fracture_surfaces = extract_fracture_surface_mesh(mesh, "Fragment1", params)
```

### Advanced Usage

```python
from src.segmentation import compare_fracture_detection_methods, visualize_detection_comparison

# Compare all methods
results = compare_fracture_detection_methods(tri_mesh, params)

# Visualize comparison
visualize_detection_comparison(o3d_mesh, results, "Fragment1")
```

### Confidence Analysis

```python
from src.segmentation import get_fracture_surface_confidence

# Analyze a specific region
confidence, indicators = get_fracture_surface_confidence(tri_mesh, face_indices, params)
print(f"Confidence: {confidence:.3f}")
print(f"Indicators: {indicators}")
```

## Performance Considerations

1. **Bumpiness Detection**: Most expensive, scales with mesh size
2. **Advanced Analysis**: Moderate cost, good balance
3. **Statistical Analysis**: Fast for small regions, scales with region count
4. **Curvature Detection**: Fastest method
5. **Combined Approach**: Cost depends on enabled methods

## Best Practices

1. **Start Simple**: Begin with curvature detection for baseline results
2. **Add Complexity**: Enable advanced analysis for better accuracy
3. **Tune Parameters**: Adjust thresholds based on your specific data
4. **Use Combined Approach**: For critical applications, use multiple methods
5. **Validate Results**: Always visually inspect detected fracture surfaces

## Troubleshooting

### Common Issues

1. **Too Many False Positives**: Increase thresholds or use combined approach
2. **Missing Fracture Surfaces**: Decrease thresholds or enable more methods
3. **Slow Performance**: Disable bumpiness detection or reduce resolution
4. **Poor Results**: Check mesh quality and adjust parameters

### Parameter Tuning

1. **For Rough Fractures**: Use bumpiness detection with low threshold
2. **For Smooth Fractures**: Use advanced analysis with adjusted weights
3. **For Small Fragments**: Use statistical analysis with lower confidence threshold
4. **For Large Meshes**: Use curvature detection or reduce resolution

## Future Improvements

1. **Machine Learning**: Train classifiers on labeled fracture data
2. **Texture Analysis**: Incorporate surface texture information
3. **Temporal Analysis**: Use multiple scans for better detection
4. **Adaptive Parameters**: Automatically adjust parameters based on mesh properties 