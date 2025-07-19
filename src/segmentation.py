import open3d as o3d
import trimesh
import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import deque
from scipy.spatial import cKDTree
from scipy import stats
import multiprocessing as mp

def get_color(index, total_items=20, cmap_name='tab10', num_variations=3):
    """
    Gets a distinct color. Uses a base colormap and applies variations
    if the number of items exceeds the colormap's distinct colors.
    Args:
        index (int): The 0-based index of the item to color.
        total_items (int): Total number of items needing colors (helps estimate variations).
        cmap_name (str): Name of the base Matplotlib colormap.
        num_variations (int): How many brightness/saturation variations to apply for each base color.
    Returns:
        tuple: (R, G, B) color.
    """
    try:
        base_cmap = plt.cm.get_cmap(cmap_name)
        if not base_cmap:
            base_cmap = plt.cm.get_cmap('Set1')
        if not base_cmap:
            base_cmap = plt.cm.get_cmap('viridis')

        num_base_colors = base_cmap.N
        base_color_index = index % num_base_colors
        variation_cycle = (index // num_base_colors) % num_variations

        r, g, b, _ = base_cmap(base_color_index)

        if variation_cycle == 0:
            pass
        elif variation_cycle == 1:
            factor = 1.3
            r = min(1.0, r * factor + 0.1)
            g = min(1.0, g * factor + 0.1)
            b = min(1.0, b * factor + 0.1)
        elif variation_cycle == 2:
            factor = 0.7
            r *= factor
            g *= factor
            b *= factor

        return np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)

    except ImportError:
        colors = [[1,0,0],[0,0,1],[0,1,0],[1,1,0],[1,0,1],[0,1,1],
                  [0.8,0.5,0.2],[0.5,0.2,0.8],[0.2,0.8,0.5], [0.6,0.6,0.6]]
        return colors[index % len(colors)]
    except Exception as e:
        print(f"Error in get_color: {e}. Using fallback.")
        colors = [[1,0,0],[0,0,1],[0,1,0]]
        return colors[index % len(colors)]


def calculate_region_average_normal(tri_mesh, face_indices):
    """
    Calculate the area-weighted average normal for a region following the paper's formula:
    N_ave(R_k) = sum(A_j * N_j) / sum(A_j) for all j in R_k
    """
    if len(face_indices) == 0:
        return np.array([0, 0, 1])
    
    face_normals = tri_mesh.face_normals[face_indices]
    face_areas = tri_mesh.area_faces[face_indices]
    
    # Area-weighted average
    weighted_normals = face_normals * face_areas[:, np.newaxis]
    avg_normal = np.sum(weighted_normals, axis=0) / np.sum(face_areas)
    
    # Normalize
    norm = np.linalg.norm(avg_normal)
    if norm > 1e-10:
        avg_normal = avg_normal / norm
    else:
        avg_normal = np.array([0, 0, 1])
    
    return avg_normal


# --- NEW HIGH-PERFORMANCE GEOMETRIC FEATURE EXTRACTION ---

def get_adaptive_params(num_faces):
    """Adjust processing parameters based on mesh size"""
    if num_faces < 10000:
        return {'max_sample_size': None, 'early_exit': False, 'batch_processing': False}
    elif num_faces < 100000:
        return {'max_sample_size': 2000, 'early_exit': True, 'batch_processing': False}
    else:  # 100K+ faces
        return {'max_sample_size': 500, 'early_exit': True, 'batch_processing': True}


def get_representative_sample(region_faces, max_sample_size=1000):
    """Sample faces from large regions instead of processing all"""
    if max_sample_size is None or len(region_faces) <= max_sample_size:
        return region_faces  # Small region, process all
    
    # Stratified sampling for large regions
    return np.random.choice(region_faces, size=max_sample_size, replace=False)


def build_adjacency_dict(face_adjacency):
    """Build adjacency dictionary for fast neighbor lookup"""
    adjacency_dict = {}
    for face1, face2 in face_adjacency:
        if face1 not in adjacency_dict:
            adjacency_dict[face1] = []
        if face2 not in adjacency_dict:
            adjacency_dict[face2] = []
        adjacency_dict[face1].append(face2)
        adjacency_dict[face2].append(face1)
    return adjacency_dict


def compute_edge_length_cv_fast(tri_mesh, face_indices):
    """Vectorized edge length coefficient of variation (scale-invariant)"""
    if len(face_indices) == 0:
        return 0.0
        
    faces = tri_mesh.faces[face_indices]
    vertices = tri_mesh.vertices
    
    # Get all edges in one vectorized operation
    edges = np.vstack([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
    ])
    
    # Vectorized distance computation
    edge_vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    # Coefficient of variation (dimensionless, scale-invariant)
    mean_len = np.mean(edge_lengths)
    if mean_len < 1e-10:
        return 0.0
    
    return np.std(edge_lengths) / mean_len


def compute_dihedral_variance_fast(tri_mesh, face_indices):
    """Fast dihedral angle variance using precomputed adjacency"""
    if len(face_indices) < 2:
        return 0.0
    
    # Build adjacency dict if not cached
    if not hasattr(tri_mesh, '_face_adjacency_dict'):
        tri_mesh._face_adjacency_dict = build_adjacency_dict(tri_mesh.face_adjacency)
    
    dihedral_angles = []
    face_normals = tri_mesh.face_normals
    face_indices_set = set(face_indices)
    
    for face_idx in face_indices:
        if face_idx in tri_mesh._face_adjacency_dict:
            for neighbor_idx in tri_mesh._face_adjacency_dict[face_idx]:
                if neighbor_idx in face_indices_set and neighbor_idx > face_idx:  # Avoid duplicates
                    # Calculate angle between face normals
                    dot_product = np.clip(
                        np.dot(face_normals[face_idx], face_normals[neighbor_idx]), 
                        -1, 1
                    )
                    angle = np.arccos(dot_product)
                    dihedral_angles.append(angle)
    
    return np.var(dihedral_angles) if len(dihedral_angles) > 0 else 0.0


def compute_triangle_quality_metrics_fast(tri_mesh, face_indices):
    """Compute triangle quality metrics (aspect ratios, area variation)"""
    if len(face_indices) == 0:
        return {'area_cv': 0.0, 'aspect_ratio_mean': 1.0, 'aspect_ratio_std': 0.0}
    
    faces = tri_mesh.faces[face_indices]
    vertices = tri_mesh.vertices
    areas = tri_mesh.area_faces[face_indices]
    
    # Area coefficient of variation
    area_mean = np.mean(areas)
    area_cv = np.std(areas) / area_mean if area_mean > 1e-10 else 0.0
    
    # Aspect ratios (longest edge / shortest edge for each triangle)
    aspect_ratios = []
    for face in faces:
        # Get edge lengths for this triangle
        v0, v1, v2 = vertices[face]
        edge_lengths = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1), 
            np.linalg.norm(v0 - v2)
        ]
        max_edge = max(edge_lengths)
        min_edge = min(edge_lengths)
        aspect_ratio = max_edge / min_edge if min_edge > 1e-10 else 1.0
        aspect_ratios.append(aspect_ratio)
    
    aspect_ratios = np.array(aspect_ratios)
    
    return {
        'area_cv': area_cv,
        'aspect_ratio_mean': np.mean(aspect_ratios),
        'aspect_ratio_std': np.std(aspect_ratios)
    }


def compute_normal_consistency_fast(tri_mesh, face_indices):
    """Compute normal vector consistency within region"""
    if len(face_indices) < 2:
        return 1.0  # Perfect consistency for single face
    
    face_normals = tri_mesh.face_normals[face_indices]
    region_avg_normal = np.mean(face_normals, axis=0)
    region_avg_normal = region_avg_normal / (np.linalg.norm(region_avg_normal) + 1e-10)
    
    # Calculate dot products with average normal
    dot_products = np.dot(face_normals, region_avg_normal)
    
    # Consistency is 1 - standard deviation of dot products
    consistency = 1.0 - np.std(dot_products)
    return max(0.0, consistency)  # Clamp to [0, 1]


def extract_geometric_features_fast(tri_mesh, region_faces, region_area_fraction, adaptive_params):
    """
    Extract scale-invariant geometric features with performance optimizations
    """
    # Get adaptive thresholds
    adaptive_thresholds = adaptive_params.get('thresholds', {})
    min_fracture_size = adaptive_thresholds.get('min_fracture_size', 0.05)
    
    # Early exit for tiny regions (adaptive threshold based on mesh properties)
    if region_area_fraction < min_fracture_size:
        return {
            'classification': 'too_small_for_fracture',
            'confidence': 1.0,
            'features': {},
            'reason': f'Region too small ({region_area_fraction:.1%} < {min_fracture_size:.1%}) - likely mesh artifact'
        }
    
    # Sample for large regions
    sample_faces = get_representative_sample(region_faces, adaptive_params['max_sample_size'])
    
    # TIER 1: Fast geometric ratios (1-5ms on sampled data)
    features = {}
    features['edge_length_cv'] = compute_edge_length_cv_fast(tri_mesh, sample_faces)
    features['dihedral_variance'] = compute_dihedral_variance_fast(tri_mesh, sample_faces)
    
    # Early exit for obviously smooth surfaces (tightened criteria)
    if (adaptive_params['early_exit'] and 
        features['edge_length_cv'] < 0.02 and  # More restrictive
        features['dihedral_variance'] < 0.005):  # More restrictive
        return {
            'classification': 'smooth_surface',
            'confidence': 0.95,
            'features': features,
            'reason': 'Very low geometric variation - likely manufactured surface'
        }
    
    # TIER 2: More detailed features
    quality_metrics = compute_triangle_quality_metrics_fast(tri_mesh, sample_faces)
    features.update(quality_metrics)
    
    features['normal_consistency'] = compute_normal_consistency_fast(tri_mesh, sample_faces)
    
    # NEW: Detect mesh quality artifacts
    is_mesh_artifact = (
        features['aspect_ratio_std'] > 50.0 and  # Very poor triangulation
        region_area_fraction < 0.15  # Small region
    )
    
    if is_mesh_artifact:
        return {
            'classification': 'mesh_artifact',
            'confidence': 0.9,
            'features': features,
            'reason': 'High aspect ratio variance in small region - likely mesh quality issue'
        }
    
    # ADAPTIVE: Calculate fracture likelihood using mesh-aware thresholds
    normal_consistency = features['normal_consistency']
    edge_cv = features['edge_length_cv']
    dihedral_var = features['dihedral_variance']
    area_cv = features['area_cv']
    
    # Get adaptive thresholds
    edge_cv_threshold = adaptive_thresholds.get('edge_cv_threshold', 0.25)
    dihedral_var_threshold = adaptive_thresholds.get('dihedral_var_threshold', 0.008)
    area_cv_threshold = adaptive_thresholds.get('area_cv_threshold', 0.4)
    smooth_thresholds = adaptive_thresholds.get('smooth_thresholds', {
        'very_smooth': 0.95, 'quite_smooth': 0.90, 
        'penalties': {'very': 0.5, 'quite': 0.25}
    })
    
    # Adaptive smoothness penalty based on mesh characteristics
    smoothness_penalty = 0.0
    if normal_consistency > smooth_thresholds['very_smooth']:
        smoothness_penalty = smooth_thresholds['penalties']['very']
    elif normal_consistency > smooth_thresholds['quite_smooth']:
        smoothness_penalty = smooth_thresholds['penalties']['quite']
    
    # Fracture indicators (using adaptive thresholds)
    fracture_indicators = {
        # Surface roughness relative to mesh baseline
        'surface_roughness': max(0.0, (edge_cv - edge_cv_threshold) * 2.0) if edge_cv > edge_cv_threshold else 0.0,
        
        # Angular variation relative to mesh baseline
        'angular_variation': max(0.0, (dihedral_var - dihedral_var_threshold) * 100.0) if dihedral_var > dihedral_var_threshold else 0.0,
        
        # Area inconsistency relative to mesh baseline
        'area_inconsistency': max(0.0, (area_cv - area_cv_threshold) * 1.5) if area_cv > area_cv_threshold else 0.0,
        
        # Triangulation quality (penalize extreme aspect ratios)
        'triangulation_quality': max(0.0, 1.0 - features['aspect_ratio_std'] / 40.0),
        
        # Normal variation (adaptive based on smoothness thresholds)
        'normal_variation': max(0.0, min(0.5, (1.0 - normal_consistency) * 2.0)) if normal_consistency < smooth_thresholds['very_smooth'] else 0.0
    }
    
    # Calculate base fracture score
    base_fracture_score = (
        fracture_indicators['surface_roughness'] * 3.0 +
        fracture_indicators['angular_variation'] * 4.0 +
        fracture_indicators['area_inconsistency'] * 2.0 +
        fracture_indicators['triangulation_quality'] * 0.5 +
        fracture_indicators['normal_variation'] * 2.0
    ) / 11.5
    
    # Apply adaptive smoothness penalty
    fracture_likelihood = max(0.0, base_fracture_score - smoothness_penalty)
    
    # Debug information (can be removed later)
    if region_area_fraction > 0.15:  # Only show for larger regions
        print(f"      DEBUG Region (area={region_area_fraction:.1%}): edge_cv={edge_cv:.3f}>{edge_cv_threshold:.3f}, "
              f"dihedral={dihedral_var:.4f}>{dihedral_var_threshold:.4f}, "
              f"base_score={base_fracture_score:.3f}, penalty={smoothness_penalty:.3f}, "
              f"final_likelihood={fracture_likelihood:.3f}")
    
    features['fracture_likelihood'] = fracture_likelihood
    features['fracture_indicators'] = fracture_indicators
    
    # Keep old irregularity score for compatibility
    features['irregularity_score'] = fracture_likelihood
    
    return {
        'classification': 'analyzed',
        'confidence': 0.8,
        'features': features
    }


class ScaleInvariantFractureClassifier:
    """Statistical classifier for fracture surface detection"""
    
    def __init__(self):
        self.feature_stats = {}
        self.classifications = {}
    
    def classify_regions(self, all_region_features, region_properties=None, adaptive_thresholds=None):
        """
        Classify regions using adaptive fracture likelihood analysis
        """
        if len(all_region_features) < 2:
            return [True] * len(all_region_features)  # If only 1-2 regions, select all
        
        # Separate regions by classification type
        analyzed_features = []
        analyzed_indices = []
        excluded_regions = []
        
        for i, region_result in enumerate(all_region_features):
            classification = region_result['classification']
            
            if classification == 'analyzed':
                analyzed_features.append(region_result['features'])
                analyzed_indices.append(i)
            elif classification in ['too_small_for_fracture', 'mesh_artifact', 'smooth_surface']:
                excluded_regions.append({
                    'index': i,
                    'classification': classification,
                    'reason': region_result.get('reason', 'No reason provided')
                })
                print(f"    Excluding Region {i+1}: {classification} - {region_result.get('reason', '')}")
        
        if len(analyzed_features) < 1:
            print(f"    No regions suitable for analysis after filtering")
            return None
        
        # NEW APPROACH: Fracture likelihood-based classification
        fracture_candidates_full = np.zeros(len(all_region_features), dtype=bool)
        confidence_scores = []
        
        # Get fracture likelihood scores
        fracture_likelihoods = []
        region_areas = []
        
        for i, (analyzed_idx, features) in enumerate(zip(analyzed_indices, analyzed_features)):
            likelihood = features.get('fracture_likelihood', 0.0)
            fracture_likelihoods.append(likelihood)
            
            if region_properties and analyzed_idx < len(region_properties):
                area_fraction = region_properties[analyzed_idx]['area_fraction']
                region_areas.append(area_fraction)
            else:
                region_areas.append(0.1)  # Default
        
        fracture_likelihoods = np.array(fracture_likelihoods)
        region_areas = np.array(region_areas)
        
        # Calculate combined scores: fracture_likelihood + area bonus
        # Larger regions get preference (fracture surfaces are often substantial)
        area_bonus = np.log1p(region_areas * 10)  # Logarithmic area bonus
        combined_scores = fracture_likelihoods * 0.7 + area_bonus * 0.3
        
        # Adaptive classification thresholds based on mesh complexity
        if adaptive_thresholds is None:
            adaptive_thresholds = {}
        classification_thresholds = adaptive_thresholds.get('classification', {
            'min_likelihood': 0.2, 'min_combined': 0.5, 
            'percentile_likelihood': 70, 'percentile_combined': 60
        })
        
        if len(fracture_likelihoods) == 1:
            # Single region - use adaptive minimum
            min_single_threshold = classification_thresholds['min_likelihood'] * 1.2  # 20% higher for single region
            is_fracture = fracture_likelihoods[0] > min_single_threshold
            confidence_scores = [fracture_likelihoods[0]]
            print(f"    Single region threshold: {min_single_threshold:.3f} (adapted from {classification_thresholds['min_likelihood']:.3f})")
        else:
            # Multiple regions - use adaptive thresholds
            
            # Adaptive minimum thresholds based on mesh complexity
            min_fracture_likelihood = classification_thresholds['min_likelihood']
            min_combined_score = classification_thresholds['min_combined']
            
            # Adaptive relative thresholds
            percentile_likelihood = classification_thresholds['percentile_likelihood']
            percentile_combined = classification_thresholds['percentile_combined']
            
            likelihood_threshold = max(min_fracture_likelihood, 
                                     np.percentile(fracture_likelihoods, percentile_likelihood))
            combined_threshold = max(min_combined_score,
                                   np.percentile(combined_scores, percentile_combined))
            
            # Classification logic (adaptive strictness)
            mesh_complexity = adaptive_thresholds.get('mesh_complexity', 0.3)
            
            if mesh_complexity > 0.4:  # Complex mesh - use OR logic
                is_fracture_likelihood = fracture_likelihoods >= likelihood_threshold
                is_fracture_combined = combined_scores >= combined_threshold
                is_above_minimum = (fracture_likelihoods >= min_fracture_likelihood)
                is_fracture = (is_fracture_likelihood | is_fracture_combined) & is_above_minimum
                print(f"    Complex mesh: using OR logic (likelihood OR combined) + minimum")
            else:  # Simple mesh - use AND logic to be more selective
                is_fracture_likelihood = fracture_likelihoods >= likelihood_threshold
                is_fracture_combined = combined_scores >= combined_threshold
                is_above_minimum = (fracture_likelihoods >= min_fracture_likelihood) & (combined_scores >= min_combined_score)
                is_fracture = is_fracture_likelihood & is_fracture_combined & is_above_minimum
                print(f"    Simple mesh: using AND logic (likelihood AND combined)")
            
            confidence_scores = (fracture_likelihoods * 0.7 + combined_scores * 0.3).tolist()
            
            # Debug information for adaptive thresholds
            print(f"    Adaptive classification thresholds:")
            print(f"    - Likelihood threshold: {likelihood_threshold:.3f} (top {100-percentile_likelihood}% or {min_fracture_likelihood:.3f} min)")
            print(f"    - Combined threshold: {combined_threshold:.3f} (top {100-percentile_combined}% or {min_combined_score:.3f} min)")
            print(f"    - Mesh complexity: {mesh_complexity:.3f}")
        
        # Map results back to original indices
        for i, analyzed_idx in enumerate(analyzed_indices):
            if analyzed_idx < len(fracture_candidates_full):
                fracture_candidates_full[analyzed_idx] = is_fracture[i] if hasattr(is_fracture, '__len__') else is_fracture
        
        # Debug information
        print(f"    Fracture likelihood analysis:")
        for i, (analyzed_idx, likelihood, area, combined, selected) in enumerate(zip(
            analyzed_indices, fracture_likelihoods, region_areas, combined_scores, 
            is_fracture if hasattr(is_fracture, '__len__') else [is_fracture]
        )):
            print(f"    Region {analyzed_idx+1}: likelihood={likelihood:.3f}, area={area:.1%}, "
                  f"combined={combined:.3f}, selected={'YES' if selected else 'NO'}")
        
        return {
            'fracture_candidates': fracture_candidates_full,
            'confidence_scores': confidence_scores,
            'analyzed_indices': analyzed_indices,
            'excluded_regions': excluded_regions,
            'fracture_likelihoods': fracture_likelihoods.tolist(),
            'combined_scores': combined_scores.tolist(),
            'thresholds': adaptive_thresholds  # Include adaptive thresholds for debugging
        }


class FractureDetectionDebugger:
    """Debug tools for fracture detection system"""
    
    def __init__(self, tri_mesh, region_properties, all_region_features):
        self.tri_mesh = tri_mesh
        self.region_properties = region_properties
        self.all_region_features = all_region_features
        
    def print_detailed_feature_analysis(self):
        """Print detailed analysis of all features for all regions"""
        print(f"\n=== DETAILED FEATURE ANALYSIS ===")
        
        # Separate analyzed regions from pre-classified
        analyzed_features = []
        analyzed_indices = []
        
        for i, region_result in enumerate(self.all_region_features):
            if region_result['classification'] == 'analyzed':
                analyzed_features.append(region_result['features'])
                analyzed_indices.append(i)
            else:
                print(f"Region {i+1}: {region_result['classification']} (confidence: {region_result['confidence']:.2f})")
        
        if len(analyzed_features) < 2:
            print("Not enough regions for statistical analysis")
            return
        
        # Show individual region features
        feature_names = list(analyzed_features[0].keys())
        print(f"\nIndividual Region Features:")
        print(f"{'Region':<8} {'Faces':<8} {'Area%':<8}", end="")
        for fname in feature_names:
            if fname not in ['irregularity_score', 'fracture_likelihood', 'fracture_indicators']:
                print(f"{fname[:8]:<10}", end="")
        print(f"{'FractureLik':<12}")
        
        for i, (analyzed_idx, features) in enumerate(zip(analyzed_indices, analyzed_features)):
            props = self.region_properties[analyzed_idx]
            print(f"{analyzed_idx+1:<8} {props['num_faces']:<8} {props['area_fraction']*100:<6.1f}%  ", end="")
            for fname in feature_names:
                if fname not in ['irregularity_score', 'fracture_likelihood', 'fracture_indicators']:
                    print(f"{features[fname]:<10.4f}", end="")
            fracture_likelihood = features.get('fracture_likelihood', features.get('irregularity_score', 0))
            print(f"{fracture_likelihood:<12.4f}")
        
        # Statistical analysis
        print(f"\nStatistical Analysis:")
        print(f"{'Feature':<15} {'Median':<10} {'MAD':<10} {'Min':<10} {'Max':<10} {'Range':<10}")
        
        for feature_name in feature_names:
            if feature_name in ['irregularity_score', 'fracture_likelihood', 'fracture_indicators']:
                continue
            values = np.array([f[feature_name] for f in analyzed_features])
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            print(f"{feature_name[:14]:<15} {median:<10.4f} {mad:<10.4f} {np.min(values):<10.4f} {np.max(values):<10.4f} {np.max(values)-np.min(values):<10.4f}")
        
        # Outlier analysis
        print(f"\nOutlier Analysis (Modified Z-Score > 2.0):")
        print(f"{'Region':<8} {'Feature':<15} {'Value':<10} {'Z-Score':<10} {'Outlier?':<10}")
        
        for feature_name in feature_names:
            if feature_name in ['irregularity_score', 'fracture_likelihood', 'fracture_indicators']:
                continue
            values = np.array([f[feature_name] for f in analyzed_features])
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            
            if mad > 1e-10:
                modified_z_scores = 0.6745 * (values - median) / mad
                
                for i, (z_score, value, analyzed_idx) in enumerate(zip(modified_z_scores, values, analyzed_indices)):
                    is_outlier = abs(z_score) > 2.0
                    if is_outlier:
                        print(f"{analyzed_idx+1:<8} {feature_name[:14]:<15} {value:<10.4f} {z_score:<10.2f} {'YES':<10}")
    
    def analyze_voting_decisions(self, classification_result):
        """Analyze how the fracture likelihood classification made its decisions"""
        print(f"\n=== FRACTURE LIKELIHOOD DECISION ANALYSIS ===")
        
        if classification_result is None:
            print("No classification result available")
            return
        
        fracture_candidates = classification_result['fracture_candidates']
        confidence_scores = classification_result['confidence_scores']
        analyzed_indices = classification_result['analyzed_indices']
        
        # Show excluded regions first
        if 'excluded_regions' in classification_result:
            print(f"\nExcluded Regions:")
            for excluded in classification_result['excluded_regions']:
                idx = excluded['index'] + 1
                classification = excluded['classification']
                reason = excluded['reason']
                print(f"  Region {idx}: {classification} - {reason}")
        
        # Show fracture likelihood analysis
        if 'fracture_likelihoods' in classification_result and 'combined_scores' in classification_result:
            fracture_likelihoods = classification_result['fracture_likelihoods']
            combined_scores = classification_result['combined_scores']
            
            print(f"\nFracture Likelihood Analysis:")
            print(f"{'Region':<8} {'Faces':<8} {'Area%':<8} {'FracLik':<10} {'Combined':<10} {'Confidence':<10} {'Selected':<10}")
            
            for i, analyzed_idx in enumerate(analyzed_indices):
                if i < len(fracture_likelihoods) and analyzed_idx < len(self.region_properties):
                    props = self.region_properties[analyzed_idx]
                    likelihood = fracture_likelihoods[i]
                    combined = combined_scores[i] if i < len(combined_scores) else 0.0
                    confidence = confidence_scores[i] if i < len(confidence_scores) else 0.0
                    selected = fracture_candidates[analyzed_idx]
                    
                    print(f"{analyzed_idx+1:<8} {props['num_faces']:<8} {props['area_fraction']*100:<6.1f}% "
                          f"{likelihood:<10.3f} {combined:<10.3f} {confidence:<10.3f} {'YES' if selected else 'NO':<10}")
            
            # Show adaptive selection logic
            print(f"\nAdaptive Selection Logic:")
            if 'thresholds' in classification_result:
                thresholds_info = classification_result.get('thresholds', {})
                mesh_complexity = thresholds_info.get('mesh_complexity', 'unknown')
                print(f"- Mesh complexity: {mesh_complexity}")
                print(f"- Thresholds adapted based on mesh baseline properties:")
                print(f"  * Edge CV threshold: >{thresholds_info.get('edge_cv_threshold', 'N/A'):.3f}")
                print(f"  * Dihedral variance threshold: >{thresholds_info.get('dihedral_var_threshold', 'N/A'):.4f}")
                print(f"  * Smoothness penalties: {thresholds_info.get('smooth_thresholds', 'N/A')}")
                print(f"- Classification uses {'OR' if float(mesh_complexity) > 0.4 else 'AND'} logic based on complexity")
            else:
                print("- Adaptive thresholds not available in classification result")
            print("- Area bonus: log(1 + area_fraction * 10) favors larger regions")
        else:
            print("Using legacy statistical outlier method (fracture likelihood not available)")
    
    def suggest_parameter_adjustments(self, classification_result):
        """Suggest parameter adjustments based on results"""
        print(f"\n=== PARAMETER ADJUSTMENT SUGGESTIONS ===")
        
        if classification_result is None:
            print("No classification available - consider lowering region growing thresholds")
            return
        
        analyzed_indices = classification_result['analyzed_indices']
        num_selected = np.sum(classification_result['fracture_candidates'])
        
        # Get analyzed features
        analyzed_features = []
        for i in analyzed_indices:
            if i < len(self.all_region_features) and self.all_region_features[i]['classification'] == 'analyzed':
                analyzed_features.append(self.all_region_features[i]['features'])
        
        if len(analyzed_features) < 2:
            print("Too few regions for analysis - consider:")
            print("  - Lowering 'area_limit_fraction' (currently removes small regions)")
            print("  - Increasing 'max_curvature_deg' (currently splits regions aggressively)")
            return
        
        # Analysis of feature distributions
        feature_names = list(analyzed_features[0].keys())
        low_variance_features = []
        
        for feature_name in feature_names:
            if feature_name in ['irregularity_score', 'fracture_likelihood', 'fracture_indicators']:
                continue
            
            # Get feature values and check if they're numeric
            try:
                values = [f[feature_name] for f in analyzed_features]
                # Skip if any value is not numeric (e.g., dictionary)
                if any(not isinstance(v, (int, float, np.number)) for v in values):
                    continue
                    
                values = np.array(values)
                cv = np.std(values) / (np.mean(values) + 1e-10)  # Coefficient of variation
                if cv < 0.1:  # Very low variation
                    low_variance_features.append(feature_name)
            except (TypeError, ValueError):
                # Skip features that can't be processed numerically
                continue
        
        if len(low_variance_features) > len(feature_names) // 2:
            print("Most features show low variation - suggests:")
            print("  - Regions are too similar (increase segmentation sensitivity)")
            print("  - Mesh might be uniformly rough/smooth")
        
        # Selection ratio analysis
        selection_ratio = num_selected / len(self.region_properties)
        if selection_ratio == 0:
            print("No regions selected - suggestions:")
            print("  - Features may not be discriminative enough")
            print("  - Statistical thresholds may be too strict")
            print("  - Consider lowering outlier threshold from 2.0 to 1.5")
        elif selection_ratio > 0.7:
            print("Most regions selected - suggestions:")
            print("  - Outlier threshold may be too lenient")
            print("  - Consider increasing threshold from 2.0 to 2.5")
        else:
            print(f"Selection ratio: {selection_ratio:.1%} - seems reasonable")
    
    def interactive_region_analysis(self):
        """Allow manual inspection of individual regions"""
        print(f"\n=== INTERACTIVE REGION ANALYSIS ===")
        print("Enter region numbers to analyze (1-based), or 'q' to quit:")
        
        while True:
            try:
                user_input = input("Region to analyze (or 'q'): ").strip().lower()
                if user_input == 'q':
                    break
                
                region_num = int(user_input)
                if region_num < 1 or region_num > len(self.region_properties):
                    print(f"Invalid region number. Valid range: 1-{len(self.region_properties)}")
                    continue
                
                # Get region info
                region_idx = region_num - 1
                props = self.region_properties[region_idx]
                feature_info = self.all_region_features[region_idx] if region_idx < len(self.all_region_features) else None
                
                print(f"\n--- Region {region_num} Analysis ---")
                print(f"Faces: {props['num_faces']}")
                print(f"Area: {props['area_fraction']*100:.2f}% of total")
                print(f"Avg Normal: [{props['avg_normal'][0]:.3f}, {props['avg_normal'][1]:.3f}, {props['avg_normal'][2]:.3f}]")
                
                if feature_info:
                    if feature_info['classification'] == 'analyzed':
                        features = feature_info['features']
                        print(f"Features:")
                        for fname, value in features.items():
                            print(f"  {fname}: {value:.4f}")
                    else:
                        print(f"Classification: {feature_info['classification']} (confidence: {feature_info['confidence']:.2f})")
                
                # Ask user if this looks like a fracture
                user_opinion = input("Does this region look like a fracture surface? (y/n): ").strip().lower()
                if user_opinion in ['y', 'yes']:
                    print("You think this IS a fracture surface")
                elif user_opinion in ['n', 'no']:
                    print("You think this is NOT a fracture surface")
                print()
                
            except ValueError:
                print("Invalid input. Enter a number or 'q'")
            except KeyboardInterrupt:
                print("\nExiting interactive analysis...")
                break

def debug_fracture_detection(tri_mesh, region_properties, params, all_region_features=None, classification_result=None):
    """
    Comprehensive debugging function for fracture detection
    """
    print(f"\n" + "="*60)
    print(f"FRACTURE DETECTION DEBUG SESSION")
    print(f"="*60)
    
    # Calculate adaptive thresholds for debugging
    baseline_props = calculate_mesh_baseline_properties(tri_mesh, region_properties)
    adaptive_thresholds = calculate_adaptive_thresholds(baseline_props)
    
    # Extract features if not provided
    if all_region_features is None:
        adaptive_params = get_adaptive_params(len(tri_mesh.faces))
        adaptive_params['thresholds'] = adaptive_thresholds
        all_region_features = []
        for props in region_properties:
            feature_result = extract_geometric_features_fast(
                tri_mesh, props['faces'], props['area_fraction'], adaptive_params
            )
            all_region_features.append(feature_result)
    
    # Create debugger
    debugger = FractureDetectionDebugger(tri_mesh, region_properties, all_region_features)
    
    # Run all debugging analyses
    debugger.print_detailed_feature_analysis()
    
    if classification_result is None:
        classifier = ScaleInvariantFractureClassifier()
        classification_result = classifier.classify_regions(all_region_features, region_properties, adaptive_thresholds)
    
    if classification_result is not None:
        debugger.analyze_voting_decisions(classification_result)
        debugger.suggest_parameter_adjustments(classification_result)
    
    # Offer interactive region analysis
    if params.get('interactive_debug', False):
        debugger.interactive_region_analysis()
    else:
        print("\nTip: Set 'interactive_debug': true in params to manually inspect regions")
    
    print(f"\n" + "="*60)
    print(f"END DEBUG SESSION")
    print(f"="*60)
    
    return debugger


def calculate_mesh_baseline_properties(tri_mesh, region_properties):
    """
    Calculate mesh-wide baseline properties to set adaptive thresholds
    """
    print(f"    Calculating mesh baseline properties...")
    
    # Sample faces from across the mesh for baseline calculation
    total_faces = len(tri_mesh.faces)
    sample_size = min(1000, total_faces // 10)  # Sample 10% or max 1000 faces
    sampled_faces = np.random.choice(total_faces, size=sample_size, replace=False)
    
    # Calculate baseline geometric properties
    baseline_edge_cv = compute_edge_length_cv_fast(tri_mesh, sampled_faces)
    baseline_dihedral_var = compute_dihedral_variance_fast(tri_mesh, sampled_faces)
    baseline_quality = compute_triangle_quality_metrics_fast(tri_mesh, sampled_faces)
    baseline_normal_consistency = compute_normal_consistency_fast(tri_mesh, sampled_faces)
    
    # Also analyze region-level properties
    region_sizes = [props['area_fraction'] for props in region_properties]
    region_face_counts = [props['num_faces'] for props in region_properties]
    
    baseline_props = {
        'edge_cv': baseline_edge_cv,
        'dihedral_variance': baseline_dihedral_var,
        'area_cv': baseline_quality['area_cv'],
        'aspect_ratio_std': baseline_quality['aspect_ratio_std'],
        'normal_consistency': baseline_normal_consistency,
        'typical_region_size': np.median(region_sizes),
        'typical_region_faces': np.median(region_face_counts),
        'total_faces': total_faces
    }
    
    print(f"    Baseline properties: edge_cv={baseline_edge_cv:.3f}, "
          f"dihedral_var={baseline_dihedral_var:.4f}, normal_consistency={baseline_normal_consistency:.3f}")
    
    return baseline_props


def calculate_adaptive_thresholds(baseline_props):
    """
    Calculate adaptive thresholds based on mesh baseline properties
    """
    # Surface roughness thresholds (more sensitive - closer to baseline)
    edge_cv_threshold = max(0.05, baseline_props['edge_cv'] * 0.9)  # 10% BELOW baseline (more sensitive)
    dihedral_var_threshold = max(0.002, baseline_props['dihedral_variance'] * 0.8)  # 20% BELOW baseline (more sensitive)  
    area_cv_threshold = max(0.15, baseline_props['area_cv'] * 0.9)  # 10% BELOW baseline (more sensitive)
    
    # Smoothness detection (adaptive to mesh's normal consistency)
    baseline_normal_consistency = baseline_props['normal_consistency']
    
    # Adaptive smoothness penalties based on mesh's natural consistency
    if baseline_normal_consistency > 0.95:  # Very consistent mesh overall
        smooth_thresholds = {'very_smooth': 0.98, 'quite_smooth': 0.96, 'penalties': {'very': 0.4, 'quite': 0.2}}
    elif baseline_normal_consistency > 0.90:  # Moderately consistent mesh
        smooth_thresholds = {'very_smooth': 0.95, 'quite_smooth': 0.92, 'penalties': {'very': 0.5, 'quite': 0.25}}
    else:  # Naturally rough/inconsistent mesh
        smooth_thresholds = {'very_smooth': 0.92, 'quite_smooth': 0.88, 'penalties': {'very': 0.6, 'quite': 0.3}}
    
    # Region size considerations
    min_fracture_size = min(0.05, baseline_props['typical_region_size'] * 0.5)  # Smaller than typical region
    
    # Classification thresholds (adaptive to mesh complexity)
    mesh_complexity = (baseline_props['edge_cv'] + baseline_props['dihedral_variance'] * 10 + 
                      baseline_props['area_cv']) / 3
    
    if mesh_complexity > 0.4:  # Complex/rough mesh
        classification_thresholds = {'min_likelihood': 0.1, 'min_combined': 0.3, 'percentile_likelihood': 50, 'percentile_combined': 40}
    elif mesh_complexity > 0.2:  # Moderate mesh  
        classification_thresholds = {'min_likelihood': 0.12, 'min_combined': 0.35, 'percentile_likelihood': 60, 'percentile_combined': 50}
    else:  # Simple/smooth mesh
        classification_thresholds = {'min_likelihood': 0.15, 'min_combined': 0.4, 'percentile_likelihood': 70, 'percentile_combined': 60}
    
    adaptive_thresholds = {
        'edge_cv_threshold': edge_cv_threshold,
        'dihedral_var_threshold': dihedral_var_threshold,
        'area_cv_threshold': area_cv_threshold,
        'smooth_thresholds': smooth_thresholds,
        'min_fracture_size': min_fracture_size,
        'classification': classification_thresholds,
        'mesh_complexity': mesh_complexity
    }
    
    print(f"    Adaptive thresholds (SENSITIVE): edge_cv>{edge_cv_threshold:.3f}, dihedral_var>{dihedral_var_threshold:.4f}")
    print(f"    Mesh complexity: {mesh_complexity:.3f}, classification min_likelihood: {classification_thresholds['min_likelihood']:.3f}")
    print(f"    Feature thresholds are now BELOW baseline for sensitivity!")
    
    return adaptive_thresholds


def detect_fracture_regions_geometric(tri_mesh, region_properties, params):
    """
    Main geometric fracture detection function using adaptive, mesh-aware thresholds
    """
    print(f"    Starting geometric fracture detection on {len(region_properties)} regions...")
    
    # Calculate mesh baseline properties for adaptive thresholds
    baseline_props = calculate_mesh_baseline_properties(tri_mesh, region_properties)
    adaptive_thresholds = calculate_adaptive_thresholds(baseline_props)
    
    # Get adaptive parameters based on mesh size
    adaptive_params = get_adaptive_params(len(tri_mesh.faces))
    adaptive_params['thresholds'] = adaptive_thresholds  # Add adaptive thresholds
    print(f"    Adaptive params for {len(tri_mesh.faces)} faces: {adaptive_params}")
    
    # Extract features for all regions (now with adaptive thresholds)
    all_region_features = []
    
    for i, props in enumerate(region_properties):
        region_faces = props['faces']
        region_area_fraction = props['area_fraction']
        
        feature_result = extract_geometric_features_fast(
            tri_mesh, region_faces, region_area_fraction, adaptive_params
        )
        
        all_region_features.append(feature_result)
        
        # Progress feedback for large meshes
        if adaptive_params['batch_processing'] and (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(region_properties)} regions...")
    
    # Classify regions using adaptive fracture likelihood analysis
    classifier = ScaleInvariantFractureClassifier()
    classification_result = classifier.classify_regions(all_region_features, region_properties, adaptive_thresholds)
    
    if classification_result is None:
        print(f"    Not enough regions for statistical analysis, falling back to interactive selection")
        # Run debug analysis even when classification fails
        if params.get('debug_fracture_detection', False):
            debug_fracture_detection(tri_mesh, region_properties, params, all_region_features, None)
        return None  # Fall back to interactive selection
    
    # Report results
    fracture_candidates = classification_result['fracture_candidates'] 
    confidence_scores = classification_result['confidence_scores']
    analyzed_indices = classification_result['analyzed_indices']
    num_selected = np.sum(fracture_candidates)
    
    print(f"    Geometric detection selected {num_selected}/{len(region_properties)} regions as fracture candidates")
    print(f"    DEBUG: Analyzed {len(analyzed_indices)} regions, confidence scores: {len(confidence_scores)}")
    
    # Show selected regions with confidence  
    for i, (is_fracture, props) in enumerate(zip(fracture_candidates, region_properties)):
        if is_fracture:
            # Find the confidence score for this region
            try:
                analyzed_idx_position = analyzed_indices.index(i) if i in analyzed_indices else -1
                if analyzed_idx_position >= 0 and analyzed_idx_position < len(confidence_scores):
                    confidence = confidence_scores[analyzed_idx_position]
                    print(f"    Region {i+1}: FRACTURE (confidence: {confidence:.2f})")
                else:
                    print(f"    Region {i+1}: FRACTURE (pre-selected)")
            except (ValueError, IndexError) as e:
                print(f"    Region {i+1}: FRACTURE (confidence: N/A, debug: {e})")
    
    # Run detailed debugging if requested
    if params.get('debug_fracture_detection', False):
        debug_fracture_detection(tri_mesh, region_properties, params, all_region_features, classification_result)
    
    return fracture_candidates


def region_growing_segmentation(tri_mesh, params):
    """
    Implements the region growing algorithm from the paper.
    
    Args:
        tri_mesh: trimesh object
        params: dictionary containing:
            - 'max_curvature_deg': maximum allowed angle between normals in same region (default 30)
            - 'area_limit_fraction': minimum region area as fraction of total (default 0.02)
    
    Returns:
        list of np.arrays containing face indices for each region
    """
    # Get parameters
    max_curvature_deg = params.get('max_curvature_deg', 30.0)
    area_limit_fraction = params.get('area_limit_fraction', 0.02)
    
    # Calculate Ne threshold from max curvature (Ne = cos(q_max))
    Ne = np.cos(np.radians(max_curvature_deg))
    
    num_faces = len(tri_mesh.faces)
    face_visited = np.zeros(num_faces, dtype=bool)
    regions = []
    
    # Precompute face adjacency if not available
    if not hasattr(tri_mesh, 'face_adjacency') or tri_mesh.face_adjacency is None:
        tri_mesh.face_adjacency = trimesh.graph.face_adjacency(tri_mesh.faces)
    
    # Build adjacency list for faster lookup
    adjacency_list = [[] for _ in range(num_faces)]
    for face1, face2 in tri_mesh.face_adjacency:
        adjacency_list[face1].append(face2)
        adjacency_list[face2].append(face1)
    
    # Region growing main loop
    for start_face in range(num_faces):
        if face_visited[start_face]:
            continue
            
        # Start new region
        current_region = []
        queue = deque([start_face])
        face_visited[start_face] = True
        
        while queue:
            current_face = queue.popleft()
            current_region.append(current_face)
            
            # Update region average normal
            region_avg_normal = calculate_region_average_normal(tri_mesh, current_region)
            
            # Check all neighbors
            for neighbor_face in adjacency_list[current_face]:
                if face_visited[neighbor_face]:
                    continue
                
                # Check if neighbor normal satisfies similarity criterion
                neighbor_normal = tri_mesh.face_normals[neighbor_face]
                dot_product = np.dot(neighbor_normal, region_avg_normal)
                
                if dot_product >= Ne:  # N_i · N_ave(R_k) >= Ne
                    face_visited[neighbor_face] = True
                    queue.append(neighbor_face)
        
        if len(current_region) > 0:
            regions.append(np.array(current_region))
    
    # Clean-up stage: eliminate small regions
    total_area = tri_mesh.area
    area_threshold = area_limit_fraction * total_area
    
    # Calculate region areas
    region_areas = []
    for region in regions:
        region_area = np.sum(tri_mesh.area_faces[region])
        region_areas.append(region_area)
    
    # Sort regions by area (largest first)
    sorted_indices = np.argsort(region_areas)[::-1]
    sorted_regions = [regions[i] for i in sorted_indices]
    sorted_areas = [region_areas[i] for i in sorted_indices]
    
    # Keep only significant regions
    significant_regions = []
    for i, (region, area) in enumerate(zip(sorted_regions, sorted_areas)):
        if area >= area_threshold:
            significant_regions.append(region)
    
    # Reassign small regions to adjacent larger regions
    if len(significant_regions) < len(regions):
        # Create a face-to-region mapping for significant regions
        face_to_region = np.full(num_faces, -1, dtype=int)
        for region_idx, region in enumerate(significant_regions):
            face_to_region[region] = region_idx
        
        # Process small regions
        for region_idx in sorted_indices:
            region = regions[region_idx]
            area = region_areas[region_idx]
            
            if area >= area_threshold:
                continue
            
            # Find adjacent significant regions
            adjacent_regions = set()
            for face in region:
                for neighbor in adjacency_list[face]:
                    neighbor_region = face_to_region[neighbor]
                    if neighbor_region >= 0:
                        adjacent_regions.add(neighbor_region)
            
            # Assign to the most similar adjacent region
            if adjacent_regions:
                best_region = None
                best_similarity = -1
                region_avg_normal = calculate_region_average_normal(tri_mesh, region)
                
                for adj_region_idx in adjacent_regions:
                    adj_region_normal = calculate_region_average_normal(
                        tri_mesh, significant_regions[adj_region_idx]
                    )
                    similarity = np.dot(region_avg_normal, adj_region_normal)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_region = adj_region_idx
                
                if best_region is not None:
                    # Merge with best region
                    significant_regions[best_region] = np.concatenate([
                        significant_regions[best_region], region
                    ])
                    face_to_region[region] = best_region
    
    return significant_regions


def extract_fracture_surface_mesh(o3d_mesh_fragment, fragment_name="Unnamed", params=None):
    """
    Main segmentation function using the paper's region growing approach.
    """
    params = params or {}
    
    print(f"\n=== Segmenting {fragment_name} using Region Growing Algorithm ===")
    
    # Parameter setup with paper's recommendations
    default_params = {
        'max_curvature_deg': params.get('max_curvature_deg', 30.0),  # Paper suggests this range
        'area_limit_fraction': params.get('area_limit_fraction', 0.02),  # 2% as paper suggests
        'visualize_segmentation': params.get('visualize_segmentation', False),
        'use_geometric_detection': params.get('use_geometric_detection', True),  # New geometric feature system
    }
    
    # Update params with defaults
    for key, value in default_params.items():
        if key not in params:
            params[key] = value
    
    # Convert to trimesh
    if not o3d_mesh_fragment.has_triangles() or not o3d_mesh_fragment.has_vertices():
        print(f"    Segmenter: Input mesh {fragment_name} has no triangles/vertices.")
        return None
        
    try:
        tri_mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh_fragment.vertices),
            faces=np.asarray(o3d_mesh_fragment.triangles),
            vertex_normals=np.asarray(o3d_mesh_fragment.vertex_normals) if o3d_mesh_fragment.has_vertex_normals() else None,
            process=False
        )
        tri_mesh.metadata['name'] = fragment_name
        
        # Ensure we have face normals and areas
        if not hasattr(tri_mesh, 'face_normals') or tri_mesh.face_normals is None:
            tri_mesh.face_normals
        if not hasattr(tri_mesh, 'area_faces') or tri_mesh.area_faces is None:
            _ = tri_mesh.area_faces
            
    except Exception as e:
        print(f"    Segmenter: Error converting O3D mesh {fragment_name} to Trimesh: {e}")
        return None
    
    total_faces = len(tri_mesh.faces)
    print(f"    Total faces: {total_faces}")
    print(f"    Max curvature threshold: {params['max_curvature_deg']}°")
    print(f"    Min region area: {params['area_limit_fraction']*100:.1f}% of total")
    
    # Perform region growing segmentation
    print(f"\n    Starting region growing segmentation...")
    regions = region_growing_segmentation(tri_mesh, params)
    print(f"    Found {len(regions)} regions after segmentation and cleanup")
    
    # Calculate region properties
    region_properties = []
    for i, region in enumerate(regions):
        avg_normal = calculate_region_average_normal(tri_mesh, region)
        area = np.sum(tri_mesh.area_faces[region])
        area_fraction = area / tri_mesh.area
        
        props = {
            'index': i,
            'faces': region,
            'num_faces': len(region),
            'area': area,
            'area_fraction': area_fraction,
            'avg_normal': avg_normal
        }
        
        region_properties.append(props)
        
        print(f"    Region {i+1}: {len(region)} faces ({area_fraction*100:.1f}% of area), "
              f"avg_normal: [{avg_normal[0]:.2f}, {avg_normal[1]:.2f}, {avg_normal[2]:.2f}]")
    
    # Sort regions by area (largest first)
    region_properties.sort(key=lambda x: x['area'], reverse=True)
    
    # Identify fracture candidates using geometric features
    face_is_fracture_candidate = np.zeros(len(tri_mesh.faces), dtype=bool)
    selected_regions = []
    
    # Try geometric detection first (new system)
    geometric_features_info = None
    if params.get('use_geometric_detection', True):
        # Get adaptive parameters and extract features for visualization
        adaptive_params = get_adaptive_params(len(tri_mesh.faces))
        
        # Extract features for all regions (for visualization)
        geometric_features_info = []
        for i, props in enumerate(region_properties):
            feature_result = extract_geometric_features_fast(
                tri_mesh, props['faces'], props['area_fraction'], adaptive_params
            )
            geometric_features_info.append(feature_result)
        
        geometric_result = detect_fracture_regions_geometric(tri_mesh, region_properties, params)
        
        if geometric_result is not None:
            # Geometric detection succeeded
            for i, is_fracture in enumerate(geometric_result):
                if is_fracture:
                    selected_regions.append(i)
                    face_is_fracture_candidate[region_properties[i]['faces']] = True
            
            print(f"    Geometric detection completed: {len(selected_regions)} regions selected")
        else:
            print(f"    Geometric detection insufficient, falling back to interactive selection")
    
    # Interactive visualization if enabled
    if params['visualize_segmentation'] and len(regions) > 0:
        print(f"\n    Visualizing {len(regions)} regions for interactive selection...")
        
        shared_state = {'confirmed_selection': False, 'quit_without_selection': False, 'current_page': 0}
        PAGE_SIZE = 10
        
        drawable_segment_infos = []
        highlight_color = np.array([0.0, 0.0, 0.0])  # Black highlight
        
        mesh_vis = copy.deepcopy(o3d_mesh_fragment)
        
        for i, props in enumerate(region_properties):
            seg_mesh = o3d.geometry.TriangleMesh()
            seg_mesh.vertices = o3d_mesh_fragment.vertices
            seg_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces[props['faces']])
            seg_mesh.remove_unreferenced_vertices()
            
            if not seg_mesh.has_vertices() or not seg_mesh.has_triangles():
                continue
                
            seg_mesh.compute_vertex_normals()
            base_color = get_color(i, len(regions))
            seg_mesh.paint_uniform_color(base_color)
            
            # Add feature information if available
            feature_info = None
            if geometric_features_info and i < len(geometric_features_info):
                feature_info = geometric_features_info[i]
            
            drawable_segment_infos.append({
                'mesh': seg_mesh,
                'id': props['index'],
                'base_color': base_color,
                'selected': props['index'] in selected_regions,
                'properties': props,
                'features': feature_info
            })
        
        if drawable_segment_infos:
            num_total_segments = len(drawable_segment_infos)
            num_pages = (num_total_segments + PAGE_SIZE - 1) // PAGE_SIZE
            
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(
                window_name=f"Select: {fragment_name} (Page 1/{num_pages}. N/P=Page. S=Confirm. Q=Skip.)",
                width=1280, height=960
            )
            
            for info in drawable_segment_infos:
                vis.add_geometry(info['mesh'])
                if info['selected']:
                    info['mesh'].paint_uniform_color(highlight_color)
            
            def print_current_page_and_selection():
                page_idx = shared_state['current_page']
                global_start = page_idx * PAGE_SIZE + 1
                global_end = min((page_idx + 1) * PAGE_SIZE, num_total_segments)
                
                print(f"\n  --- Page {page_idx + 1}/{num_pages} (Regions {global_start}-{global_end}) ---")
                print(f"  Keys 1-9, 0 (for 10th) toggle selection.")
                
                # Show properties for visible regions
                for i in range(page_idx * PAGE_SIZE, min((page_idx + 1) * PAGE_SIZE, num_total_segments)):
                    if i < len(drawable_segment_infos):
                        info = drawable_segment_infos[i]
                        props = info['properties']
                        selected_marker = "*" if info['selected'] else " "
                        print(f"  {selected_marker}[{(i % PAGE_SIZE) + 1}] Region {props['index']+1}: "
                              f"{props['num_faces']} faces ({props['area_fraction']*100:.1f}%)")
                        
                        # Show feature information if available
                        if info.get('features') and info['features']['classification'] == 'analyzed':
                            features = info['features']['features']
                            print(f"       Features: EdgeCV={features.get('edge_length_cv', 0):.3f}, "
                                  f"DihedralVar={features.get('dihedral_variance', 0):.3f}, "
                                  f"Irregularity={features.get('irregularity_score', 0):.3f}")
                        elif info.get('features'):
                            classification = info['features']['classification']
                            confidence = info['features']['confidence']
                            print(f"       Status: {classification} (confidence: {confidence:.2f})")
                
                selected_ids = sorted([info['id'] + 1 for info in drawable_segment_infos if info['selected']])
                print(f"  Selected: {selected_ids if selected_ids else 'None'}")
            
            print_current_page_and_selection()
            
            def toggle_segment_on_current_page(visualizer, key_idx):
                page_idx = shared_state['current_page']
                segment_idx = page_idx * PAGE_SIZE + key_idx
                
                if 0 <= segment_idx < num_total_segments:
                    info = drawable_segment_infos[segment_idx]
                    info['selected'] = not info['selected']
                    
                    if info['selected']:
                        info['mesh'].paint_uniform_color(highlight_color)
                    else:
                        info['mesh'].paint_uniform_color(info['base_color'])
                    
                    visualizer.update_geometry(info['mesh'])
                    print_current_page_and_selection()
                    
                return False
            
            # Register key callbacks
            for i in range(PAGE_SIZE):
                key_char = str((i + 1) % 10)
                vis.register_key_callback(ord(key_char), 
                    lambda v, idx=i: toggle_segment_on_current_page(v, idx))
            
            def change_page(visualizer, direction):
                old_page = shared_state['current_page']
                shared_state['current_page'] = (shared_state['current_page'] + direction + num_pages) % num_pages
                if old_page != shared_state['current_page']:
                    print_current_page_and_selection()
                return False
            
            vis.register_key_callback(ord('N'), lambda v: change_page(v, 1))
            vis.register_key_callback(ord('P'), lambda v: change_page(v, -1))
            
            def confirm_and_close(visualizer):
                shared_state['confirmed_selection'] = True
                print("\n  Selection Confirmed. Closing...")
                visualizer.close()
                return False
            
            def quit_and_close(visualizer):
                shared_state['quit_without_selection'] = True
                print("\n  Selection Aborted. Closing...")
                visualizer.close()
                return False
            
            vis.register_key_callback(ord('S'), confirm_and_close)
            vis.register_key_callback(ord('Q'), quit_and_close)
            
            print("\n=== Interactive Region Selection ===")
            print(f"  Fragment: {fragment_name}")
            print("  N/P: Navigate pages | 1-9,0: Toggle selection")
            print("  S: Save selection | Q: Quit without saving")
            
            vis.run()
            vis.destroy_window()
            
            if shared_state['confirmed_selection']:
                selected_regions = [info['id'] for info in drawable_segment_infos if info['selected']]
                face_is_fracture_candidate.fill(False)
                for info in drawable_segment_infos:
                    if info['selected']:
                        face_is_fracture_candidate[info['properties']['faces']] = True
                print(f"\n    User selected {len(selected_regions)} regions")
            elif shared_state['quit_without_selection']:
                print(f"\n    User quit selection. No regions selected.")
                return None
    
    # Console fallback for non-interactive mode (when geometric detection didn't select anything)
    elif not params['visualize_segmentation'] and len(regions) > 0 and len(selected_regions) == 0:
        print("\n=== Region Selection (Console) ===")
        for i, props in enumerate(region_properties):
            print(f"  Region {i+1}: {props['num_faces']} faces ({props['area_fraction']*100:.1f}% of area)")
        
        selection_str = input(f"Enter region numbers to select (1-{len(regions)}, comma-separated, 'all', or 'none'): ")
        
        if selection_str.lower() == 'all':
            selected_regions = list(range(len(regions)))
        elif selection_str.lower() == 'none' or not selection_str.strip():
            selected_regions = []
        else:
            try:
                selected_regions = [int(x.strip()) - 1 for x in selection_str.split(',') if x.strip()]
                selected_regions = [r for r in selected_regions if 0 <= r < len(regions)]
            except ValueError:
                print("    Invalid input. No regions selected.")
                selected_regions = []
        
        for region_idx in selected_regions:
            face_is_fracture_candidate[region_properties[region_idx]['faces']] = True
    
    # Collect selected regions' face indices and normals for merging
    selected_region_faces = []
    selected_region_normals = []
    for region_idx in range(len(region_properties)):
        if face_is_fracture_candidate[region_properties[region_idx]['faces']].any():
            selected_region_faces.append(set(region_properties[region_idx]['faces']))
            selected_region_normals.append(region_properties[region_idx]['avg_normal'])
    
    # Create output mesh
    if not np.any(face_is_fracture_candidate):
        print(f"\n    No regions selected for {fragment_name}")
        return None
    
    fracture_faces = tri_mesh.faces[face_is_fracture_candidate]
    fracture_surface_o3d = o3d.geometry.TriangleMesh()
    fracture_surface_o3d.vertices = o3d_mesh_fragment.vertices
    fracture_surface_o3d.triangles = o3d.utility.Vector3iVector(fracture_faces)
    fracture_surface_o3d.remove_unreferenced_vertices()
    fracture_surface_o3d.remove_degenerate_triangles()
    
    if not fracture_surface_o3d.has_triangles():
        print(f"    Extracted surface has no valid triangles")
        return None
    
    fracture_surface_o3d.compute_vertex_normals()
    print(f"\n    Extracted surface: {len(fracture_surface_o3d.vertices)} vertices, "
          f"{len(fracture_surface_o3d.triangles)} triangles")
    
    # --- IMPROVED MERGING: NORMAL + BOUNDARY DISTANCE ---
    all_triangles = np.asarray(o3d_mesh_fragment.triangles)
    all_vertices = np.asarray(o3d_mesh_fragment.vertices)
    def get_boundary_vertices(face_indices):
        # Find boundary edges for a set of faces
        faces = all_triangles[list(face_indices)]
        edges = np.vstack([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]])
        edges = np.sort(edges, axis=1)
        # Count occurrences
        edges_tuple = [tuple(e) for e in edges]
        from collections import Counter
        edge_counts = Counter(edges_tuple)
        boundary_edges = [e for e, c in edge_counts.items() if c == 1]
        boundary_verts = np.unique(np.array(boundary_edges).flatten())
        return all_vertices[boundary_verts]
    
    merged_clusters = []
    used = set()
    angle_thresh_deg = 10.0
    boundary_dist_thresh = 0.01 * np.linalg.norm(all_vertices.max(axis=0) - all_vertices.min(axis=0))  # 1% of mesh size
    for i, (faces_i, normal_i) in enumerate(zip(selected_region_faces, selected_region_normals)):
        if i in used:
            continue
        cluster = set(faces_i)
        cluster_normals = [normal_i]
        merged_idxs = [i]
        used.add(i)
        boundary_i = get_boundary_vertices(faces_i)
        for j, (faces_j, normal_j) in enumerate(zip(selected_region_faces, selected_region_normals)):
            if j == i or j in used:
                continue
            angle = np.degrees(np.arccos(np.clip(np.dot(normal_i, normal_j), -1, 1)))
            if angle < angle_thresh_deg and np.dot(normal_i, normal_j) > 0:
                # Check boundary proximity
                boundary_j = get_boundary_vertices(faces_j)
                if len(boundary_i) > 0 and len(boundary_j) > 0:
                    tree = cKDTree(boundary_j)
                    min_dist = tree.query(boundary_i, k=1)[0].min()
                    if min_dist < boundary_dist_thresh:
                        cluster |= faces_j
                        cluster_normals.append(normal_j)
                        merged_idxs.append(j)
                        used.add(j)
        merged_clusters.append((cluster, merged_idxs))
    # DEBUG: Visualize each merged face in its own window, with base mesh as wireframe
    for idx, (cluster_faces, merged_idxs) in enumerate(merged_clusters):
        if not cluster_faces:
            continue
        cluster_faces_arr = np.array(sorted(cluster_faces), dtype=np.int32)
        cluster_triangles = all_triangles[cluster_faces_arr]
        cluster_mesh = o3d.geometry.TriangleMesh()
        cluster_mesh.vertices = mesh_vis.vertices  # Use full vertex set
        cluster_mesh.triangles = o3d.utility.Vector3iVector(cluster_triangles)
        cluster_mesh.compute_vertex_normals()
        color = get_color(idx, len(merged_clusters))
        cluster_mesh.paint_uniform_color(color)
        print(f"[DEBUG] Merged regions {merged_idxs} into face {idx} (color {idx}), triangles: {len(cluster_triangles)}, vertices: {len(cluster_mesh.vertices)}, color: {color}")
        # Create wireframe for base mesh
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_vis)
        wireframe.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([wireframe, cluster_mesh], window_name=f"[DEBUG] Merged Face {idx} (Color {idx})")
    # Store merged faces as separate fracture surfaces for downstream processing
    merged_fracture_surfaces = []
    for cluster_faces, merged_idxs in merged_clusters:
        if not cluster_faces:
            continue
        cluster_faces_arr = np.array(sorted(cluster_faces), dtype=np.int32)
        cluster_triangles = all_triangles[cluster_faces_arr]
        cluster_mesh = o3d.geometry.TriangleMesh()
        cluster_mesh.vertices = mesh_vis.vertices
        cluster_mesh.triangles = o3d.utility.Vector3iVector(cluster_triangles)
        cluster_mesh.remove_unreferenced_vertices()
        cluster_mesh.remove_degenerate_triangles()
        if cluster_mesh.has_triangles():
            cluster_mesh.compute_vertex_normals()
            merged_fracture_surfaces.append(cluster_mesh)
    # Return merged fracture surfaces for this fragment
    return merged_fracture_surfaces


def visualize_segmentation(o3d_mesh, fracture_surface, fragment_name="Unnamed"):
    """
    Creates a visualization of the original mesh and the extracted surface.
    """
    vis_geometries = []
    
    # Original mesh in gray
    original_mesh_vis = copy.deepcopy(o3d_mesh)
    original_mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])
    original_mesh_vis.compute_vertex_normals()
    vis_geometries.append(original_mesh_vis)
    
    # Wireframe for structure
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(original_mesh_vis)
    edges.paint_uniform_color([0.5, 0.5, 0.5])
    vis_geometries.append(edges)
    
    # Selected surface in red
    if fracture_surface and fracture_surface.has_triangles():
        fracture_surface_vis = copy.deepcopy(fracture_surface)
        fracture_surface_vis.paint_uniform_color([1.0, 0.0, 0.0])
        fracture_surface_vis.compute_vertex_normals()
        vis_geometries.append(fracture_surface_vis)
    
    return vis_geometries


# Maintain compatibility with old function names
def identify_fracture_candidate_faces(tri_mesh_fragment, params=None):
    """
    Legacy function maintained for compatibility.
    Returns a boolean mask of fracture candidate faces.
    """
    if params is None:
        params = {}
    
    # Convert trimesh to Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh_fragment.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh_fragment.faces)
    o3d_mesh.compute_vertex_normals()
    
    # Run segmentation
    result_mesh = extract_fracture_surface_mesh(
        o3d_mesh, 
        tri_mesh_fragment.metadata.get('name', 'Unnamed'),
        params
    )
    
    if result_mesh is None:
        return np.zeros(len(tri_mesh_fragment.faces), dtype=bool)
    
    # Create boolean mask
    face_mask = np.zeros(len(tri_mesh_fragment.faces), dtype=bool)
    result_faces_set = set(map(tuple, np.asarray(result_mesh.triangles)))
    
    for i, face in enumerate(tri_mesh_fragment.faces):
        if tuple(sorted(face)) in result_faces_set or tuple(face) in result_faces_set:
            face_mask[i] = True
    
    return face_mask


if __name__ == '__main__':
    # Test with a simple cube
    print("Testing region growing segmentation on a cube...")
    test_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    test_mesh.compute_vertex_normals()
    
    # Test parameters
    test_params = {
        'max_curvature_deg': 45.0,
        'area_limit_fraction': 0.1,
        'visualize_segmentation': True,
        'use_bumpiness_detection': False
    }
    
    # Run segmentation
    result = extract_fracture_surface_mesh(test_mesh, "TestCube", test_params)
    
    if result:
        # Visualize results
        vis_geometries = visualize_segmentation(test_mesh, result, "TestCube")
        o3d.visualization.draw_geometries(vis_geometries, window_name="Region Growing Segmentation Test")