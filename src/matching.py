import numpy as np
import copy
import time
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.alignment import align_fragments_papaioannou, PapaioannoualignmentError

print("DEBUG: enhanced matching.py (Papaioannou method) loaded")


class ConstraintManager:
    """
    Manages various matching constraints as described in Papaioannou et al.
    """
    
    def __init__(self):
        self.material_axes = {}  # fragment_name -> material_axis_vector
        self.fracture_directions = {}  # fragment_name -> list of fracture_direction_vectors
        self.directional_tolerances = {}  # fragment_name -> tolerance values
    
    def add_material_axis_constraint(self, fragment_name, material_axis, tolerance=np.pi/36):
        """
        Add material axis constraint for a fragment.
        
        Args:
            fragment_name: Name of the fragment
            material_axis: 3D vector representing material grain direction
            tolerance: Angular tolerance for material axis alignment
        """
        self.material_axes[fragment_name] = {
            'axis': np.array(material_axis) / np.linalg.norm(material_axis),
            'tolerance': tolerance
        }
    
    def add_fracture_direction_constraint(self, fragment_name, fracture_directions, tolerance=np.pi/18):
        """
        Add fracture direction constraints for a fragment.
        
        Args:
            fragment_name: Name of the fragment
            fracture_directions: List of 3D vectors representing fracture surface normals
            tolerance: Angular tolerance for fracture direction alignment
        """
        normalized_directions = []
        for direction in fracture_directions:
            normalized_directions.append(np.array(direction) / np.linalg.norm(direction))
        
        self.fracture_directions[fragment_name] = {
            'directions': normalized_directions,
            'tolerance': tolerance
        }
    
    def get_constraints_for_pair(self, source_name, target_name):
        """
        Get applicable constraints for a fragment pair.
        
        Returns:
            dict: Constraint configuration for the pair
        """
        constraints = {}
        
        # Material axis constraints
        if source_name in self.material_axes and target_name in self.material_axes:
            constraints['material_axis_constraint'] = True
            constraints['material_axis_tolerance'] = min(
                self.material_axes[source_name]['tolerance'],
                self.material_axes[target_name]['tolerance']
            )
            constraints['source_material_axis'] = self.material_axes[source_name]['axis']
            constraints['target_material_axis'] = self.material_axes[target_name]['axis']
        
        # Fracture direction constraints
        if source_name in self.fracture_directions and target_name in self.fracture_directions:
            constraints['fracture_direction_constraint'] = True
            constraints['fracture_angle_tolerance'] = min(
                self.fracture_directions[source_name]['tolerance'],
                self.fracture_directions[target_name]['tolerance']
            )
            constraints['source_fracture_directions'] = self.fracture_directions[source_name]['directions']
            constraints['target_fracture_directions'] = self.fracture_directions[target_name]['directions']
        
        return constraints


class BiasedMatchingError:
    """
    Implements biased matching error with surface overlap and material constraints.
    """
    
    def __init__(self, base_error_calculator):
        self.base_calc = base_error_calculator
    
    def calculate_biased_error(self, source_mesh, target_mesh, transform_params, constraints=None, bias_weights=None):
        """
        Calculate biased matching error incorporating additional constraints.
        
        Args:
            source_mesh, target_mesh: Mesh objects
            transform_params: Transformation parameters
            constraints: Constraint configuration
            bias_weights: Weights for different bias terms
        
        Returns:
            biased_error: Total error including bias terms
        """
        if bias_weights is None:
            bias_weights = {'geometric': 1.0, 'material': 0.0, 'overlap': 0.0}
        
        # Calculate base geometric error
        Z1, Z2, valid_mask = self.base_calc.calculate_zbuffer_distances(
            source_mesh, target_mesh, transform_params
        )
        geometric_error = self.base_calc.calculate_matching_error(Z1, Z2, valid_mask)
        
        total_error = bias_weights['geometric'] * geometric_error
        
        # Add material axis bias
        if constraints and constraints.get('material_axis_constraint') and bias_weights['material'] > 0:
            material_error = self._calculate_material_axis_error(transform_params, constraints)
            total_error += bias_weights['material'] * material_error
        
        # Add surface overlap bias
        if bias_weights['overlap'] > 0:
            overlap_error = self._calculate_overlap_bias(Z1, Z2, valid_mask)
            total_error += bias_weights['overlap'] * overlap_error
        
        return total_error
    
    def _calculate_material_axis_error(self, transform_params, constraints):
        """Calculate material axis alignment error."""
        # This would require applying the transformation to the material axes
        # and measuring their angular deviation
        # Simplified implementation:
        source_axis = constraints['source_material_axis']
        target_axis = constraints['target_material_axis']
        
        # Apply transformation to source axis (simplified)
        theta1, phi1, rho1, x1, y1, theta2, phi2 = transform_params
        
        # Create rotation matrices for source
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta1), -np.sin(theta1)],
                       [0, np.sin(theta1), np.cos(theta1)]])
        
        Ry = np.array([[np.cos(phi1), 0, np.sin(phi1)],
                       [0, 1, 0],
                       [-np.sin(phi1), 0, np.cos(phi1)]])
        
        transformed_source_axis = Ry @ Rx @ source_axis
        
        # Calculate alignment error (1 - |dot product|)
        alignment = abs(np.dot(transformed_source_axis, target_axis))
        return 1.0 - alignment
    
    def _calculate_overlap_bias(self, Z1, Z2, valid_mask):
        """Calculate surface overlap bias to favor maximum overlap."""
        total_pixels = Z1.size
        valid_pixels = np.sum(valid_mask)
        
        if total_pixels == 0:
            return 1.0
        
        # Encourage maximum overlap
        overlap_ratio = valid_pixels / total_pixels
        return 1.0 - overlap_ratio


def _match_fragment_pair_papaioannou(i, j, frag_i_data, frag_j_data, params, constraint_manager=None):
    """
    Enhanced fragment pair matching using Papaioannou method with constraints.
    
    Args:
        i, j: Fragment indices
        frag_i_data, frag_j_data: Fragment data dictionaries
        params: Configuration parameters
        constraint_manager: ConstraintManager instance
    
    Returns:
        List of potential matches
    """
    matches = []
    
    source_mesh = frag_j_data['original_mesh']
    target_mesh = frag_i_data['original_mesh']
    source_name = frag_j_data['name']
    target_name = frag_i_data['name']
    
    if source_mesh is None or target_mesh is None:
        return []
    
    if not source_mesh.has_triangles() or not target_mesh.has_triangles():
        return []
    
    # Get constraints for this pair
    constraints = None
    if constraint_manager:
        constraints = constraint_manager.get_constraints_for_pair(source_name, target_name)
    
    # Configure bias weights based on available constraints
    bias_weights = {'geometric': 1.0, 'material': 0.0, 'overlap': 0.0}
    
    if constraints and constraints.get('material_axis_constraint'):
        bias_weights['material'] = params.get('material_bias_weight', 0.3)
        bias_weights['geometric'] = 1.0 - bias_weights['material']
    
    if params.get('encourage_surface_overlap', True):
        bias_weights['overlap'] = params.get('overlap_bias_weight', 0.1)
        # Renormalize
        total_weight = sum(bias_weights.values())
        for key in bias_weights:
            bias_weights[key] /= total_weight
    
    # Try both orientations: j->i and i->j
    for source_idx, target_idx, s_mesh, t_mesh, s_name, t_name in [
        (j, i, source_mesh, target_mesh, source_name, target_name),
        (i, j, target_mesh, source_mesh, target_name, source_name)
    ]:
        try:
            # For constrained matching, we might try multiple fracture direction combinations
            direction_combinations = [(None, None)]  # Default: no specific directions
            
            if constraints and constraints.get('fracture_direction_constraint'):
                source_directions = constraints['source_fracture_directions']
                target_directions = constraints['target_fracture_directions']
                direction_combinations = [
                    (s_dir, t_dir) for s_dir in source_directions for t_dir in target_directions
                ]
            
            best_result = None
            best_fitness = 0.0
            
            for source_dir, target_dir in direction_combinations:
                # Modify constraints for this specific direction combination
                current_constraints = constraints.copy() if constraints else {}
                if source_dir is not None and target_dir is not None:
                    current_constraints['specific_source_direction'] = source_dir
                    current_constraints['specific_target_direction'] = target_dir
                
                # Perform alignment
                if params.get('use_papaioannou_method', True):
                    result_transform, fitness, convergence_info = align_fragments_papaioannou(
                        s_mesh, t_mesh, params, current_constraints
                    )
                else:
                    # Fallback to legacy method if requested
                    from src.alignment import align_fragments_pcd
                    # Convert meshes to PCDs for legacy method
                    s_pcd = s_mesh.sample_points_poisson_disk(number_of_points=1000)
                    t_pcd = t_mesh.sample_points_poisson_disk(number_of_points=1000)
                    result_transform, fitness, rmse = align_fragments_pcd(s_pcd, t_pcd, None, None, params)
                    convergence_info = {'method': 'legacy', 'final_error': rmse}
                
                if result_transform is not None and fitness > best_fitness:
                    best_result = (result_transform, fitness, convergence_info)
                    best_fitness = fitness
            
            if best_result is not None:
                transformation, fitness, info = best_result
                
                # Calculate confidence score
                confidence = fitness / (info.get('final_error', 1.0) + 1e-6)
                
                # Create match record
                match = {
                    'source_idx': source_idx,
                    'target_idx': target_idx,
                    'transformation': transformation,
                    'score': fitness,
                    'rmse': info.get('final_error', 0.0),
                    'confidence': confidence,
                    'source_name': s_name,
                    'target_name': t_name,
                    'method': info.get('method', 'papaioannou'),
                    'convergence_info': info,
                    'constraints_used': constraints is not None
                }
                
                # Apply quality filter
                min_score = params.get("min_match_score", 0.3)
                if fitness >= min_score:
                    matches.append(match)
        
        except Exception as e:
            print(f"Error matching fragments {s_name} -> {t_name}: {e}")
            continue
    
    return matches


def find_pairwise_matches_enhanced(fragments_data, params, constraint_manager=None):
    """
    Enhanced pairwise matching using Papaioannou method with constraint support.

    Args:
        fragments_data: List of fragment data dictionaries
        params: Configuration parameters
        constraint_manager: Optional ConstraintManager for applying constraints

    Returns:
        List of potential matches with enhanced matching information
    """
    potential_matches = []
    num_fragments = len(fragments_data)

    if num_fragments < 2:
        print("Not enough fragments to find matches.")
        return []

    print(f"\nFinding pairwise matches among {num_fragments} fragments using Papaioannou method...")
    
    # Setup constraint manager if constraints are specified in params
    if constraint_manager is None and params.get('auto_detect_constraints', False):
        constraint_manager = ConstraintManager()
        # Auto-detect constraints from fragment properties (simplified)
        for frag_data in fragments_data:
            if 'material_axis' in frag_data:
                constraint_manager.add_material_axis_constraint(
                    frag_data['name'], frag_data['material_axis']
                )
            if 'fracture_directions' in frag_data:
                constraint_manager.add_fracture_direction_constraint(
                    frag_data['name'], frag_data['fracture_directions']
                )
    
    # Generate all pairs
    pairs = list(combinations(range(num_fragments), 2))
    print(f"Testing {len(pairs)} fragment pairs...")
    
    # Parallel processing
    max_workers = params.get('matching_max_workers', 4)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {
            executor.submit(
                _match_fragment_pair_papaioannou, 
                i, j, 
                fragments_data[i], 
                fragments_data[j], 
                params,
                constraint_manager
            ): (i, j)
            for i, j in pairs
        }
        
        completed = 0
        for future in as_completed(future_to_pair):
            i, j = future_to_pair[future]
            try:
                matches = future.result()
                if matches:
                    results.extend(matches)
                    print(f"Found {len(matches)} potential matches for pair ({i}, {j})")
                completed += 1
                
                if completed % 10 == 0:
                    print(f"Completed {completed}/{len(pairs)} pairs...")
                    
            except Exception as e:
                print(f"Error processing pair ({i}, {j}): {e}")
    
    # Sort by confidence score (higher is better)
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"Found {len(results)} total potential matches above threshold.")
    
    # Apply global filtering and ranking
    filtered_matches = _apply_global_match_filtering(results, params)
    
    print(f"After global filtering: {len(filtered_matches)} matches retained.")
    
    return filtered_matches


def _apply_global_match_filtering(matches, params):
    """
    Apply global filtering to remove conflicting or low-quality matches.
    """
    if not matches:
        return matches
    
    # Remove duplicate pairs (keep best scoring)
    seen_pairs = set()
    unique_matches = []
    
    for match in matches:
        pair_key = tuple(sorted([match['source_idx'], match['target_idx']]))
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_matches.append(match)
    
    # Apply confidence threshold
    confidence_threshold = params.get('min_confidence_score', 0.1)
    confident_matches = [m for m in unique_matches if m['confidence'] >= confidence_threshold]
    
    # Apply maximum matches per fragment limit
    max_matches_per_fragment = params.get('max_matches_per_fragment', 5)
    if max_matches_per_fragment > 0:
        fragment_match_counts = {}
        filtered_matches = []
        
        for match in confident_matches:
            source_idx = match['source_idx']
            target_idx = match['target_idx']
            
            source_count = fragment_match_counts.get(source_idx, 0)
            target_count = fragment_match_counts.get(target_idx, 0)
            
            if source_count < max_matches_per_fragment and target_count < max_matches_per_fragment:
                filtered_matches.append(match)
                fragment_match_counts[source_idx] = source_count + 1
                fragment_match_counts[target_idx] = target_count + 1
        
        return filtered_matches
    
    return confident_matches


# Legacy compatibility function
def find_pairwise_matches(fragments_data, params):
    """
    Legacy compatibility wrapper for the enhanced matching system.
    """
    print("Using enhanced Papaioannou-based matching system...")
    
    # Convert legacy parameters if needed
    enhanced_params = params.copy()
    
    # Set default Papaioannou-specific parameters
    enhanced_params.setdefault('use_papaioannou_method', True)
    enhanced_params.setdefault('papaioannou_resolution', 128)
    enhanced_params.setdefault('max_rotation_angle', np.pi)
    enhanced_params.setdefault('max_translation_factor', 0.1)
    enhanced_params.setdefault('esa_max_iter', 1000)
    enhanced_params.setdefault('max_acceptable_error', 0.5)
    enhanced_params.setdefault('material_bias_weight', 0.2)
    enhanced_params.setdefault('overlap_bias_weight', 0.1)
    enhanced_params.setdefault('encourage_surface_overlap', True)
    enhanced_params.setdefault('auto_detect_constraints', False)
    
    return find_pairwise_matches_enhanced(fragments_data, enhanced_params)


if __name__ == '__main__':
    print("Testing enhanced Papaioannou matching system...")
    
    # Create test data
    import open3d as o3d
    import os
    
    # Test with simple meshes
    fragments_test_data = []
    
    # Create test meshes
    mesh1 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    mesh2 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    
    # Transform mesh2 to create a matching scenario
    transform = np.array([
        [0.866, -0.5, 0, 0.8],
        [0.5, 0.866, 0, 0.2],
        [0, 0, 1, 0.1],
        [0, 0, 0, 1]
    ])
    mesh2.transform(transform)
    
    fragments_test_data = [
        {
            'name': 'test_fragment_1',
            'original_index': 0,
            'original_mesh': mesh1,
            'pcd_for_features': mesh1.sample_points_poisson_disk(1000),
            'features': None  # Not used in Papaioannou method
        },
        {
            'name': 'test_fragment_2', 
            'original_index': 1,
            'original_mesh': mesh2,
            'pcd_for_features': mesh2.sample_points_poisson_disk(1000),
            'features': None
        }
    ]
    
    # Test parameters
    test_params = {
        'use_papaioannou_method': True,
        'papaioannou_resolution': 64,  # Lower for faster testing
        'max_rotation_angle': np.pi,
        'max_translation_factor': 0.2,
        'esa_max_iter': 200,  # Reduced for testing
        'max_acceptable_error': 1.0,
        'min_match_score': 0.1,
        'matching_max_workers': 2,
        'material_bias_weight': 0.2,
        'overlap_bias_weight': 0.1
    }
    
    # Create constraint manager for testing
    constraint_mgr = ConstraintManager()
    constraint_mgr.add_fracture_direction_constraint(
        'test_fragment_1', [[0, 0, 1]], tolerance=np.pi/18
    )
    constraint_mgr.add_fracture_direction_constraint(
        'test_fragment_2', [[0, 0, -1]], tolerance=np.pi/18
    )
    
    # Run enhanced matching
    start_time = time.time()
    matches = find_pairwise_matches_enhanced(fragments_test_data, test_params, constraint_mgr)
    matching_time = time.time() - start_time
    
    print(f"\nMatching completed in {matching_time:.2f}s")
    print(f"Found {len(matches)} potential matches:")
    
    for i, match in enumerate(matches):
        print(f"\nMatch {i+1}:")
        print(f"  {match['source_name']} -> {match['target_name']}")
        print(f"  Score: {match['score']:.4f}")
        print(f"  Confidence: {match['confidence']:.4f}")
        print(f"  Method: {match['method']}")
        print(f"  Constraints used: {match['constraints_used']}")
        if 'convergence_info' in match:
            info = match['convergence_info']
            print(f"  Converged: {info.get('converged', 'Unknown')}")
            print(f"  Final error: {info.get('final_error', 'Unknown')}")