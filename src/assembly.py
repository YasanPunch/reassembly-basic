import trimesh
import numpy as np
import open3d as o3d
import copy
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from src.io_utils import combine_meshes, save_mesh

print("DEBUG: enhanced assembly.py (Papaioannou method) loaded")


class SurfaceOverlapAnalyzer:
    """
    Advanced surface overlap analysis based on Papaioannou et al. methodology.
    Uses both geometric and surface area criteria for overlap detection.
    """
    
    def __init__(self, params):
        self.params = params
        self.sample_density = params.get('overlap_sample_density', 500)
        self.penetration_tolerance = params.get('overlap_penetration_tolerance', 0.1)
        self.surface_overlap_threshold = params.get('surface_overlap_threshold', 0.8)
    
    def analyze_overlap(self, mesh1, mesh2, mesh1_name="mesh1", mesh2_name="mesh2", viz_collector=None):
        """
        Comprehensive overlap analysis using multiple criteria.
        
        Args:
            mesh1, mesh2: o3d.geometry.TriangleMesh objects
            mesh1_name, mesh2_name: Names for logging
            viz_collector: Optional visualization collector
        
        Returns:
            is_valid: Boolean indicating if overlap is acceptable
            overlap_info: Dictionary with detailed overlap information
        """
        if not mesh1.has_vertices() or not mesh2.has_vertices():
            return True, {'reason': 'empty_mesh', 'overlap_ratio': 0.0}
        
        overlap_info = {
                'mesh1_name': mesh1_name, 
                'mesh2_name': mesh2_name, 
            'analysis_time': 0.0
        }
        
        start_time = time.time()
        
        # 1. Bounding box overlap analysis
        bbox_valid, bbox_info = self._analyze_bbox_overlap(mesh1, mesh2)
        overlap_info.update(bbox_info)
        
        if not bbox_valid:
            overlap_info['analysis_time'] = time.time() - start_time
            if viz_collector:
                viz_collector.append({
                    'step': 'overlap_analysis_failed_bbox',
                    'type': 'event',
                    'mesh1_name': mesh1_name,
                    'mesh2_name': mesh2_name,
                    'reason': bbox_info.get('reason', 'bbox_overlap_excessive')
                })
            return False, overlap_info
        
        # 2. Surface-based overlap analysis
        surface_valid, surface_info = self._analyze_surface_overlap(mesh1, mesh2)
        overlap_info.update(surface_info)
        
        if not surface_valid:
            overlap_info['analysis_time'] = time.time() - start_time
            if viz_collector:
                viz_collector.append({
                    'step': 'overlap_analysis_failed_surface',
                    'type': 'event',
                    'mesh1_name': mesh1_name, 
                    'mesh2_name': mesh2_name, 
                    'reason': surface_info.get('reason', 'surface_overlap_excessive')
                })
            return False, overlap_info
        
        # 3. Volumetric penetration analysis (if requested)
        if self.params.get('enable_volumetric_analysis', True):
            volumetric_valid, volumetric_info = self._analyze_volumetric_overlap(mesh1, mesh2)
            overlap_info.update(volumetric_info)
            
            if not volumetric_valid:
                overlap_info['analysis_time'] = time.time() - start_time
                if viz_collector:
                    viz_collector.append({
                        'step': 'overlap_analysis_failed_volumetric',
                        'type': 'event',
                        'mesh1_name': mesh1_name,
                        'mesh2_name': mesh2_name,
                        'reason': volumetric_info.get('reason', 'volumetric_penetration_excessive')
                    })
                return False, overlap_info
        
        overlap_info['analysis_time'] = time.time() - start_time
        overlap_info['overall_valid'] = True
        
        return True, overlap_info
    
    def _analyze_bbox_overlap(self, mesh1, mesh2):
        """Analyze bounding box overlap using adaptive thresholds."""
        bbox1 = mesh1.get_axis_aligned_bounding_box()
        bbox2 = mesh2.get_axis_aligned_bounding_box()
        
        vol1 = bbox1.volume()
        vol2 = bbox2.volume()
        
        if vol1 < 1e-9 or vol2 < 1e-9:
            return True, {'bbox_overlap_ratio': 0.0, 'reason': 'zero_volume'}
        
        # Calculate intersection volume
        min1, max1 = bbox1.get_min_bound(), bbox1.get_max_bound()
        min2, max2 = bbox2.get_min_bound(), bbox2.get_max_bound()
        
        intersect_min = np.maximum(min1, min2)
        intersect_max = np.minimum(max1, max2)
        
        if np.any(intersect_min >= intersect_max):
            return True, {'bbox_overlap_ratio': 0.0}
        
        intersect_vol = np.prod(intersect_max - intersect_min)
        
        # Adaptive threshold based on relative sizes
        min_vol = min(vol1, vol2)
        max_vol = max(vol1, vol2)
        size_ratio = min_vol / max_vol
        
        # Adjust threshold based on size disparity
        base_threshold = self.params.get('max_assembly_overlap_factor_aabb', 0.8)
        adaptive_threshold = base_threshold * (0.5 + 0.5 * size_ratio)
        
        overlap_ratio1 = intersect_vol / vol1
        overlap_ratio2 = intersect_vol / vol2
        max_overlap = max(overlap_ratio1, overlap_ratio2)
        
        is_valid = max_overlap <= adaptive_threshold
        
        return is_valid, {
            'bbox_overlap_ratio': max_overlap,
            'bbox_threshold_used': adaptive_threshold,
            'bbox_size_ratio': size_ratio,
            'bbox_intersect_volume': intersect_vol
        }
    
    def _analyze_surface_overlap(self, mesh1, mesh2):
        """Analyze surface-to-surface overlap using sampling."""
        try:
            # Convert to trimesh for better surface analysis
            trimesh1 = trimesh.Trimesh(
                vertices=np.asarray(mesh1.vertices),
                faces=np.asarray(mesh1.triangles)
            )
            trimesh2 = trimesh.Trimesh(
                vertices=np.asarray(mesh2.vertices),
                faces=np.asarray(mesh2.triangles)
            )
            
            # Sample points from mesh1 surface
            if len(trimesh1.faces) == 0:
                return True, {'surface_overlap_ratio': 0.0, 'reason': 'no_faces_mesh1'}
            
            sample_points1, _ = trimesh.sample.sample_surface(trimesh1, self.sample_density)
            
            if len(sample_points1) == 0:
                return True, {'surface_overlap_ratio': 0.0, 'reason': 'no_samples'}
            
            # Check proximity to mesh2
            if len(trimesh2.faces) == 0:
                return True, {'surface_overlap_ratio': 0.0, 'reason': 'no_faces_mesh2'}
            
            # Use proximity queries to check surface overlap
            proximity_query = trimesh.proximity.ProximityQuery(trimesh2)
            distances = proximity_query.signed_distance(sample_points1)
            
            # Count points that are inside or very close to mesh2
            penetration_threshold = -self.penetration_tolerance
            overlapping_points = np.sum(distances < penetration_threshold)
            overlap_ratio = overlapping_points / len(sample_points1)
            
            # Surface area based analysis
            area1 = trimesh1.area
            area2 = trimesh2.area
            area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
            
            # Adaptive threshold based on area ratio
            base_surface_threshold = self.surface_overlap_threshold
            adaptive_surface_threshold = base_surface_threshold * (0.5 + 0.5 * area_ratio)
            
            is_valid = overlap_ratio <= adaptive_surface_threshold
            
            return is_valid, {
                'surface_overlap_ratio': overlap_ratio,
                'surface_threshold_used': adaptive_surface_threshold,
                'surface_area_ratio': area_ratio,
                'overlapping_sample_points': overlapping_points,
                'total_sample_points': len(sample_points1),
                'mean_distance': np.mean(distances),
                'min_distance': np.min(distances)
            }
            
        except Exception as e:
                print(f"Error in surface overlap analysis: {e}")
                return True, {'surface_overlap_ratio': 0.0, 'reason': f'analysis_error: {e}'}
        
    def _analyze_volumetric_overlap(self, mesh1, mesh2):
        """Analyze volumetric overlap using mesh intersection."""
        try:
            # Convert to trimesh
            trimesh1 = trimesh.Trimesh(
                vertices=np.asarray(mesh1.vertices),
                faces=np.asarray(mesh1.triangles)
            )
            trimesh2 = trimesh.Trimesh(
                vertices=np.asarray(mesh2.vertices),
                faces=np.asarray(mesh2.triangles)
            )
            
            # Check if meshes are watertight for volume calculations
            if not trimesh1.is_watertight or not trimesh2.is_watertight:
                # Attempt to fix or skip volumetric analysis
                try:
                    trimesh1.fill_holes()
                    trimesh2.fill_holes()
                except:
                    return True, {'volumetric_overlap_ratio': 0.0, 'reason': 'not_watertight'}
            
            # Calculate volumes
            vol1 = abs(trimesh1.volume) if trimesh1.is_watertight else 0
            vol2 = abs(trimesh2.volume) if trimesh2.is_watertight else 0
            
            if vol1 < 1e-9 or vol2 < 1e-9:
                return True, {'volumetric_overlap_ratio': 0.0, 'reason': 'zero_volume'}
            
            # Calculate intersection volume (simplified approach)
            # For a full implementation, we would use boolean operations
            # Here we use a sampling-based approximation
            bbox1 = trimesh1.bounds
            bbox2 = trimesh2.bounds
            
            # Sample points in intersection region
            intersect_bbox_min = np.maximum(bbox1[0], bbox2[0])
            intersect_bbox_max = np.minimum(bbox1[1], bbox2[1])
            
            if np.any(intersect_bbox_min >= intersect_bbox_max):
                return True, {'volumetric_overlap_ratio': 0.0}
            
            # Sample points uniformly in intersection bounding box
            n_samples = 1000
            sample_points = np.random.uniform(
                intersect_bbox_min, intersect_bbox_max, size=(n_samples, 3)
            )
            
            # Check which points are inside both meshes
            inside1 = trimesh1.contains(sample_points)
            inside2 = trimesh2.contains(sample_points)
            inside_both = inside1 & inside2
            
            intersect_bbox_vol = np.prod(intersect_bbox_max - intersect_bbox_min)
            estimated_intersect_vol = intersect_bbox_vol * np.sum(inside_both) / n_samples
            
            # Calculate overlap ratios
            overlap_ratio1 = estimated_intersect_vol / vol1
            overlap_ratio2 = estimated_intersect_vol / vol2
            max_volumetric_overlap = max(overlap_ratio1, overlap_ratio2)
            
            # Threshold for volumetric overlap
            volumetric_threshold = self.params.get('max_volumetric_overlap', 0.5)
            is_valid = max_volumetric_overlap <= volumetric_threshold
            
            return is_valid, {
                'volumetric_overlap_ratio': max_volumetric_overlap,
                'volumetric_threshold_used': volumetric_threshold,
                'estimated_intersection_volume': estimated_intersect_vol,
                'volume1': vol1,
                'volume2': vol2
            }
            
        except Exception as e:
            print(f"Error in volumetric overlap analysis: {e}")
            return True, {'volumetric_overlap_ratio': 0.0, 'reason': f'volumetric_error: {e}'}


class ConstrainedAssembler:
    """
    Enhanced assembler with constraint support and global optimization.
    """
    
    def __init__(self, fragments_data, pairwise_matches: list[dict], params, visualization_log=None, constraint_manager=None):
        self.fragments_data = copy.deepcopy(fragments_data) 
        self.pairwise_matches = sorted(pairwise_matches, key=lambda x: x.get('confidence', x['score']), reverse=True)
        self.params = params
        self.num_fragments = len(fragments_data)
        self.constraint_manager = constraint_manager
        
        self.original_meshes = [fd['original_mesh'] for fd in self.fragments_data]
        self.fragment_transforms = [np.eye(4) for _ in range(self.num_fragments)]
        self.is_fragment_placed = [False] * self.num_fragments
        self.assembly_components = [] 
        self.visualization_log = visualization_log if visualization_log is not None else []

        # Initialize overlap analyzer
        self.overlap_analyzer = SurfaceOverlapAnalyzer(params)
        
        # Track constraint satisfaction
        self.constraint_violations = []
    
    def assemble_with_constraints(self):
        """
        Perform constrained assembly with global optimization.
        """
        if self.num_fragments == 0:
            return None
        
        if self.num_fragments == 1:
            return self._handle_single_fragment()
        
        if not self.pairwise_matches:
            return self._handle_no_matches()
        
        # Phase 1: Greedy assembly with constraints
        print("\n[Phase 1: Constrained Greedy Assembly]")
        greedy_result = self._constrained_greedy_assembly()
        
        if greedy_result is None:
            return None
        
        # Phase 2: Global optimization (if multiple fragments placed)
        if np.sum(self.is_fragment_placed) > 2 and self.params.get('enable_global_optimization', True):
            print("\n[Phase 2: Global Pose Optimization]")
            optimized_result = self._global_pose_optimization()
            if optimized_result is not None:
                greedy_result = optimized_result
        
        # Phase 3: Constraint validation and correction
        if self.constraint_manager and self.params.get('validate_constraints', True):
            print("\n[Phase 3: Constraint Validation]")
            validated_result = self._validate_and_correct_constraints(greedy_result)
            if validated_result is not None:
                greedy_result = validated_result
        
        return greedy_result
    
    def _handle_single_fragment(self):
        """Handle assembly of single fragment."""
        frag_data = self.fragments_data[0]
        mesh_to_log = self.original_meshes[0]
    
        if self.visualization_log is not None:
            self.visualization_log.append({
                'step': 'assembly_single_fragment', 'type': 'mesh',
                'fragment_name': frag_data['name'],
                'original_index': frag_data['original_index'],
                'fragment_idx_in_valid_list': 0,
                'transform': np.eye(4),
                'vertices': np.asarray(mesh_to_log.vertices),
                'triangles': np.asarray(mesh_to_log.triangles)
            })
    
        return self._get_transformed_mesh(0)

    def _handle_no_matches(self):
        """Handle case with no valid pairwise matches."""
        print("No pairwise matches for assembly. Creating composite of unconnected fragments.")
        
        if self.visualization_log is not None:
                for i_log, fd_log in enumerate(self.fragments_data):
                    self.visualization_log.append({
                        'step': 'assembly_failed_no_pairwise_matches', 'type': 'mesh',
                        'fragment_name': fd_log['name'],
                        'original_index': fd_log['original_index'],
                        'fragment_idx_in_valid_list': i_log,
                    'transform': np.eye(4),
                        'vertices': np.asarray(self.original_meshes[i_log].vertices),
                        'triangles': np.asarray(self.original_meshes[i_log].triangles)
                    })
        
        # Return combined unconnected fragments
        return combine_meshes(self.original_meshes, self.fragment_transforms)
    
    def _constrained_greedy_assembly(self):
        """
        Perform greedy assembly with constraint checking.
        """
        # Select seed fragment using constraint-aware criteria
        seed_idx = self._select_constrained_seed()
        seed_name = self.fragments_data[seed_idx]['name']
        
        print(f"Starting constrained assembly with seed fragment: {seed_name} (idx: {seed_idx})")
        self.is_fragment_placed[seed_idx] = True
        
        current_assembly_components = [(self._get_transformed_mesh(seed_idx), seed_name)]
        
        if self.visualization_log is not None:
            self._log_placed_fragment(seed_idx, 'assembly_seed_placed')

        num_placed = 1
        
        # Greedy placement loop with constraint checking
        while num_placed < self.num_fragments:
            best_candidate = self._find_best_constrained_candidate(current_assembly_components)
            
            if best_candidate is None:
                print("No more valid constrained matches found.")
                break
            
            # Place the best candidate
            match_info, world_transform, candidate_idx = best_candidate
            self._place_fragment(candidate_idx, world_transform, match_info)
            
            placed_mesh = self._get_transformed_mesh(candidate_idx)
            placed_name = self.fragments_data[candidate_idx]['name']
            current_assembly_components.append((placed_mesh, placed_name))
            
            num_placed += 1
            
            print(f"  Placed fragment: {placed_name} (idx: {candidate_idx}) "
                  f"via score {match_info.get('confidence', match_info['score']):.3f}")
        
        # Log unplaced fragments
        self._log_unplaced_fragments()
        
        # Create final assembly
        return self._create_final_assembly()
    
    def _select_constrained_seed(self):
        """
        Select seed fragment considering constraints.
        """
        # Default: use the fragment with most matches
        fragment_match_counts = {}
        for match in self.pairwise_matches:
            source_idx = match['source_idx']
            target_idx = match['target_idx']
            fragment_match_counts[source_idx] = fragment_match_counts.get(source_idx, 0) + 1
            fragment_match_counts[target_idx] = fragment_match_counts.get(target_idx, 0) + 1
        
        if not fragment_match_counts:
            return 0  # Default to first fragment
        
        # Consider constraint compatibility in seed selection
        if self.constraint_manager:
            # Prefer fragments with well-defined constraints
            constrained_fragments = []
            for idx in fragment_match_counts.keys():
                frag_name = self.fragments_data[idx]['name']
                if (frag_name in self.constraint_manager.material_axes or 
                    frag_name in self.constraint_manager.fracture_directions):
                    constrained_fragments.append(idx)
            
            if constrained_fragments:
                # Choose constrained fragment with most matches
                best_constrained = max(constrained_fragments, 
                                     key=lambda x: fragment_match_counts[x])
                return best_constrained
        
        # Default: fragment with most matches
        return max(fragment_match_counts.keys(), key=fragment_match_counts.get)
    
    def _find_best_constrained_candidate(self, current_assembly_components):
        """
        Find the best candidate considering constraints and overlap.
        """
        best_candidate = None
        best_score = -1.0

        for match_info in self.pairwise_matches:
            s_idx, t_idx = match_info['source_idx'], match_info['target_idx']
                
            # Determine placement scenario
            candidate_idx, world_transform = None, None

            if self.is_fragment_placed[t_idx] and not self.is_fragment_placed[s_idx]:
                candidate_idx = s_idx
                world_transform = np.dot(self.fragment_transforms[t_idx], match_info['transformation'])
                
            elif self.is_fragment_placed[s_idx] and not self.is_fragment_placed[t_idx]:
                try:
                    inv_transform = np.linalg.inv(match_info['transformation'])
                    world_transform = np.dot(self.fragment_transforms[s_idx], inv_transform)
                    candidate_idx = t_idx
                except np.linalg.LinAlgError: 
                        continue 
            else: 
                continue
            
            if candidate_idx is None:
                continue
            
            # Check constraint compatibility
            if not self._check_constraint_compatibility(candidate_idx, world_transform, match_info):
                continue
            
            # Check overlap with existing assembly
            candidate_mesh = copy.deepcopy(self.original_meshes[candidate_idx])
            candidate_mesh.transform(world_transform)
            candidate_name = self.fragments_data[candidate_idx]['name']
            
            if not self._check_assembly_overlap(candidate_mesh, candidate_name, current_assembly_components):
                continue
            
            # Calculate combined score considering constraints
            combined_score = self._calculate_combined_score(match_info, candidate_idx, world_transform)
            
            if combined_score > best_score:
                best_candidate = (match_info, world_transform, candidate_idx)
                best_score = combined_score
        
        return best_candidate
    
    def _check_constraint_compatibility(self, candidate_idx, world_transform, match_info):
        """
        Check if placement satisfies constraints.
        """
        if not self.constraint_manager:
            return True
        
        candidate_name = self.fragments_data[candidate_idx]['name']
        
        # Check material axis constraints
        if candidate_name in self.constraint_manager.material_axes:
            if not self._validate_material_axis_constraint(candidate_idx, world_transform):
                return False
        
        # Check fracture direction constraints
        if candidate_name in self.constraint_manager.fracture_directions:
            if not self._validate_fracture_direction_constraint(candidate_idx, world_transform, match_info):
                return False
        
        return True
    
    def _validate_material_axis_constraint(self, candidate_idx, world_transform):
        """Validate material axis alignment constraint."""
        # Simplified implementation - would need proper axis transformation
        return True  # Placeholder
    
    def _validate_fracture_direction_constraint(self, candidate_idx, world_transform, match_info):
        """Validate fracture direction alignment constraint."""
        # Check if the match was created with fracture direction constraints
        return match_info.get('constraints_used', False)
    
    def _check_assembly_overlap(self, candidate_mesh, candidate_name, current_assembly_components):
        """
        Check overlap with all components in current assembly.
        """
        for placed_mesh, placed_name in current_assembly_components:
            is_valid, overlap_info = self.overlap_analyzer.analyze_overlap(
                candidate_mesh, placed_mesh, candidate_name, placed_name, self.visualization_log
            )
            
            if not is_valid:
                return False
        
        return True
    
    def _calculate_combined_score(self, match_info, candidate_idx, world_transform):
        """
        Calculate combined score considering multiple factors.
        """
        base_score = match_info.get('confidence', match_info['score'])
        
        # Constraint bonus
        constraint_bonus = 0.0
        if match_info.get('constraints_used', False):
            constraint_bonus = self.params.get('constraint_satisfaction_bonus', 0.2)
        
        # Method bonus (prefer Papaioannou method)
        method_bonus = 0.0
        if match_info.get('method') == 'papaioannou':
            method_bonus = self.params.get('papaioannou_method_bonus', 0.1)
        
        # Size compatibility bonus
        size_bonus = self._calculate_size_compatibility_bonus(match_info)
        
        combined_score = base_score + constraint_bonus + method_bonus + size_bonus
        return combined_score
    
    def _calculate_size_compatibility_bonus(self, match_info):
        """Calculate bonus for size-compatible matches."""
        source_idx = match_info['source_idx']
        target_idx = match_info['target_idx']
        
        # Simple size compatibility based on bounding box volumes
        try:
            bbox_source = self.original_meshes[source_idx].get_axis_aligned_bounding_box()
            bbox_target = self.original_meshes[target_idx].get_axis_aligned_bounding_box()
            
            vol_source = bbox_source.volume()
            vol_target = bbox_target.volume()
            
            if vol_source > 0 and vol_target > 0:
                size_ratio = min(vol_source, vol_target) / max(vol_source, vol_target)
                return size_ratio * self.params.get('size_compatibility_bonus', 0.05)
        except:
            pass
        
        return 0.0
    
    def _place_fragment(self, fragment_idx, world_transform, match_info):
        """Place a fragment and update state."""
        self.fragment_transforms[fragment_idx] = world_transform
        self.is_fragment_placed[fragment_idx] = True
                
        if self.visualization_log is not None:
            self._log_placed_fragment(fragment_idx, 'assembly_fragment_placed', match_info)
    
    def _log_placed_fragment(self, fragment_idx, step_type, match_info=None):
        """Log placement of a fragment."""
        frag_data = self.fragments_data[fragment_idx]
        placed_mesh = self._get_transformed_mesh(fragment_idx)
        
        log_entry = {
            'step': step_type, 'type': 'mesh',
            'fragment_name': frag_data['name'],
            'original_index': frag_data['original_index'],
            'fragment_idx_in_valid_list': fragment_idx,
            'transform': self.fragment_transforms[fragment_idx],
            'vertices': np.asarray(placed_mesh.vertices),
            'triangles': np.asarray(placed_mesh.triangles)
        }
        
        if match_info:
            log_entry.update({
                'matched_via_score': match_info.get('confidence', match_info['score']),
                'match_method': match_info.get('method', 'unknown'),
                'constraints_used': match_info.get('constraints_used', False)
            })
        
        self.visualization_log.append(log_entry)
    
    def _log_unplaced_fragments(self):
        """Log any unplaced fragments."""
        for idx, placed in enumerate(self.is_fragment_placed):
            if not placed:
                frag_data = self.fragments_data[idx]
                if self.visualization_log is not None:
                     self.visualization_log.append({
                        'step': 'assembly_fragment_unplaced', 'type': 'mesh',
                        'fragment_name': frag_data['name'],
                        'original_index': frag_data['original_index'],
                        'fragment_idx_in_valid_list': idx,
                        'transform': np.eye(4),
                        'vertices': np.asarray(self.original_meshes[idx].vertices),
                        'triangles': np.asarray(self.original_meshes[idx].triangles)
                    })
    
    def _create_final_assembly(self):
        """Create the final assembled mesh."""
        final_meshes = []
        final_transforms = []
        
        for i in range(self.num_fragments):
            if self.is_fragment_placed[i]:
                final_meshes.append(self.original_meshes[i])
                final_transforms.append(self.fragment_transforms[i])
        
        if not final_meshes:
            print("Error: No meshes were placed in the assembly.")
            return None

        return combine_meshes(final_meshes, final_transforms)
    
    def _get_transformed_mesh(self, fragment_idx):
        """Get transformed mesh for a fragment."""
        mesh = copy.deepcopy(self.original_meshes[fragment_idx])
        mesh.transform(self.fragment_transforms[fragment_idx])
        return mesh
    
    def _global_pose_optimization(self):
        """
        Perform global pose optimization using pose graph.
        """
        try:
            import open3d as o3d
            
            pose_graph = o3d.pipelines.registration.PoseGraph()
            
            # Add nodes for placed fragments
            placed_indices = [i for i, placed in enumerate(self.is_fragment_placed) if placed]
            
            for i, frag_idx in enumerate(placed_indices):
                if i == 0:
                    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
                else:
                    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(
                        self.fragment_transforms[frag_idx]
                    ))
            
            # Add edges from pairwise matches
            for match in self.pairwise_matches:
                source_idx = match['source_idx']
                target_idx = match['target_idx']
                
                if not (self.is_fragment_placed[source_idx] and self.is_fragment_placed[target_idx]):
                    continue
                
                # Find indices in placed_indices list
                try:
                    source_node = placed_indices.index(source_idx)
                    target_node = placed_indices.index(target_idx)
                    
                    confidence = match.get('confidence', match['score'])
                    information = np.eye(6) * confidence
                    
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            source_node, target_node, match['transformation'], False, information
                        )
                    )
                except ValueError:
                    continue
            
            # Run global optimization
            option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=self.params.get('voxel_downsample_size', 0.01) * 2.0,
                edge_prune_threshold=0.25,
                reference_node=0
            )
            
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option
            )
            
            # Update transforms with optimized poses
            for i, frag_idx in enumerate(placed_indices):
                self.fragment_transforms[frag_idx] = pose_graph.nodes[i].pose
            
            print("Global pose optimization completed successfully.")
            return self._create_final_assembly()
            
        except Exception as e:
            print(f"Global pose optimization failed: {e}")
            return None
    
    def _validate_and_correct_constraints(self, assembly_result):
        """
        Validate and optionally correct constraint violations.
        """
        if not self.constraint_manager:
            return assembly_result
        
        violations = []
        
        # Check material axis constraints
        for frag_name in self.constraint_manager.material_axes:
            violation = self._check_material_axis_violation(frag_name)
            if violation:
                violations.append(violation)
        
        # Check fracture direction constraints
        for frag_name in self.constraint_manager.fracture_directions:
            violation = self._check_fracture_direction_violation(frag_name)
            if violation:
                violations.append(violation)
        
        if violations:
            print(f"Found {len(violations)} constraint violations:")
            for violation in violations:
                print(f"  - {violation}")
            
            # Optionally attempt correction
            if self.params.get('attempt_constraint_correction', False):
                corrected_result = self._attempt_constraint_correction(assembly_result, violations)
                if corrected_result is not None:
                    return corrected_result
        
        return assembly_result
    
    def _check_material_axis_violation(self, frag_name):
        """Check for material axis constraint violations."""
        # Simplified placeholder implementation
        return None
    
    def _check_fracture_direction_violation(self, frag_name):
        """Check for fracture direction constraint violations."""
        # Simplified placeholder implementation
        return None
    
    def _attempt_constraint_correction(self, assembly_result, violations):
        """Attempt to correct constraint violations."""
        # Placeholder for constraint correction
        print("Constraint correction not implemented yet.")
        return assembly_result


# Legacy compatibility function
def check_overlap(mesh1_o3d, mesh1_name, mesh2_o3d, mesh2_name, params, viz_collector=None):
    """
    Legacy compatibility wrapper for enhanced overlap analysis.
    """
    analyzer = SurfaceOverlapAnalyzer(params)
    is_valid, overlap_info = analyzer.analyze_overlap(mesh1_o3d, mesh2_o3d, mesh1_name, mesh2_name, viz_collector)
    
    # Return legacy format
    overlap_ratio = overlap_info.get('surface_overlap_ratio', overlap_info.get('bbox_overlap_ratio', 0.0))
    return is_valid, overlap_ratio


class Assembler(ConstrainedAssembler):
    """
    Legacy compatibility class that extends ConstrainedAssembler.
    """
    
    def greedy_assembly(self):
        """Legacy method name compatibility."""
        return self.assemble_with_constraints()


if __name__ == '__main__':
    # Test the enhanced assembly system
    print("Testing enhanced assembly system...")
    
    import open3d as o3d
    import os
    
    # Create test fragments
    mesh_A = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    mesh_B = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    
    # Position mesh_B to connect with mesh_A
    transform_B = np.eye(4)
    transform_B[0, 3] = 1.0  # Move 1 unit in X direction
    mesh_B.transform(transform_B)
    
    mesh_A.compute_vertex_normals()
    mesh_B.compute_vertex_normals()
    
    # Create test fragment data
    fragments_test_data = [
        {
            'name': 'PartA', 'original_index': 0,
            'original_mesh': mesh_A,
            'pcd_for_features': None, 'features': None
        },
        {
            'name': 'PartB', 'original_index': 1,
            'original_mesh': mesh_B,
            'pcd_for_features': None, 'features': None
        }
    ]
    
    # Create test pairwise match
    test_transform = np.eye(4)
    test_transform[0, 3] = 1.0  # Transformation to align B with A
    
    test_matches = [
        {
            'source_idx': 1, 'target_idx': 0,
            'transformation': test_transform,
            'score': 0.9, 'confidence': 0.85,
            'source_name': 'PartB', 'target_name': 'PartA',
            'method': 'papaioannou', 'constraints_used': True
        }
    ]
    
    # Test parameters
    test_params = {
        'max_assembly_overlap_factor_aabb': 0.8,
        'surface_overlap_threshold': 0.7,
        'overlap_sample_density': 200,
        'overlap_penetration_tolerance': 0.05,
        'enable_global_optimization': True,
        'validate_constraints': False,  # Disabled for simple test
        'constraint_satisfaction_bonus': 0.2,
        'papaioannou_method_bonus': 0.1
    }
    
    # Create assembler
    test_visualization_log = []
    assembler = ConstrainedAssembler(
        fragments_test_data, test_matches, test_params, 
        visualization_log=test_visualization_log
    )
    
    # Run assembly
    print("\nRunning constrained assembly...")
    start_time = time.time()
    final_assembly = assembler.assemble_with_constraints()
    assembly_time = time.time() - start_time
    
    print(f"Assembly completed in {assembly_time:.2f}s")
    
    if final_assembly and final_assembly.has_vertices():
        print("Assembly successful!")
        print(f"Final mesh: {len(final_assembly.vertices)} vertices, {len(final_assembly.triangles)} triangles")
        
        # Save result
        output_dir = "data/enhanced_assembly_test"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "enhanced_assembled_model.obj")
        save_mesh(final_assembly, output_path)
        print(f"Saved to: {output_path}")
    else:
        print("Assembly failed")
    
    print(f"Visualization log has {len(test_visualization_log)} entries")