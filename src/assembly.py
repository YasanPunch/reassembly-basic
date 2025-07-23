import trimesh
import numpy as np
import open3d as o3d
import copy
from src.io_utils import combine_meshes, save_mesh # Assuming this is for if __name__ == '__main__'
from src.utils.geometry_utils import boolean_intersection_penetration_test

# Add global colormap import for fragment coloring
try:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20')
except ImportError:
    cmap = None

def check_overlap(mesh1_o3d, mesh1_name, mesh2_o3d, mesh2_name, params, viz_collector=None):
    if not mesh1_o3d.has_vertices() or not mesh2_o3d.has_vertices():
        return True, 0.0  # No overlap, confidence 0

    aabb1 = mesh1_o3d.get_axis_aligned_bounding_box()
    aabb2 = mesh2_o3d.get_axis_aligned_bounding_box()
    vol1_aabb = aabb1.volume()
    vol2_aabb = aabb2.volume()
    epsilon = 1e-9
    min_vol = min(vol1_aabb, vol2_aabb) + epsilon
    # Adaptive threshold: scale with min volume, or use a function of both
    base_overlap_factor = params.get("max_assembly_overlap_factor_aabb", 0.8)
    adaptive_overlap_factor = base_overlap_factor * (min_vol / (vol1_aabb + vol2_aabb + epsilon)) * 2.0  # Example scaling

    vol_intersection_aabb = 0.0
    if hasattr(aabb1, 'get_intersection'):
        intersection_aabb = aabb1.get_intersection(aabb2)
        current_vol_intersection = intersection_aabb.volume()
        if current_vol_intersection < 1e-9:
            return True, 0.0
        vol_intersection_aabb = current_vol_intersection
    else:
        if viz_collector is not None:
            if not hasattr(check_overlap, "_warned_manual_aabb_intersect"):
                print("    DEBUG assembly.py: aabb1.get_intersection not found by hasattr, calculating manually.")
                check_overlap._warned_manual_aabb_intersect = True
        min_b1, max_b1 = aabb1.get_min_bound(), aabb1.get_max_bound()
        min_b2, max_b2 = aabb2.get_min_bound(), aabb2.get_max_bound()
        intersect_min = np.maximum(min_b1, min_b2)
        intersect_max = np.minimum(max_b1, max_b2)
        if np.any(intersect_min >= intersect_max):
            return True, 0.0
        vol_intersection_aabb = np.prod(intersect_max - intersect_min)

    overlap_ratio1 = vol_intersection_aabb / (vol1_aabb + epsilon)
    overlap_ratio2 = vol_intersection_aabb / (vol2_aabb + epsilon)
    overlap_ratio_min = min(overlap_ratio1, overlap_ratio2)

    if (vol1_aabb > epsilon and overlap_ratio1 > adaptive_overlap_factor) or \
       (vol2_aabb > epsilon and overlap_ratio2 > adaptive_overlap_factor):
        if viz_collector is not None:
            viz_collector.append({
                'step': 'overlap_check_failed_aabb', 'type': 'event',
                'mesh1_name': mesh1_name,
                'mesh2_name': mesh2_name,
                'reason': f'Adaptive AABB overlap too high ({overlap_ratio1:.2f} of m1 or {overlap_ratio2:.2f} of m2)',
                'overlap_ratio1': overlap_ratio1,
                'overlap_ratio2': overlap_ratio2,
                'adaptive_overlap_factor': adaptive_overlap_factor
            })
        return False, overlap_ratio_min

    num_sample_points_overlap = params.get("overlap_check_sample_points", 300)
    penetration_allowance_ratio = params.get("overlap_penetration_allowance_ratio", 0.15)
    penetration_depth_factor = params.get("overlap_penetration_depth_factor", 0.25)
    voxel_size_ref = params.get("voxel_downsample_size", 0.01)

    try:
        mesh1_tri = trimesh.Trimesh(vertices=np.asarray(mesh1_o3d.vertices),
                                    faces=np.asarray(mesh1_o3d.triangles))
        if not mesh1_tri.is_watertight and len(mesh1_tri.faces) > 0:
            mesh1_tri.fill_holes()

        if len(mesh1_tri.faces) == 0:
            print("    Overlap Check (Trimesh): mesh1 has no faces for sampling. Relying on AABB.")
            return True, overlap_ratio_min

        sampled_points, _ = trimesh.sample.sample_surface(mesh1_tri, num_sample_points_overlap)

        if len(sampled_points) == 0:
            print("    Overlap Check (Trimesh): Failed to sample points from mesh1. Relying on AABB.")
            return True, overlap_ratio_min

        mesh2_tri = trimesh.Trimesh(vertices=np.asarray(mesh2_o3d.vertices),
                                    faces=np.asarray(mesh2_o3d.triangles))
        if not mesh2_tri.is_watertight and len(mesh2_tri.faces) > 0:
            mesh2_tri.fill_holes()

        if len(mesh2_tri.faces) == 0:
            print("    Overlap Check (Trimesh): mesh2 has no faces for proximity. Relying on AABB.")
            return True, overlap_ratio_min

        proximity_query_mesh2 = trimesh.proximity.ProximityQuery(mesh2_tri)
        signed_distances = proximity_query_mesh2.signed_distance(sampled_points)

        penetration_threshold = - (voxel_size_ref * penetration_depth_factor)
        num_penetrating_points = np.sum(signed_distances < penetration_threshold)

        ratio_penetrating = num_penetrating_points / len(sampled_points) if len(sampled_points) > 0 else 0

        if ratio_penetrating > penetration_allowance_ratio:
            if viz_collector is not None:
                viz_collector.append({
                    'step': 'overlap_check_failed_points', 'type': 'event',
                    'mesh1_name': mesh1_name,
                    'mesh2_name': mesh2_name,
                    'penetration_ratio': ratio_penetrating,
                })
            return False, max(overlap_ratio_min, ratio_penetrating)

    except Exception as e:
        print(f"    Error during Trimesh-based overlap check: {e}. Relying on AABB check result.")
        if viz_collector is not None:
            viz_collector.append({'step': 'overlap_check_trimesh_error', 'type': 'event',
                                   'mesh1_name': mesh1_name, 'mesh2_name': mesh2_name,
                                   'error_message': str(e)})
        return True, overlap_ratio_min
    return True, overlap_ratio_min


class Assembler:
    def __init__(self, fragments_data, pairwise_matches, params, visualization_log=None):
        self.fragments_data = copy.deepcopy(fragments_data) 
        self.pairwise_matches = sorted(pairwise_matches, key=lambda x: x['score'], reverse=True)
        self.params = params
        self.num_fragments = len(fragments_data)

        self.original_meshes = [fd['original_mesh'] for fd in self.fragments_data] 
        
        self.fragment_transforms = [np.eye(4) for _ in range(self.num_fragments)]
        self.is_fragment_placed = [False] * self.num_fragments
        self.assembly_components = [] 
        self.visualization_log = visualization_log if visualization_log is not None else []

    def _get_transformed_mesh(self, fragment_idx_in_assembler_list):
        mesh = copy.deepcopy(self.original_meshes[fragment_idx_in_assembler_list])
        mesh.transform(self.fragment_transforms[fragment_idx_in_assembler_list])
        return mesh

    def optimize_with_pose_graph(self, min_confidence=0.0):
        """
        Build and optimize a pose graph using Open3D's global_optimization.
        Updates self.fragment_transforms with optimized poses.
        """
        if self.num_fragments < 2:
            return
        import open3d as o3d
        pose_graph = o3d.pipelines.registration.PoseGraph()
        # Add nodes (one per fragment)
        for i in range(self.num_fragments):
            if i == 0:
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
            else:
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(self.fragment_transforms[i]))
        # Add edges (from pairwise matches)
        for match in self.pairwise_matches:
            if 'confidence' in match and match['confidence'] < min_confidence:
                continue
            source = match['source_idx']
            target = match['target_idx']
            transformation = match['transformation']
            uncertain = False # Set to True for non-sequential edges if desired
            information = np.eye(6) # Could be weighted by confidence/score
            # FIX: Swap 'information' and 'uncertain' to match Open3D's PoseGraphEdge signature
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source, target, transformation, information, uncertain
                )
            )
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
        # Update transforms
        for i in range(self.num_fragments):
            self.fragment_transforms[i] = pose_graph.nodes[i].pose

    def greedy_assembly(self):
        if self.num_fragments == 0: return None
        if self.num_fragments == 1:
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

        if not self.pairwise_matches:
            print("No pairwise matches for assembly. Cannot proceed with greedy strategy.")
            if self.visualization_log is not None: # Log unplaced if no matches
                for i_log, fd_log in enumerate(self.fragments_data):
                    self.visualization_log.append({
                        'step': 'assembly_failed_no_pairwise_matches', 'type': 'mesh',
                        'fragment_name': fd_log['name'],
                        'original_index': fd_log['original_index'],
                        'fragment_idx_in_valid_list': i_log,
                        'transform': np.eye(4), # At origin
                        'vertices': np.asarray(self.original_meshes[i_log].vertices),
                        'triangles': np.asarray(self.original_meshes[i_log].triangles)
                    })
            return None

        seed_idx = self.pairwise_matches[0]['target_idx'] 
        seed_name = self.fragments_data[seed_idx]['name']
        
        print(f"Starting assembly with seed fragment: {seed_name} (idx in current list: {seed_idx})")
        self.is_fragment_placed[seed_idx] = True
        
        # current_assembly_components stores tuples of (transformed_mesh_object, fragment_name)
        current_assembly_components = [(self._get_transformed_mesh(seed_idx), seed_name)]
        
        if self.visualization_log is not None:
            seed_mesh_transformed_o3d = current_assembly_components[0][0]
            self.visualization_log.append({
                'step': 'assembly_seed_placed', 'type': 'mesh',
                'fragment_name': seed_name,
                'original_index': self.fragments_data[seed_idx]['original_index'],
                'fragment_idx_in_valid_list': seed_idx,
                'transform': self.fragment_transforms[seed_idx],
                'vertices': np.asarray(seed_mesh_transformed_o3d.vertices),
                'triangles': np.asarray(seed_mesh_transformed_o3d.triangles)
            })

        num_placed = 1
        while num_placed < self.num_fragments:
            best_candidate_match_info = None
            best_candidate_score = -1.0 
            best_candidate_world_transform = None
            best_candidate_idx_to_place = -1 # This is an index into self.fragments_data

            for match_info in self.pairwise_matches:
                s_idx, t_idx = match_info['source_idx'], match_info['target_idx']
                s_surf_idx = match_info.get('source_surface_idx', 0)
                t_surf_idx = match_info.get('target_surface_idx', 0)
                # If you need to use a specific surface for overlap or context, use:
                #   self.fragments_data[s_idx]['fracture_surfaces'][s_surf_idx]
                #   self.fragments_data[t_idx]['fracture_surfaces'][t_surf_idx]
                # For now, original_mesh is used for placement, but this is where you'd use the surface if needed.
                
                # These are potential values for the current match_info being considered
                current_iteration_potential_world_transform = None
                current_iteration_idx_to_place = -1

                if self.is_fragment_placed[t_idx] and not self.is_fragment_placed[s_idx]:
                    current_iteration_potential_world_transform = np.dot(self.fragment_transforms[t_idx], match_info['transformation'])
                    current_iteration_idx_to_place = s_idx
                elif self.is_fragment_placed[s_idx] and not self.is_fragment_placed[t_idx]:
                    try:
                        inv_transform = np.linalg.inv(match_info['transformation'])
                        current_iteration_potential_world_transform = np.dot(self.fragment_transforms[s_idx], inv_transform)
                        current_iteration_idx_to_place = t_idx
                    except np.linalg.LinAlgError: 
                        # print(f"Warning: Could not invert transform for match {s_idx}<->{t_idx}. Skipping this path.")
                        continue 
                else: 
                    continue # This match doesn't connect a placed to an unplaced piece

                # If this match is better than what we've found so far in this iteration of the while loop
                if current_iteration_potential_world_transform is not None and current_iteration_idx_to_place != -1:
                    if match_info['score'] > best_candidate_score:
                        candidate_original_mesh_o3d = self.original_meshes[current_iteration_idx_to_place]
                        candidate_mesh_transformed_o3d = copy.deepcopy(candidate_original_mesh_o3d)
                        candidate_mesh_transformed_o3d.transform(current_iteration_potential_world_transform)
                        candidate_name = self.fragments_data[current_iteration_idx_to_place]['name']

                        # DEBUG VISUALIZATION: Show candidate placement if enabled
                        if self.params.get('debug_assembly', False):
                            print(f"[DEBUG] Visualizing candidate placement for {candidate_name}")
                            o3d.visualization.draw_geometries(
                                [candidate_mesh_transformed_o3d] + [m for m, _ in current_assembly_components],
                                window_name=f"Candidate: {candidate_name} (Red), Placed: Gray"
                            )

                        # TOGGLE OVERLAP CHECK
                        if self.params.get('disable_overlap_check', False):
                            overlap_ok = True
                            max_overlap_ratio = 0.0
                        else:
                            overlap_ok = True
                            max_overlap_ratio = 0.0
                            for placed_mesh_o3d, placed_name in current_assembly_components:
                                ok, overlap_ratio = check_overlap(candidate_mesh_transformed_o3d, candidate_name,
                                                                  placed_mesh_o3d, placed_name,
                                                                  self.params, viz_collector=self.visualization_log)
                                if not ok:
                                    overlap_ok = False
                                    break
                                max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

                        # Compute minimum volume among candidate and all placed fragments
                        candidate_tri = trimesh.Trimesh(vertices=np.asarray(candidate_mesh_transformed_o3d.vertices), faces=np.asarray(candidate_mesh_transformed_o3d.triangles))
                        candidate_vol = candidate_tri.volume
                        placed_vols = []
                        for placed_mesh_o3d, _ in current_assembly_components:
                            placed_tri = trimesh.Trimesh(vertices=np.asarray(placed_mesh_o3d.vertices), faces=np.asarray(placed_mesh_o3d.triangles))
                            placed_vols.append(placed_tri.volume)
                        min_volume_all = min([candidate_vol] + placed_vols)
                        # Instead of check_overlap, use boolean_intersection_penetration_test for each placed fragment.
                        # Allow up to 10% penetration (set in params as 'boolean_penetration_threshold': 0.1)
                        # Log results to visualization_log if enabled.
                        # If any test fails (penetration > threshold), reject the candidate placement.
                        penetration_ok = True
                        for placed_mesh_o3d, placed_name in current_assembly_components:
                            is_valid, penetration_ratio, intersection_mesh = boolean_intersection_penetration_test(
                                candidate_mesh_transformed_o3d, candidate_name,
                                placed_mesh_o3d, placed_name,
                                self.params, viz_collector=self.visualization_log,
                                min_volume_override=min_volume_all
                            )
                            # Print penetration test result to command line
                            penetration_pct = penetration_ratio * 100
                            threshold_pct = self.params.get('boolean_penetration_threshold', 0.1) * 100
                            print(f"    [Penetration Test] {candidate_name} vs {placed_name}: {penetration_pct:.2f}% (threshold: {threshold_pct:.2f}%) [min volume: {min_volume_all:.4f}] -> {'PASS' if is_valid else 'FAIL'}")
                            if not is_valid:
                                penetration_ok = False
                                if self.visualization_log is not None:
                                    self.visualization_log.append({
                                        'step': 'overlap_check_failed_boolean_penetration', 'type': 'event',
                                        'mesh1_name': candidate_name,
                                        'mesh2_name': placed_name,
                                        'reason': f'Boolean penetration test failed: {penetration_ratio*100:.1f}% penetration',
                                        'penetration_ratio': penetration_ratio,
                                        'max_penetration_allowed': self.params.get('boolean_penetration_threshold', 0.1) * 100,
                                        'min_volume_used': min_volume_all
                                    })
                                break # No need to check further placed fragments if one fails

                        if overlap_ok and penetration_ok: # This candidate is good and has the best score so far
                            best_candidate_match_info = match_info
                            best_candidate_score = match_info['score']
                            best_candidate_world_transform = current_iteration_potential_world_transform
                            best_candidate_idx_to_place = current_iteration_idx_to_place
                            best_candidate_overlap_ratio = max_overlap_ratio
            
            # After checking all pairwise_matches for the current assembly state
            if best_candidate_idx_to_place != -1 and best_candidate_match_info is not None:
                newly_placed_idx_in_list = best_candidate_idx_to_place
                newly_placed_name = self.fragments_data[newly_placed_idx_in_list]['name']

                self.fragment_transforms[newly_placed_idx_in_list] = best_candidate_world_transform
                self.is_fragment_placed[newly_placed_idx_in_list] = True
                
                placed_mesh_o3d_for_list = self._get_transformed_mesh(newly_placed_idx_in_list)
                current_assembly_components.append((placed_mesh_o3d_for_list, newly_placed_name))
                num_placed += 1
                print(f"  Placed fragment: {newly_placed_name} "
                      f"(idx in list: {newly_placed_idx_in_list}) via match score {best_candidate_score:.3f}.")
                
                if self.visualization_log is not None:
                    log_entry = {
                        'step': 'assembly_fragment_placed', 'type': 'mesh',
                        'fragment_name': newly_placed_name,
                        'original_index': self.fragments_data[newly_placed_idx_in_list]['original_index'],
                        'fragment_idx_in_valid_list': newly_placed_idx_in_list,
                        'transform': self.fragment_transforms[newly_placed_idx_in_list],
                        'vertices': np.asarray(placed_mesh_o3d_for_list.vertices),
                        'triangles': np.asarray(placed_mesh_o3d_for_list.triangles),
                        'matched_via_score': best_candidate_score,
                        'overlap_ratio': best_candidate_overlap_ratio
                    }
                    if best_candidate_match_info is not None:
                        log_entry['match_details'] = {
                            'source_idx': best_candidate_match_info['source_idx'],
                            'target_idx': best_candidate_match_info['target_idx'],
                            'source_name': best_candidate_match_info['source_name'],
                            'target_name': best_candidate_match_info['target_name'],
                            'score': best_candidate_match_info['score'],
                            'rmse': best_candidate_match_info['rmse'],
                        }
                    self.visualization_log.append(log_entry)
            else:
                print("No more non-overlapping, valid matches found to extend the assembly.")
                break
        
        if num_placed < self.num_fragments:
            print(f"Warning: Only {num_placed}/{self.num_fragments} fragments were assembled.")
            unplaced_indices = [i for i, placed in enumerate(self.is_fragment_placed) if not placed]
            print("Unplaced fragment indices (relative to valid_fragments_data list):", unplaced_indices)
            for idx_unplaced in unplaced_indices:
                print(f" - {self.fragments_data[idx_unplaced]['name']}")
                if self.visualization_log is not None: # Log unplaced fragments
                     self.visualization_log.append({
                        'step': 'assembly_fragment_unplaced', 'type': 'mesh',
                        'fragment_name': self.fragments_data[idx_unplaced]['name'],
                        'original_index': self.fragments_data[idx_unplaced]['original_index'],
                        'fragment_idx_in_valid_list': idx_unplaced,
                        'transform': np.eye(4), # At origin, as it wasn't placed
                        'vertices': np.asarray(self.original_meshes[idx_unplaced].vertices),
                        'triangles': np.asarray(self.original_meshes[idx_unplaced].triangles)
                    })


        final_meshes_to_combine_o3d = []
        final_transforms_for_combine = []
        fragment_colors = []
        # Assign a unique color to each fragment for visualization
        for i in range(self.num_fragments):
            if self.is_fragment_placed[i]:
                mesh = copy.deepcopy(self.original_meshes[i])
                # Assign color
                if cmap:
                    color = cmap(i % cmap.N)[:3]
                else:
                    # Fallback simple color cycle
                    color_cycle = [
                        [1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],
                        [0.8,0.5,0.2],[0.5,0.2,0.8],[0.2,0.8,0.5],[0.6,0.6,0.6]
                    ]
                    color = color_cycle[i % len(color_cycle)]
                mesh.paint_uniform_color(color)
                final_meshes_to_combine_o3d.append(mesh)
                final_transforms_for_combine.append(self.fragment_transforms[i])
                fragment_colors.append(color)
        
        if not final_meshes_to_combine_o3d:
            print("Error: No meshes were placed in the assembly.")
            return None

        # --- GLOBAL POSE GRAPH OPTIMIZATION ---
        if self.num_fragments > 2:
            print("\n[Pose Graph Optimization] Refining fragment poses globally...")
            self.optimize_with_pose_graph(min_confidence=0.0)

        # --- VISUALIZATION: After Pose Graph Optimization ---
        print("[Visualization] Showing output after pose graph optimization (before snapping)...")
        pose_graph_meshes = []
        for mesh, transform in zip(final_meshes_to_combine_o3d, final_transforms_for_combine):
            mesh_pg = copy.deepcopy(mesh)
            mesh_pg.transform(transform)
            pose_graph_meshes.append(mesh_pg)
        o3d.visualization.draw_geometries(
            pose_graph_meshes,
            window_name="Final Composite Assembly: After Pose Graph Optimization"
        )

        # --- POST-PROCESSING SNAPPING STEP ---
        print("[Post-Processing] Snapping adjacent fragments together...")
        SNAP_SCORE_THRESHOLD = 0.7
        snapped_transforms = list(final_transforms_for_combine)
        # For each high-scoring match, snap the source to the target
        for match in self.pairwise_matches:
            if match['score'] < SNAP_SCORE_THRESHOLD:
                continue
            s_idx = match['source_idx']
            t_idx = match['target_idx']
            # Only snap if both fragments are placed
            if s_idx >= len(final_meshes_to_combine_o3d) or t_idx >= len(final_meshes_to_combine_o3d):
                continue
            source_mesh = copy.deepcopy(final_meshes_to_combine_o3d[s_idx])
            target_mesh = copy.deepcopy(final_meshes_to_combine_o3d[t_idx])
            source_mesh.transform(snapped_transforms[s_idx])
            target_mesh.transform(snapped_transforms[t_idx])
            # Find closest points between source and target
            src_points = np.asarray(source_mesh.vertices)
            tgt_points = np.asarray(target_mesh.vertices)
            from scipy.spatial import cKDTree
            tree = cKDTree(tgt_points)
            dists, idxs = tree.query(src_points)
            min_dist = np.min(dists)
            min_src_idx = np.argmin(dists)
            min_tgt_idx = idxs[min_src_idx]
            # Compute translation vector to bring source closer to target
            src_pt = src_points[min_src_idx]
            tgt_pt = tgt_points[min_tgt_idx]
            vec = tgt_pt - src_pt
            # Move source fragment by this vector (minus a small epsilon to avoid overlap)
            epsilon = 1e-4
            snap_vec = vec - epsilon * (vec / (np.linalg.norm(vec) + 1e-8))
            snapped_transforms[s_idx][:3, 3] += snap_vec
        # Apply snapped transforms to meshes
        snapped_meshes = []
        for i, mesh in enumerate(final_meshes_to_combine_o3d):
            mesh_snapped = copy.deepcopy(mesh)
            mesh_snapped.transform(snapped_transforms[i])
            snapped_meshes.append(mesh_snapped)
        print("[Post-Processing] Snapping complete. Visualizing snapped fragments...")
        o3d.visualization.draw_geometries(
            snapped_meshes,
            window_name="Final Composite Assembly: Snapped Fragments"
        )

        final_mesh = combine_meshes(snapped_meshes, [np.eye(4)] * len(snapped_meshes))

        # Save the final mesh to data/output_assembly
        import os
        output_dir = os.path.join("data", "output_assembly")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "reconstructed_model.obj")
        save_mesh(final_mesh, output_path)
        print(f"Final assembled model saved to: {output_path}")

        return final_mesh

# In src/assembly.py, at the end of the file:

if __name__ == '__main__':
    import os
    import json
    # To test assembly.py directly, we need to simulate the data structures
    # that would normally be created by the preceding steps in main.py.

    # --- Configuration (Simplified for this test) ---
    # Normally loaded from config/reconstruction_params.json
    # For this test, we'll define some crucial ones directly.
    # Ensure these match the scale and expectations of your test fragments.
    test_params = {
        "voxel_downsample_size": 0.1, # Adjusted for potentially simpler test meshes
        "max_assembly_overlap_factor_aabb": 0.9,
        "overlap_check_sample_points": 100,
        "overlap_penetration_allowance_ratio": 0.20, # More lenient for simple test
        "overlap_penetration_depth_factor": 0.3,
        "boolean_penetration_threshold": 0.1, # Added for boolean penetration test
        # Add any other params directly used by Assembler or check_overlap if not defaulted
    }
    print(f"Using test parameters: {test_params}")

    # --- Create Dummy/Test Fragments Data ---
    # This would normally come from io_utils, preprocessing, feature_extraction
    
    # Example: Two simple cubes that should fit together
    # Cube 1: origin (0,0,0) to (1,1,1)
    mesh1 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    mesh1.compute_vertex_normals()
    
    # Cube 2: origin (1,0,0) to (2,1,1) - i.e., shifted by 1 unit in X
    mesh2 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    mesh2.translate([1, 0, 0]) # Position it to mate with mesh1's +X face
    mesh2.compute_vertex_normals()

    # Slightly transform mesh2 away so the alignment is non-trivial for a full pipeline test
    # For this isolated assembly test, we'll assume alignment has already happened
    # and provide a "perfect" pairwise match.

    # Structure expected by Assembler (from valid_fragments_data in main.py)
    # Key fields for Assembler: 'original_mesh', 'name', 'original_index'
    # 'pcd_for_features' and 'features' are not directly used by Assembler class itself,
    # but they are part of the structure it receives.
    fragments_for_assembler = [
        {
            'original_mesh': mesh1, 
            'name': 'CubePart1', 
            'original_index': 0,
            'pcd_for_features': None, # Dummy for this test
            'features': None          # Dummy for this test
        },
        {
            'original_mesh': mesh2, 
            'name': 'CubePart2', 
            'original_index': 1,
            'pcd_for_features': None, # Dummy
            'features': None          # Dummy
        },
    ]
    print(f"Created {len(fragments_for_assembler)} test fragments for assembler.")

    # --- Create Dummy Pairwise Matches ---
    # This list would normally come from matching.py
    # We need to define a transformation that aligns mesh2 (source_idx=1) to mesh1 (target_idx=0)
    # Since mesh2 is already at [1,0,0] relative to mesh1 at [0,0,0] to form a 2x1x1 block,
    # if mesh1 is the target, mesh2 needs to be transformed from its current position
    # to align with mesh1.
    # Let's simulate a scenario where mesh2 was initially at, say, [5,0,0] and needs to be moved.
    
    # Assume mesh2 (idx 1) is the source and mesh1 (idx 0) is the target.
    # If mesh2 was at some arbitrary pose, and we found a transform to align it to mesh1:
    # T_mesh2_to_mesh1 would be the transformation.
    # For this simple test, let's assume mesh2 is at its final correct relative pose
    # TO mesh1 IF mesh1 is at origin.
    # If mesh1 is at origin, and mesh2 should be at [1,0,0] next to it,
    # and if mesh2 is ALREADY at [1,0,0] (as created above), then the transformation
    # to bring mesh2 (source) to align with mesh1 (target, at origin) if mesh2 started at origin
    # would be a translation by [1,0,0].
    
    # Let's make a more explicit test case:
    # Fragment A (idx 0) at origin
    # Fragment B (idx 1) initially at [10,0,0], needs to be moved to [1,0,0] to connect to A's +X face.
    # So, the transformation for B is a translation by [-9,0,0] if B is the source.
    
    mesh_A_orig = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    mesh_A_orig.compute_vertex_normals()

    mesh_B_orig = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    # Initial position of mesh_B (e.g., as loaded from a file, already transformed away)
    initial_transform_B = np.eye(4)
    initial_transform_B[:3,3] = [10,0,0] 
    mesh_B_at_initial_pos = copy.deepcopy(mesh_B_orig)
    mesh_B_at_initial_pos.transform(initial_transform_B)
    mesh_B_at_initial_pos.compute_vertex_normals()


    fragments_for_assembler_test2 = [
        {'original_mesh': mesh_A_orig, 'name': 'PartA', 'original_index': 0, 'pcd_for_features': None, 'features': None},
        {'original_mesh': mesh_B_orig, 'name': 'PartB', 'original_index': 1, 'pcd_for_features': None, 'features': None},
        # Note: Assembler uses original_meshes. The fact that PartB is "loaded" at [10,0,0] is
        # captured by the transformation in pairwise_matches.
    ]

    # Transformation that takes PartB (source, idx 1) from its CURRENT conceptual space 
    # (which is effectively origin for its definition in original_meshes) and aligns it 
    # to PartA (target, idx 0), which is at origin.
    # To place PartB next to PartA's +X face, PartB needs to be at [1,0,0] in PartA's frame.
    # So, T_PartB_to_PartA is a translation by [1,0,0].
    transform_B_to_A = np.eye(4)
    transform_B_to_A[0,3] = 1.0 

    # If PartA (source, idx 0) were to be aligned to PartB (target, idx 1)
    # and PartB is considered fixed at origin (for this hypothetical match),
    # PartA would need to be moved to [-1,0,0].
    transform_A_to_B = np.eye(4)
    transform_A_to_B[0,3] = -1.0

    test_pairwise_matches = [
        { # Best match: B aligns to A
            'source_idx': 1, 'target_idx': 0, # PartB (source) to PartA (target)
            'transformation': transform_B_to_A,
            'score': 0.9, 'rmse': 0.01,
            'source_name': 'PartB', 'target_name': 'PartA'
        },
        { # A weaker match for testing sorting
            'source_idx': 0, 'target_idx': 1, # PartA (source) to PartB (target)
            'transformation': transform_A_to_B,
            'score': 0.8, 'rmse': 0.02,
            'source_name': 'PartA', 'target_name': 'PartB'
        }
    ]
    print(f"Created {len(test_pairwise_matches)} test pairwise matches.")

    # --- Initialize Visualization Log ---
    test_visualization_log = []

    # --- Create and Run Assembler ---
    print("\nInitializing and running Assembler for test...")
    assembler_test_instance = Assembler(fragments_for_assembler_test2, 
                                        test_pairwise_matches, 
                                        test_params, 
                                        visualization_log=test_visualization_log)
    
    final_assembled_mesh = assembler_test_instance.greedy_assembly()

    # --- Output Results ---
    if final_assembled_mesh and final_assembled_mesh.has_vertices():
        print("\nAssembly test successful!")
        output_dir_test = "data/output_assembly_test" # Separate dir for this test
        os.makedirs(output_dir_test, exist_ok=True)
        test_output_path = os.path.join(output_dir_test, "assembled_test_model.obj")
        save_mesh(final_assembled_mesh, test_output_path)
        print(f"  Test assembled model saved to: {test_output_path}")
        
        # Optionally visualize
        # print("  Visualizing test assembled model...")
        # o3d.visualization.draw_geometries([final_assembled_mesh], window_name="Test Assembled Model")
    else:
        print("\nAssembly test failed or resulted in an empty model.")

    print(f"\nVisualization log for assembly test has {len(test_visualization_log)} entries.")
    # You could print some log entries here for inspection if desired.
    # for log_entry_test in test_visualization_log:
    #     print(f"  Log: {log_entry_test.get('step')} - {log_entry_test.get('fragment_name', '')}")

    # To fully test the visualization replay, you'd save this log and use replay_log.py
    if test_visualization_log:
        from src.utils import visualization_utils # Import for saving
        test_log_file = os.path.join(output_dir_test, "assembly_test_log.pkl")
        visualization_utils.save_visualization_log(test_visualization_log, test_log_file)
        print(f"  Test visualization log saved to: {test_log_file}")
        print(f"  To replay, use replay_log.py with this file path.")