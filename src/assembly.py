import trimesh
import numpy as np
import open3d as o3d
import copy
from src.io_utils import combine_meshes

def check_overlap(mesh1_o3d, mesh2_o3d, params, viz_collector=None):
    """
    Checks for significant overlap between two Open3D meshes.
    Uses Trimesh for robust signed distance calculation after an initial AABB check.
    Logs failures to viz_collector if provided.
    """
    if not mesh1_o3d.has_vertices() or not mesh2_o3d.has_vertices():
        return True  # No geometry to overlap

    # --- Coarse AABB Overlap Check First (from previous robust version) ---
    aabb1 = mesh1_o3d.get_axis_aligned_bounding_box()
    aabb2 = mesh2_o3d.get_axis_aligned_bounding_box()
    max_overlap_factor_aabb = params.get("max_assembly_overlap_factor_aabb", 0.8) # Configurable AABB overlap

    vol_intersection_aabb = 0.0
    # Use hasattr to check for get_intersection for Open3D version compatibility
    if hasattr(aabb1, 'get_intersection'):
        intersection_aabb = aabb1.get_intersection(aabb2)
        current_vol_intersection = intersection_aabb.volume()
        if current_vol_intersection < 1e-9: return True 
        vol_intersection_aabb = current_vol_intersection
    else:
        # Manual AABB intersection (if get_intersection not found by hasattr)
        if viz_collector is not None: # Log that we are using manual method (first time per run is enough)
            if not hasattr(check_overlap, "_warned_manual_aabb_intersect"):
                print("    DEBUG assembly.py: aabb1.get_intersection not found by hasattr, calculating manually.")
                check_overlap._warned_manual_aabb_intersect = True # Warn only once
        min_b1, max_b1 = aabb1.get_min_bound(), aabb1.get_max_bound()
        min_b2, max_b2 = aabb2.get_min_bound(), aabb2.get_max_bound()
        intersect_min = np.maximum(min_b1, min_b2)
        intersect_max = np.minimum(max_b1, max_b2)
        if np.any(intersect_min >= intersect_max): return True
        vol_intersection_aabb = np.prod(intersect_max - intersect_min)

    vol1_aabb = aabb1.volume()
    vol2_aabb = aabb2.volume()
    epsilon = 1e-9
    
    if (vol1_aabb > epsilon and (vol_intersection_aabb / vol1_aabb) > max_overlap_factor_aabb) or \
       (vol2_aabb > epsilon and (vol_intersection_aabb / vol2_aabb) > max_overlap_factor_aabb):
        if viz_collector is not None:
            # Try to get names; they might not be set on these temporary mesh objects
            m1_name = getattr(mesh1_o3d, 'fragment_name_for_viz', 'CandidateMesh')
            m2_name = getattr(mesh2_o3d, 'fragment_name_for_viz', 'PlacedMesh')
            viz_collector.append({
                'step': 'overlap_check_failed_aabb',
                'mesh1_name': m1_name, 'mesh2_name': m2_name,
                'reason': f'AABB overlap too high ({vol_intersection_aabb/vol1_aabb:.2f} of m1 or {vol_intersection_aabb/vol2_aabb:.2f} of m2)'
                # To visualize this, we'd need the meshes and their transforms
                # For now, just log the event
            })
        return False # Too much AABB overlap, likely bad alignment or identical pieces

    # --- Finer Overlap Check using Point Sampling and Signed Distance (Trimesh) ---
    num_sample_points_overlap = params.get("overlap_check_sample_points", 300)
    penetration_allowance_ratio = params.get("overlap_penetration_allowance_ratio", 0.15) # Allow 10% of points to penetrate
    # Penetration depth relative to voxel_size (e.g., how deep can points go inside)
    penetration_depth_factor = params.get("overlap_penetration_depth_factor", 0.25) 
    voxel_size_ref = params.get("voxel_downsample_size", 0.01) # For scaling penetration depth

    try:
        # Convert candidate mesh (mesh1_o3d) to Trimesh for sampling
        # It's already transformed to its proposed assembly position
        mesh1_tri = trimesh.Trimesh(vertices=np.asarray(mesh1_o3d.vertices),
                                    faces=np.asarray(mesh1_o3d.triangles))
        if not mesh1_tri.is_watertight and len(mesh1_tri.faces) > 0 : mesh1_tri.fill_holes()
        
        # Sample points from the surface of mesh1
        # Ensure faces are available for trimesh.sample.sample_surface
        if len(mesh1_tri.faces) == 0:
             print("    Overlap Check: mesh1 has no faces for Trimesh sampling. Relying on AABB.")
             return True # If cannot sample, assume AABB check was sufficient

        sampled_points, _ = trimesh.sample.sample_surface(mesh1_tri, num_sample_points_overlap)
        
        if len(sampled_points) == 0:
            print("    Overlap Check: Failed to sample points from mesh1. Relying on AABB.")
            return True

        # Convert mesh2 (already placed part of assembly) to Trimesh for proximity query
        mesh2_tri = trimesh.Trimesh(vertices=np.asarray(mesh2_o3d.vertices),
                                    faces=np.asarray(mesh2_o3d.triangles))
        if not mesh2_tri.is_watertight and len(mesh2_tri.faces) > 0: mesh2_tri.fill_holes()

        if len(mesh2_tri.faces) == 0:
             print("    Overlap Check: mesh2 has no faces for Trimesh proximity. Relying on AABB.")
             return True

        proximity_query_mesh2 = trimesh.proximity.ProximityQuery(mesh2_tri)
        signed_distances = proximity_query_mesh2.signed_distance(sampled_points)

        # Negative distances mean points are inside mesh2.
        # Allow for slight penetration, e.g., 20% of voxel_size
        penetration_threshold = - (voxel_size_ref * penetration_depth_factor)
        num_penetrating_points = np.sum(signed_distances < penetration_threshold)
        
        ratio_penetrating = num_penetrating_points / len(sampled_points) if len(sampled_points) > 0 else 0

        if ratio_penetrating > penetration_allowance_ratio:
            # print(f"    Overlap Check: Rejected by point penetration. Ratio: {ratio_penetrating:.2f}")
            if viz_collector is not None:
                m1_name = getattr(mesh1_o3d, 'fragment_name_for_viz', 'CandidateMesh')
                m2_name = getattr(mesh2_o3d, 'fragment_name_for_viz', 'PlacedMesh')
                viz_collector.append({
                    'step': 'overlap_check_failed_points', 'type': 'event',
                    'mesh1_name': m1_name, 'mesh2_name': m2_name,
                    'penetration_ratio': ratio_penetrating,
                    # Log the points and meshes if you want to visualize this specific failure
                    # 'mesh1_verts': np.asarray(mesh1_o3d.vertices),
                    # 'mesh1_tris': np.asarray(mesh1_o3d.triangles),
                    # 'mesh2_verts': np.asarray(mesh2_o3d.vertices),
                    # 'mesh2_tris': np.asarray(mesh2_o3d.triangles),
                    # 'sampled_points_penetrating': sampled_points[signed_distances < penetration_threshold]
                })
            return False # Too many points interpenetrating
            
    except Exception as e:
        print(f"    Error during Trimesh-based overlap check: {e}. Relying on AABB check result.")
        if viz_collector is not None:
            viz_collector.append({'step': 'overlap_check_trimesh_error', 'type': 'event', 'error_message': str(e)})
        # If Trimesh fails, the AABB check already determined if it's a gross overlap.
        # If AABB passed, and Trimesh fails, we assume it's okay for now.
        # A more conservative approach would be to return False here.
        return True # Or False if you want to be safer when Trimesh fails

    return True # Overlap is acceptable

class Assembler:
    def __init__(self, fragments_data, pairwise_matches, params, visualization_log=None): # visualization_log param added
        """
        Args:
            fragments_data (list of dict): Contains 'mesh', 'name', 'original_index', etc.
                                           The 'mesh' here is the original full-resolution mesh.
            pairwise_matches (list of dict): From matching.py.
            params (dict): Configuration parameters.
        """
        self.fragments_data = copy.deepcopy(fragments_data) # Work on copies
        self.pairwise_matches = sorted(pairwise_matches, key=lambda x: x['score'], reverse=True)
        self.params = params
        self.num_fragments = len(fragments_data)

        # Store the original meshes, not the PCDs, for final assembly
        self.original_meshes = [fd['original_mesh'] for fd in self.fragments_data] #Ensure using 'original_mesh'
        
        # Keeps track of the current transformation of each fragment relative to world (or first fragment)
        self.fragment_transforms = [np.eye(4) for _ in range(self.num_fragments)]
        self.is_fragment_placed = [False] * self.num_fragments
        self.assembly_components = [] # List of lists, each sublist is a connected component
        # This line uses the 'visualization_log' parameter from the __init__ signature
        self.visualization_log = visualization_log if visualization_log is not None else [] # Store the log

    def _get_transformed_mesh(self, fragment_idx_in_assembler_list, for_viz_name=None):
        # fragment_idx_in_assembler_list is an index for self.original_meshes, self.fragment_transforms etc.
        mesh = copy.deepcopy(self.original_meshes[fragment_idx_in_assembler_list])
        mesh.transform(self.fragment_transforms[fragment_idx_in_assembler_list])
        if for_viz_name:
            mesh.fragment_name_for_viz = for_viz_name 
        return mesh

    def greedy_assembly(self):
        """
        A very basic greedy assembly strategy.
        1. Start with the fragment involved in the best match (or an arbitrary one if no matches).
        2. Iteratively add the best-matching unplaced fragment to the current assembly.
        """
        if self.num_fragments == 0:
            return None
        if self.num_fragments == 1:
            # Log single fragment "assembly"
            frag_data = self.fragments_data[0]
            mesh_to_log = self.original_meshes[0] # It's at identity transform
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

        # Initialize: Place the first fragment (e.g., largest or arbitrary)
        if not self.pairwise_matches:
            print("No pairwise matches for assembly. Cannot proceed with greedy strategy.")
            # Log all fragments as unplaced if no matches
            if self.visualization_log is not None:
                for i, frag_data in enumerate(self.fragments_data):
                    mesh_to_log = self.original_meshes[i]
                    self.visualization_log.append({
                        'step': 'assembly_no_matches_unplaced', 'type': 'mesh',
                        'fragment_name': frag_data['name'],
                        'original_index': frag_data['original_index'],
                        'fragment_idx_in_valid_list': i,
                        'transform': np.eye(4), # At origin
                        'vertices': np.asarray(mesh_to_log.vertices),
                        'triangles': np.asarray(mesh_to_log.triangles)
                    })
            return None # Or combine_meshes with identity transforms if preferred

        # seed_idx is an index into self.fragments_data (which is valid_fragments_data from main.py)
        seed_idx = self.pairwise_matches[0]['target_idx'] 
        
        print(f"Starting assembly with seed fragment: {self.fragments_data[seed_idx]['name']} (idx in current list: {seed_idx})")
        self.is_fragment_placed[seed_idx] = True
        
        # Pass name for viz logging in check_overlap
        seed_viz_name = self.fragments_data[seed_idx]['name']
        current_assembly_meshes = [self._get_transformed_mesh(seed_idx, for_viz_name=seed_viz_name)]
                
        # Log the seed piece placement
        if self.visualization_log is not None:
            seed_mesh_transformed_o3d = current_assembly_meshes[0]
            self.visualization_log.append({
                'step': 'assembly_seed_placed', 'type': 'mesh',
                'fragment_name': self.fragments_data[seed_idx]['name'],
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
            best_candidate_idx_to_place = -1

            # Find the best match connecting an unplaced fragment to any placed fragment
            for match_info in self.pairwise_matches:
                s_idx, t_idx = match_info['source_idx'], match_info['target_idx']
                
                potential_world_transform = None
                idx_to_place = -1
                
                # Case 1: Target is placed, Source is not. Try to add Source.
                if self.is_fragment_placed[t_idx] and not self.is_fragment_placed[s_idx]:
                    potential_world_transform = np.dot(self.fragment_transforms[t_idx], match_info['transformation'])
                    idx_to_place = s_idx                     

                # Case 2: Source is placed, Target is not. Add Target.
                elif self.is_fragment_placed[s_idx] and not self.is_fragment_placed[t_idx]:
                    try:
                        inv_transform = np.linalg.inv(match_info['transformation'])
                        potential_world_transform = np.dot(self.fragment_transforms[s_idx], inv_transform)
                        idx_to_place = t_idx
                    except np.linalg.LinAlgError: continue # Skip if transform not invertible
                else: # Either both placed, or both not placed (relative to this match's connection to assembly)
                    continue
                        
                if potential_world_transform is not None and idx_to_place != -1:
                    if match_info['score'] > best_candidate_score: # Prioritize higher score first
                        candidate_mesh_o3d = copy.deepcopy(self.original_meshes[idx_to_place])
                        candidate_mesh_o3d.transform(potential_world_transform)
                        # Attach name for check_overlap logging
                        candidate_mesh_o3d.fragment_name_for_viz = self.fragments_data[idx_to_place]['name']
                        
                        overlap_ok = True
                        for placed_mesh_o3d in current_assembly_meshes:
                            # placed_mesh_o3d should have its 'fragment_name_for_viz' set when added
                            if not check_overlap(candidate_mesh_o3d, placed_mesh_o3d, self.params, viz_collector=self.visualization_log):
                                overlap_ok = False
                                break
                        
                        if overlap_ok:
                            best_candidate_match_info = match_info
                            best_candidate_score = match_info['score']
                            best_candidate_world_transform = potential_world_transform
                            best_candidate_idx_to_place = idx_to_place
            
            if best_candidate_idx_to_place != -1:
                newly_placed_idx_in_list = best_candidate_idx_to_place # This is the index in self.fragments_data

                self.fragment_transforms[newly_placed_idx_in_list] = best_candidate_world_transform
                self.is_fragment_placed[newly_placed_idx_in_list] = True
                
                # Get the mesh with its name for viz logging in check_overlap
                placed_mesh_o3d_for_list = self._get_transformed_mesh(
                    newly_placed_idx_in_list, 
                    for_viz_name=self.fragments_data[newly_placed_idx_in_list]['name']
                )
                current_assembly_meshes.append(placed_mesh_o3d_for_list)
                num_placed += 1
                print(f"  Placed fragment: {self.fragments_data[newly_placed_idx_in_list]['name']} "
                      f"(idx in list: {newly_placed_idx_in_list}) via match score {best_candidate_score:.3f}.")
                
                if self.visualization_log is not None:
                    self.visualization_log.append({
                        'step': 'assembly_fragment_placed', 'type': 'mesh',
                        'fragment_name': self.fragments_data[newly_placed_idx_in_list]['name'],
                        'original_index': self.fragments_data[newly_placed_idx_in_list]['original_index'],
                        'fragment_idx_in_valid_list': newly_placed_idx_in_list,
                        'transform': self.fragment_transforms[newly_placed_idx_in_list],
                        'vertices': np.asarray(placed_mesh_o3d_for_list.vertices),
                        'triangles': np.asarray(placed_mesh_o3d_for_list.triangles),
                        'matched_via_score': best_candidate_score,
                        'match_details': { # Store serializable parts of match_info
                            'source_idx': best_candidate_match_info['source_idx'],
                            'target_idx': best_candidate_match_info['target_idx'],
                            'source_name': best_candidate_match_info['source_name'],
                            'target_name': best_candidate_match_info['target_name'],
                            'score': best_candidate_match_info['score'],
                            'rmse': best_candidate_match_info['rmse'],
                        }
                    })
            else:
                print("No more non-overlapping, valid matches found to extend the assembly.")
                break # No more pieces can be added to this component

        if num_placed < self.num_fragments:
            print(f"Warning: Only {num_placed}/{self.num_fragments} fragments were assembled.")
            unplaced_indices = [i for i, placed in enumerate(self.is_fragment_placed) if not placed]
            print("Unplaced fragment indices:", unplaced_indices)
            for idx in unplaced_indices:
                print(f" - {self.fragments_data[idx]['name']}")


        # Collect all placed original meshes and their final transformations
        final_meshes_to_combine_o3d = []
        final_transforms_for_combine = []
        for i in range(self.num_fragments):
            if self.is_fragment_placed[i]:
                final_meshes_to_combine_o3d.append(self.original_meshes[i]) 
                final_transforms_for_combine.append(self.fragment_transforms[i])
        
        if not final_meshes_to_combine_o3d:
            print("Error: No meshes were placed in the assembly.")
            return None

        return combine_meshes(final_meshes_to_combine_o3d, final_transforms_for_combine)


if __name__ == '__main__':
    from io_utils import load_fragments_from_directory
    from preprocessing import preprocess_fragment
    from feature_extraction import extract_features_from_pcd
    from matching import find_pairwise_matches
    import json
    import os

    # Config
    dummy_params = {
        "voxel_downsample_size": 0.05,
        "normal_estimation_radius": 0.1, "normal_estimation_max_nn": 30,
        "fpfh_feature_radius": 0.25, "fpfh_feature_max_nn": 100,
        "ransac_distance_threshold_factor": 1.5, "ransac_edge_length_factor": 0.9,
        "ransac_iterations": 10000, "ransac_n_points": 3, "ransac_confidence": 0.99,
        "icp_max_correspondence_distance_factor": 2.0,
        "icp_relative_fitness": 1e-6, "icp_relative_rmse": 1e-6, "icp_max_iteration": 30,
        "min_match_score": 0.4, # Lower for testing with simple cubes
        "max_assembly_overlap_factor": 0.7 # Allow more overlap for simple test
    }

    # Use dummy data from matching.py test if available, or recreate
    base_dir = '../dummy_data_assembly' # Relative to src/
    input_dir = os.path.join(base_dir, 'input_fragments')
    output_dir = os.path.join(base_dir, 'output_assembly')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create two cube parts that fit along X axis
    cube_obj_content_part1 = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3 4
f 8 7 6 5
f 1 5 6 2
f 3 7 8 4
f 4 8 5 1
# Missing face: 2 6 7 3 (positive X face)
""" # This is piece 1, open on +X side
    
    cube_obj_content_part2 = """
v 1 0 0
v 2 0 0
v 2 1 0
v 1 1 0
v 1 0 1
v 2 0 1
v 2 1 1
v 1 1 1
f 1 2 3 4 # This face is effectively at X=1 of combined
f 8 7 6 5 # This face is at X=2 of combined
# f 1 5 6 2 # This is the connecting face (negative X face of this part)
f 2 6 7 3 # Positive X face of this part
f 3 7 8 4
f 4 8 5 1
# Missing face: 1 5 6 2 (negative X face for this part, should mate with part1's open +X face)
""" # This is piece 2, starts at x=1, open on -X side (relative to its own coords)

    # To make them mate:
    # Part1: vertices as is.
    # Part2: vertices need to be shifted such that its local (0,y,z) matches Part1's (1,y,z)
    # For the test data, part2 is already defined to be next to part1 if simply concatenated.
    # We need to provide them as separate files, potentially with one transformed away.

    with open(os.path.join(input_dir, 'part1.obj'), 'w') as f:
        f.write(cube_obj_content_part1)
    
    mesh_part2_orig = o3d.geometry.TriangleMesh() # Create part2 programmatically
    verts_p2 = np.array([
        [1,0,0], [2,0,0], [2,1,0], [1,1,0],
        [1,0,1], [2,0,1], [2,1,1], [1,1,1]
    ])
    faces_p2 = np.array([
        [0,1,2], [0,2,3], # Front face at x=1 (local)
        [7,6,5], [7,5,4], # Back face at x=2 (local)
        [1,5,6], [1,6,2], # Right face (+X of part2)
        [3,2,6], [3,6,7], # Top face
        [4,0,3], [4,3,7], # Left face (-X of part2) -> THIS IS THE MATING FACE
        [0,4,5], [0,5,1]  # Bottom face
    ])
    mesh_part2_orig.vertices = o3d.utility.Vector3dVector(verts_p2)
    mesh_part2_orig.triangles = o3d.utility.Vector3iVector(faces_p2)
    mesh_part2_orig.compute_vertex_normals()
    
    # Save part2, but slightly transformed away to make the problem non-trivial
    mesh_part2_transformed = copy.deepcopy(mesh_part2_orig)
    # T = np.eye(4)
    # T[0,3] = 0.5 # Translate in X
    # T[1,3] = 0.5 # Translate in Y
    # R = mesh_part2_transformed.get_rotation_matrix_from_xyz((0, np.pi/4, 0)) # Rotate
    # mesh_part2_transformed.rotate(R, center=(mesh_part2_transformed.get_center()))
    # mesh_part2_transformed.translate(T[:3,3])

    o3d.io.write_triangle_mesh(os.path.join(input_dir, 'part2_tf.obj'), mesh_part2_transformed)
    # Also save a non-transformed version for simpler testing if needed
    # o3d.io.write_triangle_mesh(os.path.join(input_dir, 'part2_orig.obj'), mesh_part2_orig)


    loaded_frags_info = load_fragments_from_directory(input_dir)
    
    processed_fragments_data = []
    for frag_info in loaded_frags_info:
        mesh = frag_info['mesh']
        pcd = preprocess_fragment(mesh, dummy_params)
        features, pcd_for_features = extract_features_from_pcd(pcd, dummy_params)
        
        processed_fragments_data.append({
            'name': frag_info['name'],
            'original_index': frag_info['original_index'],
            'mesh': mesh, 
            'pcd': pcd, 
            'features': features, 
            'pcd_for_features': pcd_for_features
        })

    if len(processed_fragments_data) >= 1: # Need at least 1 for assembly, 2 for meaningful
        pairwise_matches = find_pairwise_matches(processed_fragments_data, dummy_params)
        
        print(f"\nFound {len(pairwise_matches)} pairwise matches for assembly.")
        if pairwise_matches: # Only proceed if matches were found
            assembler = Assembler(processed_fragments_data, pairwise_matches, dummy_params)
            print("\nStarting greedy assembly...")
            final_assembly = assembler.greedy_assembly()

            if final_assembly:
                output_path = os.path.join(output_dir, "assembled_model.obj")
                o3d.io.write_triangle_mesh(output_path, final_assembly)
                print(f"Assembly saved to {output_path}")
                # o3d.visualization.draw_geometries([final_assembly], window_name="Final Assembled Model")
            else:
                print("Assembly failed or resulted in an empty model.")
        else:
             print("No pairwise matches found, cannot proceed with assembly based on matches.")
             if len(processed_fragments_data) == 1:
                 print("Only one fragment, 'assembly' is just the fragment itself.")
                 o3d.io.write_triangle_mesh(os.path.join(output_dir, "assembled_model.obj"), processed_fragments_data[0]['mesh'])
             else: # More than one fragment, but no matches. Could save them separately or as a scene.
                 print("Multiple fragments but no good matches to guide assembly.")


    else:
        print("Not enough fragments processed for assembly test.")
    
    # import shutil
    # shutil.rmtree(base_dir)