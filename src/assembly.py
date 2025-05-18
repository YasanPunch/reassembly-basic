import numpy as np
import open3d as o3d
import copy
from io_utils import combine_meshes


def check_overlap(
    mesh1, mesh2, voxel_size, max_overlap_factor=0.5
):  # voxel_size IS used in the manual fallback or could be in future point-based checks
    if not mesh1.has_vertices() or not mesh2.has_vertices():
        return True

    aabb1 = mesh1.get_axis_aligned_bounding_box()
    aabb2 = mesh2.get_axis_aligned_bounding_box()

    vol_intersection = 0.0  # Initialize

    # Check if the method exists (for robustness across Open3D versions)
    if hasattr(aabb1, "get_intersection"):
        # Use the built-in method if available (e.g., Open3D >= 0.17)
        # print("    DEBUG assembly.py: Using aabb1.get_intersection()") # Optional debug
        intersection_aabb = aabb1.get_intersection(aabb2)

        current_vol_intersection = intersection_aabb.volume()
        if current_vol_intersection < 1e-9:  # Check against a small epsilon
            return True
        vol_intersection = current_vol_intersection
    else:
        # Manual AABB intersection calculation
        print(
            "    DEBUG assembly.py: aabb1.get_intersection not found by hasattr, calculating manually."
        )  # Keep this debug!
        min_b1 = aabb1.get_min_bound()
        max_b1 = aabb1.get_max_bound()
        min_b2 = aabb2.get_min_bound()
        max_b2 = aabb2.get_max_bound()

        intersect_min_x = max(min_b1[0], min_b2[0])
        intersect_min_y = max(min_b1[1], min_b2[1])
        intersect_min_z = max(min_b1[2], min_b2[2])

        intersect_max_x = min(max_b1[0], max_b2[0])
        intersect_max_y = min(max_b1[1], max_b2[1])
        intersect_max_z = min(max_b1[2], max_b2[2])

        if (
            intersect_min_x < intersect_max_x
            and intersect_min_y < intersect_max_y
            and intersect_min_z < intersect_max_z
        ):
            vol_intersection = (
                (intersect_max_x - intersect_min_x)
                * (intersect_max_y - intersect_min_y)
                * (intersect_max_z - intersect_min_z)
            )
        else:
            return True  # No valid intersection volume

    vol1 = aabb1.volume()
    vol2 = aabb2.volume()

    epsilon = 1e-9
    if vol1 > epsilon and (vol_intersection / vol1) > max_overlap_factor:
        return False
    if vol2 > epsilon and (vol_intersection / vol2) > max_overlap_factor:
        return False

    # If you uncomment the point-based check later, voxel_size will be used:
    # pcd1_sampled = mesh1.sample_points_uniformly(number_of_points=200)
    # if not pcd1_sampled.has_points(): return True
    # distances = pcd1_sampled.compute_point_cloud_distance(mesh2.get_oriented_bounding_box())
    # distances = np.asarray(distances)
    # close_points_threshold = voxel_size * 0.5 # <<< voxel_size used here
    # num_penetrating = np.sum(distances < close_points_threshold)
    # penetration_ratio = num_penetrating / len(pcd1_sampled.points)
    # if penetration_ratio > max_overlap_factor:
    #     print(f"Overlap detected: {penetration_ratio*100:.1f}% points penetrating.")
    #     return False

    return True


class Assembler:
    def __init__(self, fragments_data, pairwise_matches, params):
        """
        Args:
            fragments_data (list of dict): Contains 'mesh', 'name', 'original_index', etc.
                                           The 'mesh' here is the original full-resolution mesh.
            pairwise_matches (list of dict): From matching.py.
            params (dict): Configuration parameters.
        """
        self.fragments_data = copy.deepcopy(fragments_data)  # Work on copies
        self.pairwise_matches = sorted(
            pairwise_matches, key=lambda x: x["score"], reverse=True
        )
        self.params = params
        self.num_fragments = len(fragments_data)

        # Store the original meshes, not the PCDs, for final assembly
        self.original_meshes = [fd["mesh"] for fd in self.fragments_data]

        # Keeps track of the current transformation of each fragment relative to world (or first fragment)
        self.fragment_transforms = [np.eye(4) for _ in range(self.num_fragments)]
        self.is_fragment_placed = [False] * self.num_fragments
        self.assembly_components = []  # List of lists, each sublist is a connected component

    def _get_transformed_mesh(self, fragment_idx):
        mesh = copy.deepcopy(self.original_meshes[fragment_idx])
        mesh.transform(self.fragment_transforms[fragment_idx])
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
            return self._get_transformed_mesh(0)

        # Initialize: Place the first fragment (e.g., largest or arbitrary)
        # Or, use the target of the best match as the seed
        if not self.pairwise_matches:
            print("No pairwise matches found. Placing fragment 0 at origin.")
            seed_idx = 0
        else:
            # Consider the target of the best match as the initial piece
            # This helps establish a reference frame
            seed_idx = self.pairwise_matches[0]["target_idx"]

        print(
            f"Starting assembly with seed fragment: {self.fragments_data[seed_idx]['name']} (idx {seed_idx})"
        )
        self.is_fragment_placed[seed_idx] = True
        current_assembly_indices = {seed_idx}
        current_assembly_meshes = [
            self._get_transformed_mesh(seed_idx)
        ]  # Store meshes for overlap checks

        num_placed = 1
        while num_placed < self.num_fragments:
            best_candidate_match = None
            best_candidate_score = -1

            # Find the best match connecting an unplaced fragment to any placed fragment
            for match_info in self.pairwise_matches:
                s_idx, t_idx = match_info["source_idx"], match_info["target_idx"]

                # Case 1: Target is placed, Source is not. Add Source.
                if (
                    self.is_fragment_placed[t_idx]
                    and not self.is_fragment_placed[s_idx]
                ):
                    if match_info["score"] > best_candidate_score:
                        # Proposed transform for s_idx: T_world_target * T_target_source
                        # T_target_source is match_info['transformation'] (transforms source to target's frame)
                        # T_world_target is self.fragment_transforms[t_idx]

                        proposed_transform_s = np.dot(
                            self.fragment_transforms[t_idx],
                            match_info["transformation"],
                        )

                        # Create temporary transformed mesh for overlap check
                        temp_s_mesh = copy.deepcopy(self.original_meshes[s_idx])
                        temp_s_mesh.transform(proposed_transform_s)

                        # Check for overlap with *all already placed* meshes in current_assembly_meshes
                        overlap_ok = True
                        for placed_mesh in current_assembly_meshes:
                            if not check_overlap(
                                temp_s_mesh,
                                placed_mesh,
                                self.params["voxel_downsample_size"],
                                self.params.get("max_assembly_overlap_factor", 0.6),
                            ):
                                overlap_ok = False
                                # print(f"    Overlap detected for {s_idx} with a placed piece.")
                                break

                        if overlap_ok:
                            best_candidate_match = match_info
                            best_candidate_score = match_info["score"]
                            # Store the actual transform relative to world for the source
                            best_candidate_transform = proposed_transform_s
                            best_candidate_new_idx = s_idx

                # Case 2: Source is placed, Target is not. Add Target.
                # This requires inverting the transformation.
                # T_source_target = inv(match_info['transformation'])
                # Proposed transform for t_idx: T_world_source * T_source_target
                elif (
                    self.is_fragment_placed[s_idx]
                    and not self.is_fragment_placed[t_idx]
                ):
                    if match_info["score"] > best_candidate_score:
                        # T_s_t is match_info['transformation'] (transforms s to t's frame)
                        # We need T_t_s = inv(T_s_t)
                        # Then, new piece t gets transform: T_world_s * inv(T_s_t)
                        # This places t relative to s, where s is already in world frame.

                        # Let's rephrase: match_info['transformation'] aligns s to t.
                        # T_s_in_t_coord = match_info['transformation']
                        # If s is placed (at T_world_s), and we want to place t:
                        # t_in_world = T_world_s * inv(T_s_in_t_coord)
                        # This seems more consistent.

                        try:
                            inv_transform_s_to_t = np.linalg.inv(
                                match_info["transformation"]
                            )
                        except np.linalg.LinAlgError:
                            # print(f"Warning: Could not invert transformation for match {s_idx}->{t_idx}. Skipping.")
                            continue

                        proposed_transform_t = np.dot(
                            self.fragment_transforms[s_idx], inv_transform_s_to_t
                        )

                        temp_t_mesh = copy.deepcopy(self.original_meshes[t_idx])
                        temp_t_mesh.transform(proposed_transform_t)

                        overlap_ok = True
                        for placed_mesh in current_assembly_meshes:
                            if not check_overlap(
                                temp_t_mesh,
                                placed_mesh,
                                self.params["voxel_downsample_size"],
                                self.params.get("max_assembly_overlap_factor", 0.6),
                            ):
                                overlap_ok = False
                                # print(f"    Overlap detected for {t_idx} with a placed piece.")
                                break

                        if overlap_ok:
                            best_candidate_match = match_info  # Store original match
                            best_candidate_score = match_info["score"]
                            best_candidate_transform = proposed_transform_t
                            best_candidate_new_idx = t_idx

            if best_candidate_match:
                newly_placed_idx = best_candidate_new_idx
                self.fragment_transforms[newly_placed_idx] = best_candidate_transform
                self.is_fragment_placed[newly_placed_idx] = True
                current_assembly_indices.add(newly_placed_idx)
                current_assembly_meshes.append(
                    self._get_transformed_mesh(newly_placed_idx)
                )  # Add its transformed version
                num_placed += 1
                print(
                    f"  Placed fragment: {self.fragments_data[newly_placed_idx]['name']} (idx {newly_placed_idx}) "
                    f"via match score {best_candidate_score:.3f}."
                )
            else:
                print(
                    "No more non-overlapping, valid matches found to extend the assembly."
                )
                break  # No more pieces can be added to this component

        if num_placed < self.num_fragments:
            print(
                f"Warning: Only {num_placed}/{self.num_fragments} fragments were assembled."
            )
            unplaced_indices = [
                i for i, placed in enumerate(self.is_fragment_placed) if not placed
            ]
            print("Unplaced fragment indices:", unplaced_indices)
            for idx in unplaced_indices:
                print(f" - {self.fragments_data[idx]['name']}")

        # Collect all placed original meshes and their final transformations
        final_meshes_to_combine = []
        final_transformations = []
        for i in range(self.num_fragments):
            if self.is_fragment_placed[i]:
                final_meshes_to_combine.append(self.original_meshes[i])
                final_transformations.append(self.fragment_transforms[i])

        if not final_meshes_to_combine:
            print("Error: No meshes were placed in the assembly.")
            return None

        return combine_meshes(final_meshes_to_combine, final_transformations)


if __name__ == "__main__":
    from io_utils import load_fragments_from_directory
    from preprocessing import preprocess_fragment
    from feature_extraction import extract_features_from_pcd
    from matching import find_pairwise_matches
    import json
    import os

    # Config
    dummy_params = {
        "voxel_downsample_size": 0.05,
        "normal_estimation_radius": 0.1,
        "normal_estimation_max_nn": 30,
        "fpfh_feature_radius": 0.25,
        "fpfh_feature_max_nn": 100,
        "ransac_distance_threshold_factor": 1.5,
        "ransac_edge_length_factor": 0.9,
        "ransac_iterations": 10000,
        "ransac_n_points": 3,
        "ransac_confidence": 0.99,
        "icp_max_correspondence_distance_factor": 2.0,
        "icp_relative_fitness": 1e-6,
        "icp_relative_rmse": 1e-6,
        "icp_max_iteration": 30,
        "min_match_score": 0.4,  # Lower for testing with simple cubes
        "max_assembly_overlap_factor": 0.7,  # Allow more overlap for simple test
    }

    # Use dummy data from matching.py test if available, or recreate
    base_dir = "../dummy_data_assembly"  # Relative to src/
    input_dir = os.path.join(base_dir, "input_fragments")
    output_dir = os.path.join(base_dir, "output_assembly")
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
"""  # This is piece 1, open on +X side

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
"""  # This is piece 2, starts at x=1, open on -X side (relative to its own coords)

    # To make them mate:
    # Part1: vertices as is.
    # Part2: vertices need to be shifted such that its local (0,y,z) matches Part1's (1,y,z)
    # For the test data, part2 is already defined to be next to part1 if simply concatenated.
    # We need to provide them as separate files, potentially with one transformed away.

    with open(os.path.join(input_dir, "part1.obj"), "w") as f:
        f.write(cube_obj_content_part1)

    mesh_part2_orig = o3d.geometry.TriangleMesh()  # Create part2 programmatically
    verts_p2 = np.array(
        [
            [1, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [2, 0, 1],
            [2, 1, 1],
            [1, 1, 1],
        ]
    )
    faces_p2 = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Front face at x=1 (local)
            [7, 6, 5],
            [7, 5, 4],  # Back face at x=2 (local)
            [1, 5, 6],
            [1, 6, 2],  # Right face (+X of part2)
            [3, 2, 6],
            [3, 6, 7],  # Top face
            [4, 0, 3],
            [4, 3, 7],  # Left face (-X of part2) -> THIS IS THE MATING FACE
            [0, 4, 5],
            [0, 5, 1],  # Bottom face
        ]
    )
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

    o3d.io.write_triangle_mesh(
        os.path.join(input_dir, "part2_tf.obj"), mesh_part2_transformed
    )
    # Also save a non-transformed version for simpler testing if needed
    # o3d.io.write_triangle_mesh(os.path.join(input_dir, 'part2_orig.obj'), mesh_part2_orig)

    loaded_frags_info = load_fragments_from_directory(input_dir)

    processed_fragments_data = []
    for frag_info in loaded_frags_info:
        mesh = frag_info["mesh"]
        pcd = preprocess_fragment(mesh, dummy_params)
        features, pcd_for_features = extract_features_from_pcd(pcd, dummy_params)

        processed_fragments_data.append(
            {
                "name": frag_info["name"],
                "original_index": frag_info["original_index"],
                "mesh": mesh,
                "pcd": pcd,
                "features": features,
                "pcd_for_features": pcd_for_features,
            }
        )

    if (
        len(processed_fragments_data) >= 1
    ):  # Need at least 1 for assembly, 2 for meaningful
        pairwise_matches = find_pairwise_matches(processed_fragments_data, dummy_params)

        print(f"\nFound {len(pairwise_matches)} pairwise matches for assembly.")
        if pairwise_matches:  # Only proceed if matches were found
            assembler = Assembler(
                processed_fragments_data, pairwise_matches, dummy_params
            )
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
            print(
                "No pairwise matches found, cannot proceed with assembly based on matches."
            )
            if len(processed_fragments_data) == 1:
                print("Only one fragment, 'assembly' is just the fragment itself.")
                o3d.io.write_triangle_mesh(
                    os.path.join(output_dir, "assembled_model.obj"),
                    processed_fragments_data[0]["mesh"],
                )
            else:  # More than one fragment, but no matches. Could save them separately or as a scene.
                print("Multiple fragments but no good matches to guide assembly.")

    else:
        print("Not enough fragments processed for assembly test.")

    # import shutil
    # shutil.rmtree(base_dir)

