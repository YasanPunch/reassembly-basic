import trimesh
import numpy as np
import open3d as o3d
import copy
from src.io_utils import combine_meshes, save_mesh
from src.utils.geometry_utils import boolean_intersection_penetration_test
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os

cmap = plt.get_cmap("tab20")

class Assembler:

    def __init__(self, fragments_data, pairwise_matches, params):
        self.fragments_data = copy.deepcopy(fragments_data)
        self.pairwise_matches = sorted(
            pairwise_matches, key=lambda x: x["score"], reverse=True
        )
        self.params = params
        self.num_fragments = len(fragments_data)

        self.original_meshes = [fd["original_mesh"] for fd in self.fragments_data]

        self.fragment_transforms = [np.eye(4) for _ in range(self.num_fragments)]
        self.is_fragment_placed = [False] * self.num_fragments
        self.assembly_components = []

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
        pose_graph = o3d.pipelines.registration.PoseGraph()
        # Add nodes (one per fragment)
        for i in range(self.num_fragments):
            if i == 0:
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.eye(4))
                )
            else:
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        self.fragment_transforms[i]
                    )
                )
        # Add edges (from pairwise matches)
        for match in self.pairwise_matches:
            if "confidence" in match and match["confidence"] < min_confidence:
                continue
            source = match["source_idx"]
            target = match["target_idx"]
            transformation = match["transformation"]
            uncertain = False  # Set to True for non-sequential edges if desired
            information = np.eye(6)  # Could be weighted by confidence/score
            # FIX: Swap 'information' and 'uncertain' to match Open3D's PoseGraphEdge signature
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source, target, transformation, information, uncertain
                )
            )
        # Run global optimization
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.params.get("voxel_downsample_size", 0.01)
            * 2.0,
            edge_prune_threshold=0.25,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
        # Update transforms
        for i in range(self.num_fragments):
            self.fragment_transforms[i] = pose_graph.nodes[i].pose

    def greedy_assembly(self):
        if self.num_fragments == 0:
            return None
        if self.num_fragments == 1:
            return self._get_transformed_mesh(0)

        if not self.pairwise_matches:
            print(
                "No pairwise matches for assembly. Cannot proceed with greedy strategy."
            )
            return None

        seed_idx = self.pairwise_matches[0]["target_idx"]
        seed_name = self.fragments_data[seed_idx]["name"]

        print(
            f"Starting assembly with seed fragment: {seed_name} (idx in current list: {seed_idx})"
        )
        self.is_fragment_placed[seed_idx] = True

        # current_assembly_components stores tuples of (transformed_mesh_object, fragment_name)
        current_assembly_components = [
            (self._get_transformed_mesh(seed_idx), seed_name)
        ]

        num_placed = 1
        rejected_matches = []
        finalized = False
        used_matches = set()
        while num_placed < self.num_fragments and not finalized:
            best_candidate_match_info = None
            best_candidate_score = -1.0
            best_candidate_world_transform = None
            best_candidate_idx_to_place = -1

            # Try all pairwise matches that haven't been used or rejected
            for i, match_info in enumerate(self.pairwise_matches):
                if id(match_info) in used_matches:
                    continue
                s_idx, t_idx = match_info["source_idx"], match_info["target_idx"]
                current_iteration_potential_world_transform = None
                current_iteration_idx_to_place = -1

                if (
                    self.is_fragment_placed[t_idx]
                    and not self.is_fragment_placed[s_idx]
                ):
                    current_iteration_potential_world_transform = np.dot(
                        self.fragment_transforms[t_idx], match_info["transformation"]
                    )
                    current_iteration_idx_to_place = s_idx
                elif (
                    self.is_fragment_placed[s_idx]
                    and not self.is_fragment_placed[t_idx]
                ):
                    try:
                        inv_transform = np.linalg.inv(match_info["transformation"])
                        current_iteration_potential_world_transform = np.dot(
                            self.fragment_transforms[s_idx], inv_transform
                        )
                        current_iteration_idx_to_place = t_idx
                    except np.linalg.LinAlgError:
                        continue
                else:
                    continue

                if (
                    current_iteration_potential_world_transform is not None
                    and current_iteration_idx_to_place != -1
                ):
                    if match_info["score"] > best_candidate_score:
                        candidate_original_mesh_o3d = self.original_meshes[
                            current_iteration_idx_to_place
                        ]
                        candidate_mesh_transformed_o3d = copy.deepcopy(
                            candidate_original_mesh_o3d
                        )
                        candidate_mesh_transformed_o3d.transform(
                            current_iteration_potential_world_transform
                        )
                        candidate_name = self.fragments_data[
                            current_iteration_idx_to_place
                        ]["name"]

                        candidate_tri = trimesh.Trimesh(
                            vertices=np.asarray(
                                candidate_mesh_transformed_o3d.vertices
                            ),
                            faces=np.asarray(candidate_mesh_transformed_o3d.triangles),
                        )
                        candidate_vol = candidate_tri.volume
                        placed_vols = []
                        for placed_mesh_o3d, _ in current_assembly_components:
                            placed_tri = trimesh.Trimesh(
                                vertices=np.asarray(placed_mesh_o3d.vertices),
                                faces=np.asarray(placed_mesh_o3d.triangles),
                            )
                            placed_vols.append(placed_tri.volume)
                        min_volume_all = min([candidate_vol] + placed_vols)
                        penetration_ok = True
                        for placed_mesh_o3d, placed_name in current_assembly_components:
                            is_valid, penetration_ratio, intersection_mesh = (
                                boolean_intersection_penetration_test(
                                    candidate_mesh_transformed_o3d,
                                    candidate_name,
                                    placed_mesh_o3d,
                                    placed_name,
                                    self.params,
                                    min_volume_override=min_volume_all,
                                )
                            )
                            penetration_pct = penetration_ratio * 100
                            threshold_pct = self.params.get(
                                "boolean_penetration_threshold", 0.1
                            )
                            print(
                                f"    [Penetration Test] {candidate_name} vs {placed_name}: {penetration_pct:.2f}% (threshold: {threshold_pct:.2f}%) [min volume: {min_volume_all:.4f}] -> {'PASS' if is_valid else 'FAIL'}"
                            )
                            if not is_valid:
                                penetration_ok = False
                                break

                        if penetration_ok:
                            best_candidate_match_info = match_info
                            best_candidate_score = match_info["score"]
                            best_candidate_world_transform = (
                                current_iteration_potential_world_transform
                            )
                            best_candidate_idx_to_place = current_iteration_idx_to_place

            # If a candidate is found, prompt the user with interactive visualization
            if (
                best_candidate_idx_to_place != -1
                and best_candidate_match_info is not None
            ):
                newly_placed_idx_in_list = best_candidate_idx_to_place
                newly_placed_name = self.fragments_data[newly_placed_idx_in_list][
                    "name"
                ]

                # Create candidate mesh for visualization
                candidate_mesh = copy.deepcopy(self.original_meshes[newly_placed_idx_in_list])
                candidate_mesh.transform(best_candidate_world_transform)
                candidate_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Red for candidate

                # Create placed meshes for visualization
                placed_meshes = []
                for mesh, _ in current_assembly_components:
                    placed_mesh = copy.deepcopy(mesh)
                    placed_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for placed
                    placed_meshes.append(placed_mesh)

                print(f"\n=== Interactive Candidate Selection ===")
                print(
                    f"Candidate: {newly_placed_name} (score: {best_candidate_score:.3f})"
                )
                print("A: Accept match | R: Reject match | F: Finalize assembly")
                print("Q: Quit without selection")

                # Create interactive visualizer
                shared_state = {"action": None}

                vis = o3d.visualization.VisualizerWithKeyCallback()
                vis.create_window(
                    window_name=f"Candidate: {newly_placed_name} (Red), Placed: Gray",
                    width=1280,
                    height=960,
                )

                # Add geometries
                vis.add_geometry(candidate_mesh)
                for mesh in placed_meshes:
                    vis.add_geometry(mesh)

                def accept_match(visualizer):
                    shared_state["action"] = "accept"
                    print("\n  Match accepted. Closing...")
                    visualizer.close()
                    return False

                def reject_match(visualizer):
                    shared_state["action"] = "reject"
                    print("\n  Match rejected. Closing...")
                    visualizer.close()
                    return False

                def finalize_assembly(visualizer):
                    shared_state["action"] = "finalize"
                    print("\n  Assembly finalized. Closing...")
                    visualizer.close()
                    return False

                def quit_selection(visualizer):
                    shared_state["action"] = "quit"
                    print("\n  Selection aborted. Closing...")
                    visualizer.close()
                    return False

                # Register key callbacks
                vis.register_key_callback(ord("A"), accept_match)
                vis.register_key_callback(ord("R"), reject_match)
                vis.register_key_callback(ord("F"), finalize_assembly)
                vis.register_key_callback(ord("Q"), quit_selection)

                vis.run()
                vis.destroy_window()

                # Handle user action
                action = shared_state["action"]
                if action == "accept":
                    # Accept the match
                    self.fragment_transforms[newly_placed_idx_in_list] = (
                        best_candidate_world_transform
                    )
                    self.is_fragment_placed[newly_placed_idx_in_list] = True
                    current_assembly_components.append(
                        (
                            self._get_transformed_mesh(newly_placed_idx_in_list),
                            newly_placed_name,
                        )
                    )
                    num_placed += 1
                    used_matches.add(id(best_candidate_match_info))
                    print(
                        f"  Placed fragment: {newly_placed_name} "
                        f"(idx in list: {newly_placed_idx_in_list}) via match score {best_candidate_score:.3f}."
                    )
                elif action == "reject":
                    # Reject the match, add to rejected_matches
                    rejected_matches.append(best_candidate_match_info)
                    used_matches.add(id(best_candidate_match_info))
                    print(f"  Rejected match for fragment: {newly_placed_name}")
                elif action == "finalize":
                    finalized = True
                    print("  Finalizing assembly as per user request.")
                elif action == "quit":
                    finalized = True
                    print("  Assembly aborted by user.")
                    break
            else:
                # No more valid matches, cycle through rejected matches
                if rejected_matches:
                    print("\nNo more new matches. Cycling through rejected matches...")
                    # Remove already placed fragments from rejected_matches
                    rejected_matches = [m for m in rejected_matches if not self.is_fragment_placed[m["source_idx"]] or not self.is_fragment_placed[m["target_idx"]]]
                    if not rejected_matches:
                        print("No rejected matches left to reconsider.")
                        break
                    reconsidered = False
                    for match_info in rejected_matches[:]:
                        s_idx, t_idx = match_info["source_idx"], match_info["target_idx"]
                        if self.is_fragment_placed[t_idx] and not self.is_fragment_placed[s_idx]:
                            world_transform = np.dot(self.fragment_transforms[t_idx], match_info["transformation"])
                            idx_to_place = s_idx
                        elif self.is_fragment_placed[s_idx] and not self.is_fragment_placed[t_idx]:
                            try:
                                inv_transform = np.linalg.inv(match_info["transformation"])
                                world_transform = np.dot(self.fragment_transforms[s_idx], inv_transform)
                                idx_to_place = t_idx
                            except np.linalg.LinAlgError:
                                continue
                        else:
                            continue
                        candidate_name = self.fragments_data[idx_to_place]["name"]
                        candidate_mesh = copy.deepcopy(self.original_meshes[idx_to_place])
                        candidate_mesh.transform(world_transform)
                        candidate_mesh.paint_uniform_color(
                            [1.0, 0.0, 0.0]
                        )  # Red for candidate

                        # Create placed meshes for visualization
                        placed_meshes = []
                        for mesh, _ in current_assembly_components:
                            placed_mesh = copy.deepcopy(mesh)
                            placed_mesh.paint_uniform_color(
                                [0.7, 0.7, 0.7]
                            )  # Gray for placed
                            placed_meshes.append(placed_mesh)

                        print(f"\n=== Reconsidering Rejected Match ===")
                        print(
                            f"Candidate: {candidate_name} (score: {match_info['score']:.3f})"
                        )
                        print(
                            "A: Accept match | R: Reject match | F: Finalize assembly"
                        )
                        print("Q: Quit without selection")

                        # Create interactive visualizer for rejected match
                        shared_state = {"action": None}

                        vis = o3d.visualization.VisualizerWithKeyCallback()
                        vis.create_window(
                            window_name=f"Reconsidered: {candidate_name} (Red), Placed: Gray",
                            width=1280,
                            height=960,
                        )

                        # Add geometries
                        vis.add_geometry(candidate_mesh)
                        for mesh in placed_meshes:
                            vis.add_geometry(mesh)

                        def accept_rejected_match(visualizer):
                            shared_state["action"] = "accept"
                            print("\n  Rejected match accepted. Closing...")
                            visualizer.close()
                            return False

                        def reject_again(visualizer):
                            shared_state["action"] = "reject"
                            print("\n  Still rejected. Closing...")
                            visualizer.close()
                            return False

                        def finalize_from_rejected(visualizer):
                            shared_state["action"] = "finalize"
                            print("\n  Assembly finalized. Closing...")
                            visualizer.close()
                            return False

                        def quit_from_rejected(visualizer):
                            shared_state["action"] = "quit"
                            print("\n  Selection aborted. Closing...")
                            visualizer.close()
                            return False

                        # Register key callbacks
                        vis.register_key_callback(ord("A"), accept_rejected_match)
                        vis.register_key_callback(ord("R"), reject_again)
                        vis.register_key_callback(ord("F"), finalize_from_rejected)
                        vis.register_key_callback(ord("Q"), quit_from_rejected)

                        vis.run()
                        vis.destroy_window()

                        # Handle user action for rejected match
                        action = shared_state["action"]
                        if action == "accept":
                            self.fragment_transforms[idx_to_place] = world_transform
                            self.is_fragment_placed[idx_to_place] = True
                            current_assembly_components.append(
                                (
                                    self._get_transformed_mesh(idx_to_place),
                                    candidate_name,
                                )
                            )
                            num_placed += 1
                            rejected_matches.remove(match_info)
                            print(
                                f"  Placed fragment: {candidate_name} (from rejected list)"
                            )
                            reconsidered = True
                            break
                        elif action == "reject":
                            print(f"  Still rejected: {candidate_name}")
                        elif action == "finalize":
                            finalized = True
                            print("  Finalizing assembly as per user request.")
                            break
                        elif action == "quit":
                            finalized = True
                            print("  Assembly aborted by user.")
                            break

                    if finalized or reconsidered:
                        break
                    if not reconsidered and not finalized:
                        print("No more matches can be placed. Ending assembly.")
                        break
                else:
                    print(
                        "No more non-overlapping, valid matches found to extend the assembly."
                    )
                    break

        if num_placed < self.num_fragments:
            print(
                f"Warning: Only {num_placed}/{self.num_fragments} fragments were assembled."
            )
            unplaced_indices = [
                i for i, placed in enumerate(self.is_fragment_placed) if not placed
            ]
            print(
                "Unplaced fragment indices (relative to valid_fragments_data list):",
                unplaced_indices,
            )
            for idx_unplaced in unplaced_indices:
                print(f" - {self.fragments_data[idx_unplaced]['name']}")

            # --- MANUAL REVIEW OF UNPLACED FRAGMENTS ---
            print(f"\n=== Manual Review of Unplaced Fragments ===")
            print(
                f"Found {len(unplaced_indices)} unplaced fragments. Reviewing each one..."
            )

            # Create placed meshes for visualization
            placed_meshes = []
            for mesh, _ in current_assembly_components:
                placed_mesh = copy.deepcopy(mesh)
                placed_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for placed
                placed_meshes.append(placed_mesh)

            for unplaced_idx in unplaced_indices[
                :
            ]:  # Copy list to allow modification during iteration
                unplaced_name = self.fragments_data[unplaced_idx]["name"]
                print(f"\n--- Reviewing: {unplaced_name} ---")

                # Find all possible matches for this unplaced fragment
                possible_matches = []
                for match in self.pairwise_matches:
                    s_idx, t_idx = match["source_idx"], match["target_idx"]
                    if s_idx == unplaced_idx and self.is_fragment_placed[t_idx]:
                        # Unplaced fragment is source, can connect to placed target
                        world_transform = np.dot(
                            self.fragment_transforms[t_idx], match["transformation"]
                        )
                        possible_matches.append(
                            {
                                "match": match,
                                "transform": world_transform,
                                "target_name": self.fragments_data[t_idx]["name"],
                                "score": match["score"],
                            }
                        )
                    elif t_idx == unplaced_idx and self.is_fragment_placed[s_idx]:
                        # Unplaced fragment is target, can connect to placed source
                        try:
                            inv_transform = np.linalg.inv(match["transformation"])
                            world_transform = np.dot(
                                self.fragment_transforms[s_idx], inv_transform
                            )
                            possible_matches.append(
                                {
                                    "match": match,
                                    "transform": world_transform,
                                    "target_name": self.fragments_data[s_idx]["name"],
                                    "score": match["score"],
                                }
                            )
                        except np.linalg.LinAlgError:
                            continue

                if not possible_matches:
                    print(f"  No possible matches found for {unplaced_name}")
                    continue

                # Sort matches by score
                possible_matches.sort(key=lambda x: x["score"], reverse=True)

                print(f"  Found {len(possible_matches)} possible matches:")
                for i, pm in enumerate(possible_matches):
                    print(
                        f"    {i+1}. Connect to {pm['target_name']} (score: {pm['score']:.3f})"
                    )

                # Show each match option to user
                for i, pm in enumerate(possible_matches):
                    match_info = pm["match"]
                    world_transform = pm["transform"]
                    target_name = pm["target_name"]
                    score = pm["score"]

                    # Create candidate mesh for visualization
                    candidate_mesh = copy.deepcopy(self.original_meshes[unplaced_idx])
                    candidate_mesh.transform(world_transform)
                    candidate_mesh.paint_uniform_color(
                        [1.0, 0.0, 0.0]
                    )  # Red for candidate

                    print(f"\n  === Match Option {i+1}/{len(possible_matches)} ===")
                    print(
                        f"  Connecting {unplaced_name} to {target_name} (score: {score:.3f})"
                    )
                    print(
                        "  A: Accept this match | N: Next option | S: Skip this fragment | F: Finalize assembly"
                    )

                    # Create interactive visualizer
                    shared_state = {"action": None}

                    vis = o3d.visualization.VisualizerWithKeyCallback()
                    vis.create_window(
                        window_name=f"Manual Review: {unplaced_name} -> {target_name} (Red), Placed: Gray",
                        width=1280,
                        height=960,
                    )

                    # Add geometries
                    vis.add_geometry(candidate_mesh)
                    for mesh in placed_meshes:
                        vis.add_geometry(mesh)

                    def accept_manual_match(visualizer):
                        shared_state["action"] = "accept"
                        print("\n    Match accepted. Closing...")
                        visualizer.close()
                        return False

                    def next_option(visualizer):
                        shared_state["action"] = "next"
                        print("\n    Next option. Closing...")
                        visualizer.close()
                        return False

                    def skip_fragment(visualizer):
                        shared_state["action"] = "skip"
                        print("\n    Skipping fragment. Closing...")
                        visualizer.close()
                        return False

                    def finalize_manual(visualizer):
                        shared_state["action"] = "finalize"
                        print("\n    Finalizing assembly. Closing...")
                        visualizer.close()
                        return False

                    # Register key callbacks
                    vis.register_key_callback(ord("A"), accept_manual_match)
                    vis.register_key_callback(ord("N"), next_option)
                    vis.register_key_callback(ord("S"), skip_fragment)
                    vis.register_key_callback(ord("F"), finalize_manual)

                    vis.run()
                    vis.destroy_window()

                    # Handle user action
                    action = shared_state["action"]
                    if action == "accept":
                        # Accept the match
                        self.fragment_transforms[unplaced_idx] = world_transform
                        self.is_fragment_placed[unplaced_idx] = True
                        current_assembly_components.append(
                            (self._get_transformed_mesh(unplaced_idx), unplaced_name)
                        )
                        num_placed += 1
                        unplaced_indices.remove(unplaced_idx)
                        print(f"    Placed fragment: {unplaced_name} via manual review")
                        break  # Move to next unplaced fragment
                    elif action == "next":
                        # Continue to next option
                        continue
                    elif action == "skip":
                        # Skip this fragment entirely
                        print(f"    Skipped fragment: {unplaced_name}")
                        break  # Move to next unplaced fragment
                    elif action == "finalize":
                        # Finalize assembly
                        print("    Finalizing assembly as per user request.")
                        break  # Exit manual review loop

            # Update final count after manual review
            print(
                f"\nAfter manual review: {num_placed}/{self.num_fragments} fragments assembled."
            )
            if unplaced_indices:
                print("Remaining unplaced fragments:")
                for idx_unplaced in unplaced_indices:
                    print(f" - {self.fragments_data[idx_unplaced]['name']}")

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
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [0.8, 0.5, 0.2],
                        [0.5, 0.2, 0.8],
                        [0.2, 0.8, 0.5],
                        [0.6, 0.6, 0.6],
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
        print(
            "[Visualization] Showing output after pose graph optimization (before snapping)..."
        )
        pose_graph_meshes = []
        for mesh, transform in zip(
            final_meshes_to_combine_o3d, final_transforms_for_combine
        ):
            mesh_pg = copy.deepcopy(mesh)
            mesh_pg.paint_uniform_color(
                [0.8, 0.8, 0.8]
            )  # Light gray color for all parts
            mesh_pg.transform(transform)
            pose_graph_meshes.append(mesh_pg)
        o3d.visualization.draw_geometries(
            pose_graph_meshes,
            window_name="Final Composite Assembly: After Pose Graph Optimization",
        )

        # --- POST-PROCESSING SNAPPING STEP ---
        enable_snapping = self.params.get("enable_post_processing_snapping", False)
        snap_score_threshold = self.params.get("snap_score_threshold", 0.7)

        if enable_snapping:
            print("[Post-Processing] Snapping adjacent fragments together...")
            snapped_transforms = list(final_transforms_for_combine)
            # For each high-scoring match, snap the source to the target
            for match in self.pairwise_matches:
                if match["score"] < snap_score_threshold:
                    continue
                s_idx = match["source_idx"]
                t_idx = match["target_idx"]
                # Only snap if both fragments are placed
                if s_idx >= len(final_meshes_to_combine_o3d) or t_idx >= len(
                    final_meshes_to_combine_o3d
                ):
                    continue
                source_mesh = copy.deepcopy(final_meshes_to_combine_o3d[s_idx])
                target_mesh = copy.deepcopy(final_meshes_to_combine_o3d[t_idx])
                source_mesh.transform(snapped_transforms[s_idx])
                target_mesh.transform(snapped_transforms[t_idx])
                # Find closest points between source and target
                src_points = np.asarray(source_mesh.vertices)
                tgt_points = np.asarray(target_mesh.vertices)
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
            print(
                "[Post-Processing] Snapping complete. Visualizing snapped fragments..."
            )
            o3d.visualization.draw_geometries(
                snapped_meshes,
                window_name="Final Composite Assembly: Snapped Fragments",
            )
            final_mesh = combine_meshes(
                snapped_meshes, [np.eye(4)] * len(snapped_meshes)
            )
        else:
            print(
                "[Post-Processing] Snapping disabled. Using pose graph optimized fragments..."
            )
            # Use the pose graph optimized meshes directly without snapping
            final_mesh = combine_meshes(
                pose_graph_meshes, [np.eye(4)] * len(pose_graph_meshes)
            )

        # Save the final mesh to data/output_assembly
        output_dir = os.path.join("data", "output_assembly")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "reconstructed_model.obj")
        save_mesh(final_mesh, output_path)
        print(f"Final assembled model saved to: {output_path}")

        return final_mesh
