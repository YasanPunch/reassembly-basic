import os
import copy
import numpy as np

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import src.io_utils
import src.preprocessing
import src.matching
import src.assembly

class ProcessingPanel:
    def __init__(self, app):
        self.app = app

        w = app.window
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        self.processed_fragments_pipeline_data = []

        # Add state for asynchronous segmentation
        self.segmentation_queue = []  # Queue of fragments to process
        self.current_segmentation_fragment = None
        self.segmentation_results = {}  # Store results for each fragment

        self.label = gui.Label("Processing Panel")
        self._panel.add_child(self.label)
        self._panel.add_fixed(separation_height)

        process_ctrls = gui.CollapsableVert(
            "Process controls", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )

        self._segmentation_button = gui.Button("Segmentation")
        self._segmentation_button.horizontal_padding_em = 0.5
        self._segmentation_button.vertical_padding_em = 0
        self._segmentation_button.set_on_clicked(self._on_segmentation)

        self._pairwise_matching_button = gui.Button("Pairwise Matching")
        self._pairwise_matching_button.horizontal_padding_em = 0.5
        self._pairwise_matching_button.vertical_padding_em = 0
        self._pairwise_matching_button.set_on_clicked(self._on_pairwise_matching)

        self._multipiece_matching_button = gui.Button("Multipiece Matching")
        self._multipiece_matching_button.horizontal_padding_em = 0.5
        self._multipiece_matching_button.vertical_padding_em = 0
        self._multipiece_matching_button.set_on_clicked(self._on_multipiece_matching)

        process_ctrls.add_child(self._segmentation_button)
        process_ctrls.add_child(self._pairwise_matching_button)
        process_ctrls.add_child(self._multipiece_matching_button)

        self._panel.add_child(process_ctrls)
        self._panel.add_fixed(separation_height)

    def _on_segmentation(self):
        fragments_data_raw = (
            self.app._left_panel.item_tree.get_all_visible_base_model_items()
        )
        print(fragments_data_raw)

        print("\n[3. Preprocessing, Segmentation, and Feature Extraction]")

        # Initialize the segmentation queue
        self.segmentation_queue = fragments_data_raw.copy()
        self.current_segmentation_fragment = None
        self.segmentation_results = {}
        self.processed_fragments_pipeline_data = []

        # Start processing the first fragment
        self._process_next_fragment_in_queue()

    def _on_classification(self):
        pass
        """Get the currently selected preprocessed items."""
        selected_items = []

        # Get all preprocessing batches
        preprocessing_batches = self.app._left_panel.item_tree.get_items_by_type('preprocessing_results')

        for batch in preprocessing_batches:
            if batch.is_visible:  # Checked batches are visible
                # Get all preprocessed items from this batch
                for preprocessed_item in batch.preprocessed_items:
                    if preprocessed_item.is_visible:  # Checked items are visible
                        selected_items.append(preprocessed_item)

        return selected_items

    def _on_pairwise_matching(self):
        """Execute pairwise matching using the processed fragments data."""
        print("\n[4. Finding Pairwise Matches]")

        # Check if we have processed fragments data
        if not self.processed_fragments_pipeline_data:
            print(
                "No processed fragments data available. Please run segmentation first."
            )
            return

        # Get valid fragments (those with features)
        valid_fragments_data = [
            fd
            for fd in self.processed_fragments_pipeline_data
            if fd.get("features_list")
            and any(f is not None and f.num() > 0 for f in fd["features_list"])
        ]

        if len(valid_fragments_data) < 2:
            print(
                "Not enough valid fragments for pairwise matching. Need at least 2 fragments with features."
            )
            # Save unaligned fragments
            self._save_unaligned_fragments(
                valid_fragments_data, "insufficient_fragments"
            )
            return

        print(
            f"  Processing {len(valid_fragments_data)} valid fragments for pairwise matching..."
        )

        # Call the matching function
        pairwise_matches = src.matching.find_pairwise_matches(
            valid_fragments_data,
            self.app.params,
            debug=self.app.params.get("debug_pairwise_matching", False),
            top_n_per_pair=self.app.params.get("top_n_matches_per_pair", 3),
            processing_panel=self,  # Pass self for GUI visualization
        )

        # Handle results and store them
        self._handle_pairwise_matching_results(valid_fragments_data, pairwise_matches)

    def _on_multipiece_matching(self):
        """Execute global assembly using the pairwise matches data."""
        print("\n[5. Performing Global Assembly]")

        # Check if we have pairwise matches data
        if not hasattr(self, "pairwise_matches") or not hasattr(
            self, "valid_fragments_data"
        ):
            print(
                "No pairwise matches data available. Please run pairwise matching first."
            )
            return

        if not self.pairwise_matches:
            print("No pairwise matches found. Cannot perform global assembly.")
            return

        print(
            f"  Starting global assembly with {len(self.valid_fragments_data)} fragments and {len(self.pairwise_matches)} matches..."
        )

        # Create the assembler
        assembler = src.assembly.Assembler(
            self.valid_fragments_data, self.pairwise_matches, self.app.params
        )

        # Start the greedy assembly with GUI support
        self._start_greedy_assembly(assembler)

    def _process_next_fragment_in_queue(self):
        """Process the next fragment in the segmentation queue."""
        if not self.segmentation_queue:
            # All fragments processed, finish the pipeline
            self._finish_segmentation_pipeline()
            return

        # Get the next fragment to process
        frag_info_raw = self.segmentation_queue.pop(0)
        self.current_segmentation_fragment = frag_info_raw

        print(
            f"  Processing fragment: {frag_info_raw['name']} ({len(self.segmentation_queue) + 1} remaining)"
        )

        # Start preprocessing and segmentation for this fragment
        # This will trigger the interactive dialog if visualization is enabled
        fracture_surfaces = src.segmentation.extract_fracture_surface_mesh(
            frag_info_raw["mesh"], frag_info_raw["name"], self.app.params, self
        )

        # If no interactive visualization, process immediately
        if fracture_surfaces is not None:
            # Continue with preprocessing
            self._continue_fragment_processing(frag_info_raw, fracture_surfaces)

    def continue_segmentation_pipeline(self, fragment_name, selected_regions):
        """Called by the segmentation dialog when user confirms selection."""
        if self.current_segmentation_fragment is None:
            print(f"Error: No current fragment being processed")
            return

        frag_info_raw = self.current_segmentation_fragment

        # Store the selection results
        self.segmentation_results[fragment_name] = {
            "selected_regions": selected_regions,
            "fragment_info": frag_info_raw,
        }

        # Reconstruct fracture surfaces based on selected regions
        # We need to extract the fracture surfaces from the selected regions
        fracture_surfaces = self._reconstruct_fracture_surfaces_from_selection(
            frag_info_raw, selected_regions
        )

        self._continue_fragment_processing(frag_info_raw, fracture_surfaces)

    def _continue_fragment_processing(self, frag_info_raw, fracture_surfaces):
        """Continue processing a fragment after segmentation."""
        # Preprocessing now returns lists: (pcds_for_features_list, features_list, fracture_surfaces)
        pcds_for_features_list, features_list, _ = (
            src.preprocessing.preprocess_fragment(
                frag_info_raw, self.app.params, self, fracture_surfaces
            )
        )

        # If no valid surfaces, store empty lists and continue
        if not pcds_for_features_list or all(
            pcd is None or not pcd.has_points() for pcd in pcds_for_features_list
        ):
            print(
                f"    Warning: Preprocessing resulted in no valid point clouds for features for {frag_info_raw['name']}. Skipping."
            )
            self.processed_fragments_pipeline_data.append(
                {
                    "name": frag_info_raw["name"],
                    "original_index": frag_info_raw["original_index"],
                    "original_mesh": frag_info_raw["mesh"],
                    "fracture_surfaces": fracture_surfaces,
                    "pcds_for_features": [],
                    "features_list": [],
                }
            )
        else:
            # Store lists for each fragment
            self.processed_fragments_pipeline_data.append(
                {
                    "name": frag_info_raw["name"],
                    "original_index": frag_info_raw["original_index"],
                    "original_mesh": frag_info_raw["mesh"],
                    "fracture_surfaces": fracture_surfaces,
                    "pcds_for_features": pcds_for_features_list,
                    "features_list": features_list,
                }
            )

        # Process next fragment
        self._process_next_fragment_in_queue()

    def _reconstruct_fracture_surfaces_from_selection(
        self, frag_info_raw, selected_regions
    ):
        """Reconstruct fracture surfaces from selected regions."""
        if not selected_regions:
            print(f"    No regions selected for {frag_info_raw['name']}")
            return None

        # We need to get the original segmentation results to reconstruct the surfaces
        # For now, let's call extract_fracture_surface_mesh without the dialog
        # but we need to modify the segmentation function to support this

        # Call segmentation with pre-selected regions to reconstruct fracture surfaces
        fracture_surfaces = src.segmentation.extract_fracture_surface_mesh(
            frag_info_raw["mesh"],
            frag_info_raw["name"],
            self.app.params,
            self,  # Pass self as processing_panel
            pre_selected_regions=selected_regions,  # Pass the selected regions
        )

        return fracture_surfaces

    def show_debug_visualization(self, geometries, window_name="Debug Visualization"):
        """Show debug visualization using the GUI system instead of o3d.visualization.draw_geometries."""
        try:
            print(f"[DEBUG] Creating debug visualization: {window_name}")
            print(f"[DEBUG] Number of geometries: {len(geometries)}")

            # Create a unique scene ID for this debug visualization
            # Clean the window name to create a safe scene ID
            safe_name = (
                window_name.replace(" ", "_")
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace("→", "_to_")
                .replace(":", "_")
                .replace(".", "_")
            )
            scene_id = f"debug_{safe_name}"
            print(f"[DEBUG] Scene ID: {scene_id}")

            # Create scene widget
            scene_widget = self.app.add_scene_widget(scene_id, window_name)

            # Add all geometries to the scene
            for i, geom in enumerate(geometries):
                print(
                    f"[DEBUG] Adding geometry {i}: type={type(geom)}, vertices={len(geom.vertices) if hasattr(geom, 'vertices') else 'N/A'}"
                )

                material = rendering.MaterialRecord()
                material.shader = "defaultLit"

                # Check if geometry has been painted with a uniform color
                if hasattr(geom, "paint_uniform_color") and hasattr(geom, "get_color"):
                    # Try to get the color from the geometry
                    try:
                        # For meshes that have been painted, we need to extract the color
                        # Open3D stores colors in vertex colors, so we need to check if they exist
                        if (
                            hasattr(geom, "vertex_colors")
                            and len(geom.vertex_colors) > 0
                        ):
                            # Get the first vertex color as the uniform color
                            color = geom.vertex_colors[0]
                            material.base_color = [color[0], color[1], color[2], 1.0]
                            print(
                                f"[DEBUG] Geometry {i} color from vertex_colors: {color}"
                            )
                        else:
                            # Fallback to default color
                            material.base_color = [0.8, 0.8, 0.8, 1.0]
                            print(
                                f"[DEBUG] Geometry {i} using default color - no vertex_colors found"
                            )
                            print(
                                f"[DEBUG] Geometry {i} has vertex_colors: {hasattr(geom, 'vertex_colors')}"
                            )
                            if hasattr(geom, "vertex_colors"):
                                print(
                                    f"[DEBUG] Geometry {i} vertex_colors length: {len(geom.vertex_colors)}"
                                )
                    except Exception as e:
                        print(
                            f"Warning: Could not extract color from geometry {i}: {e}"
                        )
                        material.base_color = [0.8, 0.8, 0.8, 1.0]
                else:
                    # Default color for geometries without color information
                    material.base_color = [0.8, 0.8, 0.8, 1.0]
                    print(f"[DEBUG] Geometry {i} using default color (no color info)")

                scene_widget.scene.add_geometry(f"debug_geom_{i}", geom, material)

            # Create a modal dialog to close the visualization
            dlg = gui.Dialog(window_name)
            dlg_layout = gui.Vert()

            info_label = gui.Label("Debug visualization - click Close to continue")
            close_btn = gui.Button("Close")

            def on_close():
                try:
                    self.app.remove_scene_widget(scene_id)
                    self.app.window.close_dialog()
                except Exception as e:
                    print(f"Error closing visualization: {e}")

            close_btn.set_on_clicked(on_close)
            dlg_layout.add_child(info_label)
            dlg_layout.add_child(close_btn)
            dlg.add_child(dlg_layout)

            # Show modal dialog (this will block until user closes it)
            self.app.window.show_dialog(dlg)

            # Try to set a good camera view
            try:
                # Calculate the center of all geometries
                all_vertices = []
                for geom in geometries:
                    if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                        all_vertices.extend(geom.vertices)

                if all_vertices:
                    import numpy as np

                    vertices_array = np.array(all_vertices)
                    center = np.mean(vertices_array, axis=0)
                    print(f"[DEBUG] Calculated center: {center}")

                    # Set camera to look at the center
                    scene_widget.setup_camera(
                        60, scene_widget.scene.bounding_box, center
                    )
            except Exception as e:
                print(f"[DEBUG] Could not set camera: {e}")
        except Exception as e:
            print(f"Error creating debug visualization: {e}")
            # Fallback: just print the window name
            print(f"Debug visualization failed for: {window_name}")

    def _finish_segmentation_pipeline(self):
        """Called when all fragments have been processed."""
        print("\n=== Segmentation Pipeline Complete ===")

        # Filter out fragments that failed feature extraction (essential for matching)
        valid_fragments_data = [
            fd
            for fd in self.processed_fragments_pipeline_data
            if fd.get("features_list")
            and any(f is not None and f.num() > 0 for f in fd["features_list"])
        ]
        if len(valid_fragments_data) < len(self.processed_fragments_pipeline_data):
            print(
                f"  Warning: {len(self.processed_fragments_pipeline_data) - len(valid_fragments_data)} fragments had no valid features and were excluded from matching."
            )

        if (
            len(valid_fragments_data) < 2
        ):  # Need at least 2 fragments for pairwise matching
            print(
                "Not enough valid fragments with features for pairwise matching. Exiting or saving unaligned."
            )
            # Save unaligned original meshes if any loaded
            if self.processed_fragments_pipeline_data:
                os.makedirs(self.app.params["output_dir"], exist_ok=True)
                all_original_meshes = [
                    fd["original_mesh"] for fd in self.processed_fragments_pipeline_data
                ]
                combined_unaligned = src.io_utils.combine_meshes(all_original_meshes)
                output_path = os.path.join(
                    self.app.params["output_dir"],
                    "reconstructed_model_unaligned_originals.obj",
                )
                src.io_utils.save_mesh(combined_unaligned, output_path)
                print(f"  Saved all original unaligned fragments to {output_path}")

        # Reset state
        self.segmentation_queue = []
        self.current_segmentation_fragment = None
        self.segmentation_results = {}

    def _handle_pairwise_matching_results(self, valid_fragments_data, pairwise_matches):
        """Handle the results of pairwise matching."""
        if not pairwise_matches:
            print("No suitable pairwise matches found above threshold.")
            # Save unaligned fragments
            self._save_unaligned_fragments(valid_fragments_data, "no_matches")
            return

        print(
            f"Found {len(pairwise_matches)} potential pairwise matches above threshold."
        )

        # Store results for next steps
        self.pairwise_matches = pairwise_matches
        self.valid_fragments_data = valid_fragments_data

        # Show results dialog
        self._show_pairwise_matching_results_dialog(
            pairwise_matches, valid_fragments_data
        )

    def _save_unaligned_fragments(self, fragments_data, reason):
        """Save unaligned fragments to output directory."""
        if not fragments_data:
            print("No fragments to save.")
            return

        try:
            os.makedirs(self.app.params["output_dir"], exist_ok=True)
            all_original_meshes = [fd["original_mesh"] for fd in fragments_data]
            combined_unaligned = src.io_utils.combine_meshes(all_original_meshes)

            filename = f"reconstructed_model_{reason}.obj"
            output_path = os.path.join(self.app.params["output_dir"], filename)
            src.io_utils.save_mesh(combined_unaligned, output_path)
            print(f"  Saved unaligned fragments to {output_path}")
        except Exception as e:
            print(f"  Error saving unaligned fragments: {e}")

    def _show_pairwise_matching_results_dialog(
        self, pairwise_matches, valid_fragments_data
    ):
        """Show a dialog with pairwise matching results and visualization options."""
        # Create dialog showing results
        dlg = gui.Dialog("Pairwise Matching Results")
        dlg_layout = gui.Vert()

        # Info label
        info_label = gui.Label(f"Found {len(pairwise_matches)} pairwise matches")
        dlg_layout.add_child(info_label)

        # Create scrollable area for match list
        matches_layout = gui.Vert()

        # Sort matches by score
        sorted_matches = sorted(
            pairwise_matches, key=lambda x: x["score"], reverse=True
        )

        for i, match in enumerate(sorted_matches[:10]):  # Show top 10 matches
            source_name = valid_fragments_data[match["source_idx"]]["name"]
            target_name = valid_fragments_data[match["target_idx"]]["name"]
            score = match["score"]

            match_label = gui.Label(
                f"{i+1}. {source_name} → {target_name} (Score: {score:.3f})"
            )
            matches_layout.add_child(match_label)

        # Add scrollable area
        scroll = gui.ScrollableVert()
        scroll.add_child(matches_layout)
        dlg_layout.add_child(scroll)

        # Buttons
        buttons_layout = gui.Horiz()

        visualize_btn = gui.Button("Visualize Top Match")

        def on_visualize():
            if sorted_matches:
                self._visualize_pairwise_match(sorted_matches[0], valid_fragments_data)
            # Don't close the dialog immediately - let user close it manually after viewing

        visualize_btn.set_on_clicked(on_visualize)

        proceed_btn = gui.Button("Proceed to Assembly")

        def on_proceed():
            self.app.window.close_dialog()
            # Start global assembly pipeline
            self._on_multipiece_matching()

        proceed_btn.set_on_clicked(on_proceed)

        close_btn = gui.Button("Close")

        def on_close():
            self.app.window.close_dialog()

        close_btn.set_on_clicked(on_close)

        buttons_layout.add_child(visualize_btn)
        buttons_layout.add_child(proceed_btn)
        buttons_layout.add_child(close_btn)
        dlg_layout.add_child(buttons_layout)

        dlg.add_child(dlg_layout)
        self.app.window.show_dialog(dlg)

    def _visualize_pairwise_match(self, match_data, valid_fragments_data):
        """Visualize a specific pairwise match using the GUI system."""
        source_idx = match_data["source_idx"]
        target_idx = match_data["target_idx"]
        transformation = match_data["transformation"]

        source_data = valid_fragments_data[source_idx]
        target_data = valid_fragments_data[target_idx]

        print(
            f"[DEBUG] Visualizing match: {source_data['name']} -> {target_data['name']}"
        )
        print(
            f"[DEBUG] Source mesh vertices: {len(source_data['original_mesh'].vertices)}"
        )
        print(
            f"[DEBUG] Target mesh vertices: {len(target_data['original_mesh'].vertices)}"
        )
        print(f"[DEBUG] Transformation matrix:\n{transformation}")

        # Create copies for visualization
        source_mesh = copy.deepcopy(source_data["original_mesh"])
        target_mesh = copy.deepcopy(target_data["original_mesh"])

        # Validate meshes
        if len(source_mesh.vertices) == 0 or len(target_mesh.vertices) == 0:
            print(
                f"[ERROR] Invalid mesh: source vertices={len(source_mesh.vertices)}, target vertices={len(target_mesh.vertices)}"
            )
            return

        # Ensure normals for display
        if not source_mesh.has_vertex_normals():
            source_mesh.compute_vertex_normals()
        if not target_mesh.has_vertex_normals():
            target_mesh.compute_vertex_normals()

        # Validate transformation matrix
        if transformation is None or transformation.shape != (4, 4):
            print(f"[ERROR] Invalid transformation matrix: {transformation}")
            return

        # Apply transformation to source mesh
        source_mesh.transform(transformation)

        # Debug bounding boxes
        source_bbox = source_mesh.get_axis_aligned_bounding_box()
        target_bbox = target_mesh.get_axis_aligned_bounding_box()
        print(
            f"[DEBUG] Source bbox: min={source_bbox.min_bound}, max={source_bbox.max_bound}"
        )
        print(
            f"[DEBUG] Target bbox: min={target_bbox.min_bound}, max={target_bbox.max_bound}"
        )

        # Color the meshes
        source_mesh.paint_uniform_color([1, 0, 0])  # Red for source
        target_mesh.paint_uniform_color([0, 1, 0])  # Green for target

        # Show visualization
        window_name = f"Pairwise_Match_{source_data['name']}_to_{target_data['name']}_Score_{match_data['score']:.3f}"
        print(f"[DEBUG] Showing visualization: {window_name}")
        self.show_debug_visualization(
            [source_mesh, target_mesh],
            window_name,
        )

    def _start_greedy_assembly(self, assembler):
        """Start the greedy assembly process with GUI support."""
        # Store the assembler for the assembly process
        self.current_assembler = assembler
        self.assembly_state = {
            "current_assembly_components": [],
            "used_matches": set(),
            "rejected_matches": [],
            "finalized": False,
            "num_placed": 0,
        }

        # Start the assembly process
        self._continue_assembly_process()

    def _continue_assembly_process(self):
        """Continue the assembly process, handling one candidate at a time."""
        if self.assembly_state["finalized"]:
            self._finish_assembly()
            return

        # Find the next best candidate
        best_candidate = self._find_next_assembly_candidate()

        if best_candidate is None:
            print("No more valid candidates for assembly.")
            self._finish_assembly()
            return

        # Debug information
        candidate_name = self.current_assembler.fragments_data[best_candidate["idx"]][
            "name"
        ]
        print(
            f"  Found candidate: {candidate_name} (score: {best_candidate['score']:.3f})"
        )
        if best_candidate["match_info"]:
            match_info = best_candidate["match_info"]
            source_name = self.current_assembler.fragments_data[
                match_info["source_idx"]
            ]["name"]
            target_name = self.current_assembler.fragments_data[
                match_info["target_idx"]
            ]["name"]
            print(f"    Match: {source_name} -> {target_name}")

        # Show the candidate for user interaction
        self._show_assembly_candidate_dialog(best_candidate)

    def _find_next_assembly_candidate(self):
        """Find the next best candidate for assembly."""
        assembler = self.current_assembler
        current_components = self.assembly_state["current_assembly_components"]
        used_matches = self.assembly_state["used_matches"]

        best_candidate_idx = -1
        best_candidate_match_info = None
        best_candidate_world_transform = None
        best_candidate_score = 0.0

        # If no fragments placed yet, start with the first fragment
        if not current_components:
            best_candidate_idx = 0
            best_candidate_world_transform = np.eye(4)
            best_candidate_score = 1.0
            return {
                "idx": best_candidate_idx,
                "match_info": None,
                "world_transform": best_candidate_world_transform,
                "score": best_candidate_score,
            }

        # Find the best candidate from available matches
        for match_info in assembler.pairwise_matches:
            if id(match_info) in used_matches:
                continue

            s_idx, t_idx = match_info["source_idx"], match_info["target_idx"]

            # Check if one fragment is placed and the other isn't
            if (
                assembler.is_fragment_placed[t_idx]
                and not assembler.is_fragment_placed[s_idx]
            ):
                world_transform = np.dot(
                    assembler.fragment_transforms[t_idx], match_info["transformation"]
                )
                idx_to_place = s_idx
                match_score = match_info["score"]
            elif (
                assembler.is_fragment_placed[s_idx]
                and not assembler.is_fragment_placed[t_idx]
            ):
                try:
                    inv_transform = np.linalg.inv(match_info["transformation"])
                    world_transform = np.dot(
                        assembler.fragment_transforms[s_idx], inv_transform
                    )
                    idx_to_place = t_idx
                    match_score = match_info["score"]
                except np.linalg.LinAlgError:
                    continue
            else:
                continue

            # Check if this candidate has a better score
            if match_score > best_candidate_score:
                best_candidate_idx = idx_to_place
                best_candidate_match_info = match_info
                best_candidate_world_transform = world_transform
                best_candidate_score = match_score

        if best_candidate_idx != -1:
            return {
                "idx": best_candidate_idx,
                "match_info": best_candidate_match_info,
                "world_transform": best_candidate_world_transform,
                "score": best_candidate_score,
            }

        return None

    def _show_assembly_candidate_dialog(self, candidate):
        """Show a dialog for the user to accept/reject an assembly candidate."""
        candidate_idx = candidate["idx"]
        candidate_name = self.current_assembler.fragments_data[candidate_idx]["name"]
        candidate_score = candidate["score"]
        match_info = candidate["match_info"]

        # Create candidate mesh for visualization
        candidate_mesh = copy.deepcopy(
            self.current_assembler.original_meshes[candidate_idx]
        )
        candidate_mesh.transform(candidate["world_transform"])
        candidate_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Red for candidate

        # Create placed meshes for visualization
        placed_meshes = []
        for mesh, _ in self.assembly_state["current_assembly_components"]:
            placed_mesh = copy.deepcopy(mesh)
            placed_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for placed
            placed_meshes.append(placed_mesh)

        # Create visualization
        all_geometries = [candidate_mesh] + placed_meshes

        # Create window name with more context
        if match_info:
            source_name = self.current_assembler.fragments_data[
                match_info["source_idx"]
            ]["name"]
            target_name = self.current_assembler.fragments_data[
                match_info["target_idx"]
            ]["name"]
            window_name = f"Assembly Candidate {candidate_name} matched to {target_name if candidate_idx == match_info['source_idx'] else source_name} Score {candidate_score:.3f}"
        else:
            window_name = f"Assembly Candidate {candidate_name} (first fragment) Score {candidate_score:.3f}"

        # Show visualization
        self.show_debug_visualization(all_geometries, window_name)

        # Create dialog for user decision
        dlg = gui.Dialog(f"Assembly Candidate: {candidate_name}")
        dlg_layout = gui.Vert()

        # Info
        info_label = gui.Label(f"Candidate: {candidate_name}")
        score_label = gui.Label(f"Score: {candidate_score:.3f}")
        dlg_layout.add_child(info_label)
        dlg_layout.add_child(score_label)

        # Show match information if available
        if match_info:
            source_name = self.current_assembler.fragments_data[
                match_info["source_idx"]
            ]["name"]
            target_name = self.current_assembler.fragments_data[
                match_info["target_idx"]
            ]["name"]
            if candidate_idx == match_info["source_idx"]:
                match_label = gui.Label(f"Matched to: {target_name}")
            else:
                match_label = gui.Label(f"Matched to: {source_name}")
            dlg_layout.add_child(match_label)

        # Instructions
        instructions = gui.Label("Red mesh = candidate, Gray meshes = already placed")
        dlg_layout.add_child(instructions)

        # Buttons
        buttons_layout = gui.Horiz()

        accept_btn = gui.Button("Accept")

        def on_accept():
            self._handle_assembly_decision(candidate, "accept")
            self.app.window.close_dialog()

        accept_btn.set_on_clicked(on_accept)

        reject_btn = gui.Button("Reject")

        def on_reject():
            self._handle_assembly_decision(candidate, "reject")
            self.app.window.close_dialog()

        reject_btn.set_on_clicked(on_reject)

        finalize_btn = gui.Button("Finalize Assembly")

        def on_finalize():
            self._handle_assembly_decision(candidate, "finalize")
            self.app.window.close_dialog()

        finalize_btn.set_on_clicked(on_finalize)

        quit_btn = gui.Button("Quit")

        def on_quit():
            self._handle_assembly_decision(candidate, "quit")
            self.app.window.close_dialog()

        quit_btn.set_on_clicked(on_quit)

        buttons_layout.add_child(accept_btn)
        buttons_layout.add_child(reject_btn)
        buttons_layout.add_child(finalize_btn)
        buttons_layout.add_child(quit_btn)
        dlg_layout.add_child(buttons_layout)

        dlg.add_child(dlg_layout)
        self.app.window.show_dialog(dlg)

    def _handle_assembly_decision(self, candidate, decision):
        """Handle the user's decision for an assembly candidate."""
        assembler = self.current_assembler
        candidate_idx = candidate["idx"]
        candidate_name = assembler.fragments_data[candidate_idx]["name"]
        match_info = candidate["match_info"]

        if decision == "accept":
            # Accept the match
            assembler.fragment_transforms[candidate_idx] = candidate["world_transform"]
            assembler.is_fragment_placed[candidate_idx] = True
            self.assembly_state["current_assembly_components"].append(
                (assembler._get_transformed_mesh(candidate_idx), candidate_name)
            )
            self.assembly_state["num_placed"] += 1
            if match_info:
                self.assembly_state["used_matches"].add(id(match_info))
            print(
                f"  Placed fragment: {candidate_name} via match score {candidate['score']:.3f}"
            )

            # Continue with next candidate
            self._continue_assembly_process()

        elif decision == "reject":
            # Reject the match, add to rejected_matches
            if match_info:
                self.assembly_state["rejected_matches"].append(match_info)
                self.assembly_state["used_matches"].add(id(match_info))
            print(f"  Rejected match for fragment: {candidate_name}")

            # Continue with next candidate
            self._continue_assembly_process()

        elif decision == "finalize":
            # Finalize assembly
            self.assembly_state["finalized"] = True
            print("  Finalizing assembly as per user request.")
            self._finish_assembly()

        elif decision == "quit":
            # Quit assembly
            self.assembly_state["finalized"] = True
            print("  Assembly aborted by user.")
            self._finish_assembly()

    def _finish_assembly(self):
        """Finish the assembly process and save results."""
        assembler = self.current_assembler
        num_placed = self.assembly_state["num_placed"]

        print(f"\n=== Assembly Complete ===")
        print(
            f"Placed {num_placed} fragments out of {len(assembler.fragments_data)} total fragments."
        )

        if num_placed > 0:
            # Combine all placed fragments
            placed_meshes = []
            for i in range(len(assembler.fragments_data)):
                if assembler.is_fragment_placed[i]:
                    placed_meshes.append(assembler._get_transformed_mesh(i))

            if placed_meshes:
                reconstructed_model = src.io_utils.combine_meshes(placed_meshes)

                # Save the reconstructed model
                try:
                    os.makedirs(self.app.params["output_dir"], exist_ok=True)
                    output_path = os.path.join(
                        self.app.params["output_dir"],
                        "reconstructed_model_assembly.obj",
                    )
                    src.io_utils.save_mesh(reconstructed_model, output_path)
                    print(f"  Saved reconstructed model to {output_path}")

                    # Show final result
                    self._show_final_assembly_result(reconstructed_model, num_placed)

                except Exception as e:
                    print(f"  Error saving reconstructed model: {e}")
        else:
            print("  No fragments were placed. Assembly failed.")

        # Clean up
        self.current_assembler = None
        self.assembly_state = None

    def _show_final_assembly_result(self, reconstructed_model, num_placed):
        """Show the final assembly result."""
        # Color the reconstructed model
        reconstructed_model.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray

        # Show visualization
        self.show_debug_visualization(
            [reconstructed_model],
            f"Final Assembly Result - {num_placed} fragments combined",
        )
