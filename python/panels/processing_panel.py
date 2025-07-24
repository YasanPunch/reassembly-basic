import os
import copy

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import src.io_utils
import src.preprocessing
import src.matching

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

        self._classification_button = gui.Button("Classification")
        self._classification_button.horizontal_padding_em = 0.5
        self._classification_button.vertical_padding_em = 0
        self._classification_button.set_on_clicked(self._on_classification)

        self._pairwise_matching_button = gui.Button("Pairwise Matching")
        self._pairwise_matching_button.horizontal_padding_em = 0.5
        self._pairwise_matching_button.vertical_padding_em = 0
        self._pairwise_matching_button.set_on_clicked(self._on_pairwise_matching)

        self._multipiece_matching_button = gui.Button("Multipiece Matching")
        self._multipiece_matching_button.horizontal_padding_em = 0.5
        self._multipiece_matching_button.vertical_padding_em = 0
        self._multipiece_matching_button.set_on_clicked(self._on_multipiece_matching)

        process_ctrls.add_child(self._segmentation_button)
        process_ctrls.add_child(self._classification_button)
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
        pass

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
        # Create a unique scene ID for this debug visualization
        scene_id = f"debug_{window_name.replace(' ', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '')}"

        # Create scene widget
        scene_widget = self.app.add_scene_widget(scene_id, window_name)

        # Add all geometries to the scene
        for i, geom in enumerate(geometries):
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"

            # If geometry has uniform color, use it
            if hasattr(geom, "paint_uniform_color") and hasattr(geom, "get_color"):
                # For geometries that have been painted, we need to create a material with that color
                # This is a simplified approach - in practice, you might need to extract the color differently
                material.base_color = [0.8, 0.8, 0.8, 1.0]  # Default gray

            scene_widget.scene.add_geometry(f"debug_geom_{i}", geom, material)

        # Create a modal dialog to close the visualization
        dlg = gui.Dialog(window_name)
        dlg_layout = gui.Vert()

        info_label = gui.Label("Debug visualization - click Close to continue")
        close_btn = gui.Button("Close")

        def on_close():
            self.app.remove_scene_widget(scene_id)
            self.app.window.close_dialog()

        close_btn.set_on_clicked(on_close)
        dlg_layout.add_child(info_label)
        dlg_layout.add_child(close_btn)
        dlg.add_child(dlg_layout)

        # Show modal dialog (this will block until user closes it)
        self.app.window.show_dialog(dlg)

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
            dlg.close()

        visualize_btn.set_on_clicked(on_visualize)

        proceed_btn = gui.Button("Proceed to Assembly")

        def on_proceed():
            dlg.close()
            # TODO: Start global assembly pipeline
            print("Proceeding to global assembly...")

        proceed_btn.set_on_clicked(on_proceed)

        close_btn = gui.Button("Close")

        def on_close():
            dlg.close()

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

        # Create copies for visualization
        source_mesh = copy.deepcopy(source_data["original_mesh"])
        target_mesh = copy.deepcopy(target_data["original_mesh"])

        # Ensure normals for display
        if not source_mesh.has_vertex_normals():
            source_mesh.compute_vertex_normals()
        if not target_mesh.has_vertex_normals():
            target_mesh.compute_vertex_normals()

        # Apply transformation to source mesh
        source_mesh.transform(transformation)

        # Color the meshes
        source_mesh.paint_uniform_color([1, 0, 0])  # Red for source
        target_mesh.paint_uniform_color([0, 1, 0])  # Green for target

        # Show visualization
        self.show_debug_visualization(
            [source_mesh, target_mesh],
            f"Pairwise Match: {source_data['name']} → {target_data['name']} (Score: {match_data['score']:.3f})",
        )
