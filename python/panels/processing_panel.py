import os

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import src.io_utils
import src.preprocessing

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
        """Test with new window approach"""

        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering

        # Create dummy meshes
        candidate_mesh = o3d.geometry.TriangleMesh.create_box(
            width=1.0, height=1.0, depth=1.0
        )
        candidate_mesh.paint_uniform_color([0.8, 0.2, 0.2])  # Red
        candidate_mesh.compute_vertex_normals()

        placed_meshes = []
        for i in range(3):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            sphere.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
            sphere.compute_vertex_normals()
            sphere.translate([i * 2.0, 0, 0])
            placed_meshes.append(sphere)

        # Create a new window with scene widget
        scene_id = "pairwise_test"
        scene_widget = self.app.add_scene_widget(
            scene_id, title="Pairwise Matching Test", width=800, height=600
        )

        # Add geometries to the new scene
        candidate_material = rendering.MaterialRecord()
        candidate_material.shader = "defaultLit"
        candidate_material.base_color = [0.8, 0.2, 0.2, 1.0]  # Red

        placed_material = rendering.MaterialRecord()
        placed_material.shader = "defaultLit"
        placed_material.base_color = [0.7, 0.7, 0.7, 1.0]  # Gray

        scene_widget.scene.add_geometry("candidate", candidate_mesh, candidate_material)

        for i, mesh in enumerate(placed_meshes):
            scene_widget.scene.add_geometry(f"placed_{i}", mesh, placed_material)

        # Set camera for the new scene
        bounds = scene_widget.scene.bounding_box
        scene_widget.setup_camera(60, bounds, bounds.get_center())

        # Create a dialog for user interaction
        em = self.app.window.theme.font_size
        dlg = gui.Dialog("Pairwise Matching Test")

        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Red cube: Candidate fragment"))
        dlg_layout.add_child(gui.Label("Gray spheres: Placed fragments"))
        dlg_layout.add_child(gui.Label("Look at the new window for visualization"))

        # Add buttons
        button_layout = gui.Horiz()

        accept_btn = gui.Button("Accept")
        reject_btn = gui.Button("Reject")
        finalize_btn = gui.Button("Finalize")
        quit_btn = gui.Button("Quit")

        def cleanup_and_close():
            """Clean up scene and close dialog"""
            # Close the window
            self.app.remove_scene_widget(scene_id)
            self.app.window.close_dialog()

        def on_accept():
            print("Match accepted!")
            cleanup_and_close()

        def on_reject():
            print("Match rejected!")
            cleanup_and_close()

        def on_finalize():
            print("Assembly finalized!")
            cleanup_and_close()

        def on_quit():
            print("Selection aborted!")
            cleanup_and_close()

        accept_btn.set_on_clicked(on_accept)
        reject_btn.set_on_clicked(on_reject)
        finalize_btn.set_on_clicked(on_finalize)
        quit_btn.set_on_clicked(on_quit)

        button_layout.add_child(accept_btn)
        button_layout.add_child(reject_btn)
        button_layout.add_child(finalize_btn)
        button_layout.add_child(quit_btn)

        dlg_layout.add_child(button_layout)
        dlg.add_child(dlg_layout)

        self.app.window.show_dialog(dlg)

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
            src.preprocessing.preprocess_fragment(frag_info_raw, self.app.params, self)
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
                    fd["mesh"] for fd in self.processed_fragments_pipeline_data
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
