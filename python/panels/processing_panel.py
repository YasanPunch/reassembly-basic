import open3d.visualization.gui as gui

class ProcessingPanel:
    def __init__(self, app):
        self.app = app

        w = app.window
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        self.label = gui.Label("Processing Panel")
        self._panel.add_child(self.label)
        self._panel.add_fixed(separation_height)

        process_ctrls = gui.CollapsableVert(
            "Process controls", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )

        self._preprocessing_button = gui.Button("Pre-processing")
        self._preprocessing_button.horizontal_padding_em = 0.5
        self._preprocessing_button.vertical_padding_em = 0
        self._preprocessing_button.set_on_clicked(self._on_preprocessing)

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

        process_ctrls.add_child(self._preprocessing_button)
        process_ctrls.add_child(self._segmentation_button)
        process_ctrls.add_child(self._classification_button)
        process_ctrls.add_child(self._pairwise_matching_button)
        process_ctrls.add_child(self._multipiece_matching_button)

        self._panel.add_child(process_ctrls)
        self._panel.add_fixed(separation_height)

    def _on_preprocessing(self):
        pass

    def _on_segmentation(self):
        pass

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
