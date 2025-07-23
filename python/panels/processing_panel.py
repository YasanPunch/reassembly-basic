import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering
from typing import List
from panels.preprocessing_dialog import PreprocessingDialog
from panels.segmentation_dialog import SegmentationDialog

class ProcessingPanel:
    def __init__(self, app):
        self.app = app

        w = app.window  # to make the code more concise
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

        # 1. Pre-processing
        self._preprocessing_button = gui.Button("Pre-processing")
        self._preprocessing_button.horizontal_padding_em = 0.5
        self._preprocessing_button.vertical_padding_em = 0
        self._preprocessing_button.set_on_clicked(self._on_preprocessing)

        # 2. Segmentation
        self._segmentation_button = gui.Button("Segmentation")
        self._segmentation_button.horizontal_padding_em = 0.5
        self._segmentation_button.vertical_padding_em = 0
        self._segmentation_button.set_on_clicked(self._on_segmentation)

        # 3. Pairwise Matching
        self._pairwise_matching_button = gui.Button("Pairwise Matching")
        self._pairwise_matching_button.horizontal_padding_em = 0.5
        self._pairwise_matching_button.vertical_padding_em = 0
        self._pairwise_matching_button.set_on_clicked(self._on_pairwise_matching)

        # 4. Multipiece Matching
        self._multipiece_matching_button = gui.Button("Multipiece Matching")
        self._multipiece_matching_button.horizontal_padding_em = 0.5
        self._multipiece_matching_button.vertical_padding_em = 0
        self._multipiece_matching_button.set_on_clicked(self._on_multipiece_matching)

        # Add buttons in order
        process_ctrls.add_child(self._preprocessing_button)
        process_ctrls.add_child(self._segmentation_button)
        process_ctrls.add_child(self._pairwise_matching_button)
        process_ctrls.add_child(self._multipiece_matching_button)

        self._segment_mesh_button = gui.Button("Segment")
        self._segment_mesh_button.horizontal_padding_em = 0.5
        self._segment_mesh_button.vertical_padding_em = 0
        self._segment_mesh_button.set_on_clicked(self._on_segment)

        self._boundary_lines_button = gui.Button("Boundary Lines")
        self._boundary_lines_button.horizontal_padding_em = 0.5
        self._boundary_lines_button.vertical_padding_em = 0
        self._boundary_lines_button.set_on_clicked(self._on_boundary_lines)

        process_ctrls.add_child(self._segment_mesh_button)
        process_ctrls.add_child(self._boundary_lines_button)
        self._panel.add_child(process_ctrls)
        self._panel.add_fixed(separation_height)

    def _on_preprocessing(self):
        """Handle pre-processing step - shows dialog with parameters"""
        # Get selected base model items
        selected_items = self._get_selected_base_models()
        if not selected_items:
            # Show error dialog
            self._show_error_dialog("No base models selected", "Please select base models in the item tree before preprocessing.")
            return

        # Show preprocessing dialog
        preprocessing_dialog = PreprocessingDialog(self.app)
        preprocessing_dialog.show_dialog(selected_items)

    def _get_selected_base_models(self) -> List:
        """Get the currently selected base model items."""
        selected_items = []
        base_model_items = self.app._left_panel.item_tree.get_items_by_type('base_model')

        for item in base_model_items:
            if item.is_visible:  # Checked items are visible
                selected_items.append(item)

        return selected_items

    def _show_error_dialog(self, title: str, message: str):
        """Show an error dialog."""
        dialog = gui.Dialog(title)
        content = gui.Vert(0, gui.Margins(1.0, 1.0, 1.0, 1.0))
        content.add_child(gui.Label(message))

        button_layout = gui.Horiz(0.25)
        ok_button = gui.Button("OK")
        ok_button.set_on_clicked(lambda: self.app.window.close_dialog())
        button_layout.add_child(ok_button)
        content.add_child(button_layout)

        dialog.add_child(content)
        self.app.window.show_dialog(dialog)

    def _on_segmentation(self):
        """Handle segmentation step - shows dialog with parameters"""
        # Get selected preprocessed items
        selected_items = self._get_selected_preprocessed_items()
        if not selected_items:
            # Show error dialog
            self._show_error_dialog("No preprocessed items selected", "Please select preprocessed items in the item tree before segmentation.")
            return

        # Show segmentation dialog
        segmentation_dialog = SegmentationDialog(self.app)
        segmentation_dialog.show_dialog(selected_items)

    def _get_selected_preprocessed_items(self) -> List:
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
        """Handle pairwise matching step"""
        print("Pairwise Matching step initiated")
        selected_objects = self.app._left_panel.get_selected_objects()
        for i, obj in enumerate(self.app._loaded_objects):
            if obj in selected_objects:
                # TODO: Implement actual pairwise matching
                # For now, just create a copy of the mesh as a demo
                result_mesh = o3d.io.read_triangle_mesh(obj['path'])
                # Add result to DB Tree
                self.app._left_panel.add_processing_result(i, 'pairwise_matching', result_mesh)

    def _on_multipiece_matching(self):
        """Handle multipiece matching step"""
        print("Multipiece Matching step initiated")
        selected_objects = self.app._left_panel.get_selected_objects()
        for i, obj in enumerate(self.app._loaded_objects):
            if obj in selected_objects:
                # TODO: Implement actual multipiece matching
                # For now, just create a copy of the mesh as a demo
                result_mesh = o3d.io.read_triangle_mesh(obj['path'])
                # Add result to DB Tree
                self.app._left_panel.add_processing_result(i, 'multipiece_matching', result_mesh)

    def _on_segment(self):
        fractureSurface = FractureSurfaceMesh()

        for i in self.app._scenes_selected:
            if i == 0:
                continue

            path = self.app._scenes_paths[i]
            res = fractureSurface.extract_fracture_surface_mesh(path)
            o3d.visualization.draw_geometries([res])
            mesh = o3d.visualization.rendering.TriangleMeshModel()
            mesh.materials.append(self.app.settings.material)
            mesh.meshes.append(res)
            self.app._scenes[i].scene.clear_geometry()
            self.app._scenes[i].scene.add_model("_model_processed", mesh)

    def _on_boundary_lines(self):
        boundaryCurves = BoundaryCurves()

        line_material = rendering.MaterialRecord()
        line_material.shader = "unlitLine"  # This is the key shader for lines
        line_material.line_width = 2.0  # Adjust line thickness as needed
        line_material.base_color = [0.0, 0.0, 0.0, 1.0]

        for i in self.app._scenes_selected:
            if i == 0:
                continue

            path = self.app._scenes_paths[i]
            point_cloud, line_sets = boundaryCurves.extract_pointcloud_boundaries(path)
            self.app._scenes[i].scene.clear_geometry()
            self.app._scenes[i].scene.add_geometry(
                "PointCloud", point_cloud, self.app.settings.material
            )
            for i, line_set in enumerate(line_sets):
                self.app._scenes[i].scene.add_geometry(
                    f"LineSetPCD_{i}",
                    create_point_cloud_from_lineset(line_set),
                    self.app.settings.material,
                )


def create_point_cloud_from_lineset(line_set):
    """
    Creates a PointCloud object from a LineSet's points.
    If the LineSet has colors, those colors are also transferred to the PointCloud.
    """
    if not line_set.has_points():
        print("Warning: LineSet has no points. Returning empty PointCloud.")
        return o3d.geometry.PointCloud()

    pcd = o3d.geometry.PointCloud()
    pcd.points = line_set.points  # Directly assign the Vector3dVector of points
    pcd.paint_uniform_color([0.0, 0.0, 0.0])

    return pcd
