import open3d.visualization.gui as gui
import threading
import time
from typing import List, Dict, Any

from python.left_panel.item_tree.items import (
    SegmentedItem,
    SegmentationResultItem,
)
from python.pipeline.segmentation_engine import SegmentationEngine


class SegmentationDialog:
    """
    Dialog for segmentation operations with parameter configuration.
    """

    def __init__(self, app):
        self.app = app
        self.is_processing = False

    def show_dialog(self, selected_preprocessed_items: List):
        """Show segmentation dialog with parameters."""
        # Create dialog
        dialog = gui.Dialog("Segmentation Parameters")

        # Create dialog content
        em = self.app.window.theme.font_size
        content = gui.Vert(0, gui.Margins(em, em, em, em))

        # Title
        title = gui.Label(f"Segment {len(selected_preprocessed_items)} selected preprocessed models")
        title.text_color = gui.Color(0.2, 0.6, 1.0)
        content.add_child(title)
        content.add_fixed(int(round(0.5 * em)))

        # Parameters section
        params_section = gui.CollapsableVert("Parameters", 0.25 * em)

        # Max curvature
        curvature_layout = gui.Horiz(0.25 * em)
        curvature_layout.add_child(gui.Label("Max Curvature (deg):"))
        max_curvature_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        max_curvature_edit.set_value(30.0)
        max_curvature_edit.set_limits(5.0, 90.0)
        curvature_layout.add_child(max_curvature_edit)
        params_section.add_child(curvature_layout)

        # Area limit fraction
        area_layout = gui.Horiz(0.25 * em)
        area_layout.add_child(gui.Label("Min Area Fraction (%):"))
        area_fraction_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        area_fraction_edit.set_value(2.0)
        area_fraction_edit.set_limits(0.1, 50.0)
        area_layout.add_child(area_fraction_edit)
        params_section.add_child(area_layout)

        # Use bumpiness detection
        bumpiness_checkbox = gui.Checkbox("Use Bumpiness Detection")
        bumpiness_checkbox.checked = False
        params_section.add_child(bumpiness_checkbox)

        # Bumpiness threshold
        bumpiness_layout = gui.Horiz(0.25 * em)
        bumpiness_layout.add_child(gui.Label("Bumpiness Threshold:"))
        bumpiness_threshold_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        bumpiness_threshold_edit.set_value(0.2)
        bumpiness_threshold_edit.set_limits(0.01, 1.0)
        bumpiness_layout.add_child(bumpiness_threshold_edit)
        params_section.add_child(bumpiness_layout)

        # Elevation map resolution
        resolution_layout = gui.Horiz(0.25 * em)
        resolution_layout.add_child(gui.Label("Elevation Map Resolution:"))
        elevation_resolution_edit = gui.NumberEdit(gui.NumberEdit.Type.INT)
        elevation_resolution_edit.set_value(64)
        elevation_resolution_edit.set_limits(16, 256)
        resolution_layout.add_child(elevation_resolution_edit)
        params_section.add_child(resolution_layout)

        content.add_child(params_section)
        content.add_fixed(int(round(0.5 * em)))

        # Progress section (initially hidden)
        progress_section = gui.Vert(0)
        progress_label = gui.Label("Ready to start segmentation...")
        progress_bar = gui.ProgressBar()
        progress_bar.visible = False
        progress_section.add_child(progress_label)
        progress_section.add_child(progress_bar)
        content.add_child(progress_section)

        # Buttons
        button_layout = gui.Horiz(0.25 * em)

        ok_button = gui.Button("Start Segmentation")
        cancel_button = gui.Button("Cancel")

        button_layout.add_child(ok_button)
        button_layout.add_child(cancel_button)
        content.add_child(button_layout)

        # Store dialog content for access in callbacks
        dialog_content = {
            'dialog': dialog,
            'progress_label': progress_label,
            'progress_bar': progress_bar,
            'ok_button': ok_button,
            'cancel_button': cancel_button,
            'max_curvature_edit': max_curvature_edit,
            'area_fraction_edit': area_fraction_edit,
            'bumpiness_checkbox': bumpiness_checkbox,
            'bumpiness_threshold_edit': bumpiness_threshold_edit,
            'elevation_resolution_edit': elevation_resolution_edit
        }

        # Set up button callbacks
        def on_ok_clicked():
            if self.is_processing:
                return

            # Get parameters
            parameters = {
                'max_curvature_deg': max_curvature_edit.double_value,
                'area_limit_fraction': area_fraction_edit.double_value / 100.0,  # Convert % to fraction
                'use_bumpiness_detection': bumpiness_checkbox.checked,
                'bumpiness_threshold': bumpiness_threshold_edit.double_value,
                'elevation_map_resolution': int(elevation_resolution_edit.int_value),
                'visualize_segmentation': False  # Disable interactive visualization in batch mode
            }

            # Start segmentation
            self._start_segmentation_in_dialog(dialog_content, parameters, selected_preprocessed_items)

        def on_cancel_clicked():
            if self.is_processing:
                self.is_processing = False
                dialog_content['progress_label'].text = "Cancelling..."
            else:
                self.app.window.close_dialog()

        ok_button.set_on_clicked(on_ok_clicked)
        cancel_button.set_on_clicked(on_cancel_clicked)

        # Show dialog
        dialog.add_child(content)
        self.app.window.show_dialog(dialog)

    def _start_segmentation_in_dialog(self, dialog_content: Dict, parameters: Dict[str, Any], selected_items: List):
        """Start segmentation process in dialog."""
        self.is_processing = True

        # Update UI
        dialog_content['ok_button'].enabled = False
        dialog_content['cancel_button'].text = "Cancel"
        dialog_content['progress_bar'].visible = True
        dialog_content['progress_label'].text = "Initializing segmentation..."
        dialog_content['progress_bar'].value = 0.0

        # Create segmentation engine
        def progress_callback(step: str, progress: float, message: str):
            gui.Application.instance.post_to_main_thread(
                self.app.window, lambda: self._on_dialog_progress_update(dialog_content, step, progress, message)
            )

        segmentation_engine = SegmentationEngine(progress_callback=progress_callback)

        # Run segmentation in background thread
        thread = threading.Thread(
            target=self._run_segmentation_in_dialog,
            args=(dialog_content, parameters, selected_items, segmentation_engine)
        )
        thread.daemon = True
        thread.start()

    def _run_segmentation_in_dialog(self, dialog_content: Dict, parameters: Dict[str, Any], 
                                  selected_items: List, segmentation_engine):
        """Run segmentation in background thread."""
        try:
            # Perform segmentation
            results = segmentation_engine.segment_preprocessed_items(selected_items, parameters)

            # Add results to UI in main thread
            gui.Application.instance.post_to_main_thread(
                self.app.window, lambda: self._add_segmentation_results_from_dialog(dialog_content, results)
            )

        except Exception as e:
            # Handle error in main thread
            gui.Application.instance.post_to_main_thread(
                self.app.window, lambda: self._handle_segmentation_error_in_dialog(dialog_content, str(e))
            )

    def _on_dialog_progress_update(self, dialog_content: Dict, step: str, progress: float, message: str):
        """Handle progress updates from segmentation engine in dialog."""
        gui.Application.instance.post_to_main_thread(
            self.app.window, lambda: self._update_dialog_progress(dialog_content, progress, message)
        )

    def _update_dialog_progress(self, dialog_content: Dict, progress: float, message: str):
        """Update progress bar and status in dialog."""
        dialog_content['progress_bar'].value = progress / 100.0
        dialog_content['progress_label'].text = message

    def _add_segmentation_results_from_dialog(self, dialog_content: Dict, results: List[Dict]):
        """Add segmentation results to the item tree from dialog."""
        successful_results = 0
        segmentation_results = []  # Collect all successful results

        for result in results:
            if result['success']:
                preprocessed_item = result['item']
                label = f"Segmented: {preprocessed_item.label}"

                # Create SegmentedItems for each segment
                segmented_items = []
                for segment_data in result['segments']:
                    # Create unique scene path for each segment
                    base_name = f"segment_{preprocessed_item.label}_{segment_data['index']}"
                    counter = 0
                    scene_path = base_name
                    while True:
                        try:
                            self.app._scene_widget.scene.get_geometry(scene_path)
                            counter += 1
                            scene_path = f"{base_name}_{counter}"
                        except:
                            break

                    # Create SegmentedItem
                    segmented_item = SegmentedItem(
                        label=segment_data['label'],
                        segment_faces=segment_data.get('face_indices'),
                        segment_mesh=segment_data['segment_mesh'],
                        original_mesh=preprocessed_item.original_mesh,
                        preprocessed_item=preprocessed_item,
                        segment_properties=segment_data['segment_properties'],
                        segment_color=segment_data['segment_color'],
                        scene_path=scene_path
                    )

                    # Add segment mesh to scene with its color
                    if segment_data['segment_mesh']:
                        self.app._scene_widget.scene.add_geometry(
                            scene_path,
                            segment_data['segment_mesh'],
                            self.app.settings.material if hasattr(self.app.settings, 'material') else None
                        )
                        # Apply segment color
                        segment_data['segment_mesh'].paint_uniform_color(segment_data['segment_color'])
                        self.app._scene_widget.scene.show_geometry(scene_path, True)
                        print(f"[DEBUG] Added segment to scene: {scene_path} with color {segment_data['segment_color']}")

                    segmented_items.append(segmented_item)

                # Create SegmentationResultItem
                segmentation_result = SegmentationResultItem(
                    label=label,
                    segmented_items=segmented_items,
                    original_mesh=preprocessed_item.original_mesh,
                    preprocessed_item=preprocessed_item,
                    segmentation_parameters=result['parameters_used']
                )
                segmentation_result.total_processing_time = result['processing_time']

                successful_results += 1
                segmentation_results.append(segmentation_result)

        # Create batch result item if we have successful results
        if segmentation_results:
            timestamp = int(time.time())
            batch_label = f"Segmentation Batch {timestamp}"

            # Create the batch container
            batch_item = self.app._left_panel.item_tree.add_segmentation_batch_item(
                label=batch_label,
                segmentation_results=segmentation_results,
                batch_parameters=results[0]['parameters_used'] if results else {}
            )

            print(f"[DEBUG] Created segmentation batch: {batch_label} with {len(segmentation_results)} results")

        # Update dialog with completion
        dialog_content['progress_label'].text = f"Completed: {successful_results}/{len(results)} models segmented successfully"
        dialog_content['ok_button'].enabled = True
        dialog_content['cancel_button'].text = "Close"

        # Auto-close dialog after a short delay
        def auto_close():
            self.app.window.close_dialog()

        # Schedule auto-close after 2 seconds
        timer = threading.Timer(2.0, auto_close)
        timer.daemon = True
        timer.start()

        # Reset processing state
        self.is_processing = False

    def _handle_segmentation_error_in_dialog(self, dialog_content: Dict, error_message: str):
        """Handle segmentation errors in dialog."""
        dialog_content['progress_label'].text = f"Error: {error_message}"
        dialog_content['ok_button'].enabled = True
        dialog_content['cancel_button'].text = "Close"
        self.is_processing = False 
