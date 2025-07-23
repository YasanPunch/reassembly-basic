import open3d.visualization.gui as gui  # type: ignore
import threading
import time
from typing import List, Dict, Any, Optional
from python.left_panel.item_tree.items import PreprocessedItem
from python.pipeline.preprocessing_engine import PreprocessingEngine

class PreprocessingDialog:
    """
    Dialog for preprocessing operations with parameter configuration.
    
    This dialog integrates with the app's parameter system to provide a consistent
    interface for configuring preprocessing parameters. It automatically loads
    current parameters from the app and allows users to modify them for the
    current preprocessing operation.
    
    The dialog includes:
    - Parameter fields initialized with current app parameters
    - Reset to defaults functionality
    - Load from config file functionality
    - Automatic parameter saving when preprocessing starts
    """

    def __init__(self, app):
        self.app = app
        self.is_processing = False

    def show_dialog(self, selected_items: List):
        """Show preprocessing dialog with parameters."""
        # Create dialog
        dialog = gui.Dialog("Preprocessing Parameters")

        # Create dialog content
        em = self.app.window.theme.font_size
        content = gui.Vert(0, gui.Margins(em, em, em, em))

        # Title
        title = gui.Label(f"Preprocess {len(selected_items)} selected models")
        title.text_color = gui.Color(0.2, 0.6, 1.0)
        content.add_child(title)
        content.add_fixed(int(round(0.5 * em)))

        # Get current parameters from app
        current_params = self.app.get_all_parameters()

        # Parameters section
        params_section = gui.CollapsableVert("Parameters", 0.25 * em)

        # Voxel downsampling
        voxel_layout = gui.Horiz(0.25 * em)
        voxel_layout.add_child(gui.Label("Voxel Size:"))
        voxel_size_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        voxel_size_edit.set_value(current_params.get("voxel_downsample_size", 7.0))
        voxel_size_edit.set_limits(0.1, 50.0)
        voxel_layout.add_child(voxel_size_edit)
        params_section.add_child(voxel_layout)

        # Normal estimation radius
        normal_radius_layout = gui.Horiz(0.25 * em)
        normal_radius_layout.add_child(gui.Label("Normal Radius:"))
        normal_radius_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        normal_radius_edit.set_value(current_params.get("normal_estimation_radius", 14.0))
        normal_radius_edit.set_limits(1.0, 100.0)
        normal_radius_layout.add_child(normal_radius_edit)
        params_section.add_child(normal_radius_layout)

        # Normal estimation max neighbors
        normal_nn_layout = gui.Horiz(0.25 * em)
        normal_nn_layout.add_child(gui.Label("Max Neighbors:"))
        normal_max_nn_edit = gui.NumberEdit(gui.NumberEdit.Type.INT)
        normal_max_nn_edit.set_value(current_params.get("normal_estimation_max_nn", 30))
        normal_max_nn_edit.set_limits(5, 200)
        normal_nn_layout.add_child(normal_max_nn_edit)
        params_section.add_child(normal_nn_layout)

        # Sample points
        sample_layout = gui.Horiz(0.25 * em)
        sample_layout.add_child(gui.Label("Sample Points:"))
        sample_points_edit = gui.NumberEdit(gui.NumberEdit.Type.INT)
        sample_points_edit.set_value(current_params.get("fracture_surface_dense_sample_points", 10000))
        sample_points_edit.set_limits(1000, 100000)
        sample_layout.add_child(sample_points_edit)
        params_section.add_child(sample_layout)

        # Add noise checkbox
        add_noise_checkbox = gui.Checkbox("Add Noise")
        add_noise_checkbox.checked = current_params.get("add_preprocessing_noise", True)
        params_section.add_child(add_noise_checkbox)

        # Noise factor
        noise_layout = gui.Horiz(0.25 * em)
        noise_layout.add_child(gui.Label("Noise Factor:"))
        noise_factor_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        noise_factor_edit.set_value(current_params.get("preprocessing_noise_factor", 0.01))
        noise_factor_edit.set_limits(0.001, 0.1)
        noise_layout.add_child(noise_factor_edit)
        params_section.add_child(noise_layout)

        # Orient normals k parameter
        orient_normals_layout = gui.Horiz(0.25 * em)
        orient_normals_layout.add_child(gui.Label("Orient Normals K:"))
        orient_normals_k_edit = gui.NumberEdit(gui.NumberEdit.Type.INT)
        orient_normals_k_edit.set_value(current_params.get("orient_normals_k", 15))
        orient_normals_k_edit.set_limits(5, 50)
        orient_normals_layout.add_child(orient_normals_k_edit)
        params_section.add_child(orient_normals_layout)

        # FPFH feature radius (for feature extraction)
        fpfh_radius_layout = gui.Horiz(0.25 * em)
        fpfh_radius_layout.add_child(gui.Label("FPFH Radius:"))
        fpfh_radius_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        fpfh_radius_edit.set_value(current_params.get("fpfh_feature_radius", 35.0))
        fpfh_radius_edit.set_limits(5.0, 100.0)
        fpfh_radius_layout.add_child(fpfh_radius_edit)
        params_section.add_child(fpfh_radius_layout)

        # FPFH max neighbors
        fpfh_nn_layout = gui.Horiz(0.25 * em)
        fpfh_nn_layout.add_child(gui.Label("FPFH Max Neighbors:"))
        fpfh_max_nn_edit = gui.NumberEdit(gui.NumberEdit.Type.INT)
        fpfh_max_nn_edit.set_value(current_params.get("fpfh_feature_max_nn", 100))
        fpfh_max_nn_edit.set_limits(20, 500)
        fpfh_nn_layout.add_child(fpfh_max_nn_edit)
        params_section.add_child(fpfh_nn_layout)

        content.add_child(params_section)
        content.add_fixed(int(round(0.5 * em)))

        # Progress section (initially hidden)
        progress_section = gui.Vert(0)
        progress_label = gui.Label("Ready to start preprocessing...")
        progress_bar = gui.ProgressBar()
        progress_bar.visible = False
        progress_section.add_child(progress_label)
        progress_section.add_child(progress_bar)
        content.add_child(progress_section)

        # Buttons
        button_layout = gui.Horiz(0.25 * em)

        ok_button = gui.Button("Start Preprocessing")
        cancel_button = gui.Button("Cancel")
        reset_button = gui.Button("Reset to Defaults")
        load_config_button = gui.Button("Load from Config")

        # Store references for the button handlers
        dialog_content = {
            'selected_items': selected_items,
            'voxel_size_edit': voxel_size_edit,
            'normal_radius_edit': normal_radius_edit,
            'normal_max_nn_edit': normal_max_nn_edit,
            'sample_points_edit': sample_points_edit,
            'add_noise_checkbox': add_noise_checkbox,
            'noise_factor_edit': noise_factor_edit,
            'orient_normals_k_edit': orient_normals_k_edit,
            'fpfh_radius_edit': fpfh_radius_edit,
            'fpfh_max_nn_edit': fpfh_max_nn_edit,
            'progress_label': progress_label,
            'progress_bar': progress_bar,
            'ok_button': ok_button,
            'cancel_button': cancel_button,
            'dialog': dialog
        }

        def on_ok_clicked():
            """Handle OK button click - start preprocessing"""
            # Get parameters from dialog
            parameters = {
                "voxel_downsample_size": voxel_size_edit.double_value,
                "normal_estimation_radius": normal_radius_edit.double_value,
                "normal_estimation_max_nn": normal_max_nn_edit.int_value,
                "fracture_surface_dense_sample_points": sample_points_edit.int_value,
                "add_preprocessing_noise": add_noise_checkbox.checked,
                "preprocessing_noise_factor": noise_factor_edit.double_value,
                "orient_normals_k": orient_normals_k_edit.int_value,
                "fpfh_feature_radius": fpfh_radius_edit.double_value,
                "fpfh_feature_max_nn": fpfh_max_nn_edit.int_value,
            }

            # Update app's parameters with the new values
            for key, value in parameters.items():
                self.app.set_parameter(key, value)

            # Optionally save the updated parameters to the config file
            try:
                self.app.save_parameters()
                print(f"[DEBUG] Updated parameters saved to {self.app.config_file}")
            except Exception as e:
                print(f"[WARNING] Could not save updated parameters: {e}")

            # Disable OK button and show progress
            ok_button.enabled = False
            cancel_button.text = "Stop"
            progress_bar.visible = True
            progress_label.text = "Starting preprocessing..."

            # Start preprocessing in background thread
            self._start_preprocessing_in_dialog(dialog_content, parameters)

        def on_cancel_clicked():
            """Handle Cancel/Stop button click"""
            if cancel_button.text == "Stop":
                # Stop preprocessing
                self.is_processing = False
                progress_label.text = "Stopping preprocessing..."
            else:
                # Close dialog
                self.app.window.close_dialog()

        def on_reset_clicked():
            """Handle Reset to Defaults button click"""
            # Reset all fields to default values
            voxel_size_edit.set_value(7.0)
            normal_radius_edit.set_value(14.0)
            normal_max_nn_edit.set_value(30)
            sample_points_edit.set_value(10000)
            add_noise_checkbox.checked = True
            noise_factor_edit.set_value(0.01)
            orient_normals_k_edit.set_value(15)
            fpfh_radius_edit.set_value(35.0)
            fpfh_max_nn_edit.set_value(100)

        def on_load_config_clicked():
            """Handle Load from Config button click"""
            # Reload parameters from config file and update UI
            self.app.reload_parameters()
            current_params = self.app.get_all_parameters()

            voxel_size_edit.set_value(current_params.get("voxel_downsample_size", 7.0))
            normal_radius_edit.set_value(current_params.get("normal_estimation_radius", 14.0))
            normal_max_nn_edit.set_value(current_params.get("normal_estimation_max_nn", 30))
            sample_points_edit.set_value(current_params.get("fracture_surface_dense_sample_points", 10000))
            add_noise_checkbox.checked = current_params.get("add_preprocessing_noise", True)
            noise_factor_edit.set_value(current_params.get("preprocessing_noise_factor", 0.01))
            orient_normals_k_edit.set_value(current_params.get("orient_normals_k", 15))
            fpfh_radius_edit.set_value(current_params.get("fpfh_feature_radius", 35.0))
            fpfh_max_nn_edit.set_value(current_params.get("fpfh_feature_max_nn", 100))

        ok_button.set_on_clicked(on_ok_clicked)
        cancel_button.set_on_clicked(on_cancel_clicked)
        reset_button.set_on_clicked(on_reset_clicked)
        load_config_button.set_on_clicked(on_load_config_clicked)

        # Create button rows for better layout
        button_row1 = gui.Horiz(0.25 * em)
        button_row1.add_child(ok_button)
        button_row1.add_child(cancel_button)

        button_row2 = gui.Horiz(0.25 * em)
        button_row2.add_child(reset_button)
        button_row2.add_child(load_config_button)

        button_layout.add_child(button_row1)
        button_layout.add_child(button_row2)
        content.add_child(button_layout)

        # Set dialog content and show
        dialog.add_child(content)
        self.app.window.show_dialog(dialog)

    def _start_preprocessing_in_dialog(self, dialog_content: Dict, parameters: Dict[str, Any]):
        """Start preprocessing in a separate thread for the dialog."""
        self.is_processing = True

        # Create preprocessing engine with progress callback
        preprocessing_engine = PreprocessingEngine(
            progress_callback=lambda step, progress, message: self._on_dialog_progress_update(
                dialog_content, step, progress, message
            )
        )

        # Start preprocessing in a separate thread
        thread = threading.Thread(
            target=self._run_preprocessing_in_dialog,
            args=(dialog_content, parameters, preprocessing_engine)
        )
        thread.daemon = True
        thread.start()

    def _run_preprocessing_in_dialog(self, dialog_content: Dict, parameters: Dict[str, Any], preprocessing_engine):
        """Run preprocessing in a separate thread for the dialog."""
        try:
            # Run preprocessing
            results = preprocessing_engine.preprocess_base_models(dialog_content['selected_items'], parameters)

            # Add results to the item tree (must be done in main thread)
            gui.Application.instance.post_to_main_thread(
                self.app.window, lambda: self._add_preprocessing_results_from_dialog(dialog_content, results)
            )

        except Exception as e:
            # Handle errors in main thread
            gui.Application.instance.post_to_main_thread(
                self.app.window, lambda: self._handle_preprocessing_error_in_dialog(dialog_content, str(e))
            )

    def _on_dialog_progress_update(self, dialog_content: Dict, step: str, progress: float, message: str):
        """Handle progress updates from preprocessing engine in dialog."""
        gui.Application.instance.post_to_main_thread(
            self.app.window, lambda: self._update_dialog_progress(dialog_content, progress, message)
        )

    def _update_dialog_progress(self, dialog_content: Dict, progress: float, message: str):
        """Update progress bar and status in dialog."""
        dialog_content['progress_bar'].value = progress / 100.0
        dialog_content['progress_label'].text = message

    def _add_preprocessing_results_from_dialog(self, dialog_content: Dict, results: List[Dict]):
        """Add preprocessing results to the item tree from dialog."""
        successful_results = 0
        preprocessed_items = []  # Collect all successful items
        batch_parameters = results[0]['parameters_used'] if results else {}  # Use first result's parameters as batch parameters

        for result in results:
            if result['success']:
                original_item = result['item']
                label = f"Preprocessed: {original_item.label}"

                # Create unique scene path to avoid conflicts
                base_name = f"preprocessed_{original_item.label}"

                # Find a unique name by adding a counter if needed
                counter = 0
                scene_path = base_name
                while True:
                    try:
                        # Check if geometry exists by trying to get its visibility
                        self.app._scene_widget.scene.get_geometry(scene_path)
                        # If we get here, geometry exists, try next number
                        counter += 1
                        scene_path = f"{base_name}_{counter}"
                    except:
                        # Geometry doesn't exist, we can use this name
                        break

                print(f"[DEBUG] Adding preprocessed geometry to scene: {scene_path}")

                # Add to scene (point cloud or mesh)
                if result['preprocessed_mesh']:
                    self.app._scene_widget.scene.add_geometry(
                        scene_path, 
                        result['preprocessed_mesh'],
                        self.app.settings.material if hasattr(self.app.settings, 'material') else None
                    )
                    # Show the preprocessed result by default
                    self.app._scene_widget.scene.show_geometry(scene_path, True)
                    print(f"[DEBUG] Preprocessed geometry should be visible: {scene_path}")

                # Create PreprocessedItem (but don't add to tree yet)
                preprocessed_item = PreprocessedItem(
                    label=label,
                    preprocessed_mesh=result['preprocessed_mesh'],
                    original_mesh=original_item.mesh,
                    original_mesh_path=original_item.mesh_path,
                    preprocessing_parameters=result['parameters_used'],
                    preprocessing_steps=result['preprocessing_steps'],
                    quality_metrics=result['quality_metrics'],
                    scene_path=scene_path
                )
                preprocessed_item.processing_time = result['processing_time']
                quality_score = self._calculate_overall_quality(result['quality_metrics'])
                preprocessed_item.mesh_quality_score = quality_score
                successful_results += 1
                preprocessed_items.append(preprocessed_item)

        # Create batch result item if we have successful results
        if preprocessed_items:
            timestamp = int(time.time())
            batch_label = f"Preprocessing Batch {timestamp}"

            # Create the batch container
            batch_item = self.app._left_panel.item_tree.add_preprocessing_result_item(
                label=batch_label,
                preprocessed_items=preprocessed_items,
                batch_parameters=batch_parameters
            )

            print(f"[DEBUG] Created preprocessing batch: {batch_label} with {len(preprocessed_items)} items")

        # Update dialog with completion
        dialog_content['progress_label'].text = f"Completed: {successful_results}/{len(results)} models preprocessed successfully"
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

    def _handle_preprocessing_error_in_dialog(self, dialog_content: Dict, error_message: str):
        """Handle preprocessing errors in dialog."""
        dialog_content['progress_label'].text = f"Error: {error_message}"
        dialog_content['ok_button'].enabled = True
        dialog_content['cancel_button'].text = "Close"
        self.is_processing = False

    def _calculate_overall_quality(self, quality_metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        if not quality_metrics:
            return 0.5

        # Simple weighted average of available metrics
        weights = {
            'reduction_ratio': 0.3,
            'normal_consistency': 0.4,
            'surface_area': 0.3
        }

        total_score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in quality_metrics:
                # Normalize each metric to 0-1 range
                if metric == 'reduction_ratio':
                    # Prefer reduction ratios around 0.1-0.3
                    ratio = quality_metrics[metric]
                    normalized = max(0, 1 - abs(ratio - 0.2) / 0.2)
                elif metric == 'normal_consistency':
                    # Higher is better
                    normalized = min(1.0, quality_metrics[metric])
                elif metric == 'surface_area':
                    # Higher is better, but normalize
                    normalized = min(1.0, quality_metrics[metric] / 1000.0)
                else:
                    normalized = min(1.0, max(0.0, quality_metrics[metric]))

                total_score += normalized * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def get_preprocessing_parameters(self) -> Dict[str, Any]:
        """
        Get preprocessing-specific parameters from the app.
        
        Returns:
            dict: Dictionary containing preprocessing parameters
        """
        return self.app.get_parameters_subset([
            "voxel_downsample_size",
            "normal_estimation_radius", 
            "normal_estimation_max_nn",
            "fracture_surface_dense_sample_points",
            "add_preprocessing_noise",
            "preprocessing_noise_factor",
            "orient_normals_k",
            "fpfh_feature_radius",
            "fpfh_feature_max_nn"
        ])
