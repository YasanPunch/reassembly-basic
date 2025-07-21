import open3d.visualization.gui as gui  # type: ignore
import os


class LeftPanel:
    def __init__(self, app):
        self.app = app
        w = app.window  # to make the code more concise

        # ---- Left panel ----
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Main left panel layout - using a vertical layout to stack the two sections
        self._left_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        # DB Tree Section (Upper half)
        self._db_tree_section = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )
        
        # Add a label for the DB Tree section
        self._db_tree_label = gui.Label("DB Tree")
        self._db_tree_label.text_color = gui.Color(0.2, 0.6, 1.0)  # Blue color for header
        self._db_tree_section.add_child(self._db_tree_label)
        self._db_tree_section.add_fixed(separation_height)

        # Container for model collapsible sections
        self._model_container = gui.Vert(0.15 * em)
        self._db_tree_section.add_child(self._model_container)

        # Placeholder label for when no objects are loaded
        self._placeholder_label = gui.Label("No objects loaded")
        self._model_container.add_child(self._placeholder_label)

        # Store loaded objects and their visibility states
        self._loaded_objects = []  # List of dictionaries: {path, name, visible, scene_index, collapsible, checkbox, processing_results}
        self._selected_object_index = -1
        
        # Add the DB Tree section to the main panel
        self._left_panel.add_child(self._db_tree_section)
        self._left_panel.add_fixed(separation_height)

        # Properties Section (Lower half)
        self._properties_section = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )
        
        # Add a label for the Properties section
        self._properties_label = gui.Label("Properties")
        self._properties_label.text_color = gui.Color(0.2, 0.8, 0.2)  # Green color for header
        self._properties_section.add_child(self._properties_label)
        self._properties_section.add_fixed(separation_height)
        
        # Add a placeholder for properties display
        self._properties_text = gui.TextEdit()
        self._properties_text.placeholder_text = "Select an object to view properties"
        self._properties_text.enabled = False
        self._properties_section.add_child(self._properties_text)
        
        # Add the Properties section to the main panel
        self._left_panel.add_child(self._properties_section)

    def add_object(self, file_path, scene_index):
        w = self.app.window
        em = w.theme.font_size
        import os
        # Hide placeholder when adding the first object
        self._placeholder_label.visible = False
        # Extract filename from path
        file_name = os.path.basename(file_path)
        # Create collapsible section for this object
        object_section = gui.CollapsableVert(file_name, 0.25 * em, gui.Margins(0.25 * em, 0, 0, 0))
        # Create checkbox for visibility
        cb = gui.Checkbox("Show/Hide")
        cb.checked = True
        def handle_click(checked, idx=len(self._loaded_objects)):
            self.toggle_object_visibility(idx)
        cb.set_on_checked(handle_click)
        object_section.add_child(cb)
        # Create object entry
        object_entry = {
            'path': file_path,
            'name': file_name,
            'visible': True,
            'scene_index': scene_index,
            'collapsible': object_section,
            'checkbox': cb,
            'processing_results': {
                'preprocessing': None,
                'segmentation': None,
                'pairwise_matching': None,
                'multipiece_matching': None
            }
        }
        # Add to container and store
        self._model_container.add_child(object_section)
        self._loaded_objects.append(object_entry)
        self._update_properties_display(object_entry)

    def clear_objects(self):
        # Show placeholder if all objects are removed
        self._placeholder_label.visible = True
        self._loaded_objects.clear()
        # Optionally, remove all children except the placeholder
        for child in list(self._model_container.get_children()):
            if child is not self._placeholder_label:
                self._model_container.remove_child(child)

    def add_processing_result(self, object_index, process_type, result_mesh):
        w = self.app.window
        em = w.theme.font_size
        """Add a processing result to an object's collapsible section"""
        if 0 <= object_index < len(self._loaded_objects):
            obj = self._loaded_objects[object_index]
            
            # Create checkbox for the processing result
            cb = gui.Checkbox(f"{process_type} result")
            cb.checked = False
            
            # Store the result in memory
            result_entry = {
                'mesh': result_mesh,
                'visible': False,
                'checkbox': cb
            }
            
            # Add to processing results
            obj['processing_results'][process_type] = result_entry
            
            # Add checkbox to the collapsible section
            def handle_result_click(checked):
                self.toggle_result_visibility(object_index, process_type)
            cb.set_on_checked(handle_result_click)
            obj['collapsible'].add_child(cb)
            
            # Update properties display
            self._update_properties_display(obj)

    def toggle_result_visibility(self, object_index, process_type):
        """Toggle visibility of a processing result"""
        if 0 <= object_index < len(self._loaded_objects):
            obj = self._loaded_objects[object_index]
            result = obj['processing_results'].get(process_type)
            if result:
                result['visible'] = not result['visible']
                result['checkbox'].checked = result['visible']
                # TODO: Update scene visibility when we implement result visualization
                self.app.window.set_needs_layout()

    def toggle_object_visibility(self, object_index):
        """Toggle visibility of an object"""
        if 0 <= object_index < len(self._loaded_objects):
            obj = self._loaded_objects[object_index]
            obj['visible'] = not obj['visible']
            
            # Update checkbox state
            if obj['checkbox']:
                obj['checkbox'].checked = obj['visible']
            
            # Show/hide geometry in the main scene
            self.app._scene_widget.scene.show_geometry(obj['path'], obj['visible'])

            # Update the camera to re-frame all visible objects
            self.app._update_camera_bounds()
            
            # Update properties if this object is selected
            if self._selected_object_index == object_index:
                self._update_properties_display(obj)

    def _update_properties_display(self, obj=None):
        """Update the properties display with object and processing info"""
        if obj is None:
            obj = next((o for o in self._loaded_objects if o['visible']), None)
        if obj is None:
            self._properties_text.text_value = "No object selected."
            self._properties_text.enabled = False
            return

        # Build tree-like structure for properties
        properties_text = f"{obj['name']}\n"
        properties_text += f"  └─ File size: {os.path.getsize(obj['path']) / (1024 * 1024):.2f} MB\n"
        
        # Add processing results info
        for process_type, result in obj['processing_results'].items():
            if result:
                properties_text += f"  └─ {process_type}: {'Visible' if result['visible'] else 'Hidden'}\n"

        self._properties_text.text_value = properties_text
        self._properties_text.enabled = True

    def get_visible_objects(self):
        return [obj for obj in self._loaded_objects if obj['visible']]

    def get_all_objects(self):
        return self._loaded_objects.copy()

    def get_selected_objects(self):
        """Get all objects that are currently visible"""
        return [obj for obj in self._loaded_objects if obj['visible']] 