import open3d.visualization.gui as gui  # type: ignore
import os

class ItemTree:
    """
    ItemTree manages the item tree (formerly db tree) UI and logic for the left panel.
    This is a basic implementation with dropdowns for each major result type.
    """
    def __init__(self, app):
        self.app = app
        em = app.window.theme.font_size
        self.section = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.label = gui.Label("Item Tree")
        self.label.text_color = gui.Color(0.2, 0.6, 1.0)
        self.section.add_child(self.label)
        self.section.add_fixed(int(round(0.5 * em)))

        # Dropdowns for each item type
        self.base_model_dropdown = gui.CollapsableVert("Base Model", 0.25 * em)
        self.segmentation_results_dropdown = gui.CollapsableVert("Segmentation Results", 0.25 * em)
        self.pairwise_results_dropdown = gui.CollapsableVert("Pairwise Results", 0.25 * em)
        self.global_reassembly_results_dropdown = gui.CollapsableVert("Global Reassembly Results", 0.25 * em)
        self.classification_results_dropdown = gui.CollapsableVert("Classification Results", 0.25 * em)

        self.section.add_child(self.base_model_dropdown)
        self.section.add_child(self.segmentation_results_dropdown)
        self.section.add_child(self.pairwise_results_dropdown)
        self.section.add_child(self.global_reassembly_results_dropdown)
        self.section.add_child(self.classification_results_dropdown)

        # Track base model objects for get_all_objects
        self.base_model_objects = []

    def add_object(self, path, name=None):
        if name is None:
            name = os.path.basename(path)
        cb = gui.Checkbox(name)
        cb.checked = True
        # Connect checkbox to show/hide the object in the 3D scene
        def on_checked(checked, obj_path=path):
            print(f"[DEBUG] Checkbox for {obj_path} set to {checked}")
            self.app._scene_widget.scene.show_geometry(obj_path, checked)
        cb.set_on_checked(on_checked)
        self.base_model_dropdown.add_child(cb)
        self.base_model_objects.append({'path': path, 'name': name, 'checkbox': cb})
        # Ensure the object is shown by default
        self.app._scene_widget.scene.show_geometry(path, True)

    def get_all_objects(self):
        # Return a list of dicts with 'path' and 'visible' for each base model object
        return [
            {'path': obj['path'], 'visible': obj['checkbox'].checked}
            for obj in self.base_model_objects
        ] 