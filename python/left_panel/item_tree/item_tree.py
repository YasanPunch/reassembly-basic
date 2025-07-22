import open3d.visualization.gui as gui  # type: ignore
import os
from typing import List, Dict, Any, Optional
from .items import (
    BaseItem, BaseModelItem, SegmentationResultItem, 
    ClassificationResultItem, PairwiseResultItem, AssemblyResultItem
)


class ItemTree:
    """
    ItemTree manages the item tree UI and logic for the left panel.
    Uses a structured item system with different types of items.
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

        # Store all items by type
        self._items: Dict[str, List[BaseItem]] = {
            'base_model': [],
            'segmentation': [],
            'pairwise': [],
            'assembly': [],
            'classification': []
        }
        
        # Store UI widgets for each item
        self._item_widgets: Dict[str, gui.Widget] = {}

    def add_base_model_item(self, mesh_path: str, mesh=None, label: str = None, is_visible: bool = True) -> BaseModelItem:
        """Add a base model item to the tree."""
        if label is None:
            label = os.path.basename(mesh_path)
        
        item = BaseModelItem(label=label, mesh_path=mesh_path, mesh=mesh, is_visible=is_visible)
        
        # Create UI widget
        cb = gui.Checkbox(label)
        cb.checked = is_visible
        
        # Connect checkbox to show/hide the object in the 3D scene
        def on_checked(checked, item_id=item.id):
            print(f"[DEBUG] Checkbox for {item_id} set to {checked}")
            item.is_visible = checked
            self.app._scene_widget.scene.show_geometry(mesh_path, checked)
        
        cb.set_on_checked(on_checked)
        item.set_ui_widget(cb)
        
        # Add to UI and storage
        self.base_model_dropdown.add_child(cb)
        self._items['base_model'].append(item)
        self._item_widgets[item.id] = cb
        
        # Ensure the object is shown by default
        self.app._scene_widget.scene.show_geometry(mesh_path, is_visible)
        
        return item

    def add_segmentation_item(self, label: str, segmented_mesh=None, original_mesh_path: str = "",
                             segmentation_parameters: Dict[str, Any] = None, is_visible: bool = True) -> SegmentationResultItem:
        """Add a segmentation result item to the tree."""
        item = SegmentationResultItem(
            label=label, segmented_mesh=segmented_mesh, 
            original_mesh_path=original_mesh_path,
            segmentation_parameters=segmentation_parameters, 
            is_visible=is_visible
        )
        
        # Create UI widget
        cb = gui.Checkbox(label)
        cb.checked = is_visible
        
        def on_checked(checked, item_id=item.id):
            item.is_visible = checked
            # TODO: Show/hide segmented mesh in scene
        
        cb.set_on_checked(on_checked)
        item.set_ui_widget(cb)
        
        # Add to UI and storage
        self.segmentation_results_dropdown.add_child(cb)
        self._items['segmentation'].append(item)
        self._item_widgets[item.id] = cb
        
        return item

    def add_classification_item(self, label: str, classified_mesh=None, original_mesh_path: str = "",
                               classification_results: Dict[str, Any] = None, 
                               confidence_scores: Dict[str, float] = None,
                               is_visible: bool = True) -> ClassificationResultItem:
        """Add a classification result item to the tree."""
        item = ClassificationResultItem(
            label=label, classified_mesh=classified_mesh,
            original_mesh_path=original_mesh_path,
            classification_results=classification_results,
            confidence_scores=confidence_scores,
            is_visible=is_visible
        )
        
        # Create UI widget
        cb = gui.Checkbox(label)
        cb.checked = is_visible
        
        def on_checked(checked, item_id=item.id):
            item.is_visible = checked
            # TODO: Show/hide classified mesh in scene
        
        cb.set_on_checked(on_checked)
        item.set_ui_widget(cb)
        
        # Add to UI and storage
        self.classification_results_dropdown.add_child(cb)
        self._items['classification'].append(item)
        self._item_widgets[item.id] = cb
        
        return item

    def add_pairwise_item(self, label: str, fragment1_path: str = "", fragment2_path: str = "",
                         matched_mesh=None, transformation_matrix=None, matching_score: float = 0.0,
                         matching_parameters: Dict[str, Any] = None, is_visible: bool = True) -> PairwiseResultItem:
        """Add a pairwise result item to the tree."""
        item = PairwiseResultItem(
            label=label, fragment1_path=fragment1_path, fragment2_path=fragment2_path,
            matched_mesh=matched_mesh, transformation_matrix=transformation_matrix,
            matching_score=matching_score, matching_parameters=matching_parameters,
            is_visible=is_visible
        )
        
        # Create UI widget
        cb = gui.Checkbox(label)
        cb.checked = is_visible
        
        def on_checked(checked, item_id=item.id):
            item.is_visible = checked
            # TODO: Show/hide matched mesh in scene
        
        cb.set_on_checked(on_checked)
        item.set_ui_widget(cb)
        
        # Add to UI and storage
        self.pairwise_results_dropdown.add_child(cb)
        self._items['pairwise'].append(item)
        self._item_widgets[item.id] = cb
        
        return item

    def add_assembly_item(self, label: str, assembled_mesh=None, fragment_paths: List[str] = None,
                         assembly_parameters: Dict[str, Any] = None, assembly_score: float = 0.0,
                         fragment_count: int = 0, is_visible: bool = True) -> AssemblyResultItem:
        """Add an assembly result item to the tree."""
        item = AssemblyResultItem(
            label=label, assembled_mesh=assembled_mesh, fragment_paths=fragment_paths,
            assembly_parameters=assembly_parameters, assembly_score=assembly_score,
            fragment_count=fragment_count, is_visible=is_visible
        )
        
        # Create UI widget
        cb = gui.Checkbox(label)
        cb.checked = is_visible
        
        def on_checked(checked, item_id=item.id):
            item.is_visible = checked
            # TODO: Show/hide assembled mesh in scene
        
        cb.set_on_checked(on_checked)
        item.set_ui_widget(cb)
        
        # Add to UI and storage
        self.global_reassembly_results_dropdown.add_child(cb)
        self._items['assembly'].append(item)
        self._item_widgets[item.id] = cb
        
        return item

    def get_item_by_id(self, item_id: str) -> Optional[BaseItem]:
        """Get an item by its ID."""
        for item_list in self._items.values():
            for item in item_list:
                if item.id == item_id:
                    return item
        return None

    def get_items_by_type(self, item_type: str) -> List[BaseItem]:
        """Get all items of a specific type."""
        return self._items.get(item_type, [])

    def get_all_items(self) -> List[BaseItem]:
        """Get all items from all categories."""
        all_items = []
        for item_list in self._items.values():
            all_items.extend(item_list)
        return all_items

    def remove_item(self, item_id: str) -> bool:
        """Remove an item by its ID."""
        for item_type, item_list in self._items.items():
            for i, item in enumerate(item_list):
                if item.id == item_id:
                    # Remove from UI
                    if item_id in self._item_widgets:
                        widget = self._item_widgets[item_id]
                        # Find the parent dropdown and remove the widget
                        self._remove_widget_from_dropdown(widget, item_type)
                        del self._item_widgets[item_id]
                    
                    # Remove from storage
                    item_list.pop(i)
                    return True
        return False

    def _remove_widget_from_dropdown(self, widget: gui.Widget, item_type: str):
        """Remove a widget from its parent dropdown."""
        dropdown_map = {
            'base_model': self.base_model_dropdown,
            'segmentation': self.segmentation_results_dropdown,
            'pairwise': self.pairwise_results_dropdown,
            'assembly': self.global_reassembly_results_dropdown,
            'classification': self.classification_results_dropdown
        }
        
        dropdown = dropdown_map.get(item_type)
        if dropdown:
            # Note: Open3D GUI doesn't have a direct remove_child method
            # This is a limitation - we'd need to rebuild the dropdown
            print(f"[WARNING] Cannot remove widget from dropdown for {item_type}")

    # Legacy compatibility methods
    def add_object(self, path, name=None):
        """Legacy method for backward compatibility."""
        return self.add_base_model_item(mesh_path=path, label=name)

    def get_all_objects(self):
        """Legacy method for backward compatibility."""
        # Return a list of dicts with 'path' and 'visible' for each base model object
        result = []
        for item in self._items['base_model']:
            if isinstance(item, BaseModelItem):
                result.append({
                    'path': item.mesh_path, 
                    'visible': item.is_visible
                })
        return result 