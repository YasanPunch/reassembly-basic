import open3d.visualization.gui as gui
import os
from typing import List, Dict, Any, Optional
from .items import (
    BaseItem,
    BaseModelItem,
    SegmentationResultItem,
    ClassificationResultItem,
    PairwiseResultItem,
    AssemblyResultItem,
    PreprocessedItem,
    PreprocessedResultItem,
    SegmentationBatchItem,
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
        self.preprocessing_results_dropdown = gui.CollapsableVert("Preprocessing Results", 0.25 * em)
        self.segmentation_results_dropdown = gui.CollapsableVert("Segmentation Results", 0.25 * em)
        self.segmentation_batches_dropdown = gui.CollapsableVert("Segmentation Batches", 0.25 * em)
        self.pairwise_results_dropdown = gui.CollapsableVert("Pairwise Results", 0.25 * em)
        self.global_reassembly_results_dropdown = gui.CollapsableVert("Global Reassembly Results", 0.25 * em)
        self.classification_results_dropdown = gui.CollapsableVert("Classification Results", 0.25 * em)

        self.section.add_child(self.base_model_dropdown)
        self.section.add_child(self.preprocessing_results_dropdown)
        self.section.add_child(self.segmentation_results_dropdown)
        self.section.add_child(self.segmentation_batches_dropdown)
        self.section.add_child(self.pairwise_results_dropdown)
        self.section.add_child(self.global_reassembly_results_dropdown)
        self.section.add_child(self.classification_results_dropdown)

        # Store all items by type
        self._items: Dict[str, List[BaseItem]] = {
            'base_model': [],
            'preprocessing': [],
            'preprocessing_results': [],  # Container for PreprocessedResultItem
            'segmentation': [],
            'segmentation_results': [],  # Container for SegmentationResultItem
            'segmentation_batches': [],  # Container for SegmentationBatchItem
            'pairwise': [],
            'assembly': [],
            'classification': []
        }

        # Store UI widgets for each item
        self._item_widgets: Dict[str, gui.Widget] = {}

    def add_base_model_item(
        self,
        mesh_path: str,
        mesh=None,
        label: str = None,
        name: str = None,
        original_index: int = None,
        is_visible: bool = True,
    ) -> BaseModelItem:
        """Add a base model item to the tree."""
        if label is None:
            label = os.path.basename(mesh_path)

        item = BaseModelItem(
            label=label,
            mesh_path=mesh_path,
            mesh=mesh,
            name=name,
            original_index=original_index,
            is_visible=is_visible,
        )

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

        # Trigger layout update to refresh the UI
        self.app.window.set_needs_layout()

        return item

    def add_preprocessing_item(self, label: str, preprocessed_mesh=None, original_mesh=None, original_mesh_path: str = "",
                             preprocessing_parameters: Dict[str, Any] = None, preprocessing_steps: List[str] = None,
                             quality_metrics: Dict[str, float] = None, scene_path: str = "", is_visible: bool = True) -> PreprocessedItem:
        """Add a preprocessing result item to the tree."""
        item = PreprocessedItem(
            label=label, preprocessed_mesh=preprocessed_mesh,
            original_mesh=original_mesh,
            original_mesh_path=original_mesh_path,
            preprocessing_parameters=preprocessing_parameters,
            preprocessing_steps=preprocessing_steps,
            quality_metrics=quality_metrics,
            scene_path=scene_path,
            is_visible=is_visible
        )

        # Create UI widget
        cb = gui.Checkbox(label)
        cb.checked = is_visible

        def on_checked(checked, item_id=item.id):
            item.is_visible = checked
            # Show/hide preprocessed mesh in scene using stored scene path
            if item.preprocessed_mesh and item.scene_path:
                self.app._scene_widget.scene.show_geometry(item.scene_path, checked)

        cb.set_on_checked(on_checked)
        item.set_ui_widget(cb)

        # Add to UI and storage
        self.preprocessing_results_dropdown.add_child(cb)
        self._items['preprocessing'].append(item)
        self._item_widgets[item.id] = cb

        # Trigger layout update to refresh the UI
        self.app.window.set_needs_layout()

        return item

    def add_preprocessing_result_item(self, label: str, preprocessed_items: List[PreprocessedItem] = None,
                                    batch_parameters: Dict[str, Any] = None, is_visible: bool = True) -> PreprocessedResultItem:
        """Add a preprocessing result item (batch container) to the tree."""
        item = PreprocessedResultItem(
            label=label,
            preprocessed_items=preprocessed_items or [],
            batch_parameters=batch_parameters or {},
            is_visible=is_visible
        )

        # Create a collapsible container for the batch
        batch_container = gui.CollapsableVert(label, 0.25 * self.app.window.theme.font_size)
        batch_container.set_is_open(True)  # Start expanded

        # Create UI widget for the batch (checkbox inside the container)
        cb = gui.Checkbox(label)
        cb.checked = is_visible

        def on_checked(checked, item_id=item.id):
            item.is_visible = checked
            # Show/hide all preprocessed items in this batch
            for preprocessed_item in item.preprocessed_items:
                if preprocessed_item.preprocessed_mesh and preprocessed_item.scene_path:
                    self.app._scene_widget.scene.show_geometry(preprocessed_item.scene_path, checked)

        cb.set_on_checked(on_checked)
        item.set_ui_widget(cb)

        # Add batch checkbox to the container
        batch_container.add_child(cb)

        # Add child items (individual preprocessed items) inside the container
        for preprocessed_item in item.preprocessed_items:
            child_cb = gui.Checkbox(preprocessed_item.label)  # No need for manual indentation
            child_cb.checked = preprocessed_item.is_visible

            def on_child_checked(checked, child_item=preprocessed_item):
                child_item.is_visible = checked
                # Show/hide individual preprocessed mesh in scene
                if child_item.preprocessed_mesh and child_item.scene_path:
                    self.app._scene_widget.scene.show_geometry(child_item.scene_path, checked)

            child_cb.set_on_checked(on_child_checked)
            preprocessed_item.set_ui_widget(child_cb)

            # Add child to the batch container (nested)
            batch_container.add_child(child_cb)
            self._item_widgets[preprocessed_item.id] = child_cb

        # Add the batch container to the main dropdown
        self.preprocessing_results_dropdown.add_child(batch_container)
        self._items['preprocessing_results'].append(item)
        self._item_widgets[item.id] = batch_container

        # Store reference to the batch container in the item
        item.batch_container = batch_container

        # Trigger layout update
        self.app.window.set_needs_layout()

        return item

    def add_segmentation_batch_item(self, label: str, segmentation_results: List[SegmentationResultItem] = None,
                                  batch_parameters: Dict[str, Any] = None, is_visible: bool = True) -> SegmentationBatchItem:
        """Add a segmentation batch item (batch container) to the tree."""
        item = SegmentationBatchItem(
            label=label,
            segmentation_results=segmentation_results or [],
            batch_parameters=batch_parameters or {},
            is_visible=is_visible
        )

        # Create a collapsible container for the batch
        batch_container = gui.CollapsableVert(label, 0.25 * self.app.window.theme.font_size)
        batch_container.set_is_open(True)  # Start expanded

        # Create UI widget for the batch (checkbox inside the container)
        cb = gui.Checkbox(label)
        cb.checked = is_visible

        def on_checked(checked, item_id=item.id):
            item.is_visible = checked
            # Show/hide all segmentation results in this batch
            for segmentation_result in item.segmentation_results:
                for segmented_item in segmentation_result.segmented_items:
                    if segmented_item.segment_mesh and segmented_item.scene_path:
                        self.app._scene_widget.scene.show_geometry(segmented_item.scene_path, checked)

        cb.set_on_checked(on_checked)
        item.set_ui_widget(cb)

        # Add batch checkbox to the container
        batch_container.add_child(cb)

        # Add child items (individual segmentation results) inside the container
        for segmentation_result in item.segmentation_results:
            # Create container for this segmentation result
            result_container = gui.CollapsableVert(segmentation_result.label, 0.25 * self.app.window.theme.font_size)
            result_container.set_is_open(True)

            # Create checkbox for the segmentation result
            result_cb = gui.Checkbox(segmentation_result.label)
            result_cb.checked = segmentation_result.is_visible

            def on_result_checked(checked, result_item=segmentation_result):
                result_item.is_visible = checked
                # Show/hide all segments in this result
                for segmented_item in result_item.segmented_items:
                    if segmented_item.segment_mesh and segmented_item.scene_path:
                        self.app._scene_widget.scene.show_geometry(segmented_item.scene_path, checked)

            result_cb.set_on_checked(on_result_checked)
            segmentation_result.set_ui_widget(result_cb)

            # Add result checkbox to result container
            result_container.add_child(result_cb)

            # Add individual segments
            for segmented_item in segmentation_result.segmented_items:
                segment_cb = gui.Checkbox(segmented_item.label)
                segment_cb.checked = segmented_item.is_visible

                def on_segment_checked(checked, segment_item=segmented_item):
                    segment_item.is_visible = checked
                    # Show/hide individual segment in scene
                    if segment_item.segment_mesh and segment_item.scene_path:
                        self.app._scene_widget.scene.show_geometry(segment_item.scene_path, checked)

                segment_cb.set_on_checked(on_segment_checked)
                segmented_item.set_ui_widget(segment_cb)

                # Add segment to result container
                result_container.add_child(segment_cb)
                self._item_widgets[segmented_item.id] = segment_cb

            # Add result container to batch container
            batch_container.add_child(result_container)
            self._item_widgets[segmentation_result.id] = result_container

            # Store reference to the result container in the item
            segmentation_result.segmentation_container = result_container

        # Add the batch container to the main dropdown
        self.segmentation_batches_dropdown.add_child(batch_container)
        self._items['segmentation_batches'].append(item)
        self._item_widgets[item.id] = batch_container

        # Store reference to the batch container in the item
        item.batch_container = batch_container

        # Trigger layout update
        self.app.window.set_needs_layout()

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

        # Trigger layout update to refresh the UI
        self.app.window.set_needs_layout()

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

        # Trigger layout update to refresh the UI
        self.app.window.set_needs_layout()

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

        # Trigger layout update to refresh the UI
        self.app.window.set_needs_layout()

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

        # Trigger layout update to refresh the UI
        self.app.window.set_needs_layout()

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
            'preprocessing': self.preprocessing_results_dropdown,
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

    def refresh_layout(self):
        """Trigger a layout update for the item tree section."""
        self.app.window.set_needs_layout()

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

    def get_all_visible_base_model_items(self):
        """
        Returns a list of dicts for all visible base model items, each with:
            - 'mesh': the mesh object
            - 'name': the filename
            - 'original_index': the original index in the fragment list
            - 'path': the file path
        """
        result = []
        for item in self._items["base_model"]:
            if isinstance(item, BaseModelItem) and item.is_visible:
                result.append(
                    {
                        "mesh": item.mesh,
                        "name": getattr(item, "_name", None),
                        "original_index": getattr(item, "_original_index", None),
                        "path": item.mesh_path,
                    }
                )
        return result
