import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from python.panels.preprocessing_dialog import PreprocessingDialog
from python.panels.segmentation_dialog import SegmentationDialog

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
        pass

    def _on_multipiece_matching(self):
        pass
