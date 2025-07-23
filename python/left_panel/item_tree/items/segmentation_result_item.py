import uuid
from typing import List, Dict, Any, Optional
import numpy as np
import open3d as o3d
from .base_item import BaseItem
from .segmented_item import SegmentedItem


class SegmentationResultItem(BaseItem):
    """
    A container item that represents all segments from one preprocessed item.
    Contains multiple SegmentedItem instances.
    """
    
    def __init__(self, label: str, segmented_items: List[SegmentedItem] = None, 
                 original_mesh=None, preprocessed_item=None, segmentation_parameters: Dict[str, Any] = None,
                 is_visible: bool = True):
        """
        Initialize a segmentation result item.
        
        Args:
            label (str): Display label for the segmentation result (e.g., "Segmented: brick_part01.obj")
            segmented_items (list): List of SegmentedItem instances
            original_mesh: Original mesh that was segmented
            preprocessed_item: Reference to the preprocessed item this segmentation came from
            segmentation_parameters (dict): Parameters used for segmentation
            is_visible (bool): Whether the segmentation result is visible by default
        """
        super().__init__(label, is_visible)
        self._segmented_items = segmented_items or []
        self._original_mesh = original_mesh
        self._preprocessed_item = preprocessed_item
        self._segmentation_parameters = segmentation_parameters or {}
        self._total_processing_time = 0.0
        self._segment_count = 0
        self._fracture_candidate_count = 0
        self._total_area = 0.0
        self._segmentation_container = None  # Reference to the UI container
    
    @property
    def segmented_items(self) -> List[SegmentedItem]:
        """Get the list of segmented items."""
        return self._segmented_items
    
    @segmented_items.setter
    def segmented_items(self, value: List[SegmentedItem]):
        """Set the list of segmented items."""
        self._segmented_items = value
        self._update_segmentation_metrics()
    
    @property
    def original_mesh(self):
        """Get the original mesh that was segmented."""
        return self._original_mesh
    
    @original_mesh.setter
    def original_mesh(self, value):
        """Set the original mesh that was segmented."""
        self._original_mesh = value
    
    @property
    def preprocessed_item(self):
        """Get the preprocessed item this segmentation came from."""
        return self._preprocessed_item
    
    @preprocessed_item.setter
    def preprocessed_item(self, value):
        """Set the preprocessed item this segmentation came from."""
        self._preprocessed_item = value
    
    @property
    def segmentation_parameters(self) -> Dict[str, Any]:
        """Get the parameters used for segmentation."""
        return self._segmentation_parameters
    
    @segmentation_parameters.setter
    def segmentation_parameters(self, value: Dict[str, Any]):
        """Set the parameters used for segmentation."""
        self._segmentation_parameters = value
    
    @property
    def total_processing_time(self) -> float:
        """Get the total processing time for segmentation."""
        return self._total_processing_time
    
    @total_processing_time.setter
    def total_processing_time(self, value: float):
        """Set the total processing time for segmentation."""
        self._total_processing_time = max(0.0, value)
    
    @property
    def segment_count(self) -> int:
        """Get the number of segments."""
        return self._segment_count
    
    @property
    def fracture_candidate_count(self) -> int:
        """Get the number of fracture candidate segments."""
        return self._fracture_candidate_count
    
    @property
    def total_area(self) -> float:
        """Get the total area of all segments."""
        return self._total_area
    
    @property
    def segmentation_container(self):
        """Get the UI container for this segmentation result."""
        return self._segmentation_container
    
    @segmentation_container.setter
    def segmentation_container(self, value):
        """Set the UI container for this segmentation result."""
        self._segmentation_container = value
    
    def add_segmented_item(self, item: SegmentedItem):
        """Add a segmented item to this result."""
        self._segmented_items.append(item)
        self._update_segmentation_metrics()
    
    def remove_segmented_item(self, item_id: str) -> bool:
        """Remove a segmented item from this result by ID."""
        for i, item in enumerate(self._segmented_items):
            if item.id == item_id:
                self._segmented_items.pop(i)
                self._update_segmentation_metrics()
                return True
        return False
    
    def get_segmented_item(self, item_id: str) -> Optional[SegmentedItem]:
        """Get a segmented item by ID."""
        for item in self._segmented_items:
            if item.id == item_id:
                return item
        return None
    
    def get_visible_items(self) -> List[SegmentedItem]:
        """Get all visible segmented items."""
        return [item for item in self._segmented_items if item.is_visible]
    
    def get_visible_count(self) -> int:
        """Get the number of visible items."""
        return len(self.get_visible_items())
    
    def get_fracture_candidates(self) -> List[SegmentedItem]:
        """Get all fracture candidate segments."""
        return [item for item in self._segmented_items if item.is_fracture_candidate()]
    
    def _update_segmentation_metrics(self):
        """Update segmentation-level metrics based on contained items."""
        self._segment_count = len(self._segmented_items)
        self._fracture_candidate_count = len(self.get_fracture_candidates())
        self._total_area = sum(item.get_area() for item in self._segmented_items)
    
    def get_item_type(self) -> str:
        """Get the type of this item."""
        return "segmentation_result"
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'segmentation_parameters': self._segmentation_parameters,
            'total_processing_time': self._total_processing_time,
            'segment_count': self._segment_count,
            'fracture_candidate_count': self._fracture_candidate_count,
            'total_area': self._total_area,
            'visible_count': self.get_visible_count(),
            'has_original_mesh': self._original_mesh is not None,
            'has_preprocessed_item': self._preprocessed_item is not None,
            'has_segmentation_container': self._segmentation_container is not None,
            'segmented_items': [item.to_dict() for item in self._segmented_items]
        })
        return base_dict
    
    def __str__(self) -> str:
        visible_count = self.get_visible_count()
        return f"SegmentationResultItem(id={self._id}, label='{self._label}', segments={self._segment_count}/{self._fracture_candidate_count} fracture candidates, visible={visible_count}, area={self._total_area:.3f}, time={self._total_processing_time:.2f}s)" 