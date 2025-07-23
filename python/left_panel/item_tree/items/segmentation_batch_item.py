import uuid
from typing import List, Dict, Any, Optional
import numpy as np
import open3d as o3d
from .base_item import BaseItem
from .segmentation_result_item import SegmentationResultItem


class SegmentationBatchItem(BaseItem):
    """
    A container item that represents a batch of segmentation results.
    Contains multiple SegmentationResultItem instances.
    """
    
    def __init__(self, label: str, segmentation_results: List[SegmentationResultItem] = None, 
                 batch_parameters: Dict[str, Any] = None, is_visible: bool = True):
        """
        Initialize a segmentation batch item.
        
        Args:
            label (str): Display label for the batch (e.g., "Segmentation Batch 1")
            segmentation_results (list): List of SegmentationResultItem instances in this batch
            batch_parameters (dict): Common parameters used for this batch
            is_visible (bool): Whether the batch is visible by default
        """
        super().__init__(label, is_visible)
        self._segmentation_results = segmentation_results or []
        self._batch_parameters = batch_parameters or {}
        self._total_processing_time = 0.0
        self._total_segments = 0
        self._total_fracture_candidates = 0
        self._success_count = 0
        self._total_count = 0
        self._batch_container = None  # Reference to the UI container
    
    @property
    def segmentation_results(self) -> List[SegmentationResultItem]:
        """Get the list of segmentation results in this batch."""
        return self._segmentation_results
    
    @segmentation_results.setter
    def segmentation_results(self, value: List[SegmentationResultItem]):
        """Set the list of segmentation results in this batch."""
        self._segmentation_results = value
        self._update_batch_metrics()
    
    @property
    def batch_parameters(self) -> Dict[str, Any]:
        """Get the common parameters used for this batch."""
        return self._batch_parameters
    
    @batch_parameters.setter
    def batch_parameters(self, value: Dict[str, Any]):
        """Set the common parameters used for this batch."""
        self._batch_parameters = value
    
    @property
    def total_processing_time(self) -> float:
        """Get the total processing time for all items in this batch."""
        return self._total_processing_time
    
    @total_processing_time.setter
    def total_processing_time(self, value: float):
        """Set the total processing time for all items in this batch."""
        self._total_processing_time = max(0.0, value)
    
    @property
    def total_segments(self) -> int:
        """Get the total number of segments across all results."""
        return self._total_segments
    
    @property
    def total_fracture_candidates(self) -> int:
        """Get the total number of fracture candidates across all results."""
        return self._total_fracture_candidates
    
    @property
    def success_count(self) -> int:
        """Get the number of successfully segmented items."""
        return self._success_count
    
    @property
    def total_count(self) -> int:
        """Get the total number of items in this batch."""
        return self._total_count
    
    @property
    def batch_container(self):
        """Get the UI container for this batch."""
        return self._batch_container
    
    @batch_container.setter
    def batch_container(self, value):
        """Set the UI container for this batch."""
        self._batch_container = value
    
    def add_segmentation_result(self, item: SegmentationResultItem):
        """Add a segmentation result to this batch."""
        self._segmentation_results.append(item)
        self._update_batch_metrics()
    
    def remove_segmentation_result(self, item_id: str) -> bool:
        """Remove a segmentation result from this batch by ID."""
        for i, item in enumerate(self._segmentation_results):
            if item.id == item_id:
                self._segmentation_results.pop(i)
                self._update_batch_metrics()
                return True
        return False
    
    def get_segmentation_result(self, item_id: str) -> Optional[SegmentationResultItem]:
        """Get a segmentation result by ID."""
        for item in self._segmentation_results:
            if item.id == item_id:
                return item
        return None
    
    def get_visible_results(self) -> List[SegmentationResultItem]:
        """Get all visible segmentation results in this batch."""
        return [item for item in self._segmentation_results if item.is_visible]
    
    def get_visible_count(self) -> int:
        """Get the number of visible results in this batch."""
        return len(self.get_visible_results())
    
    def get_all_fracture_candidates(self) -> List:
        """Get all fracture candidate segments from all results."""
        all_candidates = []
        for result in self._segmentation_results:
            all_candidates.extend(result.get_fracture_candidates())
        return all_candidates
    
    def _update_batch_metrics(self):
        """Update batch-level metrics based on contained items."""
        if not self._segmentation_results:
            self._total_processing_time = 0.0
            self._total_segments = 0
            self._total_fracture_candidates = 0
            self._success_count = 0
            self._total_count = 0
            return
        
        # Count successful items
        self._success_count = len([item for item in self._segmentation_results if item.segment_count > 0])
        self._total_count = len(self._segmentation_results)
        
        # Calculate total processing time
        self._total_processing_time = sum(item.total_processing_time for item in self._segmentation_results)
        
        # Calculate total segments and fracture candidates
        self._total_segments = sum(item.segment_count for item in self._segmentation_results)
        self._total_fracture_candidates = sum(item.fracture_candidate_count for item in self._segmentation_results)
    
    def get_item_type(self) -> str:
        """Get the type of this item."""
        return "segmentation_batch"
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'batch_parameters': self._batch_parameters,
            'total_processing_time': self._total_processing_time,
            'total_segments': self._total_segments,
            'total_fracture_candidates': self._total_fracture_candidates,
            'success_count': self._success_count,
            'total_count': self._total_count,
            'visible_count': self.get_visible_count(),
            'has_batch_container': self._batch_container is not None,
            'segmentation_results': [item.to_dict() for item in self._segmentation_results]
        })
        return base_dict
    
    def __str__(self) -> str:
        visible_count = self.get_visible_count()
        return f"SegmentationBatchItem(id={self._id}, label='{self._label}', results={self._total_count}/{self._success_count} successful, segments={self._total_segments}/{self._total_fracture_candidates} fracture candidates, visible={visible_count}, time={self._total_processing_time:.2f}s)" 