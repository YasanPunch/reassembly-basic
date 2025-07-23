import uuid
from typing import List, Dict, Any, Optional
from .base_item import BaseItem
from .preprocessed_item import PreprocessedItem


class PreprocessedResultItem(BaseItem):
    """
    A container item that represents a batch of preprocessing results.
    Contains multiple PreprocessedItem instances.
    """
    
    def __init__(self, label: str, preprocessed_items: List[PreprocessedItem] = None, 
                 batch_parameters: Dict[str, Any] = None, is_visible: bool = True):
        """
        Initialize a preprocessed result item.
        
        Args:
            label (str): Display label for the batch (e.g., "Preprocessing Batch 1")
            preprocessed_items (list): List of PreprocessedItem instances in this batch
            batch_parameters (dict): Common parameters used for this batch
            is_visible (bool): Whether the batch is visible by default
        """
        super().__init__(label, is_visible)
        self._preprocessed_items = preprocessed_items or []
        self._batch_parameters = batch_parameters or {}
        self._total_processing_time = 0.0
        self._average_quality_score = 0.0
        self._success_count = 0
        self._total_count = 0
        self._batch_container = None
    
    @property
    def preprocessed_items(self) -> List[PreprocessedItem]:
        """Get the list of preprocessed items in this batch."""
        return self._preprocessed_items
    
    @preprocessed_items.setter
    def preprocessed_items(self, value: List[PreprocessedItem]):
        """Set the list of preprocessed items in this batch."""
        self._preprocessed_items = value
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
    def average_quality_score(self) -> float:
        """Get the average quality score across all items in this batch."""
        return self._average_quality_score
    
    @average_quality_score.setter
    def average_quality_score(self, value: float):
        """Set the average quality score across all items in this batch."""
        self._average_quality_score = max(0.0, min(1.0, value))
    
    @property
    def success_count(self) -> int:
        """Get the number of successfully preprocessed items."""
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
    
    def add_preprocessed_item(self, item: PreprocessedItem):
        """Add a preprocessed item to this batch."""
        self._preprocessed_items.append(item)
        self._update_batch_metrics()
    
    def remove_preprocessed_item(self, item_id: str) -> bool:
        """Remove a preprocessed item from this batch by ID."""
        for i, item in enumerate(self._preprocessed_items):
            if item.id == item_id:
                self._preprocessed_items.pop(i)
                self._update_batch_metrics()
                return True
        return False
    
    def get_preprocessed_item(self, item_id: str) -> Optional[PreprocessedItem]:
        """Get a preprocessed item by ID."""
        for item in self._preprocessed_items:
            if item.id == item_id:
                return item
        return None
    
    def get_visible_items(self) -> List[PreprocessedItem]:
        """Get all visible preprocessed items in this batch."""
        return [item for item in self._preprocessed_items if item.is_visible]
    
    def get_visible_count(self) -> int:
        """Get the number of visible items in this batch."""
        return len(self.get_visible_items())
    
    def _update_batch_metrics(self):
        """Update batch-level metrics based on contained items."""
        if not self._preprocessed_items:
            self._total_processing_time = 0.0
            self._average_quality_score = 0.0
            self._success_count = 0
            self._total_count = 0
            return
        
        # Count successful items
        self._success_count = len([item for item in self._preprocessed_items if item.preprocessed_mesh is not None])
        self._total_count = len(self._preprocessed_items)
        
        # Calculate total processing time
        self._total_processing_time = sum(item.processing_time for item in self._preprocessed_items)
        
        # Calculate average quality score
        quality_scores = [item.mesh_quality_score for item in self._preprocessed_items if item.mesh_quality_score > 0]
        if quality_scores:
            self._average_quality_score = sum(quality_scores) / len(quality_scores)
        else:
            self._average_quality_score = 0.0
    
    def get_item_type(self) -> str:
        """Get the type of this item."""
        return "preprocessed_result"
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'batch_parameters': self._batch_parameters,
            'total_processing_time': self._total_processing_time,
            'average_quality_score': self._average_quality_score,
            'success_count': self._success_count,
            'total_count': self._total_count,
            'visible_count': self.get_visible_count(),
            'has_batch_container': self._batch_container is not None,
            'preprocessed_items': [item.to_dict() for item in self._preprocessed_items]
        })
        return base_dict
    
    def __str__(self) -> str:
        visible_count = self.get_visible_count()
        return f"PreprocessedResultItem(id={self._id}, label='{self._label}', items={self._total_count}/{self._success_count} successful, visible={visible_count}, avg_quality={self._average_quality_score:.3f}, time={self._total_processing_time:.2f}s)" 