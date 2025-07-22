from .base_item import BaseItem
from typing import List, Dict, Any, Optional


class ClassificationResultItem(BaseItem):
    """
    Represents a classification result item in the item tree.
    Contains classification data and metadata.
    """
    
    def __init__(self, label: str, classified_mesh=None, original_mesh_path: str = "",
                 classification_results: Dict[str, Any] = None, confidence_scores: Dict[str, float] = None,
                 is_visible: bool = True):
        """
        Initialize a classification result item.
        
        Args:
            label (str): Display label for the item
            classified_mesh: Open3D mesh or point cloud object (classified result)
            original_mesh_path (str): Path to the original mesh that was classified
            classification_results (dict): Classification results with labels and categories
            confidence_scores (dict): Confidence scores for each classification
            is_visible (bool): Whether the item is visible by default
        """
        super().__init__(label, is_visible)
        self._classified_mesh = classified_mesh
        self._original_mesh_path = original_mesh_path
        self._classification_results = classification_results or {}
        self._confidence_scores = confidence_scores or {}
        self._primary_class = ""
        self._primary_confidence = 0.0
        
    @property
    def classified_mesh(self):
        """Get the classified Open3D mesh or point cloud object."""
        return self._classified_mesh
    
    @classified_mesh.setter
    def classified_mesh(self, value):
        """Set the classified Open3D mesh or point cloud object."""
        self._classified_mesh = value
    
    @property
    def original_mesh_path(self) -> str:
        """Get the path to the original mesh that was classified."""
        return self._original_mesh_path
    
    @original_mesh_path.setter
    def original_mesh_path(self, value: str):
        """Set the path to the original mesh that was classified."""
        self._original_mesh_path = value
    
    @property
    def classification_results(self) -> Dict[str, Any]:
        """Get the classification results."""
        return self._classification_results.copy()
    
    @classification_results.setter
    def classification_results(self, value: Dict[str, Any]):
        """Set the classification results."""
        self._classification_results = value.copy() if value else {}
    
    @property
    def confidence_scores(self) -> Dict[str, float]:
        """Get the confidence scores for classifications."""
        return self._confidence_scores.copy()
    
    @confidence_scores.setter
    def confidence_scores(self, value: Dict[str, float]):
        """Set the confidence scores for classifications."""
        self._confidence_scores = value.copy() if value else {}
    
    @property
    def primary_class(self) -> str:
        """Get the primary classification class."""
        return self._primary_class
    
    @primary_class.setter
    def primary_class(self, value: str):
        """Set the primary classification class."""
        self._primary_class = value
    
    @property
    def primary_confidence(self) -> float:
        """Get the confidence score for the primary classification."""
        return self._primary_confidence
    
    @primary_confidence.setter
    def primary_confidence(self, value: float):
        """Set the confidence score for the primary classification."""
        self._primary_confidence = value
    
    def get_item_type(self) -> str:
        """Get the type of this item."""
        return "ClassificationResultItem"
    
    def add_classification(self, class_name: str, confidence: float, metadata: Dict[str, Any] = None):
        """Add a classification result."""
        self._classification_results[class_name] = metadata or {}
        self._confidence_scores[class_name] = confidence
        
        # Update primary class if this has higher confidence
        if confidence > self._primary_confidence:
            self._primary_class = class_name
            self._primary_confidence = confidence
    
    def get_top_classifications(self, top_k: int = 5) -> List[tuple]:
        """Get the top k classifications sorted by confidence."""
        sorted_classes = sorted(self._confidence_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_classes[:top_k]
    
    def get_classification_count(self) -> int:
        """Get the number of classification results."""
        return len(self._classification_results)
    
    def get_vertex_count(self) -> int:
        """Get the number of vertices in the classified mesh."""
        if self._classified_mesh is None:
            return 0
        if hasattr(self._classified_mesh, 'vertices'):
            return len(self._classified_mesh.vertices)
        elif hasattr(self._classified_mesh, 'points'):
            return len(self._classified_mesh.points)
        return 0
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'original_mesh_path': self._original_mesh_path,
            'classification_results': self._classification_results,
            'confidence_scores': self._confidence_scores,
            'primary_class': self._primary_class,
            'primary_confidence': self._primary_confidence,
            'classification_count': self.get_classification_count(),
            'vertex_count': self.get_vertex_count()
        })
        return base_dict
    
    def __str__(self) -> str:
        return f"ClassificationResultItem(id={self._id}, label='{self._label}', primary_class='{self._primary_class}', confidence={self._primary_confidence:.2f}, visible={self._is_visible})" 