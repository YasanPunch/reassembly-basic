from .base_item import BaseItem
from typing import List, Dict, Any, Optional
import numpy as np


class PreprocessedItem(BaseItem):
    """
    Represents a preprocessed item in the item tree.
    Contains preprocessed mesh data and preprocessing metadata.
    """
    
    def __init__(self, label: str, preprocessed_mesh=None, original_mesh_path: str = "",
                 preprocessing_parameters: Dict[str, Any] = None, preprocessing_steps: List[str] = None,
                 quality_metrics: Dict[str, float] = None, is_visible: bool = True):
        """
        Initialize a preprocessed item.
        
        Args:
            label (str): Display label for the item
            preprocessed_mesh: Open3D mesh or point cloud object (preprocessed result)
            original_mesh_path (str): Path to the original mesh that was preprocessed
            preprocessing_parameters (dict): Parameters used for preprocessing
            preprocessing_steps (list): List of preprocessing steps applied
            quality_metrics (dict): Quality metrics after preprocessing
            is_visible (bool): Whether the item is visible by default
        """
        super().__init__(label, is_visible)
        self._preprocessed_mesh = preprocessed_mesh
        self._original_mesh_path = original_mesh_path
        self._preprocessing_parameters = preprocessing_parameters or {}
        self._preprocessing_steps = preprocessing_steps or []
        self._quality_metrics = quality_metrics or {}
        self._processing_time = 0.0
        self._mesh_quality_score = 0.0
        
    @property
    def preprocessed_mesh(self):
        """Get the preprocessed Open3D mesh or point cloud object."""
        return self._preprocessed_mesh
    
    @preprocessed_mesh.setter
    def preprocessed_mesh(self, value):
        """Set the preprocessed Open3D mesh or point cloud object."""
        self._preprocessed_mesh = value
    
    @property
    def original_mesh_path(self) -> str:
        """Get the path to the original mesh that was preprocessed."""
        return self._original_mesh_path
    
    @original_mesh_path.setter
    def original_mesh_path(self, value: str):
        """Set the path to the original mesh that was preprocessed."""
        self._original_mesh_path = value
    
    @property
    def preprocessing_parameters(self) -> Dict[str, Any]:
        """Get the parameters used for preprocessing."""
        return self._preprocessing_parameters.copy()
    
    @preprocessing_parameters.setter
    def preprocessing_parameters(self, value: Dict[str, Any]):
        """Set the parameters used for preprocessing."""
        self._preprocessing_parameters = value.copy() if value else {}
    
    @property
    def preprocessing_steps(self) -> List[str]:
        """Get the list of preprocessing steps applied."""
        return self._preprocessing_steps.copy()
    
    @preprocessing_steps.setter
    def preprocessing_steps(self, value: List[str]):
        """Set the list of preprocessing steps applied."""
        self._preprocessing_steps = value.copy() if value else []
    
    @property
    def quality_metrics(self) -> Dict[str, float]:
        """Get the quality metrics after preprocessing."""
        return self._quality_metrics.copy()
    
    @quality_metrics.setter
    def quality_metrics(self, value: Dict[str, float]):
        """Set the quality metrics after preprocessing."""
        self._quality_metrics = value.copy() if value else {}
    
    @property
    def processing_time(self) -> float:
        """Get the processing time in seconds."""
        return self._processing_time
    
    @processing_time.setter
    def processing_time(self, value: float):
        """Set the processing time in seconds."""
        self._processing_time = max(0.0, value)
    
    @property
    def mesh_quality_score(self) -> float:
        """Get the overall mesh quality score."""
        return self._mesh_quality_score
    
    @mesh_quality_score.setter
    def mesh_quality_score(self, value: float):
        """Set the overall mesh quality score."""
        self._mesh_quality_score = max(0.0, min(1.0, value))
    
    def get_item_type(self) -> str:
        """Get the type of this item."""
        return "PreprocessedItem"
    
    def add_preprocessing_step(self, step: str):
        """Add a preprocessing step to the list."""
        if step not in self._preprocessing_steps:
            self._preprocessing_steps.append(step)
    
    def set_preprocessing_steps(self, steps: List[str]):
        """Set all preprocessing steps at once."""
        self._preprocessing_steps = steps.copy() if steps else []
    
    def add_quality_metric(self, metric_name: str, value: float):
        """Add a quality metric."""
        self._quality_metrics[metric_name] = value
    
    def get_quality_metric(self, metric_name: str) -> Optional[float]:
        """Get a specific quality metric."""
        return self._quality_metrics.get(metric_name)
    
    def get_step_count(self) -> int:
        """Get the number of preprocessing steps applied."""
        return len(self._preprocessing_steps)
    
    def has_step(self, step_name: str) -> bool:
        """Check if a specific preprocessing step was applied."""
        return step_name in self._preprocessing_steps
    
    def get_vertex_count(self) -> int:
        """Get the number of vertices in the preprocessed mesh."""
        if self._preprocessed_mesh is None:
            return 0
        if hasattr(self._preprocessed_mesh, 'vertices'):
            return len(self._preprocessed_mesh.vertices)
        elif hasattr(self._preprocessed_mesh, 'points'):
            return len(self._preprocessed_mesh.points)
        return 0
    
    def get_triangle_count(self) -> int:
        """Get the number of triangles in the preprocessed mesh (0 for point clouds)."""
        if self._preprocessed_mesh is None or not hasattr(self._preprocessed_mesh, 'triangles'):
            return 0
        return len(self._preprocessed_mesh.triangles)
    
    def get_face_count(self) -> int:
        """Get the number of faces in the preprocessed mesh."""
        return self.get_triangle_count()
    
    def is_mesh(self) -> bool:
        """Check if this item contains a mesh (vs point cloud)."""
        return self._preprocessed_mesh is not None and hasattr(self._preprocessed_mesh, 'vertices')
    
    def is_point_cloud(self) -> bool:
        """Check if this item contains a point cloud (vs mesh)."""
        return self._preprocessed_mesh is not None and hasattr(self._preprocessed_mesh, 'points')
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of the preprocessing operation."""
        return {
            'steps_applied': self._preprocessing_steps,
            'step_count': self.get_step_count(),
            'processing_time': self._processing_time,
            'quality_score': self._mesh_quality_score,
            'vertex_count': self.get_vertex_count(),
            'triangle_count': self.get_triangle_count(),
            'quality_metrics': self._quality_metrics
        }
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'original_mesh_path': self._original_mesh_path,
            'preprocessing_parameters': self._preprocessing_parameters,
            'preprocessing_steps': self._preprocessing_steps,
            'quality_metrics': self._quality_metrics,
            'processing_time': self._processing_time,
            'mesh_quality_score': self._mesh_quality_score,
            'step_count': self.get_step_count(),
            'vertex_count': self.get_vertex_count(),
            'triangle_count': self.get_triangle_count(),
            'is_mesh': self.is_mesh(),
            'is_point_cloud': self.is_point_cloud()
        })
        return base_dict
    
    def __str__(self) -> str:
        return f"PreprocessedItem(id={self._id}, label='{self._label}', steps={self.get_step_count()}, quality={self._mesh_quality_score:.3f}, visible={self._is_visible})" 