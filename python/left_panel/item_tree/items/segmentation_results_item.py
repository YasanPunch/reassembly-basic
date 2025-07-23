from .base_item import BaseItem
import open3d as o3d
from typing import List, Dict, Any, Optional


class SegmentationResultItem(BaseItem):
    """
    Represents a segmentation result item in the item tree.
    Contains segmented mesh data and segmentation metadata.
    """
    
    def __init__(self, label: str, segmented_mesh=None, original_mesh_path: str = "", 
                 segmentation_parameters: Dict[str, Any] = None, is_visible: bool = True):
        """
        Initialize a segmentation result item.
        
        Args:
            label (str): Display label for the item
            segmented_mesh: Open3D mesh or point cloud object (segmented result)
            original_mesh_path (str): Path to the original mesh that was segmented
            segmentation_parameters (dict): Parameters used for segmentation
            is_visible (bool): Whether the item is visible by default
        """
        super().__init__(label, is_visible)
        self._segmented_mesh = segmented_mesh
        self._original_mesh_path = original_mesh_path
        self._segmentation_parameters = segmentation_parameters or {}
        self._segment_count = 0
        self._segment_labels = []
        
    @property
    def segmented_mesh(self):
        """Get the segmented Open3D mesh or point cloud object."""
        return self._segmented_mesh
    
    @segmented_mesh.setter
    def segmented_mesh(self, value):
        """Set the segmented Open3D mesh or point cloud object."""
        self._segmented_mesh = value
    
    @property
    def original_mesh_path(self) -> str:
        """Get the path to the original mesh that was segmented."""
        return self._original_mesh_path
    
    @original_mesh_path.setter
    def original_mesh_path(self, value: str):
        """Set the path to the original mesh that was segmented."""
        self._original_mesh_path = value
    
    @property
    def segmentation_parameters(self) -> Dict[str, Any]:
        """Get the parameters used for segmentation."""
        return self._segmentation_parameters.copy()
    
    @segmentation_parameters.setter
    def segmentation_parameters(self, value: Dict[str, Any]):
        """Set the parameters used for segmentation."""
        self._segmentation_parameters = value.copy() if value else {}
    
    @property
    def segment_count(self) -> int:
        """Get the number of segments in the result."""
        return self._segment_count
    
    @segment_count.setter
    def segment_count(self, value: int):
        """Set the number of segments in the result."""
        self._segment_count = value
    
    @property
    def segment_labels(self) -> List[str]:
        """Get the labels for each segment."""
        return self._segment_labels.copy()
    
    @segment_labels.setter
    def segment_labels(self, value: List[str]):
        """Set the labels for each segment."""
        self._segment_labels = value.copy() if value else []
    
    def get_item_type(self) -> str:
        """Get the type of this item."""
        return "SegmentationResultItem"
    
    def add_segment_label(self, label: str):
        """Add a label for a segment."""
        self._segment_labels.append(label)
        self._segment_count = len(self._segment_labels)
    
    def set_segment_labels(self, labels: List[str]):
        """Set all segment labels at once."""
        self._segment_labels = labels.copy() if labels else []
        self._segment_count = len(self._segment_labels)
    
    def get_segment_by_label(self, label: str):
        """Get a specific segment by its label. Override in subclasses if needed."""
        # This is a placeholder - actual implementation would depend on how segments are stored
        return None
    
    def get_vertex_count(self) -> int:
        """Get the number of vertices in the segmented mesh."""
        if self._segmented_mesh is None:
            return 0
        if hasattr(self._segmented_mesh, 'vertices'):
            return len(self._segmented_mesh.vertices)
        elif hasattr(self._segmented_mesh, 'points'):
            return len(self._segmented_mesh.points)
        return 0
    
    def get_triangle_count(self) -> int:
        """Get the number of triangles in the segmented mesh (0 for point clouds)."""
        if self._segmented_mesh is None or not hasattr(self._segmented_mesh, 'triangles'):
            return 0
        return len(self._segmented_mesh.triangles)
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'original_mesh_path': self._original_mesh_path,
            'segmentation_parameters': self._segmentation_parameters,
            'segment_count': self._segment_count,
            'segment_labels': self._segment_labels,
            'vertex_count': self.get_vertex_count(),
            'triangle_count': self.get_triangle_count()
        })
        return base_dict
    
    def __str__(self) -> str:
        return f"SegmentationResultItem(id={self._id}, label='{self._label}', segments={self._segment_count}, visible={self._is_visible})" 