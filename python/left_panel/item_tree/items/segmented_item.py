import uuid
from typing import List, Dict, Any, Optional
import numpy as np
import open3d as o3d
from .base_item import BaseItem


class SegmentedItem(BaseItem):
    """
    Represents an individual segment from segmentation.
    Each segment has its own color, faces, and properties.
    """
    
    def __init__(self, label: str, segment_faces: np.ndarray = None, segment_mesh: o3d.geometry.TriangleMesh = None,
                 original_mesh=None, preprocessed_item=None, segment_properties: Dict[str, Any] = None,
                 segment_color: List[float] = None, is_visible: bool = True):
        """
        Initialize a segmented item.
        
        Args:
            label (str): Display label for the segment (e.g., "Segment 1")
            segment_faces (np.ndarray): Face indices for this segment
            segment_mesh (o3d.geometry.TriangleMesh): Mesh for this segment
            original_mesh: Original mesh that was segmented
            preprocessed_item: Reference to the preprocessed item this segment came from
            segment_properties (dict): Properties of this segment (area, normal, etc.)
            segment_color (list): RGB color for this segment [r, g, b]
            is_visible (bool): Whether the segment is visible by default
        """
        super().__init__(label, is_visible)
        self._segment_faces = segment_faces
        self._segment_mesh = segment_mesh
        self._original_mesh = original_mesh
        self._preprocessed_item = preprocessed_item
        self._segment_properties = segment_properties or {}
        self._segment_color = segment_color or [1.0, 0.0, 0.0]  # Default red
        self._scene_path = ""
    
    @property
    def segment_faces(self) -> np.ndarray:
        """Get the face indices for this segment."""
        return self._segment_faces
    
    @segment_faces.setter
    def segment_faces(self, value: np.ndarray):
        """Set the face indices for this segment."""
        self._segment_faces = value
    
    @property
    def segment_mesh(self) -> o3d.geometry.TriangleMesh:
        """Get the mesh for this segment."""
        return self._segment_mesh
    
    @segment_mesh.setter
    def segment_mesh(self, value: o3d.geometry.TriangleMesh):
        """Set the mesh for this segment."""
        self._segment_mesh = value
    
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
        """Get the preprocessed item this segment came from."""
        return self._preprocessed_item
    
    @preprocessed_item.setter
    def preprocessed_item(self, value):
        """Set the preprocessed item this segment came from."""
        self._preprocessed_item = value
    
    @property
    def segment_properties(self) -> Dict[str, Any]:
        """Get the properties of this segment."""
        return self._segment_properties
    
    @segment_properties.setter
    def segment_properties(self, value: Dict[str, Any]):
        """Set the properties of this segment."""
        self._segment_properties = value
    
    @property
    def segment_color(self) -> List[float]:
        """Get the color for this segment."""
        return self._segment_color
    
    @segment_color.setter
    def segment_color(self, value: List[float]):
        """Set the color for this segment."""
        self._segment_color = value
    
    @property
    def scene_path(self) -> str:
        """Get the scene path for this segment."""
        return self._scene_path
    
    @scene_path.setter
    def scene_path(self, value: str):
        """Set the scene path for this segment."""
        self._scene_path = value
    
    def get_face_count(self) -> int:
        """Get the number of faces in this segment."""
        if self._segment_faces is not None:
            return len(self._segment_faces)
        elif self._segment_mesh is not None:
            return len(self._segment_mesh.triangles)
        return 0
    
    def get_vertex_count(self) -> int:
        """Get the number of vertices in this segment."""
        if self._segment_mesh is not None:
            return len(self._segment_mesh.vertices)
        return 0
    
    def get_area(self) -> float:
        """Get the area of this segment."""
        return self._segment_properties.get('area', 0.0)
    
    def get_area_fraction(self) -> float:
        """Get the area fraction of this segment relative to original mesh."""
        return self._segment_properties.get('area_fraction', 0.0)
    
    def get_average_normal(self) -> np.ndarray:
        """Get the average normal of this segment."""
        return self._segment_properties.get('avg_normal', np.array([0, 0, 1]))
    
    def get_bumpiness(self) -> float:
        """Get the bumpiness of this segment."""
        return self._segment_properties.get('bumpiness', 0.0)
    
    def is_fracture_candidate(self) -> bool:
        """Check if this segment is a fracture candidate."""
        return self._segment_properties.get('is_fracture_candidate', False)
    
    def get_item_type(self) -> str:
        """Get the type of this item."""
        return "segmented"
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'segment_faces_count': self.get_face_count(),
            'segment_vertex_count': self.get_vertex_count(),
            'segment_area': self.get_area(),
            'segment_area_fraction': self.get_area_fraction(),
            'segment_color': self._segment_color,
            'scene_path': self._scene_path,
            'segment_properties': self._segment_properties,
            'has_segment_mesh': self._segment_mesh is not None,
            'has_original_mesh': self._original_mesh is not None,
            'has_preprocessed_item': self._preprocessed_item is not None,
            'is_fracture_candidate': self.is_fracture_candidate()
        })
        return base_dict
    
    def __str__(self) -> str:
        return f"SegmentedItem(id={self._id}, label='{self._label}', faces={self.get_face_count()}, area={self.get_area():.3f}, color={self._segment_color}, visible={self._is_visible})" 