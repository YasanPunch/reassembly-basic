from .base_item import BaseItem
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class PairwiseResultItem(BaseItem):
    """
    Represents a pairwise matching result item in the item tree.
    Contains pairwise matching data between two fragments.
    """
    
    def __init__(self, label: str, fragment1_path: str = "", fragment2_path: str = "",
                 matched_mesh=None, transformation_matrix: np.ndarray = None,
                 matching_score: float = 0.0, matching_parameters: Dict[str, Any] = None,
                 is_visible: bool = True):
        """
        Initialize a pairwise result item.
        
        Args:
            label (str): Display label for the item
            fragment1_path (str): Path to the first fragment
            fragment2_path (str): Path to the second fragment
            matched_mesh: Open3D mesh or point cloud object (matched result)
            transformation_matrix (np.ndarray): 4x4 transformation matrix
            matching_score (float): Confidence score for the matching
            matching_parameters (dict): Parameters used for matching
            is_visible (bool): Whether the item is visible by default
        """
        super().__init__(label, is_visible)
        self._fragment1_path = fragment1_path
        self._fragment2_path = fragment2_path
        self._matched_mesh = matched_mesh
        self._transformation_matrix = transformation_matrix if transformation_matrix is not None else np.eye(4)
        self._matching_score = matching_score
        self._matching_parameters = matching_parameters or {}
        self._correspondence_points = []
        self._fitness_score = 0.0
        self._rmse_score = 0.0
        
    @property
    def fragment1_path(self) -> str:
        """Get the path to the first fragment."""
        return self._fragment1_path
    
    @fragment1_path.setter
    def fragment1_path(self, value: str):
        """Set the path to the first fragment."""
        self._fragment1_path = value
    
    @property
    def fragment2_path(self) -> str:
        """Get the path to the second fragment."""
        return self._fragment2_path
    
    @fragment2_path.setter
    def fragment2_path(self, value: str):
        """Set the path to the second fragment."""
        self._fragment2_path = value
    
    @property
    def matched_mesh(self):
        """Get the matched Open3D mesh or point cloud object."""
        return self._matched_mesh
    
    @matched_mesh.setter
    def matched_mesh(self, value):
        """Set the matched Open3D mesh or point cloud object."""
        self._matched_mesh = value
    
    @property
    def transformation_matrix(self) -> np.ndarray:
        """Get the 4x4 transformation matrix."""
        return self._transformation_matrix.copy()
    
    @transformation_matrix.setter
    def transformation_matrix(self, value: np.ndarray):
        """Set the 4x4 transformation matrix."""
        self._transformation_matrix = value.copy() if value is not None else np.eye(4)
    
    @property
    def matching_score(self) -> float:
        """Get the matching confidence score."""
        return self._matching_score
    
    @matching_score.setter
    def matching_score(self, value: float):
        """Set the matching confidence score."""
        self._matching_score = value
    
    @property
    def matching_parameters(self) -> Dict[str, Any]:
        """Get the parameters used for matching."""
        return self._matching_parameters.copy()
    
    @matching_parameters.setter
    def matching_parameters(self, value: Dict[str, Any]):
        """Set the parameters used for matching."""
        self._matching_parameters = value.copy() if value else {}
    
    @property
    def correspondence_points(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get the correspondence points between fragments."""
        return self._correspondence_points.copy()
    
    @correspondence_points.setter
    def correspondence_points(self, value: List[Tuple[np.ndarray, np.ndarray]]):
        """Set the correspondence points between fragments."""
        self._correspondence_points = value.copy() if value else []
    
    @property
    def fitness_score(self) -> float:
        """Get the fitness score for the matching."""
        return self._fitness_score
    
    @fitness_score.setter
    def fitness_score(self, value: float):
        """Set the fitness score for the matching."""
        self._fitness_score = value
    
    @property
    def rmse_score(self) -> float:
        """Get the RMSE score for the matching."""
        return self._rmse_score
    
    @rmse_score.setter
    def rmse_score(self, value: float):
        """Set the RMSE score for the matching."""
        self._rmse_score = value
    
    def get_item_type(self) -> str:
        """Get the type of this item."""
        return "PairwiseResultItem"
    
    def add_correspondence_point(self, point1: np.ndarray, point2: np.ndarray):
        """Add a correspondence point pair."""
        self._correspondence_points.append((point1.copy(), point2.copy()))
    
    def get_correspondence_count(self) -> int:
        """Get the number of correspondence points."""
        return len(self._correspondence_points)
    
    def get_translation(self) -> np.ndarray:
        """Get the translation vector from the transformation matrix."""
        return self._transformation_matrix[:3, 3]
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix from the transformation matrix."""
        return self._transformation_matrix[:3, :3]
    
    def get_vertex_count(self) -> int:
        """Get the number of vertices in the matched mesh."""
        if self._matched_mesh is None:
            return 0
        if hasattr(self._matched_mesh, 'vertices'):
            return len(self._matched_mesh.vertices)
        elif hasattr(self._matched_mesh, 'points'):
            return len(self._matched_mesh.points)
        return 0
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'fragment1_path': self._fragment1_path,
            'fragment2_path': self._fragment2_path,
            'transformation_matrix': self._transformation_matrix.tolist(),
            'matching_score': self._matching_score,
            'matching_parameters': self._matching_parameters,
            'correspondence_count': self.get_correspondence_count(),
            'fitness_score': self._fitness_score,
            'rmse_score': self._rmse_score,
            'vertex_count': self.get_vertex_count()
        })
        return base_dict
    
    def __str__(self) -> str:
        return f"PairwiseResultItem(id={self._id}, label='{self._label}', score={self._matching_score:.3f}, visible={self._is_visible})" 