from .base_item import BaseItem
from typing import List, Dict, Any, Optional
import numpy as np


class AssemblyResultItem(BaseItem):
    """
    Represents an assembly result item in the item tree.
    Contains the final assembled model and assembly metadata.
    """
    
    def __init__(self, label: str, assembled_mesh=None, fragment_paths: List[str] = None,
                 assembly_parameters: Dict[str, Any] = None, assembly_score: float = 0.0,
                 fragment_count: int = 0, is_visible: bool = True):
        """
        Initialize an assembly result item.
        
        Args:
            label (str): Display label for the item
            assembled_mesh: Open3D mesh or point cloud object (final assembly)
            fragment_paths (list): List of paths to the fragments used in assembly
            assembly_parameters (dict): Parameters used for assembly
            assembly_score (float): Overall assembly quality score
            fragment_count (int): Number of fragments in the assembly
            is_visible (bool): Whether the item is visible by default
        """
        super().__init__(label, is_visible)
        self._assembled_mesh = assembled_mesh
        self._fragment_paths = fragment_paths or []
        self._assembly_parameters = assembly_parameters or {}
        self._assembly_score = assembly_score
        self._fragment_count = fragment_count
        self._assembly_transformations = {}  # fragment_id -> transformation_matrix
        self._assembly_metadata = {}
        self._completion_percentage = 0.0
        
    @property
    def assembled_mesh(self):
        """Get the assembled Open3D mesh or point cloud object."""
        return self._assembled_mesh
    
    @assembled_mesh.setter
    def assembled_mesh(self, value):
        """Set the assembled Open3D mesh or point cloud object."""
        self._assembled_mesh = value
    
    @property
    def fragment_paths(self) -> List[str]:
        """Get the list of fragment paths used in assembly."""
        return self._fragment_paths.copy()
    
    @fragment_paths.setter
    def fragment_paths(self, value: List[str]):
        """Set the list of fragment paths used in assembly."""
        self._fragment_paths = value.copy() if value else []
    
    @property
    def assembly_parameters(self) -> Dict[str, Any]:
        """Get the parameters used for assembly."""
        return self._assembly_parameters.copy()
    
    @assembly_parameters.setter
    def assembly_parameters(self, value: Dict[str, Any]):
        """Set the parameters used for assembly."""
        self._assembly_parameters = value.copy() if value else {}
    
    @property
    def assembly_score(self) -> float:
        """Get the overall assembly quality score."""
        return self._assembly_score
    
    @assembly_score.setter
    def assembly_score(self, value: float):
        """Set the overall assembly quality score."""
        self._assembly_score = value
    
    @property
    def fragment_count(self) -> int:
        """Get the number of fragments in the assembly."""
        return self._fragment_count
    
    @fragment_count.setter
    def fragment_count(self, value: int):
        """Set the number of fragments in the assembly."""
        self._fragment_count = value
    
    @property
    def assembly_transformations(self) -> Dict[str, np.ndarray]:
        """Get the transformation matrices for each fragment."""
        return {k: v.copy() for k, v in self._assembly_transformations.items()}
    
    @assembly_transformations.setter
    def assembly_transformations(self, value: Dict[str, np.ndarray]):
        """Set the transformation matrices for each fragment."""
        self._assembly_transformations = {k: v.copy() for k, v in value.items()} if value else {}
    
    @property
    def assembly_metadata(self) -> Dict[str, Any]:
        """Get additional assembly metadata."""
        return self._assembly_metadata.copy()
    
    @assembly_metadata.setter
    def assembly_metadata(self, value: Dict[str, Any]):
        """Set additional assembly metadata."""
        self._assembly_metadata = value.copy() if value else {}
    
    @property
    def completion_percentage(self) -> float:
        """Get the completion percentage of the assembly."""
        return self._completion_percentage
    
    @completion_percentage.setter
    def completion_percentage(self, value: float):
        """Set the completion percentage of the assembly."""
        self._completion_percentage = max(0.0, min(100.0, value))
    
    def get_item_type(self) -> str:
        """Get the type of this item."""
        return "AssemblyResultItem"
    
    def add_fragment_path(self, path: str):
        """Add a fragment path to the assembly."""
        if path not in self._fragment_paths:
            self._fragment_paths.append(path)
            self._fragment_count = len(self._fragment_paths)
    
    def set_fragment_transformation(self, fragment_id: str, transformation: np.ndarray):
        """Set the transformation matrix for a specific fragment."""
        self._assembly_transformations[fragment_id] = transformation.copy()
    
    def get_fragment_transformation(self, fragment_id: str) -> Optional[np.ndarray]:
        """Get the transformation matrix for a specific fragment."""
        return self._assembly_transformations.get(fragment_id, None)
    
    def add_assembly_metadata(self, key: str, value: Any):
        """Add metadata to the assembly."""
        self._assembly_metadata[key] = value
    
    def get_vertex_count(self) -> int:
        """Get the number of vertices in the assembled mesh."""
        if self._assembled_mesh is None:
            return 0
        if hasattr(self._assembled_mesh, 'vertices'):
            return len(self._assembled_mesh.vertices)
        elif hasattr(self._assembled_mesh, 'points'):
            return len(self._assembled_mesh.points)
        return 0
    
    def get_triangle_count(self) -> int:
        """Get the number of triangles in the assembled mesh (0 for point clouds)."""
        if self._assembled_mesh is None or not hasattr(self._assembled_mesh, 'triangles'):
            return 0
        return len(self._assembled_mesh.triangles)
    
    def is_complete(self) -> bool:
        """Check if the assembly is complete (100% completion)."""
        return self._completion_percentage >= 100.0
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'fragment_paths': self._fragment_paths,
            'assembly_parameters': self._assembly_parameters,
            'assembly_score': self._assembly_score,
            'fragment_count': self._fragment_count,
            'assembly_transformations': {k: v.tolist() for k, v in self._assembly_transformations.items()},
            'assembly_metadata': self._assembly_metadata,
            'completion_percentage': self._completion_percentage,
            'vertex_count': self.get_vertex_count(),
            'triangle_count': self.get_triangle_count(),
            'is_complete': self.is_complete()
        })
        return base_dict
    
    def __str__(self) -> str:
        return f"AssemblyResultItem(id={self._id}, label='{self._label}', fragments={self._fragment_count}, score={self._assembly_score:.3f}, completion={self._completion_percentage:.1f}%, visible={self._is_visible})" 