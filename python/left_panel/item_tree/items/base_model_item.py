import os
from .base_item import BaseItem
import open3d as o3d


class BaseModelItem(BaseItem):
    """
    Represents a base model item in the item tree.
    Contains mesh data and file path information.
    """
    
    def __init__(self, label: str, mesh_path: str, mesh=None, is_visible: bool = True):
        """
        Initialize a base model item.
        
        Args:
            label (str): Display label for the item
            mesh_path (str): Path to the mesh file
            mesh: Open3D mesh or point cloud object
            is_visible (bool): Whether the item is visible by default
        """
        super().__init__(label, is_visible)
        self._mesh_path = mesh_path
        self._mesh = mesh
        
    @property
    def mesh_path(self) -> str:
        """Get the path to the mesh file."""
        return self._mesh_path
    
    @mesh_path.setter
    def mesh_path(self, value: str):
        """Set the path to the mesh file."""
        self._mesh_path = value
    
    @property
    def mesh(self):
        """Get the Open3D mesh or point cloud object."""
        return self._mesh
    
    @mesh.setter
    def mesh(self, value):
        """Set the Open3D mesh or point cloud object."""
        self._mesh = value
    
    def get_item_type(self) -> str:
        """Get the type of this item."""
        return "BaseModelItem"
    
    def get_file_name(self) -> str:
        """Get just the filename from the mesh path."""
        return os.path.basename(self._mesh_path)
    
    def get_file_extension(self) -> str:
        """Get the file extension from the mesh path."""
        return os.path.splitext(self._mesh_path)[1].lower()
    
    def is_mesh(self) -> bool:
        """Check if this item contains a mesh (vs point cloud)."""
        return self._mesh is not None and hasattr(self._mesh, 'vertices')
    
    def is_point_cloud(self) -> bool:
        """Check if this item contains a point cloud (vs mesh)."""
        return self._mesh is not None and hasattr(self._mesh, 'points')
    
    def get_vertex_count(self) -> int:
        """Get the number of vertices in the mesh or points in the point cloud."""
        if self._mesh is None:
            return 0
        if self.is_mesh():
            return len(self._mesh.vertices)
        elif self.is_point_cloud():
            return len(self._mesh.points)
        return 0
    
    def get_triangle_count(self) -> int:
        """Get the number of triangles in the mesh (0 for point clouds)."""
        if self._mesh is None or not self.is_mesh():
            return 0
        return len(self._mesh.triangles)
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'mesh_path': self._mesh_path,
            'file_name': self.get_file_name(),
            'file_extension': self.get_file_extension(),
            'vertex_count': self.get_vertex_count(),
            'triangle_count': self.get_triangle_count(),
            'is_mesh': self.is_mesh(),
            'is_point_cloud': self.is_point_cloud()
        })
        return base_dict
    
    def __str__(self) -> str:
        return f"BaseModelItem(id={self._id}, label='{self._label}', path='{self._mesh_path}', visible={self._is_visible})" 