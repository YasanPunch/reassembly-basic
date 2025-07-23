import os
import sys
import time
import numpy as np
import open3d as o3d
from typing import List, Dict, Any, Optional, Callable, Tuple
import copy

# Add src to path for segmentation imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
try:
    import segmentation
    print("Successfully imported segmentation module")
except ImportError as e:
    print(f"Warning: Could not import segmentation module: {e}")
    segmentation = None


class SegmentationEngine:
    """
    Engine for performing segmentation on preprocessed items.
    Integrates with the existing segmentation.py code.
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback or (lambda step, progress, message: None)
    
    def segment_preprocessed_items(self, preprocessed_items: List, parameters: Dict[str, Any]) -> List[Dict]:
        """
        Segment a list of preprocessed items.
        
        Args:
            preprocessed_items: List of PreprocessedItem instances
            parameters: Segmentation parameters
            
        Returns:
            List of segmentation results for each item
        """
        if not segmentation:
            print("Error: Segmentation module not available")
            return []
        
        results = []
        total_items = len(preprocessed_items)
        
        for i, preprocessed_item in enumerate(preprocessed_items):
            self.progress_callback("segmentation", (i / total_items) * 100,
                                 f"Segmenting {preprocessed_item.label} ({i+1}/{total_items})")
            
            try:
                result = self._segment_single_item(preprocessed_item, parameters)
                results.append(result)
            except Exception as e:
                print(f"Error segmenting {preprocessed_item.label}: {e}")
                results.append({
                    'success': False,
                    'item': preprocessed_item,
                    'error': str(e),
                    'segments': [],
                    'processing_time': 0.0
                })
        
        self.progress_callback("segmentation", 100.0, "Segmentation completed")
        return results
    
    def _segment_single_item(self, preprocessed_item, parameters: Dict[str, Any]) -> Dict:
        """
        Segment a single preprocessed item.
        
        Args:
            preprocessed_item: PreprocessedItem instance
            parameters: Segmentation parameters
            
        Returns:
            Dictionary with segmentation results
        """
        start_time = time.time()
        
        # Get the original mesh from the preprocessed item
        original_mesh = preprocessed_item.original_mesh
        if original_mesh is None:
            return {
                'success': False,
                'item': preprocessed_item,
                'error': 'No original mesh available',
                'segments': [],
                'processing_time': 0.0
            }
        
        # Prepare segmentation parameters
        seg_params = self._prepare_segmentation_parameters(parameters)
        
        # Run segmentation using the existing segmentation code
        try:
            # Use the extract_fracture_surface_mesh function from segmentation.py
            fracture_surfaces = segmentation.extract_fracture_surface_mesh(
                original_mesh, 
                preprocessed_item.label, 
                seg_params
            )
            
            processing_time = time.time() - start_time
            
            if fracture_surfaces is None or len(fracture_surfaces) == 0:
                return {
                    'success': False,
                    'item': preprocessed_item,
                    'error': 'No fracture surfaces found',
                    'segments': [],
                    'processing_time': processing_time
                }
            
            # Convert fracture surfaces to segments
            segments = self._convert_fracture_surfaces_to_segments(
                fracture_surfaces, original_mesh, preprocessed_item, seg_params
            )
            
            return {
                'success': True,
                'item': preprocessed_item,
                'segments': segments,
                'fracture_surfaces': fracture_surfaces,
                'parameters_used': seg_params,
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'item': preprocessed_item,
                'error': str(e),
                'segments': [],
                'processing_time': processing_time
            }
    
    def _prepare_segmentation_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare segmentation parameters with defaults.
        
        Args:
            parameters: User-provided parameters
            
        Returns:
            Complete parameter dictionary with defaults
        """
        default_params = {
            'max_curvature_deg': 30.0,
            'area_limit_fraction': 0.02,
            'visualize_segmentation': False,  # Disable interactive visualization in batch mode
            'use_bumpiness_detection': False,
            'elevation_map_resolution': 64,
            'bumpiness_threshold': 0.2
        }
        
        # Update with user parameters
        seg_params = default_params.copy()
        seg_params.update(parameters)
        
        return seg_params
    
    def _convert_fracture_surfaces_to_segments(self, fracture_surfaces: List, 
                                             original_mesh, preprocessed_item, 
                                             parameters: Dict[str, Any]) -> List[Dict]:
        """
        Convert fracture surfaces to segment dictionaries.
        
        Args:
            fracture_surfaces: List of fracture surface meshes
            original_mesh: Original mesh
            preprocessed_item: PreprocessedItem reference
            parameters: Segmentation parameters
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        # Get color function from segmentation module
        get_color = getattr(segmentation, 'get_color', None)
        if get_color is None:
            # Fallback color function
            def get_color(index, total_items=20):
                colors = [[1,0,0],[0,0,1],[0,1,0],[1,1,0],[1,0,1],[0,1,1],
                         [0.8,0.5,0.2],[0.5,0.2,0.8],[0.2,0.8,0.5],[0.6,0.6,0.6]]
                return colors[index % len(colors)]
        
        for i, fracture_surface in enumerate(fracture_surfaces):
            if fracture_surface is None or not fracture_surface.has_triangles():
                continue
            
            # Calculate segment properties
            segment_properties = self._calculate_segment_properties(
                fracture_surface, original_mesh, parameters
            )
            
            # Get color for this segment
            segment_color = get_color(i, len(fracture_surfaces))
            
            # Create segment dictionary
            segment = {
                'index': i,
                'label': f"Segment {i+1}",
                'fracture_surface': fracture_surface,
                'segment_mesh': fracture_surface,
                'segment_color': segment_color,
                'segment_properties': segment_properties,
                'is_fracture_candidate': True,  # All segments from fracture surfaces are candidates
                'face_indices': self._get_face_indices_from_mesh(fracture_surface, original_mesh)
            }
            
            segments.append(segment)
        
        return segments
    
    def _calculate_segment_properties(self, segment_mesh, original_mesh, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate properties for a segment.
        
        Args:
            segment_mesh: Segment mesh
            original_mesh: Original mesh
            parameters: Segmentation parameters
            
        Returns:
            Dictionary of segment properties
        """
        if not segment_mesh.has_triangles():
            return {}
        
        # Calculate area
        segment_area = segment_mesh.get_surface_area()
        original_area = original_mesh.get_surface_area() if original_mesh.has_triangles() else 1.0
        area_fraction = segment_area / original_area if original_area > 0 else 0.0
        
        # Calculate average normal
        if segment_mesh.has_vertex_normals():
            avg_normal = np.mean(np.asarray(segment_mesh.vertex_normals), axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal) if np.linalg.norm(avg_normal) > 0 else np.array([0, 0, 1])
        else:
            avg_normal = np.array([0, 0, 1])
        
        # Calculate bumpiness if requested
        bumpiness = 0.0
        if parameters.get('use_bumpiness_detection', False):
            try:
                bumpiness = segmentation.calculate_region_bumpiness(
                    segment_mesh, list(range(len(segment_mesh.triangles))), parameters
                )
            except:
                bumpiness = 0.0
        
        return {
            'area': segment_area,
            'area_fraction': area_fraction,
            'avg_normal': avg_normal,
            'bumpiness': bumpiness,
            'vertex_count': len(segment_mesh.vertices),
            'triangle_count': len(segment_mesh.triangles)
        }
    
    def _get_face_indices_from_mesh(self, segment_mesh, original_mesh) -> np.ndarray:
        """
        Get face indices of segment relative to original mesh.
        This is a simplified approach - in practice, you'd need to track face mappings.
        
        Args:
            segment_mesh: Segment mesh
            original_mesh: Original mesh
            
        Returns:
            Array of face indices
        """
        # For now, return empty array since we don't have face mapping
        # In a full implementation, you'd track which faces from the original mesh
        # correspond to each segment
        return np.array([]) 