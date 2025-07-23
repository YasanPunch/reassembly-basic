import os
import sys
import time
import numpy as np
import open3d as o3d
from typing import List, Dict, Any, Optional, Callable


class PreprocessingEngine:
    """
    Engine for executing preprocessing operations on base model items.
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize the preprocessing engine.
        
        Args:
            progress_callback: Optional callback function for progress updates
        """
        self.progress_callback = progress_callback or (lambda step, progress, message: None)
        
    def preprocess_base_models(self, base_model_items: List, parameters: Dict[str, Any]) -> List[Dict]:
        """
        Preprocess a list of base model items.
        
        Args:
            base_model_items: List of BaseModelItem objects to preprocess
            parameters: Preprocessing parameters
            
        Returns:
            List of preprocessing results with metadata
        """
        results = []
        
        for i, item in enumerate(base_model_items):
            self.progress_callback("preprocessing", (i / len(base_model_items)) * 100, 
                                 f"Preprocessing {item.label} ({i+1}/{len(base_model_items)})")
            
            try:
                result = self._preprocess_single_model(item, parameters)
                results.append(result)
            except Exception as e:
                print(f"Error preprocessing {item.label}: {e}")
                # Add error result
                results.append({
                    'success': False,
                    'item': item,
                    'error': str(e),
                    'preprocessed_mesh': None,
                    'processing_time': 0.0,
                    'quality_metrics': {},
                    'preprocessing_steps': []
                })
        
        self.progress_callback("preprocessing", 100, "Preprocessing completed")
        return results
    
    def _preprocess_single_model(self, item, parameters: Dict[str, Any]) -> Dict:
        """
        Preprocess a single base model item.
        
        Args:
            item: BaseModelItem to preprocess
            parameters: Preprocessing parameters
            
        Returns:
            Dictionary containing preprocessing results
        """
        start_time = time.time()
        
        # Extract the mesh from the item
        mesh = item.mesh
        if mesh is None:
            raise ValueError(f"No mesh data available for {item.label}")
        
        # Create fragment info structure expected by preprocessing module
        fragment_info = {
            'mesh': mesh,
            'name': item.label,
            'original_index': 0  # We'll use 0 for now, could be enhanced later
        }
        
        # Default parameters if not provided
        default_params = {
            "voxel_downsample_size": 7.0,
            "normal_estimation_radius": 14.0,
            "normal_estimation_max_nn": 30,
            "fracture_surface_dense_sample_points": 10000,
            "add_preprocessing_noise": True,
            "preprocessing_noise_factor": 0.01,
            "orient_normals_k": 15,
        }
        
        # Merge provided parameters with defaults
        merged_params = {**default_params, **parameters}
        
        try:
            # For now, use basic preprocessing to ensure it works
            # We can add advanced src.preprocessing integration later
            preprocessed_mesh = self._basic_preprocessing(mesh, merged_params)
            preprocessing_steps = ["voxel_downsampling", "normal_estimation"]
            
            # TODO: Add advanced preprocessing when modules are properly integrated
            # if 'preprocessing' in sys.modules:
            #     pcds_for_features_list, features_list, fracture_surfaces = preprocessing.preprocess_fragment(
            #         fragment_info, merged_params
            #     )
            #     if pcds_for_features_list and len(pcds_for_features_list) > 0:
            #         preprocessed_mesh = pcds_for_features_list[0]
            #     preprocessing_steps = ["voxel_downsampling", "normal_estimation", "fracture_surface_extraction"]
            
            processing_time = time.time() - start_time
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(mesh, preprocessed_mesh)
            
            return {
                'success': True,
                'item': item,
                'preprocessed_mesh': preprocessed_mesh,
                'processing_time': processing_time,
                'quality_metrics': quality_metrics,
                'preprocessing_steps': preprocessing_steps,
                'parameters_used': merged_params
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            raise Exception(f"Preprocessing failed: {str(e)}")
    
    def _basic_preprocessing(self, mesh: o3d.geometry.TriangleMesh, parameters: Dict[str, Any]) -> o3d.geometry.PointCloud:
        """
        Basic preprocessing fallback when src.preprocessing is not available.
        
        Args:
            mesh: Input mesh
            parameters: Preprocessing parameters
            
        Returns:
            Preprocessed point cloud
        """
        # Convert mesh to point cloud
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
        
        # Voxel downsampling
        voxel_size = parameters.get("voxel_downsample_size", 7.0)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # Estimate normals
        radius = parameters.get("normal_estimation_radius", 14.0)
        max_nn = parameters.get("normal_estimation_max_nn", 30)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        
        # Orient normals
        k = parameters.get("orient_normals_k", 15)
        pcd.orient_normals_consistent_tangent_plane(k=k)
        
        return pcd
    
    def _calculate_quality_metrics(self, original_mesh: o3d.geometry.TriangleMesh, 
                                 preprocessed_mesh) -> Dict[str, float]:
        """
        Calculate quality metrics for the preprocessing result.
        
        Args:
            original_mesh: Original input mesh
            preprocessed_mesh: Preprocessed result (mesh or point cloud)
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        if original_mesh is None or preprocessed_mesh is None:
            return metrics
        
        # Original mesh metrics
        if hasattr(original_mesh, 'vertices'):
            metrics['original_vertex_count'] = len(original_mesh.vertices)
        if hasattr(original_mesh, 'triangles'):
            metrics['original_triangle_count'] = len(original_mesh.triangles)
        
        # Preprocessed mesh metrics
        if hasattr(preprocessed_mesh, 'points'):
            metrics['preprocessed_point_count'] = len(preprocessed_mesh.points)
            metrics['reduction_ratio'] = metrics['preprocessed_point_count'] / max(metrics.get('original_vertex_count', 1), 1)
        elif hasattr(preprocessed_mesh, 'vertices'):
            metrics['preprocessed_vertex_count'] = len(preprocessed_mesh.vertices)
            metrics['reduction_ratio'] = metrics['preprocessed_vertex_count'] / max(metrics.get('original_vertex_count', 1), 1)
        
        # Normal quality (if available)
        if hasattr(preprocessed_mesh, 'normals') and len(preprocessed_mesh.normals) > 0:
            normals = np.asarray(preprocessed_mesh.normals)
            # Calculate normal consistency (how well normals are oriented)
            normal_consistency = np.mean(np.abs(normals[:, 2]))  # Simple metric
            metrics['normal_consistency'] = float(normal_consistency)
        
        # Surface quality (if mesh)
        if hasattr(preprocessed_mesh, 'vertices') and hasattr(preprocessed_mesh, 'triangles'):
            # Calculate surface area
            vertices = np.asarray(preprocessed_mesh.vertices)
            triangles = np.asarray(preprocessed_mesh.triangles)
            
            # Simple surface area calculation
            surface_area = 0.0
            for triangle in triangles:
                v1, v2, v3 = vertices[triangle]
                # Calculate triangle area
                edge1 = v2 - v1
                edge2 = v3 - v1
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                surface_area += area
            
            metrics['surface_area'] = float(surface_area)
        
        return metrics 