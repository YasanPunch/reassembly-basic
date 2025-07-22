"""
Supervised Learning Module for Fracture Detection

This module provides functionality for:
1. Interactive selection of fractured and non-fractured surfaces
2. Parameter optimization using Random Forest classification
3. Automatic threshold determination for fracture detection
"""

import numpy as np
import open3d as o3d
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from roughness_analysis import calculate_surface_roughness_characteristic

class FractureSurfaceSelector:
    def __init__(self, mesh, segmentation_results):
        self.mesh = mesh
        self.segmentation_results = segmentation_results
        self.region_results = segmentation_results['region_results']
        self.selected_fracture_regions = set()
        self.selected_non_fracture_regions = set()
        self.current_region_idx = 0
    def start_interactive_selection(self):
        print("\n" + "="*80)
        print("INTERACTIVE SURFACE SELECTION")
        print("="*80)
        print("You will now select which regions are fractured and which are not.")
        print("For each region, you can:")
        print("  - Press 'F' to mark as FRACTURED surface")
        print("  - Press 'N' to mark as NON-FRACTURED surface") 
        print("  - Press 'S' to SKIP this region")
        print("  - Press 'Q' to QUIT selection")
        print("="*80)
        
        total_regions = len(self.region_results)
        
        for i, region_result in enumerate(self.region_results):
            self.current_region_idx = i
            
            # Show progress
            print(f"\n{'='*60}")
            print(f"PROGRESS: Region {i+1}/{total_regions}")
            print(f"{'='*60}")
            
            # Use callback-based selection for this region
            selection = self._select_region_with_callbacks(i)
            
            if selection is None:
                print("Selection process terminated by user.")
                return False
            elif selection == 'F':
                self.selected_fracture_regions.add(i)
                print(f"✓ Region {i+1} marked as FRACTURED")
            elif selection == 'N':
                self.selected_non_fracture_regions.add(i)
                print(f"✓ Region {i+1} marked as NON-FRACTURED")
            elif selection == 'S':
                print(f"⏭ Region {i+1} skipped")
        
        return self._validate_selection()
    
    def _select_region_with_callbacks(self, region_idx):
        """
        Use Open3D callbacks to handle region selection during visualization.
        """
        # Shared state for callback communication
        shared_state = {
            'selection': None,
            'region_idx': region_idx
        }
        
        # Create visualization mesh
        viz_mesh = o3d.geometry.TriangleMesh()
        viz_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
        viz_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.triangles))
        viz_mesh.compute_vertex_normals()
        
        # Color the current region in bright yellow, others in grey
        num_vertices = len(np.asarray(self.mesh.vertices))
        colors = np.full((num_vertices, 3), 0.5)  # Grey for all vertices
        
        # Color current region in yellow
        region_result = self.region_results[region_idx]
        segment_faces = region_result['segment_faces']
        original_faces = np.asarray(self.mesh.triangles)
        
        region_vertices = set()
        for face_idx in segment_faces:
            face = original_faces[face_idx]
            region_vertices.update(face)
        
        for vertex_idx in region_vertices:
            if vertex_idx < num_vertices:
                colors[vertex_idx] = [1.0, 1.0, 0.0]  # Bright yellow
        
        viz_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Create visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            f"Region {region_idx+1} Selection - Press F/N/S/Q", 
            width=1200, 
            height=800
        )
        
        # Add geometries
        vis.add_geometry(viz_mesh)
        vis.add_geometry(coordinate_frame)
        
        # Set up view
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().point_show_normal = False
        
        # Define callback functions
        def mark_fractured(visualizer):
            shared_state['selection'] = 'F'
            print(f"\n  Region {region_idx+1} marked as FRACTURED")
            visualizer.close()
            return False
        
        def mark_non_fractured(visualizer):
            shared_state['selection'] = 'N'
            print(f"\n  Region {region_idx+1} marked as NON-FRACTURED")
            visualizer.close()
            return False
        
        def skip_region(visualizer):
            shared_state['selection'] = 'S'
            print(f"\n  Region {region_idx+1} skipped")
            visualizer.close()
            return False
        
        def quit_selection(visualizer):
            shared_state['selection'] = None
            print(f"\n  Selection process terminated")
            visualizer.close()
            return False
        
        # Register key callbacks
        vis.register_key_callback(ord('F'), mark_fractured)
        vis.register_key_callback(ord('f'), mark_fractured)  # Lowercase too
        vis.register_key_callback(ord('N'), mark_non_fractured)
        vis.register_key_callback(ord('n'), mark_non_fractured)  # Lowercase too
        vis.register_key_callback(ord('S'), skip_region)
        vis.register_key_callback(ord('s'), skip_region)  # Lowercase too
        vis.register_key_callback(ord('Q'), quit_selection)
        vis.register_key_callback(ord('q'), quit_selection)  # Lowercase too
        
        # Print instructions
        print(f"\n{'='*60}")
        print(f"REGION {region_idx+1} VISUALIZATION")
        print(f"{'='*60}")
        print(f"Region {region_idx+1} is highlighted in YELLOW")
        print(f"Other regions are shown in GREY")
        print(f"Region size: {region_result['num_faces']} faces, {region_result['num_vertices']} vertices")
        print(f"\nKeyboard Controls:")
        print(f"  F: Mark as FRACTURED surface")
        print(f"  N: Mark as NON-FRACTURED surface")
        print(f"  S: SKIP this region")
        print(f"  Q: QUIT selection process")
        print(f"{'='*60}")
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        
        return shared_state['selection']
    def _validate_selection(self):
        num_fracture = len(self.selected_fracture_regions)
        num_non_fracture = len(self.selected_non_fracture_regions)
        print(f"\nSelection Summary:")
        print(f"  Fractured regions: {num_fracture}")
        print(f"  Non-fractured regions: {num_non_fracture}")
        print(f"  Total labeled regions: {num_fracture + num_non_fracture}")
        if num_fracture == 0:
            print("❌ Error: No fractured regions selected. Need at least 1.")
            return False
        elif num_non_fracture == 0:
            print("❌ Error: No non-fractured regions selected. Need at least 1.")
            return False
        elif num_fracture + num_non_fracture < 3:
            print("❌ Error: Need at least 3 total regions for reliable training.")
            return False
        else:
            print("✓ Selection validated successfully!")
            return True
    def get_labeled_data(self):
        labeled_data = {
            'fracture_regions': list(self.selected_fracture_regions),
            'non_fracture_regions': list(self.selected_non_fracture_regions),
            'total_regions': len(self.region_results)
        }
        return labeled_data

class ParameterOptimizer:
    def __init__(self, mesh, segmentation_results, labeled_data):
        self.mesh = mesh
        self.segmentation_results = segmentation_results
        self.region_results = segmentation_results['region_results']
        self.labeled_data = labeled_data
    def define_parameter_search_space(self):
        k_values = [10, 30, 50, 75, 100]
        r_factors = [1, 3, 7, 15, 20]
        return k_values, r_factors
    def extract_features_from_region(self, region_idx, roughness_values):
        region_result = self.region_results[region_idx]
        segment_faces = region_result['segment_faces']
        original_faces = np.asarray(self.mesh.triangles)
        region_vertices = set()
        for face_idx in segment_faces:
            face = original_faces[face_idx]
            region_vertices.update(face)
        region_roughness = []
        for vertex_idx in region_vertices:
            if vertex_idx < len(roughness_values):
                region_roughness.append(roughness_values[vertex_idx])
        if len(region_roughness) == 0:
            return None
        features = {
            'mean_roughness': np.mean(region_roughness),
            'std_roughness': np.std(region_roughness),
            'median_roughness': np.median(region_roughness),
            'min_roughness': np.min(region_roughness),
            'max_roughness': np.max(region_roughness),
            'percentile_25': np.percentile(region_roughness, 25),
            'percentile_75': np.percentile(region_roughness, 75),
            'percentile_90': np.percentile(region_roughness, 90),
            'percentile_95': np.percentile(region_roughness, 95),
            'iqr': np.percentile(region_roughness, 75) - np.percentile(region_roughness, 25),
            'skewness': self._calculate_skewness(region_roughness),
            'kurtosis': self._calculate_kurtosis(region_roughness),
            'num_vertices': len(region_roughness)
        }
        return features
    def _calculate_skewness(self, data):
        if len(data) < 3:
            return 0.0
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        if std == 0:
            return 0.0
        return np.mean(((data_array - mean) / std) ** 3)
    def _calculate_kurtosis(self, data):
        if len(data) < 4:
            return 0.0
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        if std == 0:
            return 0.0
        return np.mean(((data_array - mean) / std) ** 4) - 3
    def prepare_training_data(self, k, r):
        print(f"Calculating surface roughness with k={k}, r={r}...")
        roughness_values = calculate_surface_roughness_characteristic(self.mesh, k, r)
        X = []
        y = []
        for region_idx in self.labeled_data['non_fracture_regions']:
            features = self.extract_features_from_region(region_idx, roughness_values)
            if features is not None:
                X.append(list(features.values()))
                y.append(0)
        for region_idx in self.labeled_data['fracture_regions']:
            features = self.extract_features_from_region(region_idx, roughness_values)
            if features is not None:
                X.append(list(features.values()))
                y.append(1)
        return np.array(X), np.array(y)
    def evaluate_parameters(self, k, r):
        try:
            X, y = self.prepare_training_data(k, r)
            
            # Validate training data
            if len(X) < 3:
                print(f"  → Insufficient data: only {len(X)} samples")
                return 0.0, None
            
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                print(f"  → Only one class present: {unique_classes}")
                return 0.0, None
            
            # Check class distribution
            class_counts = np.bincount(y)
            min_class_count = min(class_counts)
            if min_class_count < 2:
                print(f"  → Class imbalance: smallest class has {min_class_count} samples")
                return 0.0, None
            
            # Determine appropriate cross-validation folds
            n_samples = len(X)
            n_folds = min(3, min_class_count)  # Use fewer folds for small datasets
            
            if n_folds < 2:
                print(f"  → Cannot perform cross-validation with {n_folds} fold(s)")
                return 0.0, None
            
            # Train Random Forest with appropriate cross-validation
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(rf, X, y, cv=n_folds, scoring='accuracy')
            mean_accuracy = scores.mean()
            
            print(f"  → CV accuracy: {mean_accuracy:.4f} ({n_folds}-fold)")
            return mean_accuracy, rf
            
        except Exception as e:
            print(f"  → Error evaluating k={k}, r={r}: {e}")
            return 0.0, None
    def optimize_parameters(self):
        print("\n" + "="*80)
        print("PARAMETER OPTIMIZATION")
        print("="*80)
        k_values, r_factors = self.define_parameter_search_space()
        print(f"Searching {len(k_values)} k values and {len(r_factors)} r factors...")
        print(f"Total combinations to evaluate: {len(k_values) * len(r_factors)}")
        best_params = None
        best_score = 0.0
        best_model = None
        results = []
        vertices = np.asarray(self.mesh.vertices)
        faces = np.asarray(self.mesh.triangles)
        edge_lengths = []
        for face in faces:
            v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge_lengths.extend([
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v3 - v2),
                np.linalg.norm(v1 - v3)
            ])
        avg_edge_length = np.mean(edge_lengths)
        for i, k in enumerate(k_values):
            for j, r_factor in enumerate(r_factors):
                r = avg_edge_length * r_factor
                print(f"Evaluating k={k}, r={r:.6f} ({r_factor}x avg edge length) "
                      f"[{i*len(r_factors) + j + 1}/{len(k_values) * len(r_factors)}]")
                score, model = self.evaluate_parameters(k, r)
                results.append({
                    'k': k,
                    'r': r,
                    'r_factor': r_factor,
                    'score': score
                })
                if score > best_score:
                    best_score = score
                    best_params = (k, r, r_factor)
                    best_model = model
                    print(f"  → New best score: {score:.4f}")
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Check if we found valid parameters
        if best_params is None:
            print(f"\n❌ ERROR: No valid parameters found!")
            print(f"Possible causes:")
            print(f"  - Too few labeled regions (need at least 3 total)")
            print(f"  - Class imbalance (need at least 2 samples per class)")
            print(f"  - All parameter combinations failed")
            print(f"\nFalling back to default parameters...")
            
            # Use default parameters as fallback
            default_k = 20
            default_r = avg_edge_length * 3
            default_r_factor = 3
            
            print(f"Default parameters: k={default_k}, r={default_r:.6f} ({default_r_factor}x avg edge length)")
            
            # Train a simple model with default parameters
            try:
                X, y = self.prepare_training_data(default_k, default_r)
                if len(X) >= 2 and len(np.unique(y)) >= 2:
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                    best_params = (default_k, default_r, default_r_factor)
                    best_model = rf
                    best_score = 0.5  # Default score for fallback
                    print(f"✓ Fallback model trained successfully")
                else:
                    print(f"❌ Fallback training also failed")
                    return None, None, results
            except Exception as e:
                print(f"❌ Fallback training failed: {e}")
                return None, None, results
        else:
            print(f"\nOptimization Results:")
            print(f"Best parameters: k={best_params[0]}, r={best_params[1]:.6f} ({best_params[2]}x avg edge length)")
            print(f"Best cross-validation accuracy: {best_score:.4f}")
        
        print(f"\nTop 5 parameter combinations:")
        for i, result in enumerate(results[:5]):
            print(f"  {i+1}. k={result['k']}, r={result['r']:.6f} ({result['r_factor']}x) - Score: {result['score']:.4f}")
        
        return best_params, best_model, results

    def analyze_optimization_results(self, results):
        """
        Analyze the parameter optimization results to understand the impact of k and r.
        """
        if not results:
            print("No optimization results to analyze.")
            return
        
        print(f"\n" + "="*80)
        print("PARAMETER OPTIMIZATION ANALYSIS")
        print("="*80)
        
        # Filter out failed evaluations
        valid_results = [r for r in results if r['score'] > 0]
        
        if not valid_results:
            print("No valid parameter combinations found.")
            return
        
        # Group results by k and r for analysis
        k_analysis = {}
        r_analysis = {}
        
        for result in valid_results:
            k, r_factor, score = result['k'], result['r_factor'], result['score']
            
            # Analyze k impact
            if k not in k_analysis:
                k_analysis[k] = []
            k_analysis[k].append(score)
            
            # Analyze r impact
            if r_factor not in r_analysis:
                r_analysis[r_factor] = []
            r_analysis[r_factor].append(score)
        
        # Analyze k parameter impact
        print(f"\nK PARAMETER ANALYSIS:")
        print(f"{'k':>4} | {'Avg Score':>10} | {'Min Score':>10} | {'Max Score':>10} | {'Std Dev':>10}")
        print("-" * 60)
        for k in sorted(k_analysis.keys()):
            scores = k_analysis[k]
            avg_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            std_score = np.std(scores)
            print(f"{k:>4} | {avg_score:>10.4f} | {min_score:>10.4f} | {max_score:>10.4f} | {std_score:>10.4f}")
        
        # Analyze r parameter impact
        print(f"\nR PARAMETER ANALYSIS:")
        print(f"{'r_factor':>8} | {'Avg Score':>10} | {'Min Score':>10} | {'Max Score':>10} | {'Std Dev':>10}")
        print("-" * 60)
        for r_factor in sorted(r_analysis.keys()):
            scores = r_analysis[r_factor]
            avg_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            std_score = np.std(scores)
            print(f"{r_factor:>8} | {avg_score:>10.4f} | {min_score:>10.4f} | {max_score:>10.4f} | {std_score:>10.4f}")
        
        # Find best k and r independently
        best_k = max(k_analysis.keys(), key=lambda k: np.mean(k_analysis[k]))
        best_r_factor = max(r_analysis.keys(), key=lambda r: np.mean(r_analysis[r]))
        
        print(f"\nINDEPENDENT OPTIMAL VALUES:")
        print(f"  Best k (by average score): {best_k}")
        print(f"  Best r_factor (by average score): {best_r_factor}")
        
        # Analyze parameter interactions
        print(f"\nPARAMETER INTERACTION ANALYSIS:")
        print(f"Top 10 parameter combinations:")
        print(f"{'Rank':>4} | {'k':>4} | {'r_factor':>8} | {'Score':>10}")
        print("-" * 35)
        for i, result in enumerate(valid_results[:10]):
            print(f"{i+1:>4} | {result['k']:>4} | {result['r_factor']:>8} | {result['score']:>10.4f}")
        
        # Provide interpretation
        print(f"\nINTERPRETATION:")
        print(f"1. K (neighborhood size for local bending energy):")
        print(f"   - Higher k = larger local neighborhood = more smoothing")
        print(f"   - Lower k = smaller neighborhood = more sensitive to local details")
        print(f"   - Optimal k balances noise reduction vs. detail preservation")
        
        print(f"\n2. R (radius for surface roughness characteristic):")
        print(f"   - Higher r = larger averaging radius = more global roughness measure")
        print(f"   - Lower r = smaller radius = more local roughness measure")
        print(f"   - Optimal r balances local vs. global surface characteristics")
        
        print(f"\n3. Parameter Selection Strategy:")
        print(f"   - System uses cross-validation accuracy as optimization metric")
        print(f"   - Best parameters maximize classification accuracy on labeled data")
        print(f"   - Parameters are mesh-specific and depend on surface characteristics")
        
        return {
            'k_analysis': k_analysis,
            'r_analysis': r_analysis,
            'best_k': best_k,
            'best_r_factor': best_r_factor,
            'valid_results': valid_results
        }

class FractureClassifier:
    def __init__(self, optimal_params, trained_model):
        self.optimal_k, self.optimal_r, self.optimal_r_factor = optimal_params
        self.model = trained_model
    def classify_regions(self, mesh, segmentation_results):
        print(f"\nClassifying regions using optimized parameters:")
        print(f"  k={self.optimal_k}, r={self.optimal_r:.6f} ({self.optimal_r_factor}x avg edge length)")
        
        # Validate that we have a trained model
        if self.model is None:
            print("❌ Error: No trained model available for classification")
            return []
        
        try:
            roughness_values = calculate_surface_roughness_characteristic(mesh, self.optimal_k, self.optimal_r)
        except Exception as e:
            print(f"❌ Error calculating surface roughness: {e}")
            return []
        
        region_results = segmentation_results['region_results']
        classifications = []
        
        for i, region_result in enumerate(region_results):
            try:
                features = self._extract_features_from_region(region_result, roughness_values, mesh)
                
                if features is not None:
                    # Validate features before prediction
                    feature_values = list(features.values())
                    if any(np.isnan(val) or np.isinf(val) for val in feature_values):
                        print(f"  Warning: Invalid features for region {i+1}, marking as non-fractured")
                        classification = {
                            'region_id': i,
                            'is_fracture': False,
                            'fracture_probability': 0.0,
                            'features': features
                        }
                    else:
                        prediction = self.model.predict([feature_values])[0]
                        probability = self.model.predict_proba([feature_values])[0]
                        classification = {
                            'region_id': i,
                            'is_fracture': bool(prediction),
                            'fracture_probability': probability[1] if len(probability) > 1 else 0.0,
                            'features': features
                        }
                else:
                    classification = {
                        'region_id': i,
                        'is_fracture': False,
                        'fracture_probability': 0.0,
                        'features': None
                    }
            except Exception as e:
                print(f"  Warning: Error classifying region {i+1}: {e}")
                classification = {
                    'region_id': i,
                    'is_fracture': False,
                    'fracture_probability': 0.0,
                    'features': None
                }
            
            classifications.append(classification)
        
        return classifications
    def _extract_features_from_region(self, region_result, roughness_values, mesh):
        segment_faces = region_result['segment_faces']
        original_faces = np.asarray(mesh.triangles)
        region_vertices = set()
        for face_idx in segment_faces:
            face = original_faces[face_idx]
            region_vertices.update(face)
        region_roughness = []
        for vertex_idx in region_vertices:
            if vertex_idx < len(roughness_values):
                region_roughness.append(roughness_values[vertex_idx])
        if len(region_roughness) == 0:
            return None
        features = {
            'mean_roughness': np.mean(region_roughness),
            'std_roughness': np.std(region_roughness),
            'median_roughness': np.median(region_roughness),
            'min_roughness': np.min(region_roughness),
            'max_roughness': np.max(region_roughness),
            'percentile_25': np.percentile(region_roughness, 25),
            'percentile_75': np.percentile(region_roughness, 75),
            'percentile_90': np.percentile(region_roughness, 90),
            'percentile_95': np.percentile(region_roughness, 95),
            'iqr': np.percentile(region_roughness, 75) - np.percentile(region_roughness, 25),
            'skewness': self._calculate_skewness(region_roughness),
            'kurtosis': self._calculate_kurtosis(region_roughness),
            'num_vertices': len(region_roughness)
        }
        return features
    def _calculate_skewness(self, data):
        if len(data) < 3:
            return 0.0
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        if std == 0:
            return 0.0
        return np.mean(((data_array - mean) / std) ** 3)
    def _calculate_kurtosis(self, data):
        if len(data) < 4:
            return 0.0
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        if std == 0:
            return 0.0
        return np.mean(((data_array - mean) / std) ** 4) - 3

    def visualize_classification_results(self, mesh, segmentation_results, classifications):
        """
        Visualize the classification results with color-coded regions.
        """
        print(f"\n" + "="*80)
        print("CLASSIFICATION VISUALIZATION")
        print("="*80)
        
        # Create visualization mesh
        viz_mesh = o3d.geometry.TriangleMesh()
        viz_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        viz_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
        viz_mesh.compute_vertex_normals()
        
        # Color all vertices in grey initially
        num_vertices = len(np.asarray(mesh.vertices))
        colors = np.full((num_vertices, 3), 0.5)  # Grey for all vertices
        
        # Color regions based on classification
        region_results = segmentation_results['region_results']
        
        fracture_regions = []
        non_fracture_regions = []
        
        for i, classification in enumerate(classifications):
            region_id = classification['region_id']
            is_fracture = classification['is_fracture']
            probability = classification['fracture_probability']
            
            if region_id < len(region_results):
                region_result = region_results[region_id]
                segment_faces = region_result['segment_faces']
                original_faces = np.asarray(mesh.triangles)
                
                # Get vertices in this region
                region_vertices = set()
                for face_idx in segment_faces:
                    face = original_faces[face_idx]
                    region_vertices.update(face)
                
                # Color based on classification
                if is_fracture:
                    # Red for fractured regions (intensity based on confidence)
                    confidence_color = [0.5 + 0.5 * probability, 0.0, 0.0]  # Red with intensity
                    fracture_regions.append({
                        'region_id': region_id,
                        'probability': probability,
                        'num_faces': region_result['num_faces']
                    })
                else:
                    # Blue for non-fractured regions (intensity based on confidence)
                    confidence_color = [0.0, 0.0, 0.5 + 0.5 * (1 - probability)]  # Blue with intensity
                    non_fracture_regions.append({
                        'region_id': region_id,
                        'probability': probability,
                        'num_faces': region_result['num_faces']
                    })
                
                # Apply color to region vertices
                for vertex_idx in region_vertices:
                    if vertex_idx < num_vertices:
                        colors[vertex_idx] = confidence_color
        
        viz_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Print classification summary
        print(f"Classification Summary:")
        print(f"  Fractured regions: {len(fracture_regions)}")
        print(f"  Non-fractured regions: {len(non_fracture_regions)}")
        print(f"  Total regions: {len(classifications)}")
        
        if fracture_regions:
            print(f"\nFractured Regions (Red):")
            for region in fracture_regions:
                print(f"  Region {region['region_id']+1}: {region['probability']:.3f} confidence "
                      f"({region['num_faces']} faces)")
        
        if non_fracture_regions:
            print(f"\nNon-Fractured Regions (Blue):")
            for region in non_fracture_regions:
                print(f"  Region {region['region_id']+1}: {region['probability']:.3f} confidence "
                      f"({region['num_faces']} faces)")
        
        print(f"\nColor Legend:")
        print(f"  🔴 Red = Fractured surface (brighter = higher confidence)")
        print(f"  🔵 Blue = Non-fractured surface (brighter = higher confidence)")
        print(f"  ⚫ Grey = Unclassified or error")
        
        print(f"\nControls:")
        print(f"  - Mouse: Rotate, zoom, pan")
        print(f"  - Close window to continue")
        
        # Show the visualization
        o3d.visualization.draw_geometries(
            [viz_mesh, coordinate_frame],
            window_name="Supervised Learning Classification Results",
            width=1200,
            height=800,
            point_show_normal=False,
            mesh_show_back_face=True
        )
        
        return {
            'fracture_regions': fracture_regions,
            'non_fracture_regions': non_fracture_regions,
            'total_regions': len(classifications)
        }

    def visualize_label_comparison(self, mesh, segmentation_results, classifications, labeled_data):
        """
        Visualize comparison between manual labels and automatic classification.
        """
        print(f"\n" + "="*80)
        print("LABEL COMPARISON VISUALIZATION")
        print("="*80)
        
        # Create visualization mesh
        viz_mesh = o3d.geometry.TriangleMesh()
        viz_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        viz_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
        viz_mesh.compute_vertex_normals()
        
        # Color all vertices in grey initially
        num_vertices = len(np.asarray(mesh.vertices))
        colors = np.full((num_vertices, 3), 0.5)  # Grey for all vertices
        
        # Color regions based on agreement/disagreement
        region_results = segmentation_results['region_results']
        
        agreement_regions = []
        disagreement_regions = []
        
        for i, classification in enumerate(classifications):
            region_id = classification['region_id']
            auto_is_fracture = classification['is_fracture']
            probability = classification['fracture_probability']
            
            # Check if this region was manually labeled
            manual_is_fracture = None
            if region_id in labeled_data['fracture_regions']:
                manual_is_fracture = True
            elif region_id in labeled_data['non_fracture_regions']:
                manual_is_fracture = False
            
            if region_id < len(region_results):
                region_result = region_results[region_id]
                segment_faces = region_result['segment_faces']
                original_faces = np.asarray(mesh.triangles)
                
                # Get vertices in this region
                region_vertices = set()
                for face_idx in segment_faces:
                    face = original_faces[face_idx]
                    region_vertices.update(face)
                
                # Color based on agreement
                if manual_is_fracture is not None:
                    if auto_is_fracture == manual_is_fracture:
                        # Agreement: Green for fracture, Cyan for non-fracture
                        if auto_is_fracture:
                            color = [0.0, 0.8, 0.0]  # Green for fracture agreement
                        else:
                            color = [0.0, 0.8, 0.8]  # Cyan for non-fracture agreement
                        
                        agreement_regions.append({
                            'region_id': region_id,
                            'classification': 'fracture' if auto_is_fracture else 'non-fracture',
                            'probability': probability,
                            'num_faces': region_result['num_faces']
                        })
                    else:
                        # Disagreement: Yellow for disagreement
                        color = [1.0, 1.0, 0.0]  # Yellow for disagreement
                        
                        disagreement_regions.append({
                            'region_id': region_id,
                            'manual': 'fracture' if manual_is_fracture else 'non-fracture',
                            'automatic': 'fracture' if auto_is_fracture else 'non-fracture',
                            'probability': probability,
                            'num_faces': region_result['num_faces']
                        })
                else:
                    # Not manually labeled: Purple
                    color = [0.8, 0.0, 0.8]  # Purple for unlabeled
                
                # Apply color to region vertices
                for vertex_idx in region_vertices:
                    if vertex_idx < num_vertices:
                        colors[vertex_idx] = color
        
        viz_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Print comparison summary
        total_labeled = len(labeled_data['fracture_regions']) + len(labeled_data['non_fracture_regions'])
        total_regions = len(classifications)
        
        print(f"Label Comparison Summary:")
        print(f"  Total regions: {total_regions}")
        print(f"  Manually labeled: {total_labeled}")
        print(f"  Agreement regions: {len(agreement_regions)}")
        print(f"  Disagreement regions: {len(disagreement_regions)}")
        
        if total_labeled > 0:
            agreement_rate = len(agreement_regions) / total_labeled * 100
            print(f"  Agreement rate: {agreement_rate:.1f}%")
        
        if agreement_regions:
            print(f"\nAgreement Regions:")
            fracture_agreements = [r for r in agreement_regions if r['classification'] == 'fracture']
            non_fracture_agreements = [r for r in agreement_regions if r['classification'] == 'non-fracture']
            
            if fracture_agreements:
                print(f"  🔴 Fracture agreement (Green): {len(fracture_agreements)} regions")
                for region in fracture_agreements:
                    print(f"    Region {region['region_id']+1}: {region['probability']:.3f} confidence")
            
            if non_fracture_agreements:
                print(f"  🔵 Non-fracture agreement (Cyan): {len(non_fracture_agreements)} regions")
                for region in non_fracture_agreements:
                    print(f"    Region {region['region_id']+1}: {region['probability']:.3f} confidence")
        
        if disagreement_regions:
            print(f"\nDisagreement Regions (Yellow):")
            for region in disagreement_regions:
                print(f"  Region {region['region_id']+1}: Manual={region['manual']}, "
                      f"Auto={region['automatic']} ({region['probability']:.3f} confidence)")
        
        print(f"\nColor Legend:")
        print(f"  🟢 Green = Fracture agreement (manual + automatic)")
        print(f"  🔵 Cyan = Non-fracture agreement (manual + automatic)")
        print(f"  🟡 Yellow = Disagreement (manual ≠ automatic)")
        print(f"  🟣 Purple = Not manually labeled")
        print(f"  ⚫ Grey = Error or unclassified")
        
        print(f"\nControls:")
        print(f"  - Mouse: Rotate, zoom, pan")
        print(f"  - Close window to continue")
        
        # Show the visualization
        o3d.visualization.draw_geometries(
            [viz_mesh, coordinate_frame],
            window_name="Manual vs Automatic Classification Comparison",
            width=1200,
            height=800,
            point_show_normal=False,
            mesh_show_back_face=True
        )
        
        return {
            'agreement_regions': agreement_regions,
            'disagreement_regions': disagreement_regions,
            'agreement_rate': agreement_rate if total_labeled > 0 else 0.0
        }

def run_supervised_fracture_detection(mesh, segmentation_results):
    print("\n" + "="*80)
    print("SUPERVISED FRACTURE DETECTION")
    print("="*80)
    selector = FractureSurfaceSelector(mesh, segmentation_results)
    if not selector.start_interactive_selection():
        print("Surface selection failed or was cancelled.")
        return None
    labeled_data = selector.get_labeled_data()
    # Step 2: Parameter optimization
    optimizer = ParameterOptimizer(mesh, segmentation_results, labeled_data)
    optimization_result = optimizer.optimize_parameters()
    
    if optimization_result[0] is None:
        print("❌ Parameter optimization failed completely. Cannot proceed with classification.")
        return None
    
    optimal_params, trained_model, optimization_results = optimization_result
    
    # Step 2.5: Analyze optimization results
    print(f"\nAnalyzing parameter optimization results...")
    analysis_results = optimizer.analyze_optimization_results(optimization_results)
    
    # Step 3: Classify all regions
    classifier = FractureClassifier(optimal_params, trained_model)
    classifications = classifier.classify_regions(mesh, segmentation_results)
    
    # Step 4: Visualize classification results
    print(f"\nVisualizing classification results...")
    viz_results = classifier.visualize_classification_results(mesh, segmentation_results, classifications)
    
    # Step 4.5: Visualize label comparison
    print(f"\nVisualizing label comparison...")
    comparison_results = classifier.visualize_label_comparison(mesh, segmentation_results, classifications, labeled_data)
    
    # Step 5: Print results
    print(f"\n" + "="*80)
    print("CLASSIFICATION RESULTS")
    print("="*80)
    fracture_regions = []
    non_fracture_regions = []
    for classification in classifications:
        region_id = classification['region_id']
        is_fracture = classification['is_fracture']
        probability = classification['fracture_probability']
        if is_fracture:
            fracture_regions.append(region_id)
            print(f"Region {region_id+1}: FRACTURED (probability: {probability:.3f})")
        else:
            non_fracture_regions.append(region_id)
            print(f"Region {region_id+1}: Non-fractured (probability: {probability:.3f})")
    print(f"\nSummary:")
    print(f"  Fractured regions: {len(fracture_regions)}")
    print(f"  Non-fractured regions: {len(non_fracture_regions)}")
    print(f"  Total regions: {len(classifications)}")
    results = {
        'optimal_parameters': optimal_params,
        'optimization_results': optimization_results,
        'classifications': classifications,
        'fracture_regions': fracture_regions,
        'non_fracture_regions': non_fracture_regions,
        'labeled_data': labeled_data
    }
    return results 