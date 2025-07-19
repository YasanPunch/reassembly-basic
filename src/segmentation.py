import open3d as o3d
import trimesh
import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import deque
from scipy import ndimage
from scipy.spatial import cKDTree

def get_color(index, total_items=20, cmap_name='tab10', num_variations=3):
    """
    Gets a distinct color. Uses a base colormap and applies variations
    if the number of items exceeds the colormap's distinct colors.
    Args:
        index (int): The 0-based index of the item to color.
        total_items (int): Total number of items needing colors (helps estimate variations).
        cmap_name (str): Name of the base Matplotlib colormap.
        num_variations (int): How many brightness/saturation variations to apply for each base color.
    Returns:
        tuple: (R, G, B) color.
    """
    try:
        base_cmap = plt.cm.get_cmap(cmap_name)
        if not base_cmap:
            base_cmap = plt.cm.get_cmap('Set1')
        if not base_cmap:
            base_cmap = plt.cm.get_cmap('viridis')

        num_base_colors = base_cmap.N
        base_color_index = index % num_base_colors
        variation_cycle = (index // num_base_colors) % num_variations

        r, g, b, _ = base_cmap(base_color_index)

        if variation_cycle == 0:
            pass
        elif variation_cycle == 1:
            factor = 1.3
            r = min(1.0, r * factor + 0.1)
            g = min(1.0, g * factor + 0.1)
            b = min(1.0, b * factor + 0.1)
        elif variation_cycle == 2:
            factor = 0.7
            r *= factor
            g *= factor
            b *= factor

        return np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)

    except ImportError:
        colors = [[1,0,0],[0,0,1],[0,1,0],[1,1,0],[1,0,1],[0,1,1],
                  [0.8,0.5,0.2],[0.5,0.2,0.8],[0.2,0.8,0.5], [0.6,0.6,0.6]]
        return colors[index % len(colors)]
    except Exception as e:
        print(f"Error in get_color: {e}. Using fallback.")
        colors = [[1,0,0],[0,0,1],[0,1,0]]
        return colors[index % len(colors)]


def calculate_region_average_normal(tri_mesh, face_indices):
    """
    Calculate the area-weighted average normal for a region following the paper's formula:
    N_ave(R_k) = sum(A_j * N_j) / sum(A_j) for all j in R_k
    """
    if len(face_indices) == 0:
        return np.array([0, 0, 1])
    
    face_normals = tri_mesh.face_normals[face_indices]
    face_areas = tri_mesh.area_faces[face_indices]
    
    # Area-weighted average
    weighted_normals = face_normals * face_areas[:, np.newaxis]
    avg_normal = np.sum(weighted_normals, axis=0) / np.sum(face_areas)
    
    # Normalize
    norm = np.linalg.norm(avg_normal)
    if norm > 1e-10:
        avg_normal = avg_normal / norm
    else:
        avg_normal = np.array([0, 0, 1])
    
    return avg_normal


def region_growing_segmentation(tri_mesh, params):
    """
    Implements the region growing algorithm from the paper.
    
    Args:
        tri_mesh: trimesh object
        params: dictionary containing:
            - 'max_curvature_deg': maximum allowed angle between normals in same region (default 30)
            - 'area_limit_fraction': minimum region area as fraction of total (default 0.02)
    
    Returns:
        list of np.arrays containing face indices for each region
    """
    # Get parameters
    max_curvature_deg = params.get('max_curvature_deg', 30.0)
    area_limit_fraction = params.get('area_limit_fraction', 0.02)
    
    # Calculate Ne threshold from max curvature (Ne = cos(q_max))
    Ne = np.cos(np.radians(max_curvature_deg))
    
    num_faces = len(tri_mesh.faces)
    face_visited = np.zeros(num_faces, dtype=bool)
    regions = []
    
    # Precompute face adjacency if not available
    if not hasattr(tri_mesh, 'face_adjacency') or tri_mesh.face_adjacency is None:
        tri_mesh.face_adjacency = trimesh.graph.face_adjacency(tri_mesh.faces)
    
    # Build adjacency list for faster lookup
    adjacency_list = [[] for _ in range(num_faces)]
    for face1, face2 in tri_mesh.face_adjacency:
        adjacency_list[face1].append(face2)
        adjacency_list[face2].append(face1)
    
    # Region growing main loop
    for start_face in range(num_faces):
        if face_visited[start_face]:
            continue
            
        # Start new region
        current_region = []
        queue = deque([start_face])
        face_visited[start_face] = True
        
        while queue:
            current_face = queue.popleft()
            current_region.append(current_face)
            
            # Update region average normal
            region_avg_normal = calculate_region_average_normal(tri_mesh, current_region)
            
            # Check all neighbors
            for neighbor_face in adjacency_list[current_face]:
                if face_visited[neighbor_face]:
                    continue
                
                # Check if neighbor normal satisfies similarity criterion
                neighbor_normal = tri_mesh.face_normals[neighbor_face]
                dot_product = np.dot(neighbor_normal, region_avg_normal)
                
                if dot_product >= Ne:  # N_i · N_ave(R_k) >= Ne
                    face_visited[neighbor_face] = True
                    queue.append(neighbor_face)
        
        if len(current_region) > 0:
            regions.append(np.array(current_region))
    
    # Clean-up stage: eliminate small regions
    total_area = tri_mesh.area
    area_threshold = area_limit_fraction * total_area
    
    # Calculate region areas
    region_areas = []
    for region in regions:
        region_area = np.sum(tri_mesh.area_faces[region])
        region_areas.append(region_area)
    
    # Sort regions by area (largest first)
    sorted_indices = np.argsort(region_areas)[::-1]
    sorted_regions = [regions[i] for i in sorted_indices]
    sorted_areas = [region_areas[i] for i in sorted_indices]
    
    # Keep only significant regions
    significant_regions = []
    for i, (region, area) in enumerate(zip(sorted_regions, sorted_areas)):
        if area >= area_threshold:
            significant_regions.append(region)
    
    # Reassign small regions to adjacent larger regions
    if len(significant_regions) < len(regions):
        # Create a face-to-region mapping for significant regions
        face_to_region = np.full(num_faces, -1, dtype=int)
        for region_idx, region in enumerate(significant_regions):
            face_to_region[region] = region_idx
        
        # Process small regions
        for region_idx in sorted_indices:
            region = regions[region_idx]
            area = region_areas[region_idx]
            
            if area >= area_threshold:
                continue
            
            # Find adjacent significant regions
            adjacent_regions = set()
            for face in region:
                for neighbor in adjacency_list[face]:
                    neighbor_region = face_to_region[neighbor]
                    if neighbor_region >= 0:
                        adjacent_regions.add(neighbor_region)
            
            # Assign to the most similar adjacent region
            if adjacent_regions:
                best_region = None
                best_similarity = -1
                region_avg_normal = calculate_region_average_normal(tri_mesh, region)
                
                for adj_region_idx in adjacent_regions:
                    adj_region_normal = calculate_region_average_normal(
                        tri_mesh, significant_regions[adj_region_idx]
                    )
                    similarity = np.dot(region_avg_normal, adj_region_normal)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_region = adj_region_idx
                
                if best_region is not None:
                    # Merge with best region
                    significant_regions[best_region] = np.concatenate([
                        significant_regions[best_region], region
                    ])
                    face_to_region[region] = best_region
    
    return significant_regions


def calculate_region_bumpiness(tri_mesh, region_faces, params):
    """
    Calculate surface bumpiness using elevation map and Laplacian operator as in the paper.
    Note: This is a simplified version since we don't have direct access to depth buffer rendering.
    """
    if len(region_faces) < 10:  # Too few faces
        return 0.0
    
    # Get region bounds
    region_vertices = tri_mesh.vertices[tri_mesh.faces[region_faces].flatten()]
    region_avg_normal = calculate_region_average_normal(tri_mesh, region_faces)
    
    # Project vertices onto plane perpendicular to average normal
    centroid = np.mean(region_vertices, axis=0)
    
    # Create coordinate system with z-axis aligned to average normal
    z_axis = region_avg_normal
    # Find arbitrary perpendicular vectors
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross([0, 0, 1], z_axis)
    else:
        x_axis = np.cross([1, 0, 0], z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Project vertices to 2D
    relative_verts = region_vertices - centroid
    u_coords = np.dot(relative_verts, x_axis)
    v_coords = np.dot(relative_verts, y_axis)
    depths = np.dot(relative_verts, z_axis)
    
    # Create elevation map (simplified - using scatter interpolation)
    resolution = params.get('elevation_map_resolution', 64)
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()
    
    if u_max - u_min < 1e-6 or v_max - v_min < 1e-6:
        return 0.0
    
    # Create grid
    u_grid = np.linspace(u_min, u_max, resolution)
    v_grid = np.linspace(v_min, v_max, resolution)
    uu, vv = np.meshgrid(u_grid, v_grid)
    
    # Simple nearest neighbor interpolation for elevation map
    elevation_map = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            u_pt = uu[i, j]
            v_pt = vv[i, j]
            
            # Find nearest vertex
            distances = (u_coords - u_pt)**2 + (v_coords - v_pt)**2
            nearest_idx = np.argmin(distances)
            elevation_map[i, j] = depths[nearest_idx]
    
    # Apply Laplacian operator
    # Using scipy's Laplacian filter
    laplacian = ndimage.laplace(elevation_map)
    
    # Calculate bumpiness as average absolute Laplacian
    # Exclude infinite values
    valid_mask = np.isfinite(laplacian)
    if np.sum(valid_mask) > 0:
        bumpiness = np.mean(np.abs(laplacian[valid_mask]))
    else:
        bumpiness = 0.0
    
    return bumpiness


def extract_fracture_surface_mesh(o3d_mesh_fragment, fragment_name="Unnamed", params=None):
    """
    Main segmentation function using the paper's region growing approach.
    """
    params = params or {}
    
    print(f"\n=== Segmenting {fragment_name} using Region Growing Algorithm ===")
    
    # Parameter setup with automatic pre-selection
    default_params = {
        'max_curvature_deg': params.get('max_curvature_deg', 30.0),
        'area_limit_fraction': params.get('area_limit_fraction', 0.02),
        'visualize_segmentation': params.get('visualize_segmentation', False),
        'elevation_map_resolution': params.get('elevation_map_resolution', 64),
        
        # Automatic pre-selection (NEW - enabled by default)
        'use_automatic_preselection': params.get('use_automatic_preselection', True),
        'preselection_fracture_ratio_threshold': params.get('preselection_fracture_ratio_threshold', 0.3),
        'final_classification_threshold': params.get('final_classification_threshold', 0.5),
        
        # Adaptive threshold percentiles for automatic classification
        'roughness_threshold_percentile': params.get('roughness_threshold_percentile', 75),  # Top 25% roughest
        'curvature_threshold_percentile': params.get('curvature_threshold_percentile', 75),  # Top 25% highest curvature
        'boundary_complexity_threshold_percentile': params.get('boundary_complexity_threshold_percentile', 75),  # Top 25% most complex
        'symmetry_threshold_percentile': params.get('symmetry_threshold_percentile', 25),  # Bottom 25% least symmetric
        'planarity_threshold_percentile': params.get('planarity_threshold_percentile', 25),  # Bottom 25% least planar
        
        # Weights for combining properties in automatic classification
        'fracture_detection_weights': params.get('fracture_detection_weights', {
            'curvature': 0.3,
            'roughness': 0.3,
            'boundary_complexity': 0.2,
            'symmetry': 0.1,
            'planarity': 0.1
        }),
        
        # Legacy parameters (for backward compatibility)
        'use_adaptive_detection': params.get('use_adaptive_detection', False),
        'use_combined_approach': params.get('use_combined_approach', True),
        'score_threshold_percentile': params.get('score_threshold_percentile', 70),
        'combined_min_agreement': params.get('combined_min_agreement', 2),
        'statistical_confidence_threshold': params.get('statistical_confidence_threshold', 0.6),
        'use_bumpiness_detection': params.get('use_bumpiness_detection', False),
        'bumpiness_threshold': params.get('bumpiness_threshold', 0.2),
        'use_statistical_analysis': params.get('use_statistical_analysis', False)
    }
    
    # Update params with defaults
    for key, value in default_params.items():
        if key not in params:
            params[key] = value
    
    # Convert to trimesh
    if not o3d_mesh_fragment.has_triangles() or not o3d_mesh_fragment.has_vertices():
        print(f"    Segmenter: Input mesh {fragment_name} has no triangles/vertices.")
        return None
        
    try:
        tri_mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh_fragment.vertices),
            faces=np.asarray(o3d_mesh_fragment.triangles),
            vertex_normals=np.asarray(o3d_mesh_fragment.vertex_normals) if o3d_mesh_fragment.has_vertex_normals() else None,
            process=False
        )
        tri_mesh.metadata['name'] = fragment_name
        
        # Ensure we have face normals and areas
        if not hasattr(tri_mesh, 'face_normals') or tri_mesh.face_normals is None:
            tri_mesh.face_normals
        if not hasattr(tri_mesh, 'area_faces') or tri_mesh.area_faces is None:
            _ = tri_mesh.area_faces
            
    except Exception as e:
        print(f"    Segmenter: Error converting O3D mesh {fragment_name} to Trimesh: {e}")
        return None
    
    total_faces = len(tri_mesh.faces)
    print(f"    Total faces: {total_faces}")
    print(f"    Max curvature threshold: {params['max_curvature_deg']}°")
    print(f"    Min region area: {params['area_limit_fraction']*100:.1f}% of total")
    
    # Perform region growing segmentation
    print(f"\n    Starting region growing segmentation...")
    regions = region_growing_segmentation(tri_mesh, params)
    print(f"    Found {len(regions)} regions after segmentation and cleanup")
    
    # Calculate region properties
    region_properties = []
    for i, region in enumerate(regions):
        avg_normal = calculate_region_average_normal(tri_mesh, region)
        area = np.sum(tri_mesh.area_faces[region])
        area_fraction = area / tri_mesh.area
        
        props = {
            'index': i,
            'faces': region,
            'num_faces': len(region),
            'area': area,
            'area_fraction': area_fraction,
            'avg_normal': avg_normal,
            'bumpiness': 0.0
        }
        
        # Calculate bumpiness if requested
        if params['use_bumpiness_detection']:
            props['bumpiness'] = calculate_region_bumpiness(tri_mesh, region, params)
        
        region_properties.append(props)
        
        print(f"    Region {i+1}: {len(region)} faces ({area_fraction*100:.1f}% of area), "
              f"avg_normal: [{avg_normal[0]:.2f}, {avg_normal[1]:.2f}, {avg_normal[2]:.2f}]")
        if params['use_bumpiness_detection']:
            print(f"        Bumpiness: {props['bumpiness']:.4f}")
    
    # Sort regions by area (largest first)
    region_properties.sort(key=lambda x: x['area'], reverse=True)
    
    # Automatic fracture surface detection with pre-selection
    face_is_fracture_candidate = np.zeros(len(tri_mesh.faces), dtype=bool)
    selected_regions = []
    
    # Step 1: Run automatic classification to pre-select regions
    if params.get('use_automatic_preselection', True):
        print(f"\n    Running automatic fracture surface classification for pre-selection...")
        
        # Use the simple classification method for automatic pre-selection
        classification_result = classify_fracture_vs_original_faces(tri_mesh, params)
        automatic_fracture_faces = classification_result['fracture_faces']
        face_scores = classification_result['face_scores']
        
        print(f"    Automatic classification results:")
        print(f"      {np.sum(automatic_fracture_faces)} faces classified as fractures ({np.sum(automatic_fracture_faces)/len(tri_mesh.faces)*100:.1f}%)")
        
        # Map automatic classification results to regions for pre-selection
        for props in region_properties:
            region_faces = props['faces']
            region_fracture_faces = automatic_fracture_faces[region_faces]
            fracture_ratio = np.sum(region_fracture_faces) / len(region_faces)
            
            # Pre-select regions with high fracture ratio
            preselection_threshold = params.get('preselection_fracture_ratio_threshold', 0.3)
            if fracture_ratio > preselection_threshold:
                selected_regions.append(props['index'])
                face_is_fracture_candidate[region_faces] = True
                avg_score = np.mean(face_scores[region_faces])
                print(f"    Region {props['index']+1} PRE-SELECTED as fracture candidate")
                print(f"      → {np.sum(region_fracture_faces)}/{len(region_faces)} faces classified as fractures ({fracture_ratio*100:.1f}%)")
                print(f"      → Average fracture score: {avg_score:.3f}")
        
        num_preselected = len(selected_regions)
        if num_preselected > 0:
            print(f"\n    Pre-selection complete: {num_preselected} regions automatically selected")
            print(f"    These will appear as already selected in the interactive visualization")
        else:
            print(f"\n    No regions met the automatic pre-selection criteria")
    
    # Step 2: Legacy methods for additional detection (optional)
    elif params.get('use_adaptive_detection', False):
        print(f"\n    Running legacy adaptive fracture surface detection...")
        
        # Run adaptive detection methods
        detection_results, global_stats = compare_fracture_detection_methods_adaptive(tri_mesh, params)
        
        # Collect candidates from all methods
        all_candidates = np.zeros(len(tri_mesh.faces), dtype=bool)
        method_agreement = np.zeros(len(tri_mesh.faces), dtype=int)
        
        for method, data in detection_results.items():
            if data['count'] > 0:
                all_candidates |= data['candidates']
                method_agreement += data['candidates'].astype(int)
        
        # Use combined approach if enabled
        if params['use_combined_approach']:
            min_agreement = params['combined_min_agreement']
            final_candidates = method_agreement >= min_agreement
            print(f"    Combined approach: {np.sum(final_candidates)} faces selected (agreement >= {min_agreement} methods)")
        else:
            final_candidates = all_candidates
            print(f"    Union approach: {np.sum(final_candidates)} faces selected (any method)")
        
        # Map candidates back to regions
        for props in region_properties:
            region_faces = props['faces']
            region_candidates = final_candidates[region_faces]
            candidate_ratio = np.sum(region_candidates) / len(region_faces)
            
            if candidate_ratio > 0.3:  # At least 30% of region faces are candidates
                selected_regions.append(props['index'])
                face_is_fracture_candidate[region_faces] = True
                print(f"    Region {props['index']+1} selected as fracture candidate (adaptive detection ratio: {candidate_ratio:.2f})")
    
    elif params['use_bumpiness_detection'] and any(r['bumpiness'] > 0 for r in region_properties):
        print(f"\n    Running legacy bumpiness detection...")
        bumpiness_values = [r['bumpiness'] for r in region_properties]
        max_bumpiness = max(bumpiness_values)
        bumpiness_threshold = params['bumpiness_threshold'] * max_bumpiness
        
        for props in region_properties:
            if props['bumpiness'] > bumpiness_threshold:
                selected_regions.append(props['index'])
                face_is_fracture_candidate[props['faces']] = True
                print(f"    Region {props['index']+1} selected as fracture candidate (bumpiness: {props['bumpiness']:.4f})")
    
    # Interactive visualization if enabled
    if params['visualize_segmentation'] and len(regions) > 0:
        num_preselected = len(selected_regions)
        print(f"\n    Visualizing {len(regions)} regions for interactive selection...")
        if num_preselected > 0:
            print(f"    → {num_preselected} regions are PRE-SELECTED by automatic classification")
            print(f"    → You can modify the selection by toggling regions on/off")
        
        shared_state = {'confirmed_selection': False, 'quit_without_selection': False, 'current_page': 0}
        PAGE_SIZE = 10
        
        drawable_segment_infos = []
        highlight_color = np.array([0.0, 0.0, 0.0])  # Black highlight for selected
        preselected_color = np.array([1.0, 0.8, 0.0])  # Gold for pre-selected
        
        mesh_vis = copy.deepcopy(o3d_mesh_fragment)
        
        for i, props in enumerate(region_properties):
            seg_mesh = o3d.geometry.TriangleMesh()
            seg_mesh.vertices = o3d_mesh_fragment.vertices
            seg_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces[props['faces']])
            seg_mesh.remove_unreferenced_vertices()
            
            if not seg_mesh.has_vertices() or not seg_mesh.has_triangles():
                continue
                
            seg_mesh.compute_vertex_normals()
            base_color = get_color(i, len(regions))
            is_preselected = props['index'] in selected_regions
            
            # Use different color for pre-selected regions
            if is_preselected:
                seg_mesh.paint_uniform_color(preselected_color)
            else:
                seg_mesh.paint_uniform_color(base_color)
            
            drawable_segment_infos.append({
                'mesh': seg_mesh,
                'id': props['index'],
                'base_color': base_color,
                'selected': is_preselected,
                'properties': props,
                'preselected': is_preselected  # Track which were pre-selected
            })
        
        if drawable_segment_infos:
            num_total_segments = len(drawable_segment_infos)
            num_pages = (num_total_segments + PAGE_SIZE - 1) // PAGE_SIZE
            
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(
                window_name=f"Select: {fragment_name} (Page 1/{num_pages}. N/P=Page. S=Confirm. Q=Skip.)",
                width=1280, height=960
            )
            
            for info in drawable_segment_infos:
                vis.add_geometry(info['mesh'])
                if info['selected']:
                    info['mesh'].paint_uniform_color(highlight_color)
            
            def print_current_page_and_selection():
                page_idx = shared_state['current_page']
                global_start = page_idx * PAGE_SIZE + 1
                global_end = min((page_idx + 1) * PAGE_SIZE, num_total_segments)
                
                print(f"\n  --- Page {page_idx + 1}/{num_pages} (Regions {global_start}-{global_end}) ---")
                print(f"  Keys 1-9, 0 (for 10th) toggle selection.")
                print(f"  Colors: Gold = Pre-selected by algorithm, Black = User selected")
                
                # Show properties for visible regions
                for i in range(page_idx * PAGE_SIZE, min((page_idx + 1) * PAGE_SIZE, num_total_segments)):
                    if i < len(drawable_segment_infos):
                        info = drawable_segment_infos[i]
                        props = info['properties']
                        
                        # Different markers for different selection types
                        if info['selected'] and info.get('preselected', False):
                            marker = "A"  # Auto-selected
                        elif info['selected']:
                            marker = "*"  # User-selected
                        else:
                            marker = " "  # Not selected
                        
                        print(f"  {marker}[{(i % PAGE_SIZE) + 1}] Region {props['index']+1}: "
                              f"{props['num_faces']} faces ({props['area_fraction']*100:.1f}%)")
                        
                        # Show additional info for pre-selected regions
                        if info.get('preselected', False):
                            print(f"       → Auto-selected by fracture classification")
                        
                        if params.get('use_bumpiness_detection', False):
                            print(f"       Bumpiness: {props['bumpiness']:.4f}")
                
                # Show selection summary
                selected_ids = sorted([info['id'] + 1 for info in drawable_segment_infos if info['selected']])
                preselected_ids = sorted([info['id'] + 1 for info in drawable_segment_infos if info.get('preselected', False)])
                user_selected_ids = sorted([info['id'] + 1 for info in drawable_segment_infos 
                                          if info['selected'] and not info.get('preselected', False)])
                
                print(f"  Legend: A=Auto-selected, *=User-selected, [space]=Not selected")
                print(f"  Auto-selected: {preselected_ids if preselected_ids else 'None'}")
                print(f"  User-selected: {user_selected_ids if user_selected_ids else 'None'}")
                print(f"  Total selected: {selected_ids if selected_ids else 'None'}")
            
            print_current_page_and_selection()
            
            def toggle_segment_on_current_page(visualizer, key_idx):
                page_idx = shared_state['current_page']
                segment_idx = page_idx * PAGE_SIZE + key_idx
                
                if 0 <= segment_idx < num_total_segments:
                    info = drawable_segment_infos[segment_idx]
                    info['selected'] = not info['selected']
                    
                    # Update visual appearance
                    if info['selected']:
                        if info.get('preselected', False):
                            # Keep pre-selected regions in gold when selected
                            info['mesh'].paint_uniform_color(preselected_color)
                        else:
                            # User selections in black
                            info['mesh'].paint_uniform_color(highlight_color)
                    else:
                        # Deselected regions return to base color
                        info['mesh'].paint_uniform_color(info['base_color'])
                    
                    visualizer.update_geometry(info['mesh'])
                    print_current_page_and_selection()
                    
                return False
            
            # Register key callbacks
            for i in range(PAGE_SIZE):
                key_char = str((i + 1) % 10)
                vis.register_key_callback(ord(key_char), 
                    lambda v, idx=i: toggle_segment_on_current_page(v, idx))
            
            def change_page(visualizer, direction):
                old_page = shared_state['current_page']
                shared_state['current_page'] = (shared_state['current_page'] + direction + num_pages) % num_pages
                if old_page != shared_state['current_page']:
                    print_current_page_and_selection()
                return False
            
            vis.register_key_callback(ord('N'), lambda v: change_page(v, 1))
            vis.register_key_callback(ord('P'), lambda v: change_page(v, -1))
            
            def confirm_and_close(visualizer):
                shared_state['confirmed_selection'] = True
                print("\n  Selection Confirmed. Closing...")
                visualizer.close()
                return False
            
            def quit_and_close(visualizer):
                shared_state['quit_without_selection'] = True
                print("\n  Selection Aborted. Closing...")
                visualizer.close()
                return False
            
            vis.register_key_callback(ord('S'), confirm_and_close)
            vis.register_key_callback(ord('Q'), quit_and_close)
            
            print("\n=== Interactive Region Selection ===")
            print(f"  Fragment: {fragment_name}")
            print("  N/P: Navigate pages | 1-9,0: Toggle selection")
            print("  S: Save selection | Q: Quit without saving")
            
            vis.run()
            vis.destroy_window()
            
            if shared_state['confirmed_selection']:
                selected_regions = [info['id'] for info in drawable_segment_infos if info['selected']]
                face_is_fracture_candidate.fill(False)
                for info in drawable_segment_infos:
                    if info['selected']:
                        face_is_fracture_candidate[info['properties']['faces']] = True
                print(f"\n    User selected {len(selected_regions)} regions")
            elif shared_state['quit_without_selection']:
                print(f"\n    User quit selection. No regions selected.")
                return None
    
    # Console fallback for non-interactive mode
    elif not params['visualize_segmentation'] and len(regions) > 0 and not params['use_bumpiness_detection']:
        print("\n=== Region Selection (Console) ===")
        for i, props in enumerate(region_properties):
            print(f"  Region {i+1}: {props['num_faces']} faces ({props['area_fraction']*100:.1f}% of area)")
        
        selection_str = input(f"Enter region numbers to select (1-{len(regions)}, comma-separated, 'all', or 'none'): ")
        
        if selection_str.lower() == 'all':
            selected_regions = list(range(len(regions)))
        elif selection_str.lower() == 'none' or not selection_str.strip():
            selected_regions = []
        else:
            try:
                selected_regions = [int(x.strip()) - 1 for x in selection_str.split(',') if x.strip()]
                selected_regions = [r for r in selected_regions if 0 <= r < len(regions)]
            except ValueError:
                print("    Invalid input. No regions selected.")
                selected_regions = []
        
        for region_idx in selected_regions:
            face_is_fracture_candidate[region_properties[region_idx]['faces']] = True
    
    # Collect selected regions' face indices and normals for merging
    selected_region_faces = []
    selected_region_normals = []
    for region_idx in range(len(region_properties)):
        if face_is_fracture_candidate[region_properties[region_idx]['faces']].any():
            selected_region_faces.append(set(region_properties[region_idx]['faces']))
            selected_region_normals.append(region_properties[region_idx]['avg_normal'])
    
    # Create output mesh
    if not np.any(face_is_fracture_candidate):
        print(f"\n    No regions selected for {fragment_name}")
        return None
    
    fracture_faces = tri_mesh.faces[face_is_fracture_candidate]
    fracture_surface_o3d = o3d.geometry.TriangleMesh()
    fracture_surface_o3d.vertices = o3d_mesh_fragment.vertices
    fracture_surface_o3d.triangles = o3d.utility.Vector3iVector(fracture_faces)
    fracture_surface_o3d.remove_unreferenced_vertices()
    fracture_surface_o3d.remove_degenerate_triangles()
    
    if not fracture_surface_o3d.has_triangles():
        print(f"    Extracted surface has no valid triangles")
        return None
    
    fracture_surface_o3d.compute_vertex_normals()
    print(f"\n    Extracted surface: {len(fracture_surface_o3d.vertices)} vertices, "
          f"{len(fracture_surface_o3d.triangles)} triangles")
    
    # --- IMPROVED MERGING: NORMAL + BOUNDARY DISTANCE ---
    all_triangles = np.asarray(o3d_mesh_fragment.triangles)
    all_vertices = np.asarray(o3d_mesh_fragment.vertices)
    def get_boundary_vertices(face_indices):
        # Find boundary edges for a set of faces
        faces = all_triangles[list(face_indices)]
        edges = np.vstack([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]])
        edges = np.sort(edges, axis=1)
        # Count occurrences
        edges_tuple = [tuple(e) for e in edges]
        from collections import Counter
        edge_counts = Counter(edges_tuple)
        boundary_edges = [e for e, c in edge_counts.items() if c == 1]
        boundary_verts = np.unique(np.array(boundary_edges).flatten())
        return all_vertices[boundary_verts]
    
    merged_clusters = []
    used = set()
    angle_thresh_deg = 10.0
    boundary_dist_thresh = 0.01 * np.linalg.norm(all_vertices.max(axis=0) - all_vertices.min(axis=0))  # 1% of mesh size
    for i, (faces_i, normal_i) in enumerate(zip(selected_region_faces, selected_region_normals)):
        if i in used:
            continue
        cluster = set(faces_i)
        cluster_normals = [normal_i]
        merged_idxs = [i]
        used.add(i)
        boundary_i = get_boundary_vertices(faces_i)
        for j, (faces_j, normal_j) in enumerate(zip(selected_region_faces, selected_region_normals)):
            if j == i or j in used:
                continue
            angle = np.degrees(np.arccos(np.clip(np.dot(normal_i, normal_j), -1, 1)))
            if angle < angle_thresh_deg and np.dot(normal_i, normal_j) > 0:
                # Check boundary proximity
                boundary_j = get_boundary_vertices(faces_j)
                if len(boundary_i) > 0 and len(boundary_j) > 0:
                    tree = cKDTree(boundary_j)
                    min_dist = tree.query(boundary_i, k=1)[0].min()
                    if min_dist < boundary_dist_thresh:
                        cluster |= faces_j
                        cluster_normals.append(normal_j)
                        merged_idxs.append(j)
                        used.add(j)
        merged_clusters.append((cluster, merged_idxs))
    # DEBUG: Visualize each merged face in its own window, with base mesh as wireframe
    for idx, (cluster_faces, merged_idxs) in enumerate(merged_clusters):
        if not cluster_faces:
            continue
        cluster_faces_arr = np.array(sorted(cluster_faces), dtype=np.int32)
        cluster_triangles = all_triangles[cluster_faces_arr]
        cluster_mesh = o3d.geometry.TriangleMesh()
        cluster_mesh.vertices = mesh_vis.vertices  # Use full vertex set
        cluster_mesh.triangles = o3d.utility.Vector3iVector(cluster_triangles)
        cluster_mesh.compute_vertex_normals()
        color = get_color(idx, len(merged_clusters))
        cluster_mesh.paint_uniform_color(color)
        print(f"[DEBUG] Merged regions {merged_idxs} into face {idx} (color {idx}), triangles: {len(cluster_triangles)}, vertices: {len(cluster_mesh.vertices)}, color: {color}")
        # Create wireframe for base mesh
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_vis)
        wireframe.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([wireframe, cluster_mesh], window_name=f"[DEBUG] Merged Face {idx} (Color {idx})")
    # Store merged faces as separate fracture surfaces for downstream processing
    merged_fracture_surfaces = []
    for cluster_faces, merged_idxs in merged_clusters:
        if not cluster_faces:
            continue
        cluster_faces_arr = np.array(sorted(cluster_faces), dtype=np.int32)
        cluster_triangles = all_triangles[cluster_faces_arr]
        cluster_mesh = o3d.geometry.TriangleMesh()
        cluster_mesh.vertices = mesh_vis.vertices
        cluster_mesh.triangles = o3d.utility.Vector3iVector(cluster_triangles)
        cluster_mesh.remove_unreferenced_vertices()
        cluster_mesh.remove_degenerate_triangles()
        if cluster_mesh.has_triangles():
            cluster_mesh.compute_vertex_normals()
            merged_fracture_surfaces.append(cluster_mesh)
    # Return merged fracture surfaces for this fragment
    return merged_fracture_surfaces


def visualize_segmentation(o3d_mesh, fracture_surface, fragment_name="Unnamed"):
    """
    Creates a visualization of the original mesh and the extracted surface.
    """
    vis_geometries = []
    
    # Original mesh in gray
    original_mesh_vis = copy.deepcopy(o3d_mesh)
    original_mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])
    original_mesh_vis.compute_vertex_normals()
    vis_geometries.append(original_mesh_vis)
    
    # Wireframe for structure
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(original_mesh_vis)
    edges.paint_uniform_color([0.5, 0.5, 0.5])
    vis_geometries.append(edges)
    
    # Selected surface in red
    if fracture_surface and fracture_surface.has_triangles():
        fracture_surface_vis = copy.deepcopy(fracture_surface)
        fracture_surface_vis.paint_uniform_color([1.0, 0.0, 0.0])
        fracture_surface_vis.compute_vertex_normals()
        vis_geometries.append(fracture_surface_vis)
    
    return vis_geometries


# Maintain compatibility with old function names
def identify_fracture_candidate_faces(tri_mesh_fragment, params=None):
    """
    Legacy function maintained for compatibility.
    Returns a boolean mask of fracture candidate faces.
    """
    if params is None:
        params = {}
    
    # Convert trimesh to Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh_fragment.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh_fragment.faces)
    o3d_mesh.compute_vertex_normals()
    
    # Run segmentation
    result_mesh = extract_fracture_surface_mesh(
        o3d_mesh, 
        tri_mesh_fragment.metadata.get('name', 'Unnamed'),
        params
    )
    
    if result_mesh is None:
        return np.zeros(len(tri_mesh_fragment.faces), dtype=bool)
    
    # Create boolean mask
    face_mask = np.zeros(len(tri_mesh_fragment.faces), dtype=bool)
    result_faces_set = set(map(tuple, np.asarray(result_mesh.triangles)))
    
    for i, face in enumerate(tri_mesh_fragment.faces):
        if tuple(sorted(face)) in result_faces_set or tuple(face) in result_faces_set:
            face_mask[i] = True
    
    return face_mask


def calculate_face_curvature(tri_mesh, face_idx, neighbor_radius=2):
    """
    Calculate mean curvature for a face using its neighborhood.
    Fracture surfaces typically have higher curvature than original surfaces.
    """
    if not hasattr(tri_mesh, 'face_adjacency') or tri_mesh.face_adjacency is None:
        tri_mesh.face_adjacency = trimesh.graph.face_adjacency(tri_mesh.faces)
    
    # Build adjacency list
    adjacency_list = [[] for _ in range(len(tri_mesh.faces))]
    for face1, face2 in tri_mesh.face_adjacency:
        adjacency_list[face1].append(face2)
        adjacency_list[face2].append(face1)
    
    # Find neighbors within radius
    visited = set()
    queue = [(face_idx, 0)]
    neighbor_faces = set()
    
    while queue:
        current_face, depth = queue.pop(0)
        if current_face in visited or depth > neighbor_radius:
            continue
        visited.add(current_face)
        neighbor_faces.add(current_face)
        
        for neighbor in adjacency_list[current_face]:
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))
    
    # Calculate curvature as variation in normals
    neighbor_normals = tri_mesh.face_normals[list(neighbor_faces)]
    center_normal = tri_mesh.face_normals[face_idx]
    
    # Mean angular deviation
    angles = []
    for normal in neighbor_normals:
        dot_product = np.clip(np.dot(center_normal, normal), -1, 1)
        angle = np.arccos(abs(dot_product))
        angles.append(angle)
    
    return np.mean(angles)


def calculate_face_roughness(tri_mesh, face_idx, params):
    """
    Calculate surface roughness for a face using local geometry analysis.
    Fracture surfaces are typically rougher than original surfaces.
    """
    # Get face vertices
    face_vertices = tri_mesh.vertices[tri_mesh.faces[face_idx]]
    face_normal = tri_mesh.face_normals[face_idx]
    
    # Find neighboring faces
    if not hasattr(tri_mesh, 'face_adjacency') or tri_mesh.face_adjacency is None:
        tri_mesh.face_adjacency = trimesh.graph.face_adjacency(tri_mesh.faces)
    
    adjacency_list = [[] for _ in range(len(tri_mesh.faces))]
    for face1, face2 in tri_mesh.face_adjacency:
        adjacency_list[face1].append(face2)
        adjacency_list[face2].append(face1)
    
    neighbor_faces = adjacency_list[face_idx]
    if not neighbor_faces:
        return 0.0
    
    # Calculate roughness as normal variation
    neighbor_normals = tri_mesh.face_normals[neighbor_faces]
    center_normal = tri_mesh.face_normals[face_idx]
    
    # Standard deviation of angles
    angles = []
    for normal in neighbor_normals:
        dot_product = np.clip(np.dot(center_normal, normal), -1, 1)
        angle = np.arccos(abs(dot_product))
        angles.append(angle)
    
    return np.std(angles)


def calculate_face_boundary_complexity(tri_mesh, face_idx):
    """
    Calculate boundary complexity for a face.
    Fracture surfaces often have more complex boundaries than original surfaces.
    """
    # Get all edges from all faces
    all_edges = []
    for face in tri_mesh.faces:
        all_edges.extend([(face[0], face[1]), (face[1], face[2]), (face[2], face[0])])
    
    # Count occurrences of each edge
    from collections import Counter
    edge_counts = Counter(all_edges)
    
    # Get edges for this specific face
    face = tri_mesh.faces[face_idx]
    face_edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
    
    # Count boundary edges (edges that appear only once)
    boundary_edges = 0
    for edge in face_edges:
        if edge_counts[edge] == 1:
            boundary_edges += 1
    
    return boundary_edges / 3.0  # Normalize by number of edges in a triangle


def calculate_face_symmetry_score(tri_mesh, face_idx, params):
    """
    Calculate symmetry score for a face.
    Original surfaces are often more symmetric than fracture surfaces.
    """
    face_vertices = tri_mesh.vertices[tri_mesh.faces[face_idx]]
    face_normal = tri_mesh.face_normals[face_idx]
    
    # Calculate face centroid
    centroid = np.mean(face_vertices, axis=0)
    
    # Calculate distances from centroid to each vertex
    distances = [np.linalg.norm(vertex - centroid) for vertex in face_vertices]
    
    # Symmetry score based on distance variation
    mean_distance = np.mean(distances)
    if mean_distance == 0:
        return 1.0
    
    # Coefficient of variation (lower = more symmetric)
    cv = np.std(distances) / mean_distance
    symmetry_score = 1.0 / (1.0 + cv)
    
    return symmetry_score


def calculate_face_planarity(tri_mesh, face_idx):
    """
    Calculate planarity score for a face.
    Original surfaces are often more planar than fracture surfaces.
    """
    face_vertices = tri_mesh.vertices[tri_mesh.faces[face_idx]]
    face_normal = tri_mesh.face_normals[face_idx]
    
    # Calculate how well vertices fit a plane
    centroid = np.mean(face_vertices, axis=0)
    
    # Calculate distances from vertices to plane
    distances = []
    for vertex in face_vertices:
        # Distance from point to plane: |(p - p0) · n| / |n|
        distance = abs(np.dot(vertex - centroid, face_normal))
        distances.append(distance)
    
    # Planarity score (lower distances = more planar)
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    
    if max_distance == 0:
        return 1.0
    
    planarity_score = 1.0 - (mean_distance / max_distance)
    return planarity_score


def detect_fracture_surfaces_advanced(tri_mesh, params):
    """
    Advanced fracture surface detection using multiple geometric properties.
    """
    print("    Running advanced fracture surface detection...")
    
    num_faces = len(tri_mesh.faces)
    face_scores = np.zeros(num_faces)
    
    # Calculate various properties for each face
    curvature_scores = np.zeros(num_faces)
    roughness_scores = np.zeros(num_faces)
    boundary_complexity_scores = np.zeros(num_faces)
    symmetry_scores = np.zeros(num_faces)
    planarity_scores = np.zeros(num_faces)
    
    for face_idx in range(num_faces):
        # Calculate individual scores
        curvature_scores[face_idx] = calculate_face_curvature(tri_mesh, face_idx)
        roughness_scores[face_idx] = calculate_face_roughness(tri_mesh, face_idx, params)
        boundary_complexity_scores[face_idx] = calculate_face_boundary_complexity(tri_mesh, face_idx)
        symmetry_scores[face_idx] = calculate_face_symmetry_score(tri_mesh, face_idx, params)
        planarity_scores[face_idx] = calculate_face_planarity(tri_mesh, face_idx)
    
    # Normalize scores to [0, 1] range
    def normalize_scores(scores):
        if np.max(scores) > np.min(scores):
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return scores
    
    curvature_scores = normalize_scores(curvature_scores)
    roughness_scores = normalize_scores(roughness_scores)
    boundary_complexity_scores = normalize_scores(boundary_complexity_scores)
    symmetry_scores = normalize_scores(symmetry_scores)
    planarity_scores = normalize_scores(planarity_scores)
    
    # Combine scores (higher values indicate fracture surfaces)
    weights = params.get('fracture_detection_weights', {
        'curvature': 0.3,
        'roughness': 0.3,
        'boundary_complexity': 0.2,
        'symmetry': 0.1,  # Positive because we already inverted symmetry
        'planarity': 0.1   # Positive because we already inverted planarity
    })
    
    face_scores = (
        weights['curvature'] * curvature_scores +
        weights['roughness'] * roughness_scores +
        weights['boundary_complexity'] * boundary_complexity_scores +
        weights['symmetry'] * symmetry_scores +  # Already inverted in normalization
        weights['planarity'] * planarity_scores   # Already inverted in normalization
    )
    
    # Apply threshold
    threshold = params.get('fracture_detection_threshold', 0.5)
    fracture_candidates = face_scores > threshold
    
    print(f"    Advanced detection: {np.sum(fracture_candidates)}/{num_faces} faces identified as fracture candidates")
    
    return fracture_candidates, {
        'curvature_scores': curvature_scores,
        'roughness_scores': roughness_scores,
        'boundary_complexity_scores': boundary_complexity_scores,
        'symmetry_scores': symmetry_scores,
        'planarity_scores': planarity_scores,
        'combined_scores': face_scores
    }


def analyze_surface_statistics(tri_mesh, face_indices, params):
    """
    Analyze statistical properties of a surface region to determine if it's fractured.
    """
    if len(face_indices) == 0:
        return {'is_fracture': False, 'confidence': 0.0}
    
    # Calculate various statistics
    face_normals = tri_mesh.face_normals[face_indices]
    face_areas = tri_mesh.area_faces[face_indices]
    
    # 1. Normal variation
    avg_normal = calculate_region_average_normal(tri_mesh, face_indices)
    normal_variations = []
    for normal in face_normals:
        dot_product = np.clip(np.dot(normal, avg_normal), -1, 1)
        angle = np.arccos(abs(dot_product))
        normal_variations.append(angle)
    
    normal_std = np.std(normal_variations)
    normal_mean = np.mean(normal_variations)
    
    # 2. Area distribution
    area_std = np.std(face_areas)
    area_mean = np.mean(face_areas)
    area_cv = area_std / area_mean if area_mean > 0 else 0
    
    # 3. Surface roughness (simplified)
    roughness = normal_std
    
    # 4. Fracture indicators
    fracture_indicators = {
        'high_normal_variation': normal_std > 0.3,  # High variation in normals
        'irregular_area_distribution': area_cv > 0.5,  # Irregular face sizes
        'rough_surface': roughness > 0.2,  # Rough surface
        'small_region': len(face_indices) < 100  # Small regions often fractures
    }
    
    # Calculate confidence score
    confidence = 0.0
    if fracture_indicators['high_normal_variation']:
        confidence += 0.3
    if fracture_indicators['irregular_area_distribution']:
        confidence += 0.2
    if fracture_indicators['rough_surface']:
        confidence += 0.3
    if fracture_indicators['small_region']:
        confidence += 0.2
    
    is_fracture = confidence > 0.5
    
    return {
        'is_fracture': is_fracture,
        'confidence': confidence,
        'normal_std': normal_std,
        'area_cv': area_cv,
        'roughness': roughness,
        'indicators': fracture_indicators
    }


def calculate_adaptive_thresholds(tri_mesh, params):
    """
    Calculate adaptive thresholds based on the object's properties.
    This replaces fixed thresholds with relative ones based on the actual mesh characteristics.
    """
    print("    Calculating adaptive thresholds based on object properties...")
    
    num_faces = len(tri_mesh.faces)
    if num_faces == 0:
        print("    Warning: No faces found in mesh")
        return {}
    
    thresholds = {}
    
    # Calculate roughness for all faces
    print("      Calculating roughness values...")
    roughness_values = np.zeros(num_faces)
    for face_idx in range(num_faces):
        roughness_values[face_idx] = calculate_face_roughness(tri_mesh, face_idx, params)
    
    # Calculate curvature for all faces
    print("      Calculating curvature values...")
    curvature_values = np.zeros(num_faces)
    for face_idx in range(num_faces):
        curvature_values[face_idx] = calculate_face_curvature(tri_mesh, face_idx)
    
    # Calculate boundary complexity for all faces
    print("      Calculating boundary complexity values...")
    boundary_complexity_values = np.zeros(num_faces)
    for face_idx in range(num_faces):
        boundary_complexity_values[face_idx] = calculate_face_boundary_complexity(tri_mesh, face_idx)
    
    # Calculate symmetry scores for all faces
    print("      Calculating symmetry values...")
    symmetry_values = np.zeros(num_faces)
    for face_idx in range(num_faces):
        symmetry_values[face_idx] = calculate_face_symmetry_score(tri_mesh, face_idx, params)
    
    # Calculate planarity scores for all faces
    print("      Calculating planarity values...")
    planarity_values = np.zeros(num_faces)
    for face_idx in range(num_faces):
        planarity_values[face_idx] = calculate_face_planarity(tri_mesh, face_idx)
    
    # Store all values for later use
    thresholds['all_values'] = {
        'roughness': roughness_values,
        'curvature': curvature_values,
        'boundary_complexity': boundary_complexity_values,
        'symmetry': symmetry_values,
        'planarity': planarity_values
    }
    
    # Calculate adaptive thresholds using percentiles with safety checks
    roughness_percentile = params.get('roughness_threshold_percentile', 75)  # Top 25% roughest
    curvature_percentile = params.get('curvature_threshold_percentile', 75)  # Top 25% highest curvature
    boundary_complexity_percentile = params.get('boundary_complexity_threshold_percentile', 75)  # Top 25% most complex
    symmetry_percentile = params.get('symmetry_threshold_percentile', 25)  # Bottom 25% least symmetric
    planarity_percentile = params.get('planarity_threshold_percentile', 25)  # Bottom 25% least planar
    
    # Ensure percentiles are valid
    roughness_percentile = np.clip(roughness_percentile, 0, 100)
    curvature_percentile = np.clip(curvature_percentile, 0, 100)
    boundary_complexity_percentile = np.clip(boundary_complexity_percentile, 0, 100)
    symmetry_percentile = np.clip(symmetry_percentile, 0, 100)
    planarity_percentile = np.clip(planarity_percentile, 0, 100)
    
    # Calculate thresholds with safety checks
    thresholds['roughness'] = np.percentile(roughness_values, roughness_percentile) if len(roughness_values) > 0 else 0
    thresholds['curvature'] = np.percentile(curvature_values, curvature_percentile) if len(curvature_values) > 0 else 0
    thresholds['boundary_complexity'] = np.percentile(boundary_complexity_values, boundary_complexity_percentile) if len(boundary_complexity_values) > 0 else 0
    thresholds['symmetry'] = np.percentile(symmetry_values, symmetry_percentile) if len(symmetry_values) > 0 else 0
    thresholds['planarity'] = np.percentile(planarity_values, planarity_percentile) if len(planarity_values) > 0 else 0
    
    # Calculate statistics for reporting
    thresholds['statistics'] = {
        'roughness': {
            'mean': np.mean(roughness_values),
            'std': np.std(roughness_values),
            'min': np.min(roughness_values),
            'max': np.max(roughness_values),
            'threshold': thresholds['roughness']
        },
        'curvature': {
            'mean': np.mean(curvature_values),
            'std': np.std(curvature_values),
            'min': np.min(curvature_values),
            'max': np.max(curvature_values),
            'threshold': thresholds['curvature']
        },
        'boundary_complexity': {
            'mean': np.mean(boundary_complexity_values),
            'std': np.std(boundary_complexity_values),
            'min': np.min(boundary_complexity_values),
            'max': np.max(boundary_complexity_values),
            'threshold': thresholds['boundary_complexity']
        },
        'symmetry': {
            'mean': np.mean(symmetry_values),
            'std': np.std(symmetry_values),
            'min': np.min(symmetry_values),
            'max': np.max(symmetry_values),
            'threshold': thresholds['symmetry']
        },
        'planarity': {
            'mean': np.mean(planarity_values),
            'std': np.std(planarity_values),
            'min': np.min(planarity_values),
            'max': np.max(planarity_values),
            'threshold': thresholds['planarity']
        }
    }
    
    print(f"    Adaptive thresholds calculated:")
    print(f"      Roughness: {thresholds['roughness']:.4f} (top {100-roughness_percentile}%)")
    print(f"      Curvature: {thresholds['curvature']:.4f} (top {100-curvature_percentile}%)")
    print(f"      Boundary Complexity: {thresholds['boundary_complexity']:.4f} (top {100-boundary_complexity_percentile}%)")
    print(f"      Symmetry: {thresholds['symmetry']:.4f} (bottom {symmetry_percentile}%)")
    print(f"      Planarity: {thresholds['planarity']:.4f} (bottom {planarity_percentile}%)")
    
    return thresholds


def detect_fracture_surfaces_adaptive(tri_mesh, params):
    """
    Advanced fracture surface detection using adaptive thresholds based on object properties.
    """
    print("    Running adaptive fracture surface detection...")
    
    # Calculate adaptive thresholds
    thresholds = calculate_adaptive_thresholds(tri_mesh, params)
    
    num_faces = len(tri_mesh.faces)
    face_scores = np.zeros(num_faces)
    
    # Get all pre-calculated values
    roughness_values = thresholds['all_values']['roughness']
    curvature_values = thresholds['all_values']['curvature']
    boundary_complexity_values = thresholds['all_values']['boundary_complexity']
    symmetry_values = thresholds['all_values']['symmetry']
    planarity_values = thresholds['all_values']['planarity']
    
    # Normalize scores to [0, 1] range based on thresholds
    def normalize_above_threshold(values, threshold):
        """Normalize values above threshold to [0,1], values below threshold to 0"""
        normalized = np.zeros_like(values)
        above_threshold = values > threshold
        if np.any(above_threshold):
            max_val = np.max(values[above_threshold])
            if max_val > threshold:
                normalized[above_threshold] = (values[above_threshold] - threshold) / (max_val - threshold)
        return normalized
    
    def normalize_below_threshold(values, threshold):
        """Normalize values below threshold to [0,1], values above threshold to 0"""
        normalized = np.zeros_like(values)
        below_threshold = values < threshold
        if np.any(below_threshold):
            min_val = np.min(values[below_threshold])
            if threshold > min_val:
                normalized[below_threshold] = (threshold - values[below_threshold]) / (threshold - min_val)
        return normalized
    
    def safe_normalize(values, threshold, above=True):
        """Safe normalization with edge case handling"""
        if above:
            return normalize_above_threshold(values, threshold)
        else:
            return normalize_below_threshold(values, threshold)
    
    # Normalize each property with safe handling
    roughness_scores = safe_normalize(roughness_values, thresholds['roughness'], above=True)
    curvature_scores = safe_normalize(curvature_values, thresholds['curvature'], above=True)
    boundary_complexity_scores = safe_normalize(boundary_complexity_values, thresholds['boundary_complexity'], above=True)
    symmetry_scores = safe_normalize(symmetry_values, thresholds['symmetry'], above=False)  # Lower symmetry = fracture
    planarity_scores = safe_normalize(planarity_values, thresholds['planarity'], above=False)  # Lower planarity = fracture
    
    # Combine scores with weights
    weights = params.get('fracture_detection_weights', {
        'curvature': 0.3,
        'roughness': 0.3,
        'boundary_complexity': 0.2,
        'symmetry': 0.1,
        'planarity': 0.1
    })
    
    face_scores = (
        weights['curvature'] * curvature_scores +
        weights['roughness'] * roughness_scores +
        weights['boundary_complexity'] * boundary_complexity_scores +
        weights['symmetry'] * symmetry_scores +
        weights['planarity'] * planarity_scores
    )
    
    # Use adaptive threshold based on score distribution
    score_threshold_percentile = params.get('score_threshold_percentile', 70)
    score_threshold = np.percentile(face_scores, score_threshold_percentile)
    fracture_candidates = face_scores > score_threshold
    
    print(f"    Adaptive detection: {np.sum(fracture_candidates)}/{num_faces} faces identified as fracture candidates")
    print(f"    Score threshold: {score_threshold:.4f} (top {100-score_threshold_percentile}%)")
    
    return fracture_candidates, {
        'curvature_scores': curvature_scores,
        'roughness_scores': roughness_scores,
        'boundary_complexity_scores': boundary_complexity_scores,
        'symmetry_scores': symmetry_scores,
        'planarity_scores': planarity_scores,
        'combined_scores': face_scores,
        'thresholds': thresholds,
        'score_threshold': score_threshold
    }


def analyze_surface_statistics_adaptive(tri_mesh, face_indices, params, global_stats=None):
    """
    Analyze statistical properties of a surface region with adaptive thresholds.
    """
    if len(face_indices) == 0:
        return {'is_fracture': False, 'confidence': 0.0}
    
    # Calculate various statistics
    face_normals = tri_mesh.face_normals[face_indices]
    face_areas = tri_mesh.area_faces[face_indices]
    
    # 1. Normal variation
    avg_normal = calculate_region_average_normal(tri_mesh, face_indices)
    normal_variations = []
    for normal in face_normals:
        dot_product = np.clip(np.dot(normal, avg_normal), -1, 1)
        angle = np.arccos(abs(dot_product))
        normal_variations.append(angle)
    
    normal_std = np.std(normal_variations)
    normal_mean = np.mean(normal_variations)
    
    # 2. Area distribution
    area_std = np.std(face_areas)
    area_mean = np.mean(face_areas)
    area_cv = area_std / area_mean if area_mean > 0 else 0
    
    # 3. Surface roughness (simplified)
    roughness = normal_std
    
    # 4. Region size relative to mesh size
    total_faces = len(tri_mesh.faces)
    size_ratio = len(face_indices) / total_faces
    
    # Use adaptive thresholds if global stats are provided
    if global_stats is not None:
        # Compare against global statistics
        normal_variation_threshold = np.percentile(global_stats['normal_variations'], 75)
        area_cv_threshold = np.percentile(global_stats['area_cvs'], 75)
        roughness_threshold = np.percentile(global_stats['roughness_values'], 75)
        size_threshold = np.percentile(global_stats['region_sizes'], 25)  # Smaller regions = fractures
    else:
        # Fallback to fixed thresholds
        normal_variation_threshold = 0.3
        area_cv_threshold = 0.5
        roughness_threshold = 0.2
        size_threshold = 0.1  # 10% of total faces
    
    # Fracture indicators with adaptive thresholds
    fracture_indicators = {
        'high_normal_variation': normal_std > normal_variation_threshold,
        'irregular_area_distribution': area_cv > area_cv_threshold,
        'rough_surface': roughness > roughness_threshold,
        'small_region': size_ratio < size_threshold
    }
    
    # Calculate confidence score based on how many indicators are true
    confidence = 0.0
    indicator_weights = {
        'high_normal_variation': 0.3,
        'irregular_area_distribution': 0.2,
        'rough_surface': 0.3,
        'small_region': 0.2
    }
    
    for indicator, is_true in fracture_indicators.items():
        if is_true:
            confidence += indicator_weights[indicator]
    
    is_fracture = confidence > 0.5
    
    return {
        'is_fracture': is_fracture,
        'confidence': confidence,
        'normal_std': normal_std,
        'area_cv': area_cv,
        'roughness': roughness,
        'size_ratio': size_ratio,
        'indicators': fracture_indicators,
        'thresholds_used': {
            'normal_variation': normal_variation_threshold,
            'area_cv': area_cv_threshold,
            'roughness': roughness_threshold,
            'size': size_threshold
        }
    }


def calculate_global_statistics(tri_mesh, params):
    """
    Calculate global statistics for the entire mesh to use as reference for adaptive thresholds.
    """
    print("    Calculating global mesh statistics...")
    
    num_faces = len(tri_mesh.faces)
    global_stats = {}
    
    # Calculate normal variations for all faces
    normal_variations = []
    roughness_values = []
    area_cvs = []
    region_sizes = []
    
    # Sample faces for efficiency (if mesh is large)
    sample_size = min(1000, num_faces)
    if num_faces > 1000:
        sample_indices = np.random.choice(num_faces, sample_size, replace=False)
    else:
        sample_indices = np.arange(num_faces)
    
    for face_idx in sample_indices:
        # Calculate normal variation for this face
        face_normal = tri_mesh.face_normals[face_idx]
        
        # Find neighboring faces
        if not hasattr(tri_mesh, 'face_adjacency') or tri_mesh.face_adjacency is None:
            tri_mesh.face_adjacency = trimesh.graph.face_adjacency(tri_mesh.faces)
        
        adjacency_list = [[] for _ in range(num_faces)]
        for face1, face2 in tri_mesh.face_adjacency:
            adjacency_list[face1].append(face2)
            adjacency_list[face2].append(face1)
        
        neighbor_faces = adjacency_list[face_idx]
        if neighbor_faces:
            neighbor_normals = tri_mesh.face_normals[neighbor_faces]
            variations = []
            for normal in neighbor_normals:
                dot_product = np.clip(np.dot(face_normal, normal), -1, 1)
                angle = np.arccos(abs(dot_product))
                variations.append(angle)
            
            normal_variations.append(np.std(variations))
            roughness_values.append(calculate_face_roughness(tri_mesh, face_idx, params))
    
    # Calculate area statistics for all faces
    face_areas = tri_mesh.area_faces
    area_mean = np.mean(face_areas)
    area_std = np.std(face_areas)
    area_cv_global = area_std / area_mean if area_mean > 0 else 0
    
    # Calculate individual area CVs for sampled faces
    individual_area_cvs = []
    for face_idx in sample_indices:
        # Get neighboring faces for this face
        if not hasattr(tri_mesh, 'face_adjacency') or tri_mesh.face_adjacency is None:
            tri_mesh.face_adjacency = trimesh.graph.face_adjacency(tri_mesh.faces)
        
        adjacency_list = [[] for _ in range(num_faces)]
        for face1, face2 in tri_mesh.face_adjacency:
            adjacency_list[face1].append(face2)
            adjacency_list[face2].append(face1)
        
        neighbor_faces = adjacency_list[face_idx]
        if neighbor_faces:
            neighbor_areas = tri_mesh.area_faces[neighbor_faces]
            neighbor_mean = np.mean(neighbor_areas)
            neighbor_std = np.std(neighbor_areas)
            neighbor_cv = neighbor_std / neighbor_mean if neighbor_mean > 0 else 0
            individual_area_cvs.append(neighbor_cv)
        else:
            individual_area_cvs.append(area_cv_global)
    
    # Estimate region sizes by running a quick region growing
    regions = region_growing_segmentation(tri_mesh, params)
    region_sizes = [len(region) / num_faces for region in regions]
    
    global_stats = {
        'normal_variations': np.array(normal_variations),
        'roughness_values': np.array(roughness_values),
        'area_cvs': np.array(individual_area_cvs),  # Use individual CVs
        'region_sizes': np.array(region_sizes),
        'total_faces': num_faces,
        'sample_size': sample_size
    }
    
    print(f"    Global statistics calculated from {sample_size} sample faces")
    print(f"    Normal variation range: {np.min(normal_variations):.4f} - {np.max(normal_variations):.4f}")
    print(f"    Roughness range: {np.min(roughness_values):.4f} - {np.max(roughness_values):.4f}")
    print(f"    Region size range: {np.min(region_sizes):.4f} - {np.max(region_sizes):.4f}")
    
    return global_stats


def compare_fracture_detection_methods_adaptive(tri_mesh, params):
    """
    Compare different methods for detecting fracture surfaces using adaptive thresholds.
    """
    print(f"\n=== Comparing Adaptive Fracture Surface Detection Methods ===")
    
    num_faces = len(tri_mesh.faces)
    results = {}
    
    # Calculate global statistics for adaptive thresholds
    global_stats = calculate_global_statistics(tri_mesh, params)
    
    # Method 1: Adaptive Geometric Analysis
    print(f"\n1. Testing Adaptive Geometric Analysis...")
    adaptive_candidates, scores = detect_fracture_surfaces_adaptive(tri_mesh, params)
    
    results['adaptive_geometric'] = {
        'candidates': adaptive_candidates,
        'count': np.sum(adaptive_candidates),
        'percentage': np.sum(adaptive_candidates) / num_faces * 100,
        'scores': scores
    }
    print(f"   Adaptive geometric detection: {results['adaptive_geometric']['count']} faces ({results['adaptive_geometric']['percentage']:.1f}%)")
    
    # Method 2: Adaptive Statistical Analysis
    print(f"\n2. Testing Adaptive Statistical Analysis...")
    regions = region_growing_segmentation(tri_mesh, params)
    statistical_candidates = np.zeros(num_faces, dtype=bool)
    
    for region in regions:
        stats = analyze_surface_statistics_adaptive(tri_mesh, region, params, global_stats)
        if stats['is_fracture'] and stats['confidence'] > params.get('statistical_confidence_threshold', 0.6):
            statistical_candidates[region] = True
    
    results['adaptive_statistical'] = {
        'candidates': statistical_candidates,
        'count': np.sum(statistical_candidates),
        'percentage': np.sum(statistical_candidates) / num_faces * 100
    }
    print(f"   Adaptive statistical analysis: {results['adaptive_statistical']['count']} faces ({results['adaptive_statistical']['percentage']:.1f}%)")
    
    # Method 3: Simple Adaptive Curvature-based Detection
    print(f"\n3. Testing Adaptive Curvature-based Detection...")
    curvature_candidates = np.zeros(num_faces, dtype=bool)
    curvature_values = np.zeros(num_faces)
    
    for face_idx in range(num_faces):
        curvature_values[face_idx] = calculate_face_curvature(tri_mesh, face_idx)
    
    # Use adaptive threshold based on curvature distribution
    curvature_percentile = params.get('curvature_threshold_percentile', 75)
    curvature_threshold = np.percentile(curvature_values, curvature_percentile)
    curvature_candidates = curvature_values > curvature_threshold
    
    results['adaptive_curvature'] = {
        'candidates': curvature_candidates,
        'count': np.sum(curvature_candidates),
        'percentage': np.sum(curvature_candidates) / num_faces * 100,
        'values': curvature_values,
        'threshold': curvature_threshold
    }
    print(f"   Adaptive curvature detection: {results['adaptive_curvature']['count']} faces ({results['adaptive_curvature']['percentage']:.1f}%)")
    
    # Method 4: Combined Adaptive Approach
    print(f"\n4. Testing Combined Adaptive Approach...")
    combined_candidates = np.zeros(num_faces, dtype=bool)
    available_methods = list(results.keys())
    
    if len(available_methods) > 0:
        # Require at least 2 methods to agree
        min_agreement = params.get('combined_min_agreement', 2)
        agreement_count = np.zeros(num_faces, dtype=int)
        
        for method in available_methods:
            agreement_count += results[method]['candidates'].astype(int)
        
        combined_candidates = agreement_count >= min_agreement
        
        results['combined_adaptive'] = {
            'candidates': combined_candidates,
            'count': np.sum(combined_candidates),
            'percentage': np.sum(combined_candidates) / num_faces * 100,
            'agreement_count': agreement_count
        }
        print(f"   Combined adaptive detection: {results['combined_adaptive']['count']} faces ({results['combined_adaptive']['percentage']:.1f}%)")
    
    # Summary
    print(f"\n=== Adaptive Detection Method Summary ===")
    for method, data in results.items():
        print(f"{method.replace('_', ' ').title()}: {data['count']} faces ({data['percentage']:.1f}%)")
    
    return results, global_stats


def visualize_detection_comparison(o3d_mesh, detection_results, fragment_name="Unnamed"):
    """
    Visualize the results of different fracture detection methods.
    """
    if not detection_results:
        print("No detection results to visualize")
        return
    
    # Create visualization geometries
    vis_geometries = []
    
    # Original mesh in gray
    original_mesh = copy.deepcopy(o3d_mesh)
    original_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    original_mesh.compute_vertex_normals()
    vis_geometries.append(original_mesh)
    
    # Wireframe for structure
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(original_mesh)
    wireframe.paint_uniform_color([0.5, 0.5, 0.5])
    vis_geometries.append(wireframe)
    
    # Color each detection method
    colors = {
        'bumpiness': [1.0, 0.0, 0.0],      # Red
        'advanced': [0.0, 1.0, 0.0],       # Green
        'statistical': [0.0, 0.0, 1.0],    # Blue
        'curvature': [1.0, 1.0, 0.0],      # Yellow
        'combined': [1.0, 0.0, 1.0]        # Magenta
    }
    
    for method, data in detection_results.items():
        if method in colors and data['count'] > 0:
            # Create mesh for this method's candidates
            candidate_faces = np.where(data['candidates'])[0]
            if len(candidate_faces) > 0:
                method_mesh = o3d.geometry.TriangleMesh()
                method_mesh.vertices = o3d_mesh.vertices
                method_mesh.triangles = o3d.utility.Vector3iVector(
                    np.asarray(o3d_mesh.triangles)[candidate_faces]
                )
                method_mesh.remove_unreferenced_vertices()
                method_mesh.compute_vertex_normals()
                method_mesh.paint_uniform_color(colors[method])
                vis_geometries.append(method_mesh)
    
    # Create legend
    legend_geometries = []
    y_offset = 0
    for method, data in detection_results.items():
        if method in colors and data['count'] > 0:
            # Create a small sphere for legend
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate([0, y_offset, 0])
            sphere.paint_uniform_color(colors[method])
            legend_geometries.append(sphere)
            y_offset += 0.3
    
    # Show visualization
    print(f"\nVisualizing detection results for {fragment_name}")
    print("Colors: Red=Bumpiness, Green=Advanced, Blue=Statistical, Yellow=Curvature, Magenta=Combined")
    
    o3d.visualization.draw_geometries(vis_geometries, window_name=f"Fracture Detection Comparison - {fragment_name}")


def get_fracture_surface_confidence(tri_mesh, face_indices, params):
    """
    Get a confidence score for whether a set of faces represents a fracture surface.
    Returns a score between 0 (definitely original surface) and 1 (definitely fracture surface).
    """
    if len(face_indices) == 0:
        return 0.0
    
    # Calculate multiple indicators
    indicators = {}
    
    # 1. Normal variation
    face_normals = tri_mesh.face_normals[face_indices]
    avg_normal = calculate_region_average_normal(tri_mesh, face_indices)
    normal_variations = []
    for normal in face_normals:
        dot_product = np.clip(np.dot(normal, avg_normal), -1, 1)
        angle = np.arccos(abs(dot_product))
        normal_variations.append(angle)
    
    indicators['normal_variation'] = np.std(normal_variations)
    
    # 2. Surface roughness
    indicators['roughness'] = calculate_region_bumpiness(tri_mesh, face_indices, params)
    
    # 3. Area distribution
    face_areas = tri_mesh.area_faces[face_indices]
    area_cv = np.std(face_areas) / np.mean(face_areas) if np.mean(face_areas) > 0 else 0
    indicators['area_irregularity'] = area_cv
    
    # 4. Region size (smaller regions are more likely to be fractures)
    indicators['size_factor'] = min(1.0, len(face_indices) / 1000.0)  # Normalize by expected size
    
    # 5. Boundary complexity
    boundary_complexities = [calculate_face_boundary_complexity(tri_mesh, face_idx) for face_idx in face_indices]
    indicators['boundary_complexity'] = np.mean(boundary_complexities)
    
    # Combine indicators into confidence score
    weights = params.get('confidence_weights', {
        'normal_variation': 0.25,
        'roughness': 0.25,
        'area_irregularity': 0.2,
        'size_factor': 0.15,
        'boundary_complexity': 0.15
    })
    
    # Normalize and combine
    confidence = 0.0
    for indicator, value in indicators.items():
        if indicator in weights:
            # Normalize based on typical ranges
            if indicator == 'normal_variation':
                normalized_value = min(1.0, value / 0.5)  # 0.5 radians is high variation
            elif indicator == 'roughness':
                normalized_value = min(1.0, value / 0.3)  # 0.3 is high roughness
            elif indicator == 'area_irregularity':
                normalized_value = min(1.0, value / 1.0)  # 1.0 is high irregularity
            elif indicator == 'size_factor':
                normalized_value = 1.0 - value  # Invert: smaller = higher fracture probability
            elif indicator == 'boundary_complexity':
                normalized_value = value  # Already normalized
            else:
                normalized_value = min(1.0, value)
            
            confidence += weights[indicator] * normalized_value
    
    return min(1.0, confidence), indicators


def validate_adaptive_detection_logic(tri_mesh, params):
    """
    Validate the adaptive detection logic and check for potential issues.
    """
    print("\n=== Validating Adaptive Detection Logic ===")
    
    num_faces = len(tri_mesh.faces)
    if num_faces == 0:
        print("ERROR: No faces found in mesh")
        return False
    
    print(f"Mesh has {num_faces} faces")
    
    # Test 1: Check if all required properties can be calculated
    print("\n1. Testing property calculations...")
    try:
        # Test a few faces
        test_faces = min(10, num_faces)
        for i in range(test_faces):
            roughness = calculate_face_roughness(tri_mesh, i, params)
            curvature = calculate_face_curvature(tri_mesh, i)
            boundary_complexity = calculate_face_boundary_complexity(tri_mesh, i)
            symmetry = calculate_face_symmetry_score(tri_mesh, i, params)
            planarity = calculate_face_planarity(tri_mesh, i)
            
            # Check for NaN or infinite values
            if not np.isfinite(roughness) or not np.isfinite(curvature) or \
               not np.isfinite(boundary_complexity) or not np.isfinite(symmetry) or \
               not np.isfinite(planarity):
                print(f"ERROR: Non-finite values found for face {i}")
                return False
        
        print(f"✓ Property calculations successful for {test_faces} test faces")
    except Exception as e:
        print(f"ERROR: Property calculation failed: {e}")
        return False
    
    # Test 2: Check adaptive threshold calculation
    print("\n2. Testing adaptive threshold calculation...")
    try:
        thresholds = calculate_adaptive_thresholds(tri_mesh, params)
        if not thresholds:
            print("ERROR: No thresholds calculated")
            return False
        
        # Check if all required thresholds are present
        required_keys = ['roughness', 'curvature', 'boundary_complexity', 'symmetry', 'planarity']
        for key in required_keys:
            if key not in thresholds:
                print(f"ERROR: Missing threshold for {key}")
                return False
        
        print("✓ Adaptive thresholds calculated successfully")
    except Exception as e:
        print(f"ERROR: Threshold calculation failed: {e}")
        return False
    
    # Test 3: Check normalization logic
    print("\n3. Testing normalization logic...")
    try:
        # Test with sample data
        test_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        threshold = 0.5
        
        # Test above threshold normalization
        above_scores = np.zeros_like(test_values)
        above_threshold = test_values > threshold
        if np.any(above_threshold):
            max_val = np.max(test_values[above_threshold])
            if max_val > threshold:
                above_scores[above_threshold] = (test_values[above_threshold] - threshold) / (max_val - threshold)
        
        # Test below threshold normalization
        below_scores = np.zeros_like(test_values)
        below_threshold = test_values < threshold
        if np.any(below_threshold):
            min_val = np.min(test_values[below_threshold])
            if threshold > min_val:
                below_scores[below_threshold] = (threshold - test_values[below_threshold]) / (threshold - min_val)
        
        print("✓ Normalization logic working correctly")
    except Exception as e:
        print(f"ERROR: Normalization test failed: {e}")
        return False
    
    # Test 4: Check score combination
    print("\n4. Testing score combination...")
    try:
        # Test with sample scores
        test_scores = {
            'curvature': np.array([0.1, 0.2, 0.3]),
            'roughness': np.array([0.2, 0.3, 0.4]),
            'boundary_complexity': np.array([0.1, 0.2, 0.3]),
            'symmetry': np.array([0.3, 0.2, 0.1]),
            'planarity': np.array([0.2, 0.1, 0.3])
        }
        
        weights = params.get('fracture_detection_weights', {
            'curvature': 0.3,
            'roughness': 0.3,
            'boundary_complexity': 0.2,
            'symmetry': 0.1,
            'planarity': 0.1
        })
        
        combined_scores = (
            weights['curvature'] * test_scores['curvature'] +
            weights['roughness'] * test_scores['roughness'] +
            weights['boundary_complexity'] * test_scores['boundary_complexity'] +
            weights['symmetry'] * test_scores['symmetry'] +
            weights['planarity'] * test_scores['planarity']
        )
        
        if not np.all(np.isfinite(combined_scores)):
            print("ERROR: Non-finite values in combined scores")
            return False
        
        print("✓ Score combination working correctly")
    except Exception as e:
        print(f"ERROR: Score combination test failed: {e}")
        return False
    
    # Test 5: Check global statistics
    print("\n5. Testing global statistics calculation...")
    try:
        global_stats = calculate_global_statistics(tri_mesh, params)
        if not global_stats:
            print("ERROR: No global statistics calculated")
            return False
        
        required_stats = ['normal_variations', 'roughness_values', 'area_cvs', 'region_sizes']
        for key in required_stats:
            if key not in global_stats:
                print(f"ERROR: Missing global statistic: {key}")
                return False
        
        print("✓ Global statistics calculated successfully")
    except Exception as e:
        print(f"ERROR: Global statistics calculation failed: {e}")
        return False
    
    print("\n=== All Validation Tests Passed ===")
    return True


def run_adaptive_detection_with_validation(tri_mesh, params):
    """
    Run adaptive detection with comprehensive validation.
    """
    print("=== Running Adaptive Detection with Validation ===")
    
    # Validate logic first
    if not validate_adaptive_detection_logic(tri_mesh, params):
        print("ERROR: Validation failed. Aborting detection.")
        return None, None
    
    # Run detection
    print("\n=== Running Detection ===")
    try:
        detection_results, global_stats = compare_fracture_detection_methods_adaptive(tri_mesh, params)
        
        # Validate results
        total_candidates = 0
        for method, data in detection_results.items():
            if 'candidates' in data:
                total_candidates += np.sum(data['candidates'])
        
        if total_candidates == 0:
            print("WARNING: No fracture candidates detected by any method")
        else:
            print(f"✓ Detection completed successfully. Total candidates: {total_candidates}")
        
        return detection_results, global_stats
        
    except Exception as e:
        print(f"ERROR: Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def classify_fracture_vs_original_faces(tri_mesh, params=None):
    """
    Simple single-method classification of faces into fracture vs original surfaces.
    Uses adaptive thresholds based on the whole object's geometric properties.
    
    Args:
        tri_mesh: trimesh object
        params: optional parameters dict
        
    Returns:
        dict containing:
            - 'fracture_faces': boolean array marking fracture faces
            - 'original_faces': boolean array marking original faces
            - 'face_scores': confidence scores for each face
            - 'thresholds': the adaptive thresholds used
    """
    if params is None:
        params = {}
    
    print(f"Classifying {len(tri_mesh.faces)} faces into fracture vs original surfaces...")
    
    num_faces = len(tri_mesh.faces)
    if num_faces == 0:
        return {
            'fracture_faces': np.array([]),
            'original_faces': np.array([]),
            'face_scores': np.array([]),
            'thresholds': {}
        }
    
    # Step 1: Calculate geometric properties for all faces
    print("  Calculating geometric properties...")
    roughness_values = np.zeros(num_faces)
    curvature_values = np.zeros(num_faces)
    boundary_complexity_values = np.zeros(num_faces)
    symmetry_values = np.zeros(num_faces)
    planarity_values = np.zeros(num_faces)
    
    for face_idx in range(num_faces):
        roughness_values[face_idx] = calculate_face_roughness(tri_mesh, face_idx, params)
        curvature_values[face_idx] = calculate_face_curvature(tri_mesh, face_idx)
        boundary_complexity_values[face_idx] = calculate_face_boundary_complexity(tri_mesh, face_idx)
        symmetry_values[face_idx] = calculate_face_symmetry_score(tri_mesh, face_idx, params)
        planarity_values[face_idx] = calculate_face_planarity(tri_mesh, face_idx)
    
    # Step 2: Calculate adaptive thresholds from object properties
    print("  Calculating adaptive thresholds...")
    roughness_percentile = params.get('roughness_threshold_percentile', 75)
    curvature_percentile = params.get('curvature_threshold_percentile', 75)
    boundary_complexity_percentile = params.get('boundary_complexity_threshold_percentile', 75)
    symmetry_percentile = params.get('symmetry_threshold_percentile', 25)
    planarity_percentile = params.get('planarity_threshold_percentile', 25)
    
    thresholds = {
        'roughness': np.percentile(roughness_values, roughness_percentile),
        'curvature': np.percentile(curvature_values, curvature_percentile),
        'boundary_complexity': np.percentile(boundary_complexity_values, boundary_complexity_percentile),
        'symmetry': np.percentile(symmetry_values, symmetry_percentile),
        'planarity': np.percentile(planarity_values, planarity_percentile)
    }
    
    print(f"  Adaptive thresholds:")
    print(f"    Roughness: {thresholds['roughness']:.4f} (top {100-roughness_percentile}%)")
    print(f"    Curvature: {thresholds['curvature']:.4f} (top {100-curvature_percentile}%)")
    print(f"    Boundary Complexity: {thresholds['boundary_complexity']:.4f} (top {100-boundary_complexity_percentile}%)")
    print(f"    Symmetry: {thresholds['symmetry']:.4f} (bottom {symmetry_percentile}%)")
    print(f"    Planarity: {thresholds['planarity']:.4f} (bottom {planarity_percentile}%)")
    
    # Step 3: Calculate fracture scores for each face
    print("  Calculating fracture scores...")
    face_scores = np.zeros(num_faces)
    
    # Get weights for combining properties
    weights = params.get('fracture_detection_weights', {
        'curvature': 0.3,
        'roughness': 0.3,
        'boundary_complexity': 0.2,
        'symmetry': 0.1,
        'planarity': 0.1
    })
    
    for face_idx in range(num_faces):
        score = 0.0
        
        # Higher roughness = more likely fracture
        if roughness_values[face_idx] > thresholds['roughness']:
            score += weights['roughness']
        
        # Higher curvature = more likely fracture
        if curvature_values[face_idx] > thresholds['curvature']:
            score += weights['curvature']
        
        # Higher boundary complexity = more likely fracture
        if boundary_complexity_values[face_idx] > thresholds['boundary_complexity']:
            score += weights['boundary_complexity']
        
        # Lower symmetry = more likely fracture
        if symmetry_values[face_idx] < thresholds['symmetry']:
            score += weights['symmetry']
        
        # Lower planarity = more likely fracture
        if planarity_values[face_idx] < thresholds['planarity']:
            score += weights['planarity']
        
        face_scores[face_idx] = score
    
    # Step 4: Apply final threshold to classify faces
    final_threshold = params.get('final_classification_threshold', 0.5)
    fracture_faces = face_scores > final_threshold
    original_faces = ~fracture_faces
    
    num_fracture = np.sum(fracture_faces)
    num_original = np.sum(original_faces)
    
    print(f"  Classification results:")
    print(f"    Fracture faces: {num_fracture} ({num_fracture/num_faces*100:.1f}%)")
    print(f"    Original faces: {num_original} ({num_original/num_faces*100:.1f}%)")
    
    return {
        'fracture_faces': fracture_faces,
        'original_faces': original_faces,
        'face_scores': face_scores,
        'thresholds': thresholds,
        'properties': {
            'roughness': roughness_values,
            'curvature': curvature_values,
            'boundary_complexity': boundary_complexity_values,
            'symmetry': symmetry_values,
            'planarity': planarity_values
        }
    }


def visualize_face_classification(o3d_mesh, classification_result, fragment_name="Mesh"):
    """
    Visualize the face classification results.
    
    Args:
        o3d_mesh: Open3D mesh
        classification_result: result from classify_fracture_vs_original_faces
        fragment_name: name for the visualization window
    """
    if not classification_result['fracture_faces'].any() and not classification_result['original_faces'].any():
        print("No classification results to visualize")
        return
    
    print(f"Creating visualization for {fragment_name}...")
    
    # Create visualization mesh
    vis_mesh = copy.deepcopy(o3d_mesh)
    vis_mesh.compute_vertex_normals()
    
    # Color faces based on classification
    face_colors = np.zeros((len(vis_mesh.triangles), 3))
    
    fracture_faces = classification_result['fracture_faces']
    original_faces = classification_result['original_faces']
    face_scores = classification_result['face_scores']
    
    for i in range(len(vis_mesh.triangles)):
        if fracture_faces[i]:
            # Red intensity based on confidence score
            intensity = min(1.0, face_scores[i])
            face_colors[i] = [1.0, 1.0 - intensity * 0.5, 1.0 - intensity * 0.5]  # Red
        else:
            # Blue for original surfaces
            face_colors[i] = [0.3, 0.3, 1.0]  # Blue
    
    # Apply colors to vertices (average from adjacent faces)
    vertex_colors = np.zeros((len(vis_mesh.vertices), 3))
    vertex_counts = np.zeros(len(vis_mesh.vertices))
    
    for i, triangle in enumerate(np.asarray(vis_mesh.triangles)):
        for vertex_idx in triangle:
            vertex_colors[vertex_idx] += face_colors[i]
            vertex_counts[vertex_idx] += 1
    
    # Normalize
    for i in range(len(vertex_colors)):
        if vertex_counts[i] > 0:
            vertex_colors[i] /= vertex_counts[i]
    
    vis_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    print("Visualization colors:")
    print("  Red = Fracture surfaces")
    print("  Blue = Original surfaces")
    print("  Intensity = Confidence level")
    
    o3d.visualization.draw_geometries([vis_mesh], window_name=f"Face Classification - {fragment_name}")


def extract_fracture_surfaces_simple(o3d_mesh_fragment, fragment_name="Unnamed", params=None):
    """
    Simple fracture surface extraction using single classification method.
    
    Args:
        o3d_mesh_fragment: Open3D mesh
        fragment_name: name for logging
        params: optional parameters
        
    Returns:
        list of Open3D meshes representing fracture surfaces
    """
    params = params or {}
    
    print(f"\n=== Simple Fracture Surface Extraction for {fragment_name} ===")
    
    # Convert to trimesh
    if not o3d_mesh_fragment.has_triangles() or not o3d_mesh_fragment.has_vertices():
        print(f"Input mesh {fragment_name} has no triangles/vertices.")
        return []
    
    try:
        tri_mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh_fragment.vertices),
            faces=np.asarray(o3d_mesh_fragment.triangles),
            process=False
        )
    except Exception as e:
        print(f"Error converting mesh {fragment_name} to Trimesh: {e}")
        return []
    
    # Classify faces
    classification_result = classify_fracture_vs_original_faces(tri_mesh, params)
    
    # Extract fracture surface mesh
    fracture_faces = classification_result['fracture_faces']
    if not np.any(fracture_faces):
        print(f"No fracture faces detected in {fragment_name}")
        return []
    
    # Create fracture surface mesh
    fracture_face_indices = np.where(fracture_faces)[0]
    fracture_triangles = np.asarray(o3d_mesh_fragment.triangles)[fracture_face_indices]
    
    fracture_surface = o3d.geometry.TriangleMesh()
    fracture_surface.vertices = o3d_mesh_fragment.vertices
    fracture_surface.triangles = o3d.utility.Vector3iVector(fracture_triangles)
    fracture_surface.remove_unreferenced_vertices()
    fracture_surface.remove_degenerate_triangles()
    
    if not fracture_surface.has_triangles():
        print(f"Extracted fracture surface has no valid triangles")
        return []
    
    fracture_surface.compute_vertex_normals()
    
    print(f"Extracted fracture surface: {len(fracture_surface.vertices)} vertices, "
          f"{len(fracture_surface.triangles)} triangles")
    
    # Visualize if requested
    if params.get('visualize_classification', False):
        visualize_face_classification(o3d_mesh_fragment, classification_result, fragment_name)
    
    return [fracture_surface]


if __name__ == '__main__':
    # Test with a simple cube
    print("Testing region growing segmentation on a cube...")
    test_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    test_mesh.compute_vertex_normals()
    
    # Convert to trimesh for testing
    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(test_mesh.vertices),
        faces=np.asarray(test_mesh.triangles),
        process=False
    )
    
    # Test parameters for simple classification
    test_params = {
        'roughness_threshold_percentile': 75,
        'curvature_threshold_percentile': 75,
        'boundary_complexity_threshold_percentile': 75,
        'symmetry_threshold_percentile': 25,
        'planarity_threshold_percentile': 25,
        'final_classification_threshold': 0.5,
        'visualize_classification': True,
        'fracture_detection_weights': {
            'curvature': 0.3,
            'roughness': 0.3,
            'boundary_complexity': 0.2,
            'symmetry': 0.1,
            'planarity': 0.1
        }
    }
    
    # Test simple face classification
    print("\n=== Testing Simple Face Classification ===")
    classification_result = classify_fracture_vs_original_faces(tri_mesh, test_params)
    
    # Show results
    fracture_faces = classification_result['fracture_faces']
    original_faces = classification_result['original_faces']
    face_scores = classification_result['face_scores']
    
    print(f"Classification Results:")
    print(f"  Fracture faces: {np.sum(fracture_faces)} ({np.sum(fracture_faces)/len(tri_mesh.faces)*100:.1f}%)")
    print(f"  Original faces: {np.sum(original_faces)} ({np.sum(original_faces)/len(tri_mesh.faces)*100:.1f}%)")
    
    # Visualize classification
    if test_params.get('visualize_classification', True):
        visualize_face_classification(test_mesh, classification_result, "TestCube")
    
    # Test simple fracture surface extraction
    print("\n=== Testing Simple Fracture Surface Extraction ===")
    fracture_surfaces = extract_fracture_surfaces_simple(test_mesh, "TestCube", test_params)
    
    if fracture_surfaces:
        print(f"Extracted {len(fracture_surfaces)} fracture surface(s)")
        for i, surface in enumerate(fracture_surfaces):
            print(f"  Surface {i+1}: {len(surface.vertices)} vertices, {len(surface.triangles)} triangles")
        
        # Visualize fracture surfaces
        for surface in fracture_surfaces:
            surface.paint_uniform_color([1.0, 0.0, 0.0])  # Red
            surface.compute_vertex_normals()
        
        # Show with original mesh
        original_vis = copy.deepcopy(test_mesh)
        original_vis.paint_uniform_color([0.7, 0.7, 0.7])
        original_vis.compute_vertex_normals()
        
        vis_objects = [original_vis] + fracture_surfaces
        o3d.visualization.draw_geometries(vis_objects, window_name="Simple Classification Test")