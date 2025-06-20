import open3d as o3d
import trimesh
import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import deque
from scipy import ndimage

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
    
    # Parameter setup with paper's recommendations
    default_params = {
        'max_curvature_deg': params.get('max_curvature_deg', 30.0),  # Paper suggests this range
        'area_limit_fraction': params.get('area_limit_fraction', 0.02),  # 2% as paper suggests
        'visualize_segmentation': params.get('visualize_segmentation', False),
        'elevation_map_resolution': params.get('elevation_map_resolution', 64),
        'bumpiness_threshold': params.get('bumpiness_threshold', 0.2),
        'use_bumpiness_detection': params.get('use_bumpiness_detection', False)
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
    
    # Identify fracture candidates
    face_is_fracture_candidate = np.zeros(len(tri_mesh.faces), dtype=bool)
    selected_regions = []
    
    # If bumpiness detection is enabled, use it to identify rough surfaces
    if params['use_bumpiness_detection'] and any(r['bumpiness'] > 0 for r in region_properties):
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
        print(f"\n    Visualizing {len(regions)} regions for interactive selection...")
        
        shared_state = {'confirmed_selection': False, 'quit_without_selection': False, 'current_page': 0}
        PAGE_SIZE = 10
        
        drawable_segment_infos = []
        highlight_color = np.array([0.0, 0.0, 0.0])  # Black highlight
        
        for i, props in enumerate(region_properties):
            seg_mesh = o3d.geometry.TriangleMesh()
            seg_mesh.vertices = o3d_mesh_fragment.vertices
            seg_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces[props['faces']])
            seg_mesh.remove_unreferenced_vertices()
            
            if not seg_mesh.has_vertices() or not seg_mesh.has_triangles():
                continue
                
            seg_mesh.compute_vertex_normals()
            base_color = get_color(i, len(regions))
            seg_mesh.paint_uniform_color(base_color)
            
            drawable_segment_infos.append({
                'mesh': seg_mesh,
                'id': props['index'],
                'base_color': base_color,
                'selected': props['index'] in selected_regions,
                'properties': props
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
                
                # Show properties for visible regions
                for i in range(page_idx * PAGE_SIZE, min((page_idx + 1) * PAGE_SIZE, num_total_segments)):
                    if i < len(drawable_segment_infos):
                        info = drawable_segment_infos[i]
                        props = info['properties']
                        selected_marker = "*" if info['selected'] else " "
                        print(f"  {selected_marker}[{(i % PAGE_SIZE) + 1}] Region {props['index']+1}: "
                              f"{props['num_faces']} faces ({props['area_fraction']*100:.1f}%)")
                        if params['use_bumpiness_detection']:
                            print(f"       Bumpiness: {props['bumpiness']:.4f}")
                
                selected_ids = sorted([info['id'] + 1 for info in drawable_segment_infos if info['selected']])
                print(f"  Selected: {selected_ids if selected_ids else 'None'}")
            
            print_current_page_and_selection()
            
            def toggle_segment_on_current_page(visualizer, key_idx):
                page_idx = shared_state['current_page']
                segment_idx = page_idx * PAGE_SIZE + key_idx
                
                if 0 <= segment_idx < num_total_segments:
                    info = drawable_segment_infos[segment_idx]
                    info['selected'] = not info['selected']
                    
                    if info['selected']:
                        info['mesh'].paint_uniform_color(highlight_color)
                    else:
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
    
    return fracture_surface_o3d


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


if __name__ == '__main__':
    # Test with a simple cube
    print("Testing region growing segmentation on a cube...")
    test_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    test_mesh.compute_vertex_normals()
    
    # Test parameters
    test_params = {
        'max_curvature_deg': 45.0,
        'area_limit_fraction': 0.1,
        'visualize_segmentation': True,
        'use_bumpiness_detection': False
    }
    
    # Run segmentation
    result = extract_fracture_surface_mesh(test_mesh, "TestCube", test_params)
    
    if result:
        # Visualize results
        vis_geometries = visualize_segmentation(test_mesh, result, "TestCube")
        o3d.visualization.draw_geometries(vis_geometries, window_name="Region Growing Segmentation Test")