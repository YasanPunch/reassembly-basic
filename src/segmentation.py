import open3d as o3d
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy import ndimage
from scipy.spatial import cKDTree
from collections import Counter
import time

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

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


def extract_fracture_surface_mesh(
    o3d_mesh_fragment,
    fragment_name="Unnamed",
    params=None,
    processing_panel=None,
    pre_selected_regions=None,
):
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
    # Handle pre-selected regions (for reconstruction from previous selection)
    if pre_selected_regions is not None:
        print(
            f"\n    Using pre-selected regions for {fragment_name}: {pre_selected_regions}"
        )
        # Update face_is_fracture_candidate based on pre-selected regions
        face_is_fracture_candidate.fill(False)
        for region_id in pre_selected_regions:
            if region_id < len(region_properties):
                face_is_fracture_candidate[region_properties[region_id]["faces"]] = True
        # Continue to fracture surface creation without showing dialog
    # If interactive visualization is enabled, show dialog for user selection
    elif params["visualize_segmentation"] and len(regions) > 0:
        print(f"\n    Visualizing {len(regions)} regions for interactive selection...")

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

        if not drawable_segment_infos:
            print("    No valid segments to display")
            return None

        # Asynchronous approach - show dialog and let user interact
        _show_segmentation_dialog_async(
            drawable_segment_infos, fragment_name, params, processing_panel
        )
        # Return None for now - the result will be handled by the callback
        return None

    # No console fallback - only interactive visualization is supported
    elif not params['visualize_segmentation'] and len(regions) > 0 and not params['use_bumpiness_detection']:
        print(f"\n    No interactive visualization enabled for {fragment_name}")
        print(
            f"    Enable 'visualize_segmentation' in parameters to select regions interactively"
        )
        return None

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
        cluster_mesh.vertices = o3d_mesh_fragment.vertices  # Use full vertex set
        cluster_mesh.triangles = o3d.utility.Vector3iVector(cluster_triangles)
        cluster_mesh.compute_vertex_normals()
        color = get_color(idx, len(merged_clusters))
        cluster_mesh.paint_uniform_color(color)
        print(f"[DEBUG] Merged regions {merged_idxs} into face {idx} (color {idx}), triangles: {len(cluster_triangles)}, vertices: {len(cluster_mesh.vertices)}, color: {color}")
        # Create wireframe for base mesh
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh_fragment)
        wireframe.paint_uniform_color([0.5, 0.5, 0.5])

        # Skip debug visualization during reconstruction phase to avoid multiple windows
        # Debug visualization is only useful during initial segmentation
        if processing_panel is not None and pre_selected_regions is None:
            processing_panel.show_debug_visualization(
                [wireframe, cluster_mesh], f"[DEBUG] Merged Face {idx} (Color {idx})"
            )
        else:
            # Skip visualization during reconstruction or standalone mode
            print(
                f"[DEBUG] Merged Face {idx} visualization skipped (reconstruction phase or no processing panel)"
            )
    # Store merged faces as separate fracture surfaces for downstream processing
    merged_fracture_surfaces = []
    for cluster_faces, merged_idxs in merged_clusters:
        if not cluster_faces:
            continue
        cluster_faces_arr = np.array(sorted(cluster_faces), dtype=np.int32)
        cluster_triangles = all_triangles[cluster_faces_arr]
        cluster_mesh = o3d.geometry.TriangleMesh()
        cluster_mesh.vertices = o3d_mesh_fragment.vertices
        cluster_mesh.triangles = o3d.utility.Vector3iVector(cluster_triangles)
        cluster_mesh.remove_unreferenced_vertices()
        cluster_mesh.remove_degenerate_triangles()
        if cluster_mesh.has_triangles():
            cluster_mesh.compute_vertex_normals()
            merged_fracture_surfaces.append(cluster_mesh)
    # Return merged fracture surfaces for this fragment
    return merged_fracture_surfaces


# This function has been replaced by _show_segmentation_dialog_async
# and is no longer used in the new asynchronous approach


def _interactive_selection_standalone(drawable_segment_infos, fragment_name, params):
    """
    Original standalone interactive selection approach using key callbacks.
    """
    shared_state = {
        "confirmed_selection": False,
        "quit_without_selection": False,
        "current_page": 0,
    }
    PAGE_SIZE = 10
    highlight_color = np.array([0.0, 0.0, 0.0])  # Black highlight

    num_total_segments = len(drawable_segment_infos)
    num_pages = (num_total_segments + PAGE_SIZE - 1) // PAGE_SIZE

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"Select: {fragment_name} (Page 1/{num_pages}. N/P=Page. S=Confirm. Q=Skip.)",
        width=1280,
        height=960,
    )

    for info in drawable_segment_infos:
        vis.add_geometry(info["mesh"])
        if info["selected"]:
            info["mesh"].paint_uniform_color(highlight_color)

    def print_current_page_and_selection():
        page_idx = shared_state["current_page"]
        global_start = page_idx * PAGE_SIZE + 1
        global_end = min((page_idx + 1) * PAGE_SIZE, num_total_segments)

        print(
            f"\n  --- Page {page_idx + 1}/{num_pages} (Regions {global_start}-{global_end}) ---"
        )
        print(f"  Keys 1-9, 0 (for 10th) toggle selection.")

        # Show properties for visible regions
        for i in range(
            page_idx * PAGE_SIZE, min((page_idx + 1) * PAGE_SIZE, num_total_segments)
        ):
            if i < len(drawable_segment_infos):
                info = drawable_segment_infos[i]
                props = info["properties"]
                selected_marker = "*" if info["selected"] else " "
                print(
                    f"  {selected_marker}[{(i % PAGE_SIZE) + 1}] Region {props['index']+1}: "
                    f"{props['num_faces']} faces ({props['area_fraction']*100:.1f}%)"
                )
                if params.get("use_bumpiness_detection"):
                    print(f"       Bumpiness: {props['bumpiness']:.4f}")

        selected_ids = sorted(
            [info["id"] + 1 for info in drawable_segment_infos if info["selected"]]
        )
        print(f"  Selected: {selected_ids if selected_ids else 'None'}")

    print_current_page_and_selection()

    def toggle_segment_on_current_page(visualizer, key_idx):
        page_idx = shared_state["current_page"]
        segment_idx = page_idx * PAGE_SIZE + key_idx

        if 0 <= segment_idx < num_total_segments:
            info = drawable_segment_infos[segment_idx]
            info["selected"] = not info["selected"]

            if info["selected"]:
                info["mesh"].paint_uniform_color(highlight_color)
            else:
                info["mesh"].paint_uniform_color(info["base_color"])

            visualizer.update_geometry(info["mesh"])
            print_current_page_and_selection()

        return False

    # Register key callbacks
    for i in range(PAGE_SIZE):
        key_char = str((i + 1) % 10)
        vis.register_key_callback(
            ord(key_char), lambda v, idx=i: toggle_segment_on_current_page(v, idx)
        )

    def change_page(visualizer, direction):
        old_page = shared_state["current_page"]
        shared_state["current_page"] = (
            shared_state["current_page"] + direction + num_pages
        ) % num_pages
        if old_page != shared_state["current_page"]:
            print_current_page_and_selection()
        return False

    vis.register_key_callback(ord("N"), lambda v: change_page(v, 1))
    vis.register_key_callback(ord("P"), lambda v: change_page(v, -1))

    def confirm_and_close(visualizer):
        shared_state["confirmed_selection"] = True
        print("\n  Selection Confirmed. Closing...")
        visualizer.close()
        return False

    def quit_and_close(visualizer):
        shared_state["quit_without_selection"] = True
        print("\n  Selection Aborted. Closing...")
        visualizer.close()
        return False

    vis.register_key_callback(ord("S"), confirm_and_close)
    vis.register_key_callback(ord("Q"), quit_and_close)

    print("\n=== Interactive Region Selection ===")
    print(f"  Fragment: {fragment_name}")
    print("  N/P: Navigate pages | 1-9,0: Toggle selection")
    print("  S: Save selection | Q: Quit without saving")

    vis.run()
    vis.destroy_window()

    if shared_state["confirmed_selection"]:
        selected_regions = [
            info["id"] for info in drawable_segment_infos if info["selected"]
        ]
        print(f"\n    User selected {len(selected_regions)} regions")
        return selected_regions
    elif shared_state["quit_without_selection"]:
        print(f"\n    User quit selection. No regions selected.")
        return None

    return None


def _show_segmentation_dialog_async(
    drawable_segment_infos, fragment_name, params, processing_panel
):
    """
    Asynchronous segmentation dialog that shows one window at a time and waits for user input.
    This function creates the dialog and window, then returns immediately. The result is handled
    by callbacks that trigger the next step in the processing pipeline.
    """
    # Create a new window with scene widget using app.py functions
    scene_id = f"segmentation_{fragment_name}"
    scene_widget = processing_panel.app.add_scene_widget(
        scene_id, title=f"Segmentation: {fragment_name}", width=1000, height=800
    )

    # Add all segment meshes to the scene
    for info in drawable_segment_infos:
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        if info["selected"]:
            material.base_color = [0.0, 0.0, 0.0, 1.0]  # Black for selected
        else:
            material.base_color = [*info["base_color"], 1.0]

        scene_widget.scene.add_geometry(f"segment_{info['id']}", info["mesh"], material)

    # Set camera for the scene
    bounds = scene_widget.scene.bounding_box
    scene_widget.setup_camera(60, bounds, bounds.get_center())

    # Create dialog for user interaction
    em = processing_panel.app.window.theme.font_size
    dlg = gui.Dialog(f"Select Fracture Surfaces: {fragment_name}")

    dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

    # Add description
    dlg_layout.add_child(gui.Label("Select fracture surface regions:"))
    dlg_layout.add_child(
        gui.Label("Look at the visualization window to see the segments")
    )

    # Create scrollable list of segments
    scroll = gui.ScrollableVert(em, gui.Margins(0, 0, 0, 0))

    # Track selection state
    selection_state = {info["id"]: info["selected"] for info in drawable_segment_infos}

    def update_visualization():
        """Update the visualization based on current selection"""
        for info in drawable_segment_infos:
            # Remove the existing geometry
            scene_widget.scene.remove_geometry(f"segment_{info['id']}")

            # Create new material with updated color
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            if selection_state[info["id"]]:
                material.base_color = [0.0, 0.0, 0.0, 1.0]  # Black for selected
            else:
                material.base_color = [*info["base_color"], 1.0]

            # Re-add the geometry with new material
            scene_widget.scene.add_geometry(
                f"segment_{info['id']}", info["mesh"], material
            )

    # Create checkboxes for each segment
    for info in drawable_segment_infos:
        props = info["properties"]
        checkbox = gui.Checkbox(
            f"Region {props['index']+1}: {props['num_faces']} faces ({props['area_fraction']*100:.1f}%)"
        )
        checkbox.checked = info["selected"]

        def on_toggle(checked, segment_id=info["id"]):
            selection_state[segment_id] = checked
            update_visualization()

        checkbox.set_on_checked(on_toggle)
        scroll.add_child(checkbox)

        # Add bumpiness info if available
        if params.get("use_bumpiness_detection") and props.get("bumpiness", 0) > 0:
            bumpiness_label = gui.Label(f"    Bumpiness: {props['bumpiness']:.4f}")
            scroll.add_child(bumpiness_label)

    dlg_layout.add_child(scroll)

    # Add buttons
    button_layout = gui.Horiz()

    select_all_btn = gui.Button("Select All")
    select_none_btn = gui.Button("Select None")
    confirm_btn = gui.Button("Confirm")
    cancel_btn = gui.Button("Cancel")

    def on_select_all():
        for info in drawable_segment_infos:
            selection_state[info["id"]] = True
        update_visualization()
        # Update checkboxes
        for child in scroll.get_children():
            if isinstance(child, gui.Checkbox):
                child.checked = True

    def on_select_none():
        for info in drawable_segment_infos:
            selection_state[info["id"]] = False
        update_visualization()
        # Update checkboxes
        for child in scroll.get_children():
            if isinstance(child, gui.Checkbox):
                child.checked = False

    def on_confirm():
        # Update the original drawable_segment_infos with selection
        for info in drawable_segment_infos:
            info["selected"] = selection_state[info["id"]]

        # Get selected regions
        selected_regions = [
            info["id"] for info in drawable_segment_infos if info["selected"]
        ]

        print(
            f"\n    User selected {len(selected_regions)} regions for {fragment_name}"
        )

        # Reset the view before closing
        processing_panel.app.window.set_needs_layout()

        # Clean up and close
        processing_panel.app.remove_scene_widget(scene_id)
        processing_panel.app.window.close_dialog()

        # Trigger next step in processing pipeline
        processing_panel.continue_segmentation_pipeline(fragment_name, selected_regions)

    def on_cancel():
        print(
            f"\n    User cancelled selection for {fragment_name}. No regions selected."
        )

        # Clean up and close
        processing_panel.app.remove_scene_widget(scene_id)
        processing_panel.app.window.close_dialog()

        # Trigger next step in processing pipeline with no selection
        processing_panel.continue_segmentation_pipeline(fragment_name, [])

    select_all_btn.set_on_clicked(on_select_all)
    select_none_btn.set_on_clicked(on_select_none)
    confirm_btn.set_on_clicked(on_confirm)
    cancel_btn.set_on_clicked(on_cancel)

    button_layout.add_child(select_all_btn)
    button_layout.add_child(select_none_btn)
    button_layout.add_child(confirm_btn)
    button_layout.add_child(cancel_btn)

    dlg_layout.add_child(button_layout)
    dlg.add_child(dlg_layout)

    # Show dialog and return immediately (non-blocking)
    processing_panel.app.window.show_dialog(dlg)
