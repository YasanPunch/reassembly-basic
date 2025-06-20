import open3d as o3d
import trimesh
import numpy as np
import copy
import math # For cos, radians

# For color mapping if needed:
try:
    import matplotlib.pyplot as plt
    # Qualitative colormaps are better for distinct categories
    if plt.colormaps.get('tab20'): cmap_qualitative = plt.cm.get_cmap('tab20', 20)
    elif plt.colormaps.get('Pastel1'): cmap_qualitative = plt.cm.get_cmap('Pastel1', 20)
    else: cmap_qualitative = plt.cm.get_cmap('viridis', 20) # Fallback
except ImportError:
    print("Matplotlib not installed, some color features in visualization might be limited.")
    plt = None
    cmap_qualitative = None

# --- UTILITY FUNCTIONS (largely unchanged) ---
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
        # Ensure cmap_name is valid or fallback
        valid_cmap_name = cmap_name
        if plt and cmap_name not in plt.colormaps():
            print(f"Warning: Colormap '{cmap_name}' not found. Falling back to 'tab10'.")
            valid_cmap_name = 'tab10'
            if 'tab10' not in plt.colormaps(): # Further fallback
                valid_cmap_name = 'viridis'

        base_cmap = plt.cm.get_cmap(valid_cmap_name) if plt else None
        if not base_cmap: # Fallback if matplotlib or specific cmap failed
            colors_fallback = [[1,0,0],[0,0,1],[0,1,0],[1,1,0],[1,0,1],[0,1,1],
                               [0.8,0.5,0.2],[0.5,0.2,0.8],[0.2,0.8,0.5], [0.6,0.6,0.6]]
            return colors_fallback[index % len(colors_fallback)]

        num_base_colors = base_cmap.N if base_cmap.N > 0 else 20 # Default if N is not defined or 0

        base_color_index = index % num_base_colors
        variation_cycle = (index // num_base_colors) % num_variations

        r, g, b, _ = base_cmap(base_color_index / (num_base_colors -1) if num_base_colors > 1 else 0.0)


        if variation_cycle == 0: # Use original color
            pass
        elif variation_cycle == 1: # Make it slightly lighter
            factor = 1.3
            r = min(1.0, r * factor + 0.1); g = min(1.0, g * factor + 0.1); b = min(1.0, b * factor + 0.1)
        elif variation_cycle == 2: # Make it slightly darker
            factor = 0.7
            r *= factor; g *= factor; b *= factor
        # Add more variation cycles if num_variations > 3

        return np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)

    except Exception as e:
        print(f"Error in get_color: {e}. Using simple fallback.")
        colors_fallback = [[1,0,0],[0,0,1],[0,1,0]]
        return colors_fallback[index % len(colors_fallback)]

# --- NEW CORE SEGMENTATION FUNCTIONS (based on Papaioannou et al. 2000) ---

def segment_mesh_by_region_growing(tri_mesh, max_angle_deviation_deg):
    """
    Segments a mesh using region growing based on normal similarity.
    Args:
        tri_mesh (trimesh.Trimesh): The input mesh. Must have face_adjacency computed.
                                    Ensure tri_mesh.process() has been called or process=True on load.
        max_angle_deviation_deg (float): Maximum allowed angle (in degrees)
                                         between a face normal and the region's average normal.
    Returns:
        list: A list of lists, where each inner list contains the face indices
              belonging to a grown region.
    """
    num_faces = len(tri_mesh.faces)
    if num_faces == 0:
        return []

    # Ensure face_adjacency is available. Trimesh typically computes this if `process=True`
    # or after a call to `tri_mesh.process()`.
    if not hasattr(tri_mesh, 'face_adjacency') or tri_mesh.face_adjacency is None:
        print("    Segmenter: Warning - tri_mesh.face_adjacency not found. Attempting to process mesh.")
        tri_mesh.process() # This should compute adjacencies
        if not hasattr(tri_mesh, 'face_adjacency') or tri_mesh.face_adjacency is None:
            print("    Segmenter: Critical Error - tri_mesh.face_adjacency still not available after processing.")
            # As a fallback, if adjacencies can't be found, treat every face as its own region.
            return [[i] for i in range(num_faces)]


    eN_threshold = math.cos(math.radians(max_angle_deviation_deg))

    face_normals = tri_mesh.face_normals.copy()
    face_areas = tri_mesh.area_faces.copy()

    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    valid_normals_mask = norms.flatten() > 1e-9
    face_normals[valid_normals_mask] /= norms[valid_normals_mask]
    face_normals[~valid_normals_mask] = np.array([0,0,1])

    assigned_to_region = np.full(num_faces, -1, dtype=int)
    regions = []
    current_region_id = 0

    # Build an adjacency list for faster neighbor lookup
    # tri_mesh.face_adjacency is a list of pairs [face_idx_1, face_idx_2]
    adj = [[] for _ in range(num_faces)]
    for i, j in tri_mesh.face_adjacency:
        adj[i].append(j)
        adj[j].append(i)

    for seed_face_idx in range(num_faces):
        if assigned_to_region[seed_face_idx] != -1:
            continue

        current_region_faces = []
        q = [seed_face_idx]
        assigned_to_region[seed_face_idx] = current_region_id
        
        current_region_weighted_N_sum = face_normals[seed_face_idx] * face_areas[seed_face_idx]
        current_region_total_area = face_areas[seed_face_idx]
        
        if current_region_total_area > 1e-9 and np.linalg.norm(current_region_weighted_N_sum) > 1e-9:
            current_region_N_ave = current_region_weighted_N_sum / np.linalg.norm(current_region_weighted_N_sum)
        else:
            current_region_N_ave = face_normals[seed_face_idx]

        head = 0
        while head < len(q):
            face_k_idx = q[head]; head+=1
            current_region_faces.append(face_k_idx)

            # Use the precomputed adjacency list 'adj'
            for neighbor_face_idx in adj[face_k_idx]: # <--- MODIFIED LINE
                if assigned_to_region[neighbor_face_idx] == -1:
                    N_neighbor = face_normals[neighbor_face_idx]
                    
                    if np.dot(N_neighbor, current_region_N_ave) >= eN_threshold:
                        assigned_to_region[neighbor_face_idx] = current_region_id
                        q.append(neighbor_face_idx)
                        
                        current_region_weighted_N_sum += N_neighbor * face_areas[neighbor_face_idx]
                        current_region_total_area += face_areas[neighbor_face_idx]
                        
                        if current_region_total_area > 1e-9 and np.linalg.norm(current_region_weighted_N_sum) > 1e-9:
                             current_region_N_ave = current_region_weighted_N_sum / np.linalg.norm(current_region_weighted_N_sum)

        if current_region_faces:
            regions.append(np.array(current_region_faces, dtype=int))
            current_region_id += 1
            
    return regions

def cleanup_small_regions(tri_mesh, regions, min_region_area_percentage, face_adjacency_list): # Added face_adjacency_list
    """
    Merges small regions into larger adjacent regions.
    Args:
        tri_mesh (trimesh.Trimesh): The input mesh.
        regions (list of np.ndarray): List of regions (face indices).
        min_region_area_percentage (float): Minimum area percentage for a region.
        face_adjacency_list (list of lists): Precomputed adjacency list where adj[i] are neighbors of face i.
    Returns:
        list: A new list of cleaned regions.
    """
    if not regions:
        return []

    total_mesh_area = tri_mesh.area
    if total_mesh_area < 1e-9:
        return [r for r in regions if len(r) > 0]

    area_threshold = (min_region_area_percentage / 100.0) * total_mesh_area
    
    region_info = []
    face_to_region_map = {} # To quickly find which region a face belongs to

    for i, r_faces in enumerate(regions):
        if len(r_faces) == 0:
            region_info.append({'id': i, 'faces': np.array([], dtype=int), 'area': 0, 'avg_normal': np.array([0,0,0]), 'merged_into': -1})
            continue
        area = np.sum(tri_mesh.area_faces[r_faces])
        
        weighted_normals_sum = np.sum(tri_mesh.face_normals[r_faces] * tri_mesh.area_faces[r_faces, np.newaxis], axis=0)
        norm_sum = np.linalg.norm(weighted_normals_sum)
        avg_normal = weighted_normals_sum / norm_sum if norm_sum > 1e-9 else tri_mesh.face_normals[r_faces[0]]
        
        region_info.append({'id': i, 'faces': r_faces.copy(), 'area': area, 'avg_normal': avg_normal, 'merged_into': -1}) # Use copy for faces
        for face_idx in r_faces:
            face_to_region_map[face_idx] = i # Map face to its original region index (before sorting/merging)


    max_passes = max(5, len(regions) // 10) if len(regions) > 10 else 5 # Heuristic for passes
    for _pass_num in range(max_passes):
        merges_in_pass = 0
        
        # Build a current mapping from face index to *active* region_info index (in the region_info list)
        # This map needs to be dynamic if regions merge and their 'id's become complex to track
        # Simpler: work with region_info directly, which tracks merges via 'merged_into'
        
        active_region_indices_sorted_by_area = sorted(
            [idx for idx, r_info_entry in enumerate(region_info) if r_info_entry['merged_into'] == -1 and len(r_info_entry['faces']) > 0],
            key=lambda idx: region_info[idx]['area']
        )


        for r_info_idx in active_region_indices_sorted_by_area: # r_info_idx is index in region_info list
            current_small_region_info = region_info[r_info_idx]

            if current_small_region_info['area'] >= area_threshold: # No longer small enough to merge
                continue

            adjacent_region_candidates = {} # key: neighbor_region_info_idx, value: {'similarity': float}
            
            for face_in_small_r in current_small_region_info['faces']:
                # Use the passed face_adjacency_list
                for neighbor_face_orig_mesh in face_adjacency_list[face_in_small_r]: # <--- MODIFIED LINE
                    
                    # Find which active region this neighbor_face_orig_mesh belongs to
                    if neighbor_face_orig_mesh in face_to_region_map:
                        original_neighbor_region_id = face_to_region_map[neighbor_face_orig_mesh]
                        
                        # Find the current region_info entry for this original_neighbor_region_id
                        # This is tricky if merges have happened. We need to trace to the *current* active region.
                        target_region_info_idx = -1
                        temp_id_trace = original_neighbor_region_id
                        
                        # Find the current active region_info_idx that original_neighbor_region_id might have merged into
                        # We need to find the entry in region_info list whose 'id' matches original_neighbor_region_id
                        # and then check if it has been merged. If so, trace its merger.

                        # Find the list index corresponding to original_neighbor_region_id
                        # This assumes original_neighbor_region_id maps directly to an initial entry in region_info
                        # This part is complex if we sort region_info or re-index heavily.
                        # Let's find the current active region for neighbor_face_orig_mesh
                        current_neighbor_region_info_idx = -1
                        for temp_idx, temp_r_info in enumerate(region_info):
                            if temp_r_info['merged_into'] == -1 and neighbor_face_orig_mesh in temp_r_info['faces']:
                                current_neighbor_region_info_idx = temp_idx
                                break
                        
                        if current_neighbor_region_info_idx != -1 and current_neighbor_region_info_idx != r_info_idx:
                            # Target region must not be itself
                            neighbor_actual_info = region_info[current_neighbor_region_info_idx]
                            if neighbor_actual_info['area'] > current_small_region_info['area'] or \
                               neighbor_actual_info['area'] >= area_threshold:
                                
                                if current_neighbor_region_info_idx not in adjacent_region_candidates:
                                    similarity = np.dot(current_small_region_info['avg_normal'], neighbor_actual_info['avg_normal'])
                                    adjacent_region_candidates[current_neighbor_region_info_idx] = {'similarity': similarity}
            
            if adjacent_region_candidates:
                best_target_info_idx = max(adjacent_region_candidates.keys(), key=lambda k: adjacent_region_candidates[k]['similarity'])
                
                # Perform merge
                target_region_info = region_info[best_target_info_idx]
                small_region_original_faces = current_small_region_info['faces'] # Save before emptying

                # Update faces (concatenate and unique)
                target_region_info['faces'] = np.unique(np.concatenate((target_region_info['faces'], current_small_region_info['faces'])))
                
                # Update area and avg_normal of the target region
                new_weighted_sum = (target_region_info['avg_normal'] * target_region_info['area']) + \
                                   (current_small_region_info['avg_normal'] * current_small_region_info['area'])
                target_region_info['area'] += current_small_region_info['area'] # Update area first
                
                new_norm_sum = np.linalg.norm(new_weighted_sum)
                target_region_info['avg_normal'] = new_weighted_sum / new_norm_sum if new_norm_sum > 1e-9 and target_region_info['area'] > 1e-9 else target_region_info['avg_normal']

                # Mark small region as merged
                current_small_region_info['merged_into'] = target_region_info['id'] # merge into original id of target
                current_small_region_info['faces'] = np.array([], dtype=int)
                current_small_region_info['area'] = 0
                merges_in_pass += 1

                # Update face_to_region_map for faces that were in the small_region
                # They now belong to the target region (identified by its original ID)
                for face_idx_moved in small_region_original_faces:
                     face_to_region_map[face_idx_moved] = target_region_info['id']

        if merges_in_pass == 0:
            break

    final_cleaned_regions = []
    for r_info_entry in region_info:
        if r_info_entry['merged_into'] == -1 and len(r_info_entry['faces']) > 0:
            final_cleaned_regions.append(r_info_entry['faces'])
            
    return final_cleaned_regions

# --- ROUGHNESS CALCULATION (adapted from existing `calculate_cluster_curvature`) ---
def calculate_region_roughness(tri_mesh, region_face_indices):
    """
    Calculates a roughness metric for a region (std dev of normal angles to average normal).
    Args:
        tri_mesh (trimesh.Trimesh): The mesh.
        region_face_indices (np.ndarray): Indices of faces in the region.
    Returns:
        float: Roughness metric (0 for perfectly flat, higher for rougher).
    """
    if len(region_face_indices) < 2: # Need at least 2 faces for std dev
        return 0.0
        
    cluster_normals = tri_mesh.face_normals[region_face_indices]
    
    avg_normal = np.mean(cluster_normals, axis=0)
    norm_avg = np.linalg.norm(avg_normal)
    if norm_avg < 1e-9:
        return np.pi # Max roughness if average normal is zero (highly inconsistent normals)
    avg_normal /= norm_avg
    
    # Calculate angular deviations from average normal
    cos_angles = np.clip(np.dot(cluster_normals, avg_normal), -1.0, 1.0)
    deviations_rad = np.arccos(cos_angles)
    
    roughness = np.std(deviations_rad)
    return roughness

# --- MAIN SEGMENTATION ORCHESTRATOR ---
def extract_fracture_surface_mesh(o3d_mesh_fragment, fragment_name="Unnamed", params=None):
    params = params or {}
    # ... (initial print and trimesh conversion logic - ensure process=True for trimesh object) ...
    # Ensure tri_mesh is created with process=True
    try:
        tri_mesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh_fragment.vertices),
                                   faces=np.asarray(o3d_mesh_fragment.triangles),
                                   vertex_normals=np.asarray(o3d_mesh_fragment.vertex_normals) if o3d_mesh_fragment.has_vertex_normals() else None,
                                   process=True) # ENSURE THIS IS TRUE
        # ... (rest of the error checking for tri_mesh) ...
    except Exception as e:
        print(f"    Segmenter: Error converting/processing O3D mesh {fragment_name} to Trimesh: {e}")
        return None
    if tri_mesh.faces is None or len(tri_mesh.faces) == 0:
         print(f"    Segmenter: Mesh {fragment_name} has no faces after trimesh conversion.")
         return None

    # 1. Perform Region Growing Segmentation
    max_angle_dev = params.get('max_angle_deviation_deg', 30.0)
    print(f"    Segmenter [{fragment_name}]: Performing region growing (max_angle_dev: {max_angle_dev}°)...")
    raw_regions = segment_mesh_by_region_growing(tri_mesh, max_angle_dev)
    print(f"    Segmenter [{fragment_name}]: Found {len(raw_regions)} raw regions.")

    # Precompute face_adjacency_list for cleanup_small_regions
    num_faces_for_adj = len(tri_mesh.faces)
    face_adj_list_for_cleanup = [[] for _ in range(num_faces_for_adj)]
    if hasattr(tri_mesh, 'face_adjacency') and tri_mesh.face_adjacency is not None:
        for i_adj, j_adj in tri_mesh.face_adjacency:
            face_adj_list_for_cleanup[i_adj].append(j_adj)
            face_adj_list_for_cleanup[j_adj].append(i_adj)
    else:
        print("    Segmenter: Warning - face_adjacency not available for cleanup. Cleanup might be suboptimal.")


    # 2. Perform Cleanup of Small Regions
    min_area_perc = params.get('min_region_area_percentage', 2.0)
    print(f"    Segmenter [{fragment_name}]: Cleaning up small regions (min_area_perc: {min_area_perc}%)...")
    cleaned_regions = cleanup_small_regions(tri_mesh, raw_regions, min_area_perc, face_adj_list_for_cleanup) # Pass adj list
    print(f"    Segmenter [{fragment_name}]: {len(cleaned_regions)} regions after cleanup.")

    # ... (rest of extract_fracture_surface_mesh: filtering, interactive/auto selection, final mesh creation) ...
    min_final_size = params.get("min_final_segment_size_after_cleanup", 10)
    final_segments_for_selection = [seg for seg in cleaned_regions if len(seg) >= min_final_size]
    print(f"    Segmenter [{fragment_name}]: {len(final_segments_for_selection)} segments after min size filter ({min_final_size} faces).")

    if not final_segments_for_selection:
        print(f"    Segmenter [{fragment_name}]: No segments remaining after processing.")
        return None
    
    # ... (The rest of the interactive/auto selection logic remains the same) ...
    selected_segment_indices_from_user = []
    shared_state = {'confirmed_selection': False, 'quit_without_selection': False, 'current_page': 0}
    PAGE_SIZE = 10

    if params.get('visualize_segmentation', False):
        # ... (interactive selection code using drawable_segment_infos) ...
        # (This part seems okay, ensure it uses final_segments_for_selection correctly)
        print(f"    Segmenter [{fragment_name}]: Visualizing {len(final_segments_for_selection)} segments for interactive selection...")
        
        drawable_segment_infos = []
        highlight_color = np.array([0.0, 0.0, 0.0]) # BLACK highlight

        for i, seg_faces_indices in enumerate(final_segments_for_selection):
            if len(seg_faces_indices) == 0: continue
            seg_mesh_o3d = o3d.geometry.TriangleMesh()
            seg_mesh_o3d.vertices = o3d_mesh_fragment.vertices
            seg_mesh_o3d.triangles = o3d.utility.Vector3iVector(tri_mesh.faces[seg_faces_indices])
            seg_mesh_o3d.remove_unreferenced_vertices() # Important
            if not seg_mesh_o3d.has_vertices() or not seg_mesh_o3d.has_triangles(): continue
            
            seg_mesh_o3d.compute_vertex_normals()
            base_color = get_color(i, len(final_segments_for_selection))
            seg_mesh_o3d.paint_uniform_color(base_color)
            drawable_segment_infos.append({'mesh': seg_mesh_o3d, 'id': i, 
                                           'base_color': base_color, 'selected': False})
        
        if not drawable_segment_infos:
            print(f"    Segmenter [{fragment_name}]: No displayable segments for visualization.")
            shared_state['quit_without_selection'] = True # Treat as if user quit
        else:
            num_total_segments_to_display = len(drawable_segment_infos)
            num_pages = (num_total_segments_to_display + PAGE_SIZE - 1) // PAGE_SIZE

            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name=f"Select: {fragment_name} (Page {shared_state['current_page']+1}/{num_pages}. N/P. S=Save. Q=Quit.)", 
                              width=1280, height=960)
            
            for info in drawable_segment_infos:
                vis.add_geometry(info['mesh'])

            def print_current_page_and_selection(visualizer_ref=None): # Renamed vis to visualizer_ref
                page_idx = shared_state['current_page']
                global_start_seg_num = page_idx * PAGE_SIZE + 1
                global_end_seg_num = min((page_idx + 1) * PAGE_SIZE, num_total_segments_to_display)
                
                print(f"\n  --- Page {page_idx + 1}/{num_pages} (Segments {global_start_seg_num}-{global_end_seg_num} overall) ---")
                selected_ids_display = sorted([info['id'] + 1 for info in drawable_segment_infos if info['selected']])
                print(f"  Overall Selected segment numbers (1-based): {selected_ids_display if selected_ids_display else 'None'}")
                if visualizer_ref: 
                     render_option = visualizer_ref.get_render_option() # Get render options
                     if render_option: render_option.mesh_show_wireframe = True 
                     # Window title update is tricky with raw VisualizerWithKeyCallback, often set at creation.
                     # For simplicity, we'll rely on console print for title hint.
                     title_hint = f"Select: {fragment_name} (Page {shared_state['current_page']+1}/{num_pages}. N/P. S=Save. Q=Quit.)"
                     print(f"  Window Title Hint: {title_hint}")


            print_current_page_and_selection(vis) 

            def toggle_segment_on_current_page(visualizer, key_on_page_0_to_9):
                page_idx = shared_state['current_page']
                segment_idx_in_drawable_list = page_idx * PAGE_SIZE + key_on_page_0_to_9
                
                if 0 <= segment_idx_in_drawable_list < num_total_segments_to_display:
                    seg_info_item = drawable_segment_infos[segment_idx_in_drawable_list]
                    seg_info_item['selected'] = not seg_info_item['selected']
                    mesh_to_update = seg_info_item['mesh']
                    if seg_info_item['selected']:
                        mesh_to_update.paint_uniform_color(highlight_color)
                    else:
                        mesh_to_update.paint_uniform_color(seg_info_item['base_color'])
                    visualizer.update_geometry(mesh_to_update)
                    print_current_page_and_selection(visualizer)
                return False

            for i in range(PAGE_SIZE):
                key_char = str((i + 1) % 10)
                vis.register_key_callback(ord(key_char), 
                    lambda vis_cb, k_idx=i: toggle_segment_on_current_page(vis_cb, k_idx))
            
            def change_page(visualizer, direction):
                old_page = shared_state['current_page']
                shared_state['current_page'] = (shared_state['current_page'] + direction + num_pages) % num_pages
                if old_page != shared_state['current_page'] or direction == 0:
                    print_current_page_and_selection(visualizer) # Pass visualizer
                return False

            vis.register_key_callback(ord('N'), lambda v_cb: change_page(v_cb, 1)) # Pass 'v_cb'
            vis.register_key_callback(ord('P'), lambda v_cb: change_page(v_cb, -1)) # Pass 'v_cb'
            
            def confirm_and_close(visualizer):
                shared_state['confirmed_selection'] = True
                print("  Selection Confirmed ('S'). Closing window...")
                visualizer.close()
                return False
            
            def quit_and_close(visualizer):
                shared_state['quit_without_selection'] = True
                print("  Selection Aborted ('Q'). Closing window...")
                visualizer.close()
                return False

            vis.register_key_callback(ord('S'), confirm_and_close)
            vis.register_key_callback(ord('Q'), quit_and_close)
            
            vis.run()
            vis.destroy_window()

            if shared_state['confirmed_selection']:
                selected_segment_indices_from_user = [info['id'] for info in drawable_segment_infos if info['selected']]
    
    if not shared_state['confirmed_selection'] and not shared_state['quit_without_selection']:
        if params.get('visualize_segmentation', False):
            print(f"    Interactive selection for {fragment_name} was closed without S/Q. No segments selected.")
            selected_segment_indices_from_user = []
        else: 
            print(f"    Segmenter [{fragment_name}]: Attempting automatic fracture surface selection.")
            roughness_thresh = params.get('fracture_roughness_threshold', 0.15)
            auto_selected_indices = []
            for i, seg_faces in enumerate(final_segments_for_selection):
                roughness = calculate_region_roughness(tri_mesh, seg_faces)
                if roughness > roughness_thresh:
                    auto_selected_indices.append(i)
            
            if auto_selected_indices:
                print(f"    Auto-selected {len(auto_selected_indices)} segments based on roughness > {roughness_thresh}.")
                selected_segment_indices_from_user = auto_selected_indices
            else:
                print(f"    No segments met automatic roughness criterion. Falling back to console input for {fragment_name}.")
                if not final_segments_for_selection:
                    print(f"    Segmenter [{fragment_name}]: No segments available for console selection.")
                    return None

                print("\n=== Fracture Surface Selection (Console Input) ===")
                for i_prompt, seg_prompt_faces in enumerate(final_segments_for_selection):
                    print(f"  Segment {i_prompt + 1} ({len(seg_prompt_faces)} faces, Roughness: {calculate_region_roughness(tri_mesh, seg_prompt_faces):.3f})")
                
                cluster_selection_str = input(f"Enter desired segment numbers (1-{len(final_segments_for_selection)}) for '{fragment_name}' (comma separated, 'all', or 'none'): ")
                temp_selected_indices = []
                if cluster_selection_str.strip().lower() == 'all':
                    temp_selected_indices = list(range(len(final_segments_for_selection)))
                elif cluster_selection_str.strip().lower() == 'none' or not cluster_selection_str.strip():
                    temp_selected_indices = []
                else:
                    try:
                        temp_selected_indices = [int(x.strip()) - 1 for x in cluster_selection_str.split(',') if x.strip()]
                    except ValueError:
                        print("    Invalid input. Defaulting to 'none'.")
                selected_segment_indices_from_user = [idx for idx in temp_selected_indices if 0 <= idx < len(final_segments_for_selection)]

    if not selected_segment_indices_from_user:
        print(f"    Segmenter [{fragment_name}]: No segments selected as fracture surface.")
        return None

    all_selected_faces_list = [] # Changed name to avoid conflict
    for user_selected_idx in selected_segment_indices_from_user:
        if 0 <= user_selected_idx < len(final_segments_for_selection):
            all_selected_faces_list.append(final_segments_for_selection[user_selected_idx])
    
    if not all_selected_faces_list:
        print(f"    Segmenter [{fragment_name}]: No valid faces from selection.")
        return None

    fracture_candidate_faces_indices = np.concatenate(all_selected_faces_list)
    
    fracture_surface_o3d = o3d.geometry.TriangleMesh()
    fracture_surface_o3d.vertices = o3d_mesh_fragment.vertices 
    fracture_surface_o3d.triangles = o3d.utility.Vector3iVector(tri_mesh.faces[fracture_candidate_faces_indices])
    fracture_surface_o3d.remove_unreferenced_vertices()
    fracture_surface_o3d.remove_degenerate_triangles()
    
    if not fracture_surface_o3d.has_triangles():
        print(f"    Segmenter [{fragment_name}]: Extracted fracture surface has no triangles after cleaning.")
        return None
        
    fracture_surface_o3d.compute_vertex_normals()
    print(f"    Segmenter [{fragment_name}]: Extracted fracture surface with {len(fracture_surface_o3d.vertices)}V, {len(fracture_surface_o3d.triangles)}T.")
    print(f"--- Finished Segmentation for: {fragment_name} ---\n")
    return fracture_surface_o3d

# --- VISUALIZATION HELPER (for `preprocessing.py` to call if interactive segmentation shows fracture) ---
def visualize_segmentation(o3d_mesh, fracture_surface, fragment_name="Unnamed"):
    """
    Creates a visualization of the original mesh and the extracted fracture surface.
    Returns a list of Open3D geometries ready for visualization.
    (This is different from the interactive paged selector inside extract_fracture_surface_mesh)
    """
    vis_geometries = []
    
    original_mesh_vis = copy.deepcopy(o3d_mesh)
    original_mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])
    original_mesh_vis.compute_vertex_normals()
    vis_geometries.append(original_mesh_vis)
    
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(original_mesh_vis)
    edges.paint_uniform_color([0.5, 0.5, 0.5])
    vis_geometries.append(edges)
    
    if fracture_surface and fracture_surface.has_triangles():
        fracture_surface_vis = copy.deepcopy(fracture_surface)
        fracture_surface_vis.paint_uniform_color([1.0, 0.0, 0.0]) # Red for fracture
        fracture_surface_vis.compute_vertex_normals()
        vis_geometries.append(fracture_surface_vis)
    
    return vis_geometries


# --- DEPRECATED/REMOVED FUNCTIONS ---
# The following functions from the original segmentation.py are no longer needed
# with the new region-growing approach:
#
# - calculate_face_roughness
# - cluster_faces_by_normal (DBSCAN version)
# - get_segment_connected_components
# - is_segment_normals_coherent_pca
# - is_segment_planar_by_fit
# - calculate_per_face_roughness_metric
# - is_segment_roughness_homogeneous
# - split_segment_by_recluster
# - calculate_pca_badness_score
# - calculate_planar_badness_score
# - refine_segment (the large recursive function)
# - identify_fracture_surface_by_normals (old combined method)
# - identify_fracture_candidate_faces_by_boundary (can be kept if a boundary heuristic is desired *after* region growing)
# - identify_fracture_candidate_faces (old wrapper)
# - visualize_segmentation_clusters (the interactive paged selector is now part of extract_fracture_surface_mesh)

if __name__ == '__main__':
    print("--- Testing New Segmentation Logic ---")
    
    # Create a test mesh: a box with one "bumpy" face
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    vertices = np.asarray(box_mesh.vertices)
    
    # Make the top face (Z=1) bumpy
    top_face_indices_mask = np.isclose(vertices[:, 2], 1.0)
    noise = np.random.uniform(-0.1, 0.1, size=(np.sum(top_face_indices_mask), 3))
    noise[:, :2] = 0 # Only add noise in Z direction for simplicity
    vertices[top_face_indices_mask] += noise[top_face_indices_mask[top_face_indices_mask].shape[0]-noise.shape[0]:] # align indices if needed

    noisy_box_mesh = o3d.geometry.TriangleMesh()
    noisy_box_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    noisy_box_mesh.triangles = box_mesh.triangles
    noisy_box_mesh.compute_vertex_normals() # Important for trimesh conversion

    # Test parameters
    test_params = {
        "max_angle_deviation_deg": 25.0,      # For region growing
        "min_region_area_percentage": 1.0,    # For cleanup
        "visualize_segmentation": True,       # Test interactive selection
        "fracture_roughness_threshold": 0.05, # For automatic selection if interactive is skipped
        "min_final_segment_size_after_cleanup": 5 # Min faces for a segment to be shown/selectable
    }

    print(f"Test Mesh: {len(noisy_box_mesh.vertices)}V, {len(noisy_box_mesh.triangles)}T")
    
    # o3d.visualization.draw_geometries([noisy_box_mesh], window_name="Original Test Mesh")

    extracted_fracture_surf = extract_fracture_surface_mesh(noisy_box_mesh, 
                                                            fragment_name="TestNoisyBox", 
                                                            params=test_params)

    if extracted_fracture_surf and extracted_fracture_surf.has_triangles():
        print(f"Successfully extracted a surface with {len(extracted_fracture_surf.vertices)}V, {len(extracted_fracture_surf.triangles)}T")
        
        # Visualize the final result (original + selected fracture)
        # This is what preprocessing.py would do if 'visualize_segmentation' in main params is true
        # Note: `visualize_segmentation` param for extract_fracture_surface_mesh controls *interactive selection*
        # The call below shows the *result* of that selection.
        
        final_vis_geoms = visualize_segmentation(noisy_box_mesh, extracted_fracture_surf, "TestNoisyBox - Final Result")
        o3d.visualization.draw_geometries(final_vis_geoms, window_name="Segmentation Test - Final Result")
    else:
        print("No fracture surface extracted or selected.")

    print("--- Test Finished ---")