import open3d as o3d
import numpy as np
import copy
import pickle # For saving/loading the visualization log
import os

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

def get_color(index, total_items=20): # Added total_items for better cmap indexing
    if cmap_qualitative:
        return cmap_qualitative(index % cmap_qualitative.N if cmap_qualitative.N > 0 else index % total_items)[:3]
    else:
        # Fallback simple colors
        colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0.8,0.5,0.2],[0.5,0.2,0.8],[0.2,0.8,0.5]]
        return colors[index % len(colors)]
    
def _reconstruct_geometry_from_log_entry(entry_data):
    """Helper to reconstruct o3d geometry from logged numpy arrays."""
    geom_type = entry_data.get('type')
    if geom_type == 'mesh':
        mesh = o3d.geometry.TriangleMesh()
        if entry_data.get('vertices') is not None:
            mesh.vertices = o3d.utility.Vector3dVector(entry_data['vertices'])
        if entry_data.get('triangles') is not None:
            mesh.triangles = o3d.utility.Vector3iVector(entry_data['triangles'])
        if entry_data.get('vertex_colors') is not None and len(entry_data['vertex_colors']) > 0:
            mesh.vertex_colors = o3d.utility.Vector3dVector(entry_data['vertex_colors'])
        if entry_data.get('vertex_normals') is not None and len(entry_data['vertex_normals']) > 0:
            mesh.vertex_normals = o3d.utility.Vector3dVector(entry_data['vertex_normals'])
        elif mesh.has_triangles(): # Compute normals if none and has triangles
            mesh.compute_vertex_normals()
        return mesh
    elif geom_type == 'pointcloud':
        pcd = o3d.geometry.PointCloud()
        if entry_data.get('points') is not None:
            pcd.points = o3d.utility.Vector3dVector(entry_data['points'])
        if entry_data.get('colors') is not None and len(entry_data['colors']) > 0:
            pcd.colors = o3d.utility.Vector3dVector(entry_data['colors'])
        if entry_data.get('normals') is not None and len(entry_data['normals']) > 0:
            pcd.normals = o3d.utility.Vector3dVector(entry_data['normals'])
        return pcd
    return None

def draw_registration_result(source, target, transformation, window_name="Registration Result"):
    # ... (remains the same, but ensure geometries are copies if modified)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

def save_visualization_log(log_data, filepath):
    """Saves the visualization log to a file using pickle."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(log_data, f)
        print(f"Visualization log successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving visualization log: {e}")

def load_visualization_log(filepath):
    """Loads a visualization log from a file."""
    try:
        with open(filepath, 'rb') as f:
            log_data = pickle.load(f)
        print(f"Visualization log successfully loaded from {filepath}")
        return log_data
    except Exception as e:
        print(f"Error loading visualization log: {e}")
        return None

def replay_visualization_log(log_filepath, fragments_data_map=None):
    """
    Basic replay of a saved visualization log.
    This will open many windows sequentially.
    A proper interactive viewer is more complex.
    Args:
        log_filepath (str): Path to the pickled log file.
        fragments_data_map (dict, optional): Map of fragment_name to original_index,
                                             used for coloring if available.
                                             If not, colors assigned sequentially.
    """
    log_data = load_visualization_log(log_filepath)
    if not log_data:
        return

    print(f"\n--- Replaying Visualization Log ({len(log_data)} steps) ---")
    print("Press Q in each Open3D window to close it and proceed to the next step.")

    original_index_to_color = {}
    num_unique_frags_for_coloring = 0 # For consistent color cycling

    if fragments_data_map:
        # Get unique original indices to determine total number of distinct colors needed
        # This helps in making colors more consistent if not all original_indices appear early in log
        all_original_indices_in_map = sorted(list(set(fragments_data_map.values())))
        num_unique_frags_for_coloring = len(all_original_indices_in_map)
        for name, orig_idx_from_map in fragments_data_map.items():
            # Map the original_idx to its position in the sorted unique list for consistent color indexing
            if num_unique_frags_for_coloring > 0:
                 color_map_idx = all_original_indices_in_map.index(orig_idx_from_map)
                 original_index_to_color[orig_idx_from_map] = get_color(color_map_idx, num_unique_frags_for_coloring)
            else: # Should not happen if fragments_data_map is not empty
                 original_index_to_color[orig_idx_from_map] = get_color(0) # Fallback
    else: # If no map, we'll assign colors based on original_index as encountered
        print("    Replay: No fragments_data_map provided, colors will be assigned on-the-fly based on original_index.")


    current_assembly_state = {} 

    for i, entry in enumerate(log_data):
        step_type = entry.get('step')
        # Try to get name and original_index from various possible keys used during logging
        frag_name = entry.get('name', entry.get('fragment_name', entry.get('source_name', 'UnknownFrag')))
        orig_idx = entry.get('original_index', entry.get('fragment_idx_in_valid_list', -1)) # Use original_index if available

        print(f"\nLog Entry {i+1}/{len(log_data)}: Type='{step_type}', Name='{frag_name}'")
        
        geometries_to_draw = []
        window_title = f"Log {i+1}: {step_type} - {frag_name}"
        
        # Determine color for this fragment
        # If we have a map and the original_index is in it, use that color
        # Otherwise, assign a color based on the original_index encountered
        if orig_idx != -1 and orig_idx in original_index_to_color:
            frag_color = original_index_to_color[orig_idx]
        elif orig_idx != -1: # orig_idx present but not in precomputed map (e.g. map was empty)
            if orig_idx not in original_index_to_color: # Assign a new color
                 # Use a counter for new colors if no map, or if orig_idx wasn't in map
                 color_assign_idx = len(original_index_to_color)
                 original_index_to_color[orig_idx] = get_color(color_assign_idx, num_unique_frags_for_coloring or 5) # Default to 5 unique colors if num_unique not set
            frag_color = original_index_to_color[orig_idx]
        else: # No original_index available, use sequential color
            frag_color = get_color(i, len(log_data))


        if step_type == 'initial_fragment':
            # ... (rest of the replay logic from previous correct version of visualization_utils.py)
            # Ensure you use the _reconstruct_geometry_from_log_entry helper correctly
            geom = _reconstruct_geometry_from_log_entry(entry)
            if geom:
                geom.paint_uniform_color(frag_color)
                geometries_to_draw.append(geom)
        
        elif step_type == 'segmentation_result':
            orig_mesh_data = { 'type': entry.get('original_mesh_type'), 'vertices': entry.get('original_mesh_vertices'), 'triangles': entry.get('original_mesh_triangles')}
            orig_mesh = _reconstruct_geometry_from_log_entry(orig_mesh_data)
            if orig_mesh:
                orig_mesh.paint_uniform_color(frag_color) # Color original with its consistent color
                geometries_to_draw.append(orig_mesh)

            if entry.get('fracture_mesh_type'):
                fract_mesh_data = {'type': entry.get('fracture_mesh_type'), 'vertices': entry.get('fracture_mesh_vertices'), 'triangles': entry.get('fracture_mesh_triangles')}
                fract_mesh = _reconstruct_geometry_from_log_entry(fract_mesh_data)
                if fract_mesh:
                    fract_mesh.paint_uniform_color([1,0,0]) # Red for fracture
                    geometries_to_draw.append(fract_mesh)
        
        elif step_type == 'dense_sampling_result' or step_type == 'downsampled_pcd_for_features_result':
            pcd = _reconstruct_geometry_from_log_entry(entry)
            if pcd:
                pcd.paint_uniform_color(frag_color)
                geometries_to_draw.append(pcd)

        elif step_type == 'pairwise_match_success':
            source_pcd_data = {'type': 'pointcloud', 'points': entry['source_pcd_points'], 'normals': entry.get('source_pcd_normals')}
            target_pcd_data = {'type': 'pointcloud', 'points': entry['target_pcd_points'], 'normals': entry.get('target_pcd_normals')}
            source_pcd = _reconstruct_geometry_from_log_entry(source_pcd_data)
            target_pcd = _reconstruct_geometry_from_log_entry(target_pcd_data)
            transform = entry['transformation']
            
            if source_pcd and target_pcd:
                # Get original_index for source and target for consistent coloring
                source_orig_idx = fragments_data_map.get(entry['source_name'], -1) if fragments_data_map else -1
                target_orig_idx = fragments_data_map.get(entry['target_name'], -1) if fragments_data_map else -1

                source_color = original_index_to_color.get(source_orig_idx, get_color(0, num_unique_frags_for_coloring or 2))
                target_color = original_index_to_color.get(target_orig_idx, get_color(1, num_unique_frags_for_coloring or 2))
                
                source_pcd.paint_uniform_color(source_color)
                target_pcd.paint_uniform_color(target_color)
                source_pcd.transform(transform)
                geometries_to_draw.extend([source_pcd, target_pcd])
            window_title += f" (Score: {entry.get('score',0):.2f})"


        elif step_type == 'assembly_seed_placed' or step_type == 'assembly_fragment_placed':
            geom_data = {'type': 'mesh', 'vertices': entry['vertices'], 'triangles': entry['triangles']}
            placed_geom = _reconstruct_geometry_from_log_entry(geom_data)
            
            if placed_geom:
                # orig_idx for this placed piece should be in the entry
                placed_frag_color = original_index_to_color.get(orig_idx, get_color(orig_idx if orig_idx !=-1 else 0, num_unique_frags_for_coloring or 5))
                placed_geom.paint_uniform_color(placed_frag_color)
                
                if step_type == 'assembly_seed_placed':
                    current_assembly_state.clear() 
                current_assembly_state[orig_idx] = placed_geom 
            
            geometries_to_draw.extend(list(current_assembly_state.values()))

        elif step_type == 'final_assembly_result':
            current_assembly_state.clear()
            assembled_model_data = {
                'type': 'mesh', 'vertices': entry['vertices'], 'triangles': entry['triangles'],
                'vertex_colors': entry.get('vertex_colors'), 'vertex_normals': entry.get('vertex_normals')
            }
            assembled_model = _reconstruct_geometry_from_log_entry(assembled_model_data)
            if assembled_model:
                # For final assembly, color by component piece
                final_assembly_components_meshes = []
                if 'placed_fragments_info' in entry:
                    for placed_info in entry['placed_fragments_info']:
                        comp_name = placed_info['name']
                        comp_orig_idx = placed_info['original_index']
                        comp_transform = placed_info['transform']
                        
                        # Need the original UNTRANSFORMED mesh to apply new transform and color
                        # This part requires loading original meshes again if not stored in log.
                        # For simplicity in this replay, we'll just show the combined mesh.
                        # A more advanced replay would reconstruct each component.
                        pass # Placeholder for individual component coloring
                
                if not final_assembly_components_meshes: # If not colored by component
                    assembled_model.paint_uniform_color([0.6, 0.6, 0.9]) 
                    geometries_to_draw.append(assembled_model)
                else:
                    geometries_to_draw.extend(final_assembly_components_meshes)


        elif 'failed' in step_type or 'error' in step_type:
            print(f"    Event: {entry.get('reason', entry.get('error_message', 'Failure/Error event'))}")
        
        # ... (rest of drawing logic)
        if geometries_to_draw:
            valid_geoms = [g for g in geometries_to_draw if g is not None and ( (isinstance(g, o3d.geometry.PointCloud) and g.has_points()) or (isinstance(g, o3d.geometry.TriangleMesh) and g.has_vertices()) )]
            if valid_geoms:
                try:
                    o3d.visualization.draw_geometries(valid_geoms, window_name=window_title, width=1024, height=768)
                except Exception as e_draw:
                    print(f"      Error during o3d.visualization.draw_geometries: {e_draw}")
            else: print("    No valid geometries to draw for this step.")
        else: print("    No geometries to draw for this step.")
    
    print("\n--- Replay Finished ---")
    
# Add this function to src/utils/visualization_utils.py

def visualize_segmentation_results(original_mesh, fracture_surface, fragment_name=""):
    """
    Visualizes segmentation results showing original mesh and extracted fracture surface.
    Args:
        original_mesh (o3d.geometry.TriangleMesh): Original mesh
        fracture_surface (o3d.geometry.TriangleMesh): Extracted fracture surface
        fragment_name (str): Name of the fragment for window title
    """
    vis_geometries = []
    
    # Original mesh - make it semi-transparent gray
    original_mesh_vis = copy.deepcopy(original_mesh)
    original_mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
    original_mesh_vis.compute_vertex_normals()
    vis_geometries.append(original_mesh_vis)
    
    # Add wireframe for better structure visibility
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(original_mesh_vis)
    edges.paint_uniform_color([0.5, 0.5, 0.5])  # Darker gray for edges
    vis_geometries.append(edges)
    
    # Fracture surface - bright color
    if fracture_surface and fracture_surface.has_triangles():
        fracture_surface_vis = copy.deepcopy(fracture_surface)
        fracture_surface_vis.paint_uniform_color([1.0, 0.0, 0.0])  # Red for fracture
        fracture_surface_vis.compute_vertex_normals()
        vis_geometries.append(fracture_surface_vis)
    
    # Display
    window_title = f"Segmentation Result: {fragment_name}" if fragment_name else "Segmentation Result"
    o3d.visualization.draw_geometries(vis_geometries, window_name=window_title)

# Placeholder for a more advanced interactive viewer (complex to implement quickly)
def interactive_step_visualization(log_data, original_fragments_list):
    print("Interactive step visualization is not fully implemented yet.")
    print("Sequential replay will be used if available via a separate script.")
    # For a true interactive viewer, you'd typically create a custom Open3D Visualizer object,
    # add key callbacks for "Next", "Previous", and update the geometries in the scene.
    # For now, suggest using replay_visualization_log from a separate script.
    if log_data:
        print("To replay sequentially (many windows):")
        print("1. Save the log using --visualize_steps_file log.pkl")
        print("2. Run a script like this:")
        print("   from src.utils.visualization_utils import replay_visualization_log")
        print("   # Create fragments_data_map = {name: original_index for name, original_index in ...}")
        print("   # replay_visualization_log('log.pkl', fragments_data_map=fragments_data_map)")
        pass

if __name__ == '__main__':
    # Example of loading and replaying a log
    # Create a dummy log for testing
    dummy_log = [
        {'step': 'initial_fragment', 'name': 'part1', 'original_index': 0, 'geometry': o3d.geometry.TriangleMesh.create_box()},
        {'step': 'segmentation', 'name': 'part1', 
         'original_mesh': o3d.geometry.TriangleMesh.create_box(), 
         'fracture_mesh': o3d.geometry.TriangleMesh.create_sphere(radius=0.2).translate([0.5,0.5,1.2])},
        {'step': 'final_assembly_result', 'assembled_model': o3d.geometry.TriangleMesh.create_cone()}
    ]
    dummy_log_file = "dummy_viz_log.pkl"
    save_visualization_log(dummy_log, dummy_log_file)
    
    # Create a dummy fragments_data_map for coloring in replay
    dummy_fragments_map = {'part1': 0}

    replay_visualization_log(dummy_log_file, fragments_data_map=dummy_fragments_map)
    if os.path.exists(dummy_log_file):
        os.remove(dummy_log_file)