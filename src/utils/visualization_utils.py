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

def replay_visualization_log(log_filepath, fragments_name_to_original_idx_map=None):
    """
    Basic replay of a saved visualization log.
    This will open many windows sequentially.
    A proper interactive viewer is more complex.
    Args:
        log_filepath (str): Path to the pickled log file.
        fragments_data_map (dict, optional): Map of fragment_name to original_index,
                                             used for coloring.
    """
    log_data = load_visualization_log(log_filepath)
    if not log_data: return

    print(f"\n--- Replaying Visualization Log ({len(log_data)} steps) ---")
    print("Press Q in each Open3D window to close it and proceed to the next step.")

    # Build a map of original_index to a unique color for consistent coloring
    # Use fragments_name_to_original_idx_map if provided, otherwise assign colors as we see fragments
    original_index_to_color = {}
    num_unique_frags = 0
    if fragments_name_to_original_idx_map:
        # Get unique original indices to determine total number of colors needed
        unique_original_indices = sorted(list(set(fragments_name_to_original_idx_map.values())))
        num_unique_frags = len(unique_original_indices)
        for name, orig_idx in fragments_name_to_original_idx_map.items():
            # Map the original_idx to its position in the sorted unique list for consistent color indexing
            color_map_idx = unique_original_indices.index(orig_idx)
            original_index_to_color[orig_idx] = get_color(color_map_idx, num_unique_frags)

    # Store currently placed geometries for assembly steps
    # Key: original_index, Value: transformed o3d.geometry.TriangleMesh
    current_assembly_state = {} 

    for i, entry in enumerate(log_data):
        step_type = entry.get('step')
        frag_name = entry.get('name', entry.get('fragment_name', 'UnknownFrag'))
        orig_idx = entry.get('original_index', -1) # Original index from loading

        print(f"\nLog Entry {i+1}/{len(log_data)}: Type='{step_type}', Name='{frag_name}'")
        
        geometries_to_draw = []
        window_title = f"Log {i+1}: {step_type} - {frag_name}"
        
        # Determine color for this fragment based on its original_index
        frag_color = original_index_to_color.get(orig_idx, get_color(orig_idx if orig_idx != -1 else i, num_unique_frags or len(log_data)))

        if step_type == 'initial_fragment':
            geom = _reconstruct_geometry_from_log_entry(entry)
            if geom:
                geom.paint_uniform_color(frag_color)
                geometries_to_draw.append(geom)
        
        elif step_type == 'segmentation_result':
            orig_mesh_data = {
                'type': entry.get('original_mesh_type'),
                'vertices': entry.get('original_mesh_vertices'),
                'triangles': entry.get('original_mesh_triangles')
            }
            orig_mesh = _reconstruct_geometry_from_log_entry(orig_mesh_data)
            if orig_mesh:
                orig_mesh.paint_uniform_color(frag_color if np.allclose(frag_color, [0.7,0.7,0.7]) else [0.7,0.7,0.7]) # Grey if not otherwise colored
                geometries_to_draw.append(orig_mesh)

            if entry.get('fracture_mesh_type'):
                fract_mesh_data = {
                    'type': entry.get('fracture_mesh_type'),
                    'vertices': entry.get('fracture_mesh_vertices'),
                    'triangles': entry.get('fracture_mesh_triangles')
                }
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
                source_idx_for_color = fragments_name_to_original_idx_map.get(entry['source_name'], 0)
                target_idx_for_color = fragments_name_to_original_idx_map.get(entry['target_name'], 1)

                source_pcd.paint_uniform_color(original_index_to_color.get(source_idx_for_color, get_color(0)))
                target_pcd.paint_uniform_color(original_index_to_color.get(target_idx_for_color, get_color(1)))
                source_pcd.transform(transform)
                geometries_to_draw.extend([source_pcd, target_pcd])
            window_title += f" (Score: {entry['score']:.2f})"

        elif step_type == 'assembly_seed_placed' or step_type == 'assembly_fragment_placed':
            geom_data = {'type': 'mesh', 'vertices': entry['vertices'], 'triangles': entry['triangles']}
            placed_geom = _reconstruct_geometry_from_log_entry(geom_data)
            if placed_geom:
                placed_geom.paint_uniform_color(frag_color) # Color already determined by entry['original_index']
                
                if step_type == 'assembly_seed_placed':
                    current_assembly_state.clear() # Start new assembly
                current_assembly_state[orig_idx] = placed_geom # Store/update by original_index
            
            geometries_to_draw.extend(list(current_assembly_state.values())) # Draw all currently placed

        elif step_type == 'final_assembly_result':
            current_assembly_state.clear() # Show only the final combined mesh
            assembled_model = _reconstruct_geometry_from_log_entry(entry)
            if assembled_model:
                # Color final assembly by piece if possible
                final_assembly_components = []
                if 'placed_fragments_info' in entry and fragments_name_to_original_idx_map:
                    # Load original meshes to color them
                    # This assumes replay_log.py has access to the original fragments path
                    # For simplicity, we'll color based on the component list if available
                    # This part requires more careful handling of original meshes.
                    # For now, just draw the combined model.
                    assembled_model.paint_uniform_color([0.6, 0.6, 0.9]) # Light purple
                else:
                    assembled_model.paint_uniform_color([0.6, 0.6, 0.9])
                geometries_to_draw.append(assembled_model)
        
        elif 'failed' in step_type:
            print(f"    {entry.get('reason', entry.get('error', 'Failure event'))}")
            # No specific geometry to draw for failure events unless logged

        if geometries_to_draw:
            valid_geoms = [g for g in geometries_to_draw if g is not None and ( (isinstance(g, o3d.geometry.PointCloud) and g.has_points()) or (isinstance(g, o3d.geometry.TriangleMesh) and g.has_vertices()) )]
            if valid_geoms:
                o3d.visualization.draw_geometries(valid_geoms, window_name=window_title, width=1024, height=768)
            else: print("    No valid geometries to draw for this step.")
        else: print("    No geometries to draw for this step.")

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