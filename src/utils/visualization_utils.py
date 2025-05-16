import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation, window_name="Registration Result"):
    """
    Visualizes the source point cloud transformed and aligned with the target.
    Args:
        source (o3d.geometry.PointCloud): Source point cloud.
        target (o3d.geometry.PointCloud): Target point cloud.
        transformation (np.ndarray): 4x4 transformation matrix to align source to target.
        window_name (str): Name for the visualization window.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])  # Orange
    target_temp.paint_uniform_color([0, 0.651, 0.929]) # Blue

    if transformation is not None:
        source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)


def visualize_steps(step_name, fragments_data, matches_or_assembly=None, num_to_show=3):
    """
    A helper to visualize intermediate steps.
    Args:
        step_name (str): e.g., "preprocessing", "pairwise_matches".
        fragments_data (list of dict): List of fragment data.
                                      Expected keys: 'pcd_for_features' or 'mesh'.
        matches_or_assembly: Data specific to the step (e.g., list of matches, or final mesh).
        num_to_show (int): How many examples to show for list-based visualizations.
    """
    print(f"\n--- Visualizing Step: {step_name} ---")

    if step_name == "preprocessing":
        print("  Showing preprocessed point clouds (first few).")
        geometries = []
        for i, frag_data in enumerate(fragments_data):
            if i >= num_to_show: break
            if frag_data.get('pcd') and frag_data['pcd'].has_points():
                pcd_vis = copy.deepcopy(frag_data['pcd'])
                # Assign a unique color
                color = plt.cm.get_cmap('gist_rainbow')(i / min(num_to_show, len(fragments_data)))[:3]
                pcd_vis.paint_uniform_color(color)
                # Translate for better side-by-side view
                pcd_vis.translate([i * 0.5, 0, 0]) # Adjust translation factor based on model scale
                geometries.append(pcd_vis)
        if geometries:
            o3d.visualization.draw_geometries(geometries, window_name="Preprocessed Point Clouds")
        else:
            print("  No valid preprocessed PCDs to show.")

    elif step_name == "feature_points": # e.g., the PCDs used for feature extraction
        print("  Showing point clouds used for feature extraction (first few).")
        geometries = []
        for i, frag_data in enumerate(fragments_data):
            if i >= num_to_show: break
            if frag_data.get('pcd_for_features') and frag_data['pcd_for_features'].has_points():
                pcd_ff_vis = copy.deepcopy(frag_data['pcd_for_features'])
                color = plt.cm.get_cmap('viridis')(i / min(num_to_show, len(fragments_data)))[:3]
                pcd_ff_vis.paint_uniform_color(color)
                pcd_ff_vis.translate([i * 0.5, 0, 0])
                geometries.append(pcd_ff_vis)
        if geometries:
            o3d.visualization.draw_geometries(geometries, window_name="PCDs for Feature Extraction")
        else:
            print("  No valid PCDs for features to show.")
            
    elif step_name == "pairwise_matches" and matches_or_assembly:
        print(f"  Showing best {num_to_show} pairwise matches.")
        matches_to_show = sorted(matches_or_assembly, key=lambda x: x['score'], reverse=True)
        for i, match in enumerate(matches_to_show):
            if i >= num_to_show: break
            
            source_idx = match['source_idx']
            target_idx = match['target_idx']
            transform = match['transformation']
            
            # Ensure indices are valid for fragments_data
            if not (0 <= source_idx < len(fragments_data) and 0 <= target_idx < len(fragments_data)):
                print(f"    Skipping match with invalid indices: source {source_idx}, target {target_idx}")
                continue

            source_frag_data = fragments_data[source_idx]
            target_frag_data = fragments_data[target_idx]

            # Use pcd_for_features as these were used for alignment
            source_pcd = source_frag_data.get('pcd_for_features')
            target_pcd = target_frag_data.get('pcd_for_features')

            if source_pcd and target_pcd and source_pcd.has_points() and target_pcd.has_points():
                title = (f"Match {i+1}: {source_frag_data['name']} (src) to "
                         f"{target_frag_data['name']} (tgt) - Score: {match['score']:.3f}")
                print(f"    Displaying: {title}")
                draw_registration_result(source_pcd, target_pcd, transform, window_name=title)
            else:
                print(f"    Skipping match visualization for {source_frag_data['name']} <-> {target_frag_data['name']} due to missing PCDs.")


    elif step_name == "final_assembly" and matches_or_assembly:
        if isinstance(matches_or_assembly, o3d.geometry.TriangleMesh) and matches_or_assembly.has_vertices():
            print("  Showing final assembled model.")
            o3d.visualization.draw_geometries([matches_or_assembly], window_name="Final Assembled Model")
        else:
            print("  No valid final assembly mesh to show.")
    else:
        print(f"  Visualization for step '{step_name}' not implemented or no data.")

# For color mapping if needed:
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib not installed, some color features in visualization might be limited.")
    plt = None


if __name__ == '__main__':
    # Example usage (requires dummy data or actual data)
    # Create two dummy point clouds
    pcd1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.3).sample_points_poisson_disk(100)
    pcd1.translate([-0.5, 0, 0])
    pcd2 = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.5).sample_points_poisson_disk(100)
    pcd2.translate([0.5, 0, 0])
    
    # Dummy fragments_data structure
    dummy_frags = [
        {'name': 'sphere', 'pcd': pcd1, 'pcd_for_features': pcd1, 'mesh': o3d.geometry.TriangleMesh.create_sphere(radius=0.3)},
        {'name': 'box', 'pcd': pcd2, 'pcd_for_features': pcd2, 'mesh': o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.5)}
    ]
    
    # Visualize preprocessing step (if plt is available for colors)
    if plt:
        visualize_steps("preprocessing", dummy_frags)

    # Dummy match data
    # Pretend pcd1 (sphere) aligns to pcd2 (box) with a simple translation
    # (This is not a real alignment, just for visualization structure)
    # Transformation that would move pcd1 to where pcd2 is
    # T_move_pcd1_to_pcd2_origin = np.eye(4)
    # T_move_pcd1_to_pcd2_origin[0,3] = 1.0 # pcd1 is at -0.5, pcd2 is at 0.5.
    
    # For draw_registration_result, source is pcd1, target is pcd2
    # Let's assume pcd1 needs to be transformed to align with pcd2
    # The 'transformation' should bring pcd1 onto pcd2
    # If pcd1 is at -0.5 and pcd2 is at 0.5, pcd1 needs to move +1.0 in X
    # If the alignment was pcd1 (source) to pcd2 (target):
    T_align_p1_to_p2 = np.eye(4)
    T_align_p1_to_p2[0,3] = 1.0 
    
    dummy_matches = [
        {'source_idx': 0, 'target_idx': 1, 'transformation': T_align_p1_to_p2, 'score': 0.9,
         'source_name': 'sphere', 'target_name': 'box'}
    ]
    
    visualize_steps("pairwise_matches", dummy_frags, dummy_matches)

    # Dummy final assembly
    combined_mesh = dummy_frags[0]['mesh'] + dummy_frags[1]['mesh'].transform(T_align_p1_to_p2) # Simple combination
    visualize_steps("final_assembly", dummy_frags, combined_mesh)

    print("Visualization examples complete.")