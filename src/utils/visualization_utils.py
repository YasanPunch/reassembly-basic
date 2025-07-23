import open3d as o3d
import copy
import pickle
import matplotlib.pyplot as plt

if plt.colormaps.get("tab20"):
    cmap_qualitative = plt.cm.get_cmap("tab20", 20)
elif plt.colormaps.get("Pastel1"):
    cmap_qualitative = plt.cm.get_cmap("Pastel1", 20)
else:
    cmap_qualitative = plt.cm.get_cmap("viridis", 20)

def get_color(index, total_items=20): # Added total_items for better cmap indexing
    if cmap_qualitative:
        return cmap_qualitative(index % cmap_qualitative.N if cmap_qualitative.N > 0 else index % total_items)[:3]
    else:
        # Fallback simple colors
        colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0.8,0.5,0.2],[0.5,0.2,0.8],[0.2,0.8,0.5]]
        return colors[index % len(colors)]


def draw_registration_result(
    source, target, transformation, window_name="Registration Result"
):
    # ... (remains the same, but ensure geometries are copies if modified)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp], window_name=window_name
    )


def debug_visualize_voxel_downsampling(
    original_mesh, surf, pcd, fragment_name, surface_index
):
    """
    Debug visualization function to show voxel downsampling results.

    Args:
        original_mesh: The original mesh
        surf: The fracture surface mesh
        pcd: The downsampled point cloud
        fragment_name: Name of the fragment
        surface_index: Index of the current surface
    """
    print(
        f"    [DEBUG] Visualizing voxel downsampled point cloud for surface {surface_index} of {fragment_name} ({len(pcd.points)} points)"
    )

    # Create visualization geometries
    vis_geoms = []

    # Original mesh in gray
    original_mesh_vis = copy.deepcopy(original_mesh)
    original_mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
    vis_geoms.append(original_mesh_vis)

    # Add wireframe for better structure visibility
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(original_mesh_vis)
    wireframe.paint_uniform_color([0.5, 0.5, 0.5])
    vis_geoms.append(wireframe)

    # Current fracture surface in green
    surf_vis = copy.deepcopy(surf)
    surf_vis.paint_uniform_color([0.0, 1.0, 0.0])  # Green for fracture surface
    vis_geoms.append(surf_vis)

    # Downsampled point cloud in red
    if pcd.has_points():
        pcd_vis = copy.deepcopy(pcd)
        pcd_vis.paint_uniform_color([1.0, 0.0, 0.0])  # Red for downsampled points
        vis_geoms.append(pcd_vis)

    # Display with informative window title
    window_title = f"[DEBUG] Voxel Downsampling: {fragment_name} Surface {surface_index} (Gray=Original, Green=Surface, Red=Downsampled Points)"
    o3d.visualization.draw_geometries(vis_geoms, window_name=window_title)
