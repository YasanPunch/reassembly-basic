import open3d as o3d
import numpy as np
import random
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def voxel_downsample(point_cloud, voxel_size=2.0):
    return point_cloud.voxel_down_sample(voxel_size=voxel_size)

def region_growing(point_cloud, k_neighbors=30, normal_threshold=0.95, min_cluster_size=10):
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k_neighbors)
    )
    point_cloud.orient_normals_to_align_with_direction()

    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    n_points = len(points)
    print(f"Number of points: {n_points}")

    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    unvisited = set(range(n_points))
    clusters = []
    visited = [False] * n_points

    while unvisited:
        seed_index = unvisited.pop()
        if visited[seed_index]:
            continue
        current_cluster = [seed_index]
        visited[seed_index] = True
        unvisited_queue = [seed_index]

        while unvisited_queue:
            growing_index = unvisited_queue.pop(0)
            seed_point = points[growing_index]
            seed_normal = normals[growing_index]
            [k, neighbor_indices, _] = pcd_tree.search_radius_vector_3d(seed_point, radius=20)

            for neighbor_index in neighbor_indices:
                if not visited[neighbor_index]:
                    neighbor_normal = normals[neighbor_index]
                    similarity = np.dot(seed_normal, neighbor_normal)
                    if similarity > normal_threshold:
                        visited[neighbor_index] = True
                        unvisited.discard(neighbor_index)
                        current_cluster.append(neighbor_index)
                        unvisited_queue.append(neighbor_index)

        if len(current_cluster) >= min_cluster_size:
            clusters.append(current_cluster)

    print(f"Found {len(clusters)} clusters.")
    return clusters

def visualize_clusters(point_cloud, clusters):
    print(f"Visualizing {len(clusters)} clusters...")
    points = np.asarray(point_cloud.points)
    n_points = len(points)
    cluster_colors = [[random.random(), random.random(), random.random()] for _ in range(len(clusters))]
    colors = [[0, 0, 0]] * n_points
    for i, cluster_indices in enumerate(clusters):
        for point_index in cluster_indices:
            colors[point_index] = cluster_colors[i]

    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])

# ------------- ✅ NEW: Boundary Curve Extraction ----------------

def extract_pointcloud_boundaries(point_cloud, clusters, curvature_threshold=0.01, neighbor_radius=4):

    print("Extracting point cloud-based fracture boundaries with continuity...")
    all_linesets = []

    for cluster_indices in clusters:
        cluster_pcd = point_cloud.select_by_index(cluster_indices)
        if len(cluster_pcd.points) < 50:
            continue

        cluster_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
        )

        # Compute curvature
        points = np.asarray(cluster_pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(cluster_pcd)
        boundary_points = []

        for i in range(len(points)):
            [_, idx, _] = kdtree.search_radius_vector_3d(cluster_pcd.points[i], neighbor_radius)
            if len(idx) < 5:
                continue
            neighbors = np.asarray(cluster_pcd.points)[idx, :]
            cov = np.cov(neighbors.T)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.sort(eigvals)
            curvature = eigvals[0] / np.sum(eigvals)
            if curvature > curvature_threshold:
                boundary_points.append(cluster_pcd.points[i])

        if len(boundary_points) > 1:
            # Build an ordered line set including all points
            boundary_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(boundary_points))
            points_arr = np.asarray(boundary_pcd.points)
            visited = np.zeros(len(points_arr), dtype=bool)
            kdtree = o3d.geometry.KDTreeFlann(boundary_pcd)

            ordered_lines = []

            while not np.all(visited):
                # Start from an unvisited point
                unvisited_indices = np.where(visited == False)[0]
                current_idx = unvisited_indices[0]
                visited[current_idx] = True
                chain = [current_idx]

                for _ in range(len(points_arr) - 1):
                    [_, idxs, _] = kdtree.search_knn_vector_3d(points_arr[current_idx], 10)
                    found = False
                    for next_idx in idxs[1:]:  # Skip self
                        if not visited[next_idx]:
                            visited[next_idx] = True
                            ordered_lines.append([current_idx, next_idx])
                            current_idx = next_idx
                            chain.append(current_idx)
                            found = True
                            break
                    if not found:
                        break  # Start new segment

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points_arr),
                lines=o3d.utility.Vector2iVector(ordered_lines)
            )
            line_set.paint_uniform_color([1, 0, 0])  # Red lines
            all_linesets.append(line_set)

    return all_linesets



def extract_concave_convex_patches_with_labels(point_cloud, K_thresh=0.005, H_thresh=0.01, neighbor_radius=5, min_neighbors=6, min_cluster_size=20):
    print("Extracting and clustering concave and convex patches...")
    print(f"Using parameters: min_neighbors={min_neighbors}, min_cluster_size={min_cluster_size}")

    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )

    points = np.asarray(point_cloud.points)
    n_points = len(points)
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    labels = np.full(n_points, fill_value=-1)  # -1 = unclassified

    # 1. Curvature-based classification
    for i in range(n_points):
        [_, idx, _] = kdtree.search_radius_vector_3d(point_cloud.points[i], neighbor_radius)
        if len(idx) < min_neighbors:
            continue
        neighbors = points[idx]
        cov = np.cov(neighbors.T)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)[::-1]
        k1, k2 = eigvals[0], eigvals[1]

        K = k1 * k2
        H = (k1 + k2) / 2

        if K > K_thresh:
            if H > H_thresh:
                labels[i] = 1  # convex
            elif H < -H_thresh:
                labels[i] = 0  # concave

    # 2. Patch clustering using region growing
    def cluster_type(target_type):
        clustered = []
        visited = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            if labels[i] != target_type or visited[i]:
                continue
            cluster = []
            queue = [i]
            visited[i] = True

            while queue:
                current = queue.pop(0)
                cluster.append(current)
                [_, neighbors, _] = kdtree.search_radius_vector_3d(point_cloud.points[current], neighbor_radius)
                for ni in neighbors:
                    if not visited[ni] and labels[ni] == target_type:
                        visited[ni] = True
                        queue.append(ni)

            if len(cluster) >= min_cluster_size:  # minimum patch size
                clustered.append(cluster)

        return clustered

    concave_clusters = cluster_type(0)
    convex_clusters = cluster_type(1)

    # 3. Assign distinct colors using Open3D's color utilities
    def get_distinct_colors(n):
        colors = []
        for i in range(n):
            # Generate distinct colors using HSV color space
            hue = i / n
            saturation = 0.8
            value = 0.9
            # Convert HSV to RGB
            h = hue * 6
            i = int(h)
            f = h - i
            p = value * (1 - saturation)
            q = value * (1 - saturation * f)
            t = value * (1 - saturation * (1 - f))
            
            if i == 0:
                r, g, b = value, t, p
            elif i == 1:
                r, g, b = q, value, p
            elif i == 2:
                r, g, b = p, value, t
            elif i == 3:
                r, g, b = p, q, value
            elif i == 4:
                r, g, b = t, p, value
            else:
                r, g, b = value, p, q
                
            colors.append([r, g, b])
        return colors

    colors = np.full((n_points, 3), fill_value=0.8)  # Gray background for unclassified

    concave_colors = get_distinct_colors(len(concave_clusters))
    convex_colors = get_distinct_colors(len(convex_clusters))

    for i, cluster in enumerate(concave_clusters):
        color = concave_colors[i]
        for idx in cluster:
            colors[idx] = color

    for i, cluster in enumerate(convex_clusters):
        color = convex_colors[i]
        for idx in cluster:
            colors[idx] = color

    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    print(f"Detected {len(concave_clusters)} concave patches and {len(convex_clusters)} convex patches.")
    return point_cloud




def visualize_boundaries(point_cloud, line_sets):
    print("Visualizing boundary curves...")

    # Set all point cloud vertices to yellow
    n_points = np.asarray(point_cloud.points).shape[0]
    point_cloud.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]] * n_points)  # Yellow

    # Set all boundary line sets to black
    for line_set in line_sets:
        line_set.paint_uniform_color([0.0, 0.0, 0.0])  # Black

    # Visualize
    o3d.visualization.draw_geometries([point_cloud] + line_sets)

# ----------------------------------------------------------------

class ReassemblyGUI:
    def __init__(self):
        self.point_cloud = None
        self.original_point_cloud = None
        self.clusters = None
        self.line_sets = None
        self.patch_colored_cloud = None
        
        # Default parameters
        self.voxel_size = 3.0
        self.k_neighbors = 20
        self.normal_threshold = 0.90
        self.min_cluster_size = 50
        self.K_thresh = 0.003
        self.H_thresh = 0.005
        self.neighbor_radius = 5
        self.min_neighbors = 8
        self.min_patch_size = 25

        # Initialize the application
        gui.Application.instance.initialize()
        
        # Create main window
        self.window = gui.Application.instance.create_window("Reassembly GUI", 1600, 900)
        
        # Create scenes directly attached to window
        self.scenes = []
        self.scene_labels = []
        scene_names = ["Original Point Cloud", "Boundary Curves", "Patches", "Combined View"]
        
        for i in range(4):
            # Create scene widget
            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(self.window.renderer)
            self.scenes.append(scene)
            
            # Create label for scene
            label = gui.Label(scene_names[i])
            label.text_color = gui.Color(1.0, 1.0, 1.0)  # White color
            self.scene_labels.append(label)
            
            # Add directly to window
            self.window.add_child(label)
            self.window.add_child(scene)
        
        # Create control panel
        self.panel = gui.Vert(0, gui.Margins(0.25, 0.25, 0.25, 0.25))
        
        # Add file loading button
        load_button = gui.Button("Load Point Cloud")
        load_button.set_on_clicked(self.load_point_cloud)
        self.panel.add_child(load_button)

        # Add parameter controls
        self.panel.add_child(gui.Label("Parameters"))
        
        # Voxel size slider
        voxel_layout = gui.Horiz()
        voxel_layout.add_child(gui.Label("Voxel Size:"))
        self.voxel_slider = gui.Slider(gui.Slider.DOUBLE)
        self.voxel_slider.set_limits(0.1, 5.0)
        self.voxel_slider.double_value = self.voxel_size
        voxel_layout.add_child(self.voxel_slider)
        self.panel.add_child(voxel_layout)

        # K neighbors slider
        k_layout = gui.Horiz()
        k_layout.add_child(gui.Label("K Neighbors:"))
        self.k_slider = gui.Slider(gui.Slider.INT)
        self.k_slider.set_limits(5, 50)
        self.k_slider.int_value = self.k_neighbors
        k_layout.add_child(self.k_slider)
        self.panel.add_child(k_layout)

        # Normal threshold slider
        normal_layout = gui.Horiz()
        normal_layout.add_child(gui.Label("Normal Threshold:"))
        self.normal_slider = gui.Slider(gui.Slider.DOUBLE)
        self.normal_slider.set_limits(0.5, 1.0)
        self.normal_slider.double_value = self.normal_threshold
        normal_layout.add_child(self.normal_slider)
        self.panel.add_child(normal_layout)

        # Min cluster size slider
        cluster_layout = gui.Horiz()
        cluster_layout.add_child(gui.Label("Min Cluster Size:"))
        self.cluster_slider = gui.Slider(gui.Slider.INT)
        self.cluster_slider.set_limits(10, 200)
        self.cluster_slider.int_value = self.min_cluster_size
        cluster_layout.add_child(self.cluster_slider)
        self.panel.add_child(cluster_layout)

        # K threshold slider
        k_thresh_layout = gui.Horiz()
        k_thresh_layout.add_child(gui.Label("K Threshold:"))
        self.k_thresh_slider = gui.Slider(gui.Slider.DOUBLE)
        self.k_thresh_slider.set_limits(0.0001, 0.01)
        self.k_thresh_slider.double_value = self.K_thresh
        k_thresh_layout.add_child(self.k_thresh_slider)
        self.panel.add_child(k_thresh_layout)

        # H threshold slider
        h_thresh_layout = gui.Horiz()
        h_thresh_layout.add_child(gui.Label("H Threshold:"))
        self.h_thresh_slider = gui.Slider(gui.Slider.DOUBLE)
        self.h_thresh_slider.set_limits(0.0001, 0.01)
        self.h_thresh_slider.double_value = self.H_thresh
        h_thresh_layout.add_child(self.h_thresh_slider)
        self.panel.add_child(h_thresh_layout)

        # Process button
        process_button = gui.Button("Process")
        process_button.set_on_clicked(self.process_point_cloud)
        self.panel.add_child(process_button)

        # Add panel to window
        self.window.add_child(self.panel)

        # Set up layout
        self.window.set_on_layout(self._on_layout)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        em = layout_context.theme.font_size
        width = 17 * em

        # Calculate scene dimensions
        scene_width = (r.get_right() - width) / 2
        scene_height = r.height / 2

        # Position scenes in a 2x2 grid
        for i in range(4):
            row = i // 2
            col = i % 2
            x = r.x + col * scene_width
            y = r.y + row * scene_height
            
            # Position label
            self.scene_labels[i].frame = gui.Rect(x, y, scene_width, em)
            
            # Position scene
            self.scenes[i].frame = gui.Rect(x, y + em, scene_width, scene_height - em)
        
        # Position the control panel
        self.panel.frame = gui.Rect(r.get_right() - width, r.y, width, r.height)

    def update_scene(self, scene_index, geometry, material=None):
        if material is None:
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.point_size = 3.0

        # Clear existing geometry
        self.scenes[scene_index].scene.clear_geometry()
        
        # Add new geometry
        self.scenes[scene_index].scene.add_geometry("geometry", geometry, material)
        
        # Set up camera
        bounds = geometry.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = bounds.get_extent()
        radius = np.linalg.norm(extent) * 0.5
        
        # Set up camera with a good view of the geometry
        self.scenes[scene_index].setup_camera(60, bounds, center)
        self.scenes[scene_index].look_at(center, center + [0, 0, radius], [0, 1, 0])
        
        # Force redraw of the scene widget
        self.scenes[scene_index].force_redraw()

    def load_point_cloud(self):
        dialog = gui.FileDialog(gui.FileDialog.OPEN, "Choose point cloud file", self.window.theme)
        dialog.add_filter(".ply", "Point cloud files (.ply)")
        dialog.add_filter("", "All files")
        
        dialog.set_on_cancel(self._on_file_dialog_cancel)
        dialog.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dialog)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        try:
            self.window.close_dialog()
            print(f"\nLoading point cloud from: {filename}")
            self.original_point_cloud = o3d.io.read_point_cloud(filename)
            if not self.original_point_cloud.is_empty():
                print(f"Successfully loaded point cloud with {len(self.original_point_cloud.points)} points")
                self.point_cloud = self.original_point_cloud
                self.process_point_cloud()
            else:
                print("Error: Loaded point cloud is empty")
        except Exception as e:
            print(f"Error loading point cloud: {str(e)}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())

    def process_point_cloud(self):
        try:
            if self.point_cloud is None:
                print("Error: No point cloud loaded")
                return

            print("\n=== Starting Point Cloud Processing ===")
            # Update parameters from sliders
            self.voxel_size = self.voxel_slider.double_value
            self.k_neighbors = self.k_slider.int_value
            self.normal_threshold = self.normal_slider.double_value
            self.min_cluster_size = self.cluster_slider.int_value
            self.K_thresh = self.k_thresh_slider.double_value
            self.H_thresh = self.h_thresh_slider.double_value

            print(f"Current parameters:")
            print(f"- Voxel size: {self.voxel_size}")
            print(f"- K neighbors: {self.k_neighbors}")
            print(f"- Normal threshold: {self.normal_threshold}")
            print(f"- Min cluster size: {self.min_cluster_size}")
            print(f"- K threshold: {self.K_thresh}")
            print(f"- H threshold: {self.H_thresh}")

            # Reset to original point cloud if available
            if self.original_point_cloud is not None:
                print("Resetting to original point cloud...")
                self.point_cloud = self.original_point_cloud
                print(f"Original point cloud size: {len(self.original_point_cloud.points)} points")

            # Show original point cloud
            original_cloud = o3d.geometry.PointCloud(self.point_cloud)
            if not original_cloud.has_normals():
                original_cloud.estimate_normals()
            if not original_cloud.has_colors():
                original_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            self.update_scene(0, original_cloud)

            # Process point cloud
            print("\nPerforming voxel downsampling...")
            self.point_cloud = voxel_downsample(self.point_cloud, self.voxel_size)
            print(f"Downsampled point cloud size: {len(self.point_cloud.points)} points")

            print("\nPerforming region growing...")
            self.clusters = region_growing(
                self.point_cloud,
                k_neighbors=self.k_neighbors,
                normal_threshold=self.normal_threshold,
                min_cluster_size=self.min_cluster_size
            )
            print(f"Found {len(self.clusters)} clusters")

            print("\nExtracting boundaries...")
            self.line_sets = extract_pointcloud_boundaries(self.point_cloud, self.clusters)
            print(f"Generated {len(self.line_sets)} line sets")

            # Show boundary curves
            boundary_cloud = o3d.geometry.PointCloud(self.point_cloud)
            boundary_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.point_size = 3.0
            self.update_scene(1, boundary_cloud, material)
            for i, line_set in enumerate(self.line_sets):
                line_material = rendering.MaterialRecord()
                line_material.shader = "unlitLine"
                line_material.line_width = 2.0
                self.scenes[1].scene.add_geometry(f"line_set_{i}", line_set, line_material)
            self.scenes[1].force_redraw()

            print("\nExtracting concave/convex patches...")
            self.patch_colored_cloud = extract_concave_convex_patches_with_labels(
                self.point_cloud,
                K_thresh=self.K_thresh,
                H_thresh=self.H_thresh,
                neighbor_radius=self.neighbor_radius,
                min_neighbors=self.min_neighbors,
                min_cluster_size=self.min_patch_size
            )

            # Show patches
            if not self.patch_colored_cloud.has_normals():
                self.patch_colored_cloud.estimate_normals()
            self.update_scene(2, self.patch_colored_cloud)

            # Show combined view
            combined_cloud = o3d.geometry.PointCloud(self.patch_colored_cloud)
            self.update_scene(3, combined_cloud)
            for i, line_set in enumerate(self.line_sets):
                line_material = rendering.MaterialRecord()
                line_material.shader = "unlitLine"
                line_material.line_width = 2.0
                self.scenes[3].scene.add_geometry(f"line_set_{i}", line_set, line_material)
            self.scenes[3].force_redraw()

            print("=== Processing Complete ===\n")

        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())

if __name__ == "__main__":
    gui_app = ReassemblyGUI()
    gui.Application.instance.run()

