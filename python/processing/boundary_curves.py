import open3d as o3d
import numpy as np
import random


class BoundaryCurves:
    def __init__(self):
        pass

    def extract_pointcloud_boundaries(self, path):
        point_cloud = o3d.io.read_point_cloud(path)

        if point_cloud.is_empty():
            print("Error: Point cloud is empty or could not be loaded.")
            exit()

        point_cloud = voxel_downsample(point_cloud, voxel_size=1.0)
        # o3d.visualization.draw_geometries([point_cloud])

        clusters = region_growing(
            point_cloud, k_neighbors=20, normal_threshold=0.90, min_cluster_size=50
        )
        # visualize_clusters(point_cloud, clusters)

        # Use point-cloud-only boundary detection
        line_sets = extract_pointcloud_boundaries(point_cloud, clusters)

        print("Done!")

        return visualize_boundaries(point_cloud, line_sets)


def voxel_downsample(point_cloud, voxel_size=3.0):
    return point_cloud.voxel_down_sample(voxel_size=voxel_size)


def region_growing(
    point_cloud, k_neighbors=30, normal_threshold=0.95, min_cluster_size=10
):
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=k_neighbors
        )
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
            [k, neighbor_indices, _] = pcd_tree.search_radius_vector_3d(
                seed_point, radius=20
            )

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

    # print(f"Found {len(clusters)} clusters.")
    return clusters


def visualize_clusters(point_cloud, clusters):
    print(f"Visualizing {len(clusters)} clusters...")
    points = np.asarray(point_cloud.points)
    n_points = len(points)
    cluster_colors = [
        [random.random(), random.random(), random.random()]
        for _ in range(len(clusters))
    ]
    colors = [[0, 0, 0]] * n_points
    for i, cluster_indices in enumerate(clusters):
        for point_index in cluster_indices:
            colors[point_index] = cluster_colors[i]

    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])


# ------------- ✅ NEW: Boundary Curve Extraction ----------------


def extract_pointcloud_boundaries(
    point_cloud, clusters, curvature_threshold=0.01, neighbor_radius=4
):
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
            [_, idx, _] = kdtree.search_radius_vector_3d(
                cluster_pcd.points[i], neighbor_radius
            )
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
            boundary_pcd = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(boundary_points)
            )
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
                    [_, idxs, _] = kdtree.search_knn_vector_3d(
                        points_arr[current_idx], 10
                    )
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
                lines=o3d.utility.Vector2iVector(ordered_lines),
            )
            line_set.paint_uniform_color([1, 0, 0])  # Red lines
            all_linesets.append(line_set)

    return all_linesets

    print("Extracting point cloud-based fracture boundaries...")
    all_linesets = []

    for cluster_indices in clusters:
        cluster_pcd = point_cloud.select_by_index(cluster_indices)
        if len(cluster_pcd.points) < 50:
            continue

        cluster_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
        )

        # Compute curvature (approximate using PCA on neighborhood)
        points = np.asarray(cluster_pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(cluster_pcd)
        boundary_points = []

        for i in range(len(points)):
            [_, idx, _] = kdtree.search_radius_vector_3d(
                cluster_pcd.points[i], neighbor_radius
            )
            if len(idx) < 5:
                continue
            neighbors = np.asarray(cluster_pcd.points)[idx, :]
            cov = np.cov(neighbors.T)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.sort(eigvals)
            curvature = eigvals[0] / np.sum(eigvals)  # Smallest eigenvalue → curvature
            if curvature > curvature_threshold:
                boundary_points.append(cluster_pcd.points[i])

        if len(boundary_points) > 1:
            # Sort boundary points into an ordered chain
            boundary_pcd = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(boundary_points)
            )
            points_arr = np.asarray(boundary_pcd.points)
            visited = np.zeros(len(points_arr), dtype=bool)
            ordered_indices = []

            # Start from the first point
            current_idx = 0
            ordered_indices.append(current_idx)
            visited[current_idx] = True

            kdtree = o3d.geometry.KDTreeFlann(boundary_pcd)

            for _ in range(len(points_arr) - 1):
                [_, idxs, dists] = kdtree.search_knn_vector_3d(
                    points_arr[current_idx], 10
                )
                found_next = False
                for next_idx in idxs[1:]:  # skip self
                    if not visited[next_idx]:
                        ordered_indices.append(next_idx)
                        visited[next_idx] = True
                        current_idx = next_idx
                        found_next = True
                        break
                if not found_next:
                    break  # reached end of chain

            # Build connected lines from ordered points
            ordered_points = [points_arr[i] for i in ordered_indices]
            lines = [[i, i + 1] for i in range(len(ordered_points) - 1)]

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(ordered_points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.paint_uniform_color([1, 0, 0])  # red
            all_linesets.append(line_set)

    return all_linesets

    print("Extracting fracture (outer) edges...")
    all_linesets = []

    for cluster_indices in clusters:
        cluster_pcd = point_cloud.select_by_index(cluster_indices)
        if len(cluster_pcd.points) < 50:
            continue

        cluster_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
        )

        try:
            # Poisson surface reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cluster_pcd, depth=8
            )
            mesh.compute_vertex_normals()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            # Get boundary loops (each loop is a list of vertex indices)
            boundaries = mesh.get_boundaries()
            if not boundaries:
                print("No boundary found for this fragment.")
                continue

            # Convert boundary loops to line sets for visualization
            for loop in boundaries:
                if len(loop) < 2:
                    continue

                points = [mesh.vertices[i] for i in loop]
                lines = [[i, i + 1] for i in range(len(points) - 1)]
                # Optionally close the loop
                lines.append([len(points) - 1, 0])

                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                line_set.paint_uniform_color([1, 0, 0])  # Red lines
                all_linesets.append(line_set)

        except Exception as e:
            print(f"Mesh failed on cluster: {e}")

    return all_linesets

    print("Extracting fracture (outer) edges...")
    all_linesets = []

    for cluster_indices in clusters:
        cluster_pcd = point_cloud.select_by_index(cluster_indices)
        if len(cluster_pcd.points) < 50:
            continue

        cluster_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
        )

        try:
            # Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cluster_pcd, depth=8
            )
            mesh.compute_vertex_normals()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            # Compute adjacency list to find open boundary edges
            edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)

            # Extract only the boundary edges (edges with only 1 adjacent triangle)
            edge_map = mesh.get_edge_attribute_triangle_count()
            fracture_edges = []
            for edge in edges:
                # Only keep edges that are actual open boundaries (single triangle adjacent)
                if edge_map[edge] == 1:
                    fracture_edges.append(edge)

            # Map and collect the points
            if fracture_edges:
                edge_points = []
                lines = []
                vertex_map = {}
                idx = 0
                for e in fracture_edges:
                    for vid in e:
                        if vid not in vertex_map:
                            vertex_map[vid] = idx
                            edge_points.append(mesh.vertices[vid])
                            idx += 1
                    lines.append([vertex_map[e[0]], vertex_map[e[1]]])

                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(edge_points),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                line_set.paint_uniform_color([1, 0, 0])  # red = fracture edge
                all_linesets.append(line_set)

        except Exception as e:
            print(f"Mesh failed on cluster: {e}")

    return all_linesets

    print("Extracting boundary edges...")
    all_linesets = []

    for cluster_indices in clusters:
        cluster_pcd = point_cloud.select_by_index(cluster_indices)
        cluster_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
        )

        # Compute mesh (required to find edges)
        try:
            cluster_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cluster_pcd, depth=7
            )
            cluster_mesh.compute_vertex_normals()
            cluster_mesh.remove_degenerate_triangles()
            cluster_mesh.remove_duplicated_triangles()
            cluster_mesh.remove_duplicated_vertices()
            cluster_mesh.remove_non_manifold_edges()

            # Extract boundary edges
            edges = cluster_mesh.get_non_manifold_edges(allow_boundary_edges=True)
            edge_vertices = []
            lines = []
            idx_map = {}
            counter = 0
            for edge in edges:
                for v in edge:
                    if v not in idx_map:
                        idx_map[v] = counter
                        edge_vertices.append(cluster_mesh.vertices[v])
                        counter += 1
                lines.append([idx_map[edge[0]], idx_map[edge[1]]])

            if edge_vertices and lines:
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(edge_vertices),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                line_set.paint_uniform_color([1, 0, 0])  # red boundary lines
                all_linesets.append(line_set)
        except:
            print("Poisson reconstruction failed for one cluster (likely too sparse).")

    return all_linesets


def visualize_boundaries(point_cloud, line_sets):
    # print("Visualizing boundary curves...")

    # Set all point cloud vertices to yellow
    n_points = np.asarray(point_cloud.points).shape[0]
    point_cloud.colors = o3d.utility.Vector3dVector(
        [[1.0, 1.0, 0.0]] * n_points
    )  # Yellow

    # Set all boundary line sets to black
    for line_set in line_sets:
        line_set.paint_uniform_color([0.0, 0.0, 0.0])  # Black

    # Visualize
    o3d.visualization.draw_geometries([point_cloud] + line_sets)
    # geometry = [point_cloud] + line_sets
    return point_cloud, line_sets
