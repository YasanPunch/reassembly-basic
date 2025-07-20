## Segmentation

🧩 1. Load the Mesh
Start by loading a triangle mesh using Open3D.
Compute vertex normals if they aren't already available.

🔗 2. Build Vertex Adjacency
Construct a data structure (like an adjacency list) that maps each vertex to its neighboring vertices.
This is based on shared edges in the triangle faces.

🎯 3. Define a Similarity Criterion
Decide what constitutes a "similar" vertex:
Similar normal orientation (e.g., small angle between normals).
Optional: distance threshold, curvature, or color similarity.

🌱 4. Initialize Region Growing
Create a visited tracker to avoid revisiting vertices.

For each unvisited vertex:

Start a new region.

Use a queue (BFS) or stack (DFS) to iteratively explore neighbors that satisfy the similarity criterion.

Mark visited vertices and add them to the current region.

🧮 5. Grow Regions
Expand each region by checking adjacent vertices.

If a neighbor meets the similarity criterion, include it in the region and continue growing from it.

Repeat until the region can no longer grow.

📦 6. Store and Manage Regions
Save each grown region as a list of vertex indices.

Optionally filter out small regions (e.g., noise) based on size thresholds.

🎨 7. Visualize or Post-process
Assign unique colors to each region for visualization.

You can also extract sub-meshes or perform further analysis per region.
