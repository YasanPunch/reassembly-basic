import open3d as o3d
import numpy as np
import trimesh
import copy
from scipy.optimize import minimize
import time

print("DEBUG: enhanced alignment.py (Papaioannou method) loaded")

class PapaioannoualignmentError:
    """
    Implementation of the Papaioannou et al. matching error calculation
    using slope-based surface fitting with Z-buffer optimization.
    """
    
    def __init__(self, resolution=128, max_distance_factor=1.3):
        self.resolution = resolution
        self.max_distance_factor = max_distance_factor
        
    def calculate_zbuffer_distances(self, mesh1, mesh2, transform_params):
        print(f"[DEBUG] calculate_zbuffer_distances: params={transform_params}")
        """
        Calculate Z-buffer distances for two meshes at given transformation parameters.
        
        Args:
            mesh1, mesh2: o3d.geometry.TriangleMesh objects
            transform_params: [theta1, phi1, rho1, x1, y1, theta2, phi2]
        
        Returns:
            Z1, Z2: depth buffers for both objects
            valid_mask: boolean mask of valid depth values
        """
        # Apply transformations based on 7-DOF parameterization
        theta1, phi1, rho1, x1, y1, theta2, phi2 = transform_params
        
        # Create transformation matrices
        T1 = self._create_transformation_matrix(theta1, phi1, rho1, x1, y1, True)
        T2 = self._create_transformation_matrix(theta2, phi2, 0, 0, 0, False)
        
        # Transform meshes
        mesh1_transformed = copy.deepcopy(mesh1)
        mesh2_transformed = copy.deepcopy(mesh2)
        mesh1_transformed.transform(T1)
        mesh2_transformed.transform(T2)
        
        # Calculate bounding box for rendering setup
        bbox1 = mesh1_transformed.get_axis_aligned_bounding_box()
        bbox2 = mesh2_transformed.get_axis_aligned_bounding_box()
        
        # Combine bounding boxes
        combined_min = np.minimum(bbox1.min_bound, bbox2.min_bound)
        combined_max = np.maximum(bbox1.max_bound, bbox2.max_bound)
        extent = combined_max - combined_min
        max_extent = np.max(extent)
        
        # Setup rendering parameters
        center = (combined_min + combined_max) / 2
        render_distance = max_extent * self.max_distance_factor
        
        # Render depth buffers using ray casting (simplified Z-buffer simulation)
        Z1, Z2 = self._render_depth_buffers(
            mesh1_transformed, mesh2_transformed, 
            center, render_distance, max_extent
        )
        
        # Create valid mask where both objects have valid depth values
        valid_mask = (Z1 < np.inf) & (Z2 < np.inf) & (Z1 > 0) & (Z2 > 0)
        
        print("[DEBUG] Z-buffer calculation complete.")
        return Z1, Z2, valid_mask
    
    def _create_transformation_matrix(self, theta, phi, rho, x, y, is_first_object):
        """Create 4x4 transformation matrix based on paper's parameterization."""
        # Rotation matrices
        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(theta), -np.sin(theta), 0],
                       [0, np.sin(theta), np.cos(theta), 0],
                       [0, 0, 0, 1]])
        
        Ry = np.array([[np.cos(phi), 0, np.sin(phi), 0],
                       [0, 1, 0, 0],
                       [-np.sin(phi), 0, np.cos(phi), 0],
                       [0, 0, 0, 1]])
        
        if is_first_object:
            Rz = np.array([[np.cos(rho), -np.sin(rho), 0, 0],
                           [np.sin(rho), np.cos(rho), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
            
            T = np.array([[1, 0, 0, x],
                          [0, 1, 0, y],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
            
            # For first object: T * Rz * Rx * Ry (note: applied right to left)
            return T @ Rz @ Rx @ Ry
        else:
            # For second object: Rx * Ry
            return Rx @ Ry
    
    def _render_depth_buffers(self, mesh1, mesh2, center, render_distance, max_extent):
        print(f"[DEBUG] _render_depth_buffers: center={center}, render_distance={render_distance}, max_extent={max_extent}, resolution={self.resolution}")
        Z1 = np.full((self.resolution, self.resolution), np.inf)
        Z2 = np.full((self.resolution, self.resolution), np.inf)
        
        # Create rays from viewing plane
        x_range = np.linspace(-max_extent/2, max_extent/2, self.resolution) + center[0]
        y_range = np.linspace(-max_extent/2, max_extent/2, self.resolution) + center[1]
        
        # Convert meshes to trimesh for ray intersection
        try:
            trimesh1 = trimesh.Trimesh(
                vertices=np.asarray(mesh1.vertices),
                faces=np.asarray(mesh1.triangles)
            )
            trimesh2 = trimesh.Trimesh(
                vertices=np.asarray(mesh2.vertices),
                faces=np.asarray(mesh2.triangles)
            )
        except:
            return Z1, Z2
        
        # Cast rays and find intersections
        for i, x in enumerate(x_range):
            if i % 10 == 0:
                print(f"[DEBUG] Raycasting row {i+1}/{self.resolution}")
            for j, y in enumerate(y_range):
                ray_origin = np.array([x, y, center[2] - render_distance])
                ray_direction = np.array([0, 0, 1])
                
                # Intersect with mesh1
                try:
                    locations1, _, _ = trimesh1.ray.intersects_location(
                        ray_origins=[ray_origin],
                        ray_directions=[ray_direction]
                    )
                    if len(locations1) > 0:
                        Z1[j, i] = locations1[0][2]  # First intersection depth
                except:
                    pass
                
                # Intersect with mesh2 (from opposite direction for paper's setup)
                try:
                    ray_origin_2 = np.array([x, y, center[2] + render_distance])
                    ray_direction_2 = np.array([0, 0, -1])
                    locations2, _, _ = trimesh2.ray.intersects_location(
                        ray_origins=[ray_origin_2],
                        ray_directions=[ray_direction_2]
                    )
                    if len(locations2) > 0:
                        Z2[j, self.resolution - 1 - i] = center[2] + render_distance - locations2[0][2]
                except:
                    pass
        
        print("[DEBUG] Depth buffer rendering complete.")
        return Z1, Z2
    
    def calculate_matching_error(self, Z1, Z2, valid_mask):
        print(f"[DEBUG] calculate_matching_error: valid points={np.sum(valid_mask)}")
        """
        Calculate the slope-based matching error from the paper.
        
        εd = (1/Ns) Σ |∇u(d1 + d2)| + |∇v(d1 + d2)|
        """
        if np.sum(valid_mask) < 10:  # Need minimum valid points
            return np.inf

        # Mask out invalid values in Z1 and Z2
        Z1 = np.where(np.isfinite(Z1), Z1, 0)
        Z2 = np.where(np.isfinite(Z2), Z2, 0)

        # Calculate forward differences (gradients)
        du_d1 = np.zeros_like(Z1)
        du_d2 = np.zeros_like(Z2)
        du_d1[:-1, :] = Z1[1:, :] - Z1[:-1, :]
        du_d2[:-1, :] = Z2[1:, :] - Z2[:-1, :]

        dv_d1 = np.zeros_like(Z1)
        dv_d2 = np.zeros_like(Z2)
        dv_d1[:, :-1] = Z1[:, 1:] - Z1[:, :-1]
        dv_d2[:, :-1] = Z2[:, 1:] - Z2[:, :-1]

        u_component = np.abs(du_d1 + du_d2)
        v_component = np.abs(dv_d1 + dv_d2)

        # Only use valid regions for error calculation
        error_map = u_component + v_component
        valid_error = error_map[valid_mask]

        if len(valid_error) == 0:
            return np.inf

        matching_error = np.mean(valid_error)
        print(f"[DEBUG] Matching error: {matching_error}")
        return matching_error


class EnhancedSimulatedAnnealing:
    """
    Implementation of Enhanced Simulated Annealing (ESA) from the paper
    with adaptive cooling and search space partitioning.
    """
    
    def __init__(self, bounds, max_iter=2000, n_fail=4, n_attempted=80, n_accepted=8):
        self.bounds = bounds  # List of (min, max) for each parameter
        self.max_iter = max_iter
        self.n_fail = n_fail
        self.n_attempted = n_attempted
        self.n_accepted = n_accepted
        self.a_min = 0.6
        self.a_max = 0.9
        
    def optimize(self, objective_func, x0=None):
        print("[DEBUG] Starting ESA optimization loop")
        """
        Optimize using Enhanced Simulated Annealing.
        
        Args:
            objective_func: Function to minimize
            x0: Initial guess (optional)
        
        Returns:
            best_params: Best parameter vector found
            best_error: Best error value
        """
        # Initialize
        if x0 is None:
            x = self._random_initial_point()
        else:
            x = np.array(x0)
        
        # Estimate initial temperature using Monte Carlo
        T0, T_min = self._estimate_temperatures(objective_func)
        T = T0
        
        best_x = x.copy()
        best_error = objective_func(x)
        current_error = best_error
        
        fail_count = 0
        iteration = 0
        
        print(f"ESA: Starting optimization with T0={T0:.6f}, T_min={T_min:.6f}")
        
        while T > T_min and fail_count < self.n_fail and iteration < self.max_iter:
            print(f"[DEBUG] ESA iteration {iteration}, T={T:.6f}, best_error={best_error:.6f}")
            errors_at_temp = []
            attempted = 0
            accepted = 0
            
            # Equilibrium loop at current temperature
            while attempted < self.n_attempted and accepted < self.n_accepted:
                # Generate perturbation using search space partitioning (M=2 as in paper)
                new_x = self._perturb_subset(x, M=2)
                new_error = objective_func(new_x)
                
                # Accept/reject decision
                delta_error = new_error - current_error
                
                if delta_error < 0:
                    # Accept improvement
                    x = new_x
                    current_error = new_error
                    accepted += 1
                    
                    if new_error < best_error:
                        best_x = new_x.copy()
                        best_error = new_error
                        print(f"ESA: New best error {best_error:.6f} at iteration {iteration}")
                        
                elif np.random.random() < np.exp(-delta_error / T):
                    # Accept with probability
                    x = new_x
                    current_error = new_error
                    accepted += 1
                
                errors_at_temp.append(current_error)
                attempted += 1
            
            # Adaptive cooling
            if len(errors_at_temp) > 0:
                error_min = min(errors_at_temp)
                error_avg = np.mean(errors_at_temp)
                
                # Adaptive cooling rate
                alpha = max(min(error_min / error_avg if error_avg > 0 else self.a_min, 
                               self.a_max), self.a_min)
                T = alpha * T
                
                # Check for failure to improve
                if abs(error_min - best_error) < 1e-8:
                    fail_count += 1
                else:
                    fail_count = 0
            
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"[DEBUG] ESA progress: iteration {iteration}")
            
            if iteration % 100 == 0:
                print(f"ESA: Iteration {iteration}, T={T:.6f}, Best={best_error:.6f}")
        
        print(f"[DEBUG] ESA optimization finished after {iteration} iterations. Best error: {best_error}")
        return best_x, best_error
    
    def _random_initial_point(self):
        """Generate random initial point within bounds."""
        x = np.zeros(len(self.bounds))
        for i, (low, high) in enumerate(self.bounds):
            x[i] = np.random.uniform(low, high)
        return x
    
    def _estimate_temperatures(self, objective_func, n_samples=50):
        """Estimate initial and final temperatures using Monte Carlo."""
        # Sample uphill transitions
        uphill_deltas = []
        for _ in range(n_samples):
            x1 = self._random_initial_point()
            x2 = self._random_initial_point()
            e1 = objective_func(x1)
            e2 = objective_func(x2)
            if e2 > e1:
                uphill_deltas.append(e2 - e1)
        
        if len(uphill_deltas) == 0:
            return 1.0, 1e-6
        
        delta_avg = np.mean(uphill_deltas)
        P0 = 0.5  # Initial acceptance probability
        
        T0 = -delta_avg / np.log(P0)
        T_min = -1e-6 * delta_avg / np.log(1e-6 * P0 + 1e-8)
        
        return max(T0, 1e-6), max(T_min, 1e-8)
    
    def _perturb_subset(self, x, M=2):
        """
        Perturb only M variables using "least frequently used first" rule.
        For simplicity, we randomly select M variables.
        """
        new_x = x.copy()
        indices = np.random.choice(len(x), size=min(M, len(x)), replace=False)
        
        for i in indices:
            low, high = self.bounds[i]
            # Add Gaussian perturbation (adaptive step size could be added)
            perturbation = np.random.normal(0, (high - low) * 0.1)
            new_x[i] = np.clip(x[i] + perturbation, low, high)
        
        return new_x


def align_fragments_papaioannou(source_mesh, target_mesh, params, constraints=None):
    print("[DEBUG] Starting align_fragments_papaioannou")
    """
    Align two fragment meshes using the Papaioannou et al. method.
    
    Args:
        source_mesh, target_mesh: o3d.geometry.TriangleMesh objects
        params: Configuration parameters
        constraints: Optional constraints dict
    
    Returns:
        transformation: 4x4 transformation matrix or None
        fitness: Matching quality score
        convergence_info: Optimization details
    """
    print("Starting Papaioannou alignment method...")
    
    # Initialize error calculator
    resolution = params.get('papaioannou_resolution', 128)
    error_calc = PapaioannoualignmentError(resolution=resolution)
    
    # Define search bounds for 7-DOF parameterization
    # [theta1, phi1, rho1, x1, y1, theta2, phi2]
    max_angle = params.get('max_rotation_angle', np.pi)
    max_translation = params.get('max_translation_factor', 0.1)
    
    # Calculate maximum diameter for translation bounds
    bbox_source = source_mesh.get_axis_aligned_bounding_box()
    bbox_target = target_mesh.get_axis_aligned_bounding_box()
    extent_source = bbox_source.get_extent()
    extent_target = bbox_target.get_extent()
    max_diameter = max(np.max(extent_source), np.max(extent_target))
    max_trans = max_diameter * max_translation
    
    bounds = [
        (-max_angle, max_angle),  # theta1
        (-max_angle, max_angle),  # phi1  
        (-max_angle, max_angle),  # rho1
        (-max_trans, max_trans),  # x1
        (-max_trans, max_trans),  # y1
        (-max_angle, max_angle),  # theta2
        (-max_angle, max_angle),  # phi2
    ]
    
    # Apply constraints if provided
    if constraints:
        bounds = _apply_constraints(bounds, constraints)
    
    # Define objective function
    def objective_function(transform_params):
        try:
            Z1, Z2, valid_mask = error_calc.calculate_zbuffer_distances(
                source_mesh, target_mesh, transform_params
            )
            error = error_calc.calculate_matching_error(Z1, Z2, valid_mask)
            return error
        except Exception as e:
            print(f"Error in objective function: {e}")
            return np.inf
    
    # Run optimization
    optimizer = EnhancedSimulatedAnnealing(
        bounds=bounds,
        max_iter=params.get('esa_max_iter', 2000)
    )
    
    start_time = time.time()
    print("[DEBUG] Running ESA optimizer...")
    best_params, best_error = optimizer.optimize(objective_function)
    print(f"[DEBUG] ESA optimizer finished. best_error={best_error}")
    optimization_time = time.time() - start_time
    
    print(f"Optimization completed in {optimization_time:.2f}s")
    print(f"Best error: {best_error:.6f}")
    print(f"Best parameters: {best_params}")
    
    # Convert best parameters to transformation matrix
    if best_error < params.get('max_acceptable_error', 1.0):
        print(f"[DEBUG] Alignment successful. best_error={best_error}")
        theta1, phi1, rho1, x1, y1, theta2, phi2 = best_params
        
        # Create transformation for source mesh (relative to target)
        T_source = error_calc._create_transformation_matrix(theta1, phi1, rho1, x1, y1, True)
        T_target = error_calc._create_transformation_matrix(theta2, phi2, 0, 0, 0, False)
        
        # Relative transformation from source to target
        transformation = np.linalg.inv(T_target) @ T_source
        
        # Calculate fitness score (inverse of error, normalized)
        fitness = 1.0 / (1.0 + best_error)
        
        convergence_info = {
            'method': 'Papaioannou_ESA',
            'final_error': best_error,
            'optimization_time': optimization_time,
            'parameters': best_params,
            'converged': best_error < params.get('max_acceptable_error', 1.0)
        }
        
        return transformation, fitness, convergence_info
    else:
        print(f"[DEBUG] Alignment failed. best_error={best_error}")
        print(f"Alignment failed: error {best_error:.6f} above threshold")
        return None, 0.0, {
            'method': 'Papaioannou_ESA',
            'final_error': best_error,
            'optimization_time': optimization_time,
            'converged': False
        }


def _apply_constraints(bounds, constraints):
    """Apply constraints to search bounds."""
    new_bounds = bounds.copy()
    
    # Material axis constraint - lock some rotations
    if constraints.get('material_axis_constraint'):
        # Lock theta2 = theta1 and phi2 = phi1 + pi (simplified)
        # In practice, this would require more sophisticated handling
        pass
    
    # Fracture direction constraint - limit rotation ranges
    if constraints.get('fracture_direction_constraint'):
        max_deviation = constraints.get('fracture_angle_tolerance', np.pi/18)  # 10 degrees
        for i in [0, 1, 5, 6]:  # Angular parameters
            new_bounds[i] = (-max_deviation, max_deviation)
    
    return new_bounds


# Legacy compatibility function
def align_fragments_pcd(source_pcd, target_pcd, source_fpfh, target_fpfh, params):
    """
    Legacy compatibility wrapper. 
    Note: The Papaioannou method works directly with meshes, not point clouds.
    This function converts PCDs to meshes and applies the new method.
    """
    print("Warning: Using legacy PCD alignment. Consider using align_fragments_papaioannou directly with meshes.")
    
    # Convert point clouds to meshes using Poisson reconstruction
    try:
        source_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(source_pcd, depth=8)
        target_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(target_pcd, depth=8)
        
        if not source_mesh.has_triangles() or not target_mesh.has_triangles():
            print("Failed to create valid meshes from point clouds")
            return None, 0.0, 0.0
        
        # Use new Papaioannou method
        transformation, fitness, info = align_fragments_papaioannou(source_mesh, target_mesh, params)
        
        if transformation is not None:
            # For compatibility, return RMSE estimate
            rmse = info.get('final_error', 0.0)
            return transformation, fitness, rmse
        else:
            return None, 0.0, np.inf
            
    except Exception as e:
        print(f"Error in legacy alignment: {e}")
        return None, 0.0, np.inf


if __name__ == '__main__':
    # Test the enhanced alignment system
    print("Testing Papaioannou alignment method...")
    
    # Create test meshes
    mesh1 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    mesh2 = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    
    # Apply known transformation to mesh2
    known_transform = np.array([
        [0.866, -0.5, 0, 0.5],
        [0.5, 0.866, 0, 0.3],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    mesh2.transform(known_transform)
    
    # Test parameters
    test_params = {
        'papaioannou_resolution': 64,  # Lower resolution for faster testing
        'max_rotation_angle': np.pi,
        'max_translation_factor': 0.2,
        'esa_max_iter': 500,
        'max_acceptable_error': 0.5
    }
    
    # Run alignment
    result_transform, fitness, info = align_fragments_papaioannou(mesh1, mesh2, test_params)
    
    if result_transform is not None:
        print("Alignment successful!")
        print(f"Fitness: {fitness:.4f}")
        print(f"Convergence info: {info}")
        print("Known transform:")
        print(known_transform)
        print("Recovered transform:")
        print(result_transform)
    else:
        print("Alignment failed")
        print(f"Info: {info}")