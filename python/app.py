import os
import sys
import json

import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore

from configuration.configuration_panel import ConfigurationPanel
from left_panel.left_panel import LeftPanel
from models.models_panel import ModelsPanel
from panels.processing_panel import ProcessingPanel
from settings.settings import Settings
from settings.settings_panel import SettingsPanel
from left_panel.item_tree.items.base_model_item import BaseModelItem



class App:
    """
    Main application class for the 3D Model Fragment Reconstructor.
    
    This class manages the GUI application and provides access to configuration parameters
    that can be used by any module or class under the app. Parameters are automatically
    loaded from a JSON configuration file on initialization.
    
    Attributes:
        params (dict): Dictionary containing all reconstruction parameters
        config_file (str): Path to the configuration file
    """
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_SHOW_MODELS = 13
    MENU_SHOW_CONFIGS = 14
    MENU_SHOW_PCPROCESSING = 12

    MENU_ABOUT = 21

    DEFAULT_IBL = "default"
    DEFAULT_CONFIG_FILE = "config/reconstruction_params.json"

    def __init__(self, width, height, config_file=None):
        """
        Initialize the application with configuration parameters.
        
        Args:
            width (int): Window width
            height (int): Window height
            config_file (str, optional): Path to JSON configuration file. 
                                       Uses DEFAULT_CONFIG_FILE if None.
        """
        self.settings = Settings()
        
        # Load configuration parameters
        self.config_file = config_file or App.DEFAULT_CONFIG_FILE
        self.params = self._load_parameters()

        self._scene_objects = {}  # To store {'path': {'mesh' | 'geometry', 'transform'}}

        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + App.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Reassembly", width, height
        )
        w = self.window
        
        # Create the main 3D scene widget
        self._scene_widget = gui.SceneWidget()
        self._scene_widget.scene = rendering.Open3DScene(w.renderer)
        self._scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA) # Default camera control

        # --- Visualization: Set background and lighting for the scene widget ---
        self._scene_widget.scene.set_background([1, 1, 1, 1])  # White background
        self._scene_widget.scene.scene.set_sun_light([0, 1, 0], [1, 1, 1], 100000)
        self._scene_widget.scene.scene.enable_sun_light(True)
        # ---------------------------------------------------------------

        self._panels_layout = gui.ScrollableVert()

        self._left_panel = LeftPanel(self)
        self._settings_panel = SettingsPanel(self)
        self._models_panel = ModelsPanel(self)
        self._configuration_panel = ConfigurationPanel(self)
        self._processing_panel = ProcessingPanel(self)


        w.set_on_layout(self._on_layout)
        w.add_child(self._scene_widget) # Add the single scene widget
        w.add_child(self._left_panel.item_tree.section)
        w.add_child(self._left_panel.properties_panel.section)
        w.add_child(self._panels_layout)

        p = self._panels_layout

        p.add_child(self._settings_panel._settings_panel)
        p.add_child(self._models_panel._panel)
        p.add_child(self._configuration_panel._panel)
        p.add_child(self._processing_panel._panel)


        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Open...", App.MENU_OPEN)
            file_menu.add_item("Export Current Image...", App.MENU_EXPORT)
            file_menu.add_separator()
            file_menu.add_item("Quit", App.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials", App.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(App.MENU_SHOW_SETTINGS, True)
            settings_menu.add_item("Models", App.MENU_SHOW_MODELS)
            settings_menu.set_checked(App.MENU_SHOW_MODELS, True)
            settings_menu.add_item("Configurations", App.MENU_SHOW_CONFIGS)
            settings_menu.set_checked(App.MENU_SHOW_CONFIGS, True)
            settings_menu.add_item("Processing", App.MENU_SHOW_PCPROCESSING)
            settings_menu.set_checked(App.MENU_SHOW_PCPROCESSING, True)

            help_menu = gui.Menu()
            help_menu.add_item("About", App.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Panel Settings", settings_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(App.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(App.MENU_EXPORT, self._on_menu_export)
        w.set_on_menu_item_activated(App.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(
            App.MENU_SHOW_SETTINGS, self._on_menu_toggle_settings_panel
        )
        w.set_on_menu_item_activated(
            App.MENU_SHOW_PCPROCESSING, self._on_menu_toggle_processing_panel
        )
        w.set_on_menu_item_activated(
            App.MENU_SHOW_MODELS, self._on_menu_toggle_models_panel
        )
        w.set_on_menu_item_activated(
            App.MENU_SHOW_CONFIGS, self._on_menu_toggle_configs_panel
        )

        w.set_on_menu_item_activated(App.MENU_ABOUT, self._on_menu_about)
        # Menu ----

    def _load_parameters(self):
        """
        Loads configuration parameters from JSON file.
        
        Returns:
            dict: Dictionary containing reconstruction parameters
        """
        print(f"[DEBUG] Loading parameters from: {self.config_file}")
        try:
            with open(self.config_file, 'r') as f:
                params = json.load(f)
            print(f"[DEBUG] Successfully loaded {len(params)} parameters from {self.config_file}")
            return params
        except FileNotFoundError:
            print(f"[WARNING] Config file not found at {self.config_file}. Creating default configuration.")
            return self._create_default_parameters()
        except json.JSONDecodeError:
            print(f"[ERROR] Could not decode JSON from {self.config_file}. Using default configuration.")
            return self._create_default_parameters()
        except Exception as e:
            print(f"[ERROR] Unexpected error loading parameters: {e}. Using default configuration.")
            return self._create_default_parameters()

    def _create_default_parameters(self):
        """
        Creates default configuration parameters if config file is not found or invalid.
        
        Returns:
            dict: Dictionary containing default reconstruction parameters
        """
        print("[DEBUG] Creating default configuration parameters")
        default_params = {
            "voxel_downsample_size": 7.0,
            "normal_estimation_radius": 14.0,
            "normal_estimation_max_nn": 30,
            "fpfh_feature_radius": 35.0,
            "fpfh_feature_max_nn": 100,
            "ransac_distance_threshold_factor": 1.5,
            "ransac_edge_length_factor": 0.9,
            "ransac_iterations": 1000000,
            "ransac_n_points": 4,
            "ransac_confidence": 0.999,
            "icp_max_correspondence_distance_factor": 2.0,
            "icp_relative_fitness": 1e-6,
            "icp_relative_rmse": 1e-6,
            "icp_max_iteration": 50,
            "min_match_score": 0.3,
            "min_boundary_edges_for_fracture_face": 1,
            "fracture_surface_dense_sample_points": 10000,
            "add_preprocessing_noise": True,
            "preprocessing_noise_factor": 0.01,
            "orient_normals_k": 15,
            "use_boolean_intersection_test": True,
            "boolean_penetration_threshold": 0.1,
        }
        
        # Try to save the default config file
        try:
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(default_params, f, indent=4)
            print(f"[DEBUG] Created default config file at {self.config_file}")
        except Exception as e:
            print(f"[WARNING] Could not create default config file: {e}")
        
        return default_params

    def get_parameter(self, key, default=None):
        """
        Get a specific parameter value.
        
        Args:
            key (str): Parameter key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The parameter value or default if not found
        """
        return self.params.get(key, default)

    def set_parameter(self, key, value):
        """
        Set a specific parameter value.
        
        Args:
            key (str): Parameter key to set
            value: Value to set for the parameter
        """
        self.params[key] = value

    def save_parameters(self, config_file=None):
        """
        Save current parameters to JSON file.
        
        Args:
            config_file (str, optional): Path to save config file. Uses current config_file if None.
        """
        save_path = config_file or self.config_file
        try:
            config_dir = os.path.dirname(save_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(self.params, f, indent=4)
            print(f"[DEBUG] Parameters saved to {save_path}")
        except Exception as e:
            print(f"[ERROR] Could not save parameters to {save_path}: {e}")

    def reload_parameters(self, config_file=None):
        """
        Reload parameters from JSON file.
        
        Args:
            config_file (str, optional): Path to config file. Uses current config_file if None.
        """
        if config_file:
            self.config_file = config_file
        self.params = self._load_parameters()
        print(f"[DEBUG] Parameters reloaded from {self.config_file}")

    def get_all_parameters(self):
        """
        Get all configuration parameters.
        
        Returns:
            dict: Dictionary containing all reconstruction parameters
        """
        return self.params.copy()

    def get_parameters_subset(self, keys):
        """
        Get a subset of parameters by key names.
        
        Args:
            keys (list): List of parameter keys to retrieve
            
        Returns:
            dict: Dictionary containing only the requested parameters
        """
        return {key: self.params.get(key) for key in keys}

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        em = layout_context.theme.font_size
        
        panel_width = 17 * em

        # Position left panel
        self._left_panel.item_tree.section.frame = gui.Rect(r.x, r.y, panel_width, r.height / 2)
        self._left_panel.properties_panel.section.frame = gui.Rect(r.x, r.y + r.height / 2, panel_width, r.height / 2)

        # Position right panel
        self._panels_layout.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

        # Position the main scene widget in the center (ensure it's not overlapped)
        scene_x = r.x + panel_width
        scene_width = r.width - 2 * panel_width
        self._scene_widget.frame = gui.Rect(scene_x, r.y, scene_width, r.height)

        height = min(
            r.height,
            self._settings_panel._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._settings_panel._settings_panel.frame = gui.Rect(
            r.get_right() - panel_width, r.y, panel_width, height
        )

        height = min(
            r.height,
            self._configuration_panel._panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._configuration_panel._panel.frame = gui.Rect(
            r.get_right() - panel_width, r.y, panel_width, height
        )

        height = min(
            r.height,
            self._processing_panel._panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._processing_panel._panel.frame = gui.Rect(
            r.get_right() - 2 * panel_width, r.get_bottom() - height, panel_width, height
        )



    def _on_menu_open(self):
        try:
            dlg = gui.FileDialog(
                gui.FileDialog.OPEN, "Choose file to load", self.window.theme
            )
            dlg.add_filter(
                ".ply .stl .fbx .obj .off .gltf .glb",
                "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, .gltf, .glb)",
            )
            dlg.add_filter(
                ".xyz .xyzn .xyzrgb .ply .pcd .pts",
                "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, .pcd, .pts)",
            )
            dlg.add_filter(".ply", "Polygon files (.ply)")
            dlg.add_filter(".stl", "Stereolithography files (.stl)")
            dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
            dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
            dlg.add_filter(".off", "Object file format (.off)")
            dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
            dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
            dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
            dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
            dlg.add_filter(".xyzrgb", "ASCII point cloud files with colors (.xyzrgb)")
            dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
            dlg.add_filter(".pts", "3D Points files (.pts)")
            dlg.add_filter("", "All files")

            # A file dialog MUST define on_cancel and on_done functions
            dlg.set_on_cancel(self._on_file_dialog_cancel)
            dlg.set_on_done(self._on_load_dialog_done)
            self.window.show_dialog(dlg)
        except Exception as e:
            print(f"Error opening file dialog: {e}")

    def _on_file_dialog_cancel(self):
        try:
            self.window.close_dialog()
        except Exception as e:
            print(f"Error closing file dialog: {e}")

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        try:
            self.load(filename)
        except Exception as e:
            print(f"Error loading file: {e}")

    def _on_menu_export(self):
        try:
            dlg = gui.FileDialog(
                gui.FileDialog.SAVE, "Choose file to save", self.window.theme
            )
            dlg.add_filter(".png", "PNG files (.png)")
            dlg.set_on_cancel(self._on_file_dialog_cancel)
            dlg.set_on_done(self._on_export_dialog_done)
            self.window.show_dialog(dlg)
        except Exception as e:
            print(f"Error opening export dialog: {e}")

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        try:
            # frame = self._scene.frame
            # self.export_image(filename, frame.width, frame.height)
            pass
        except Exception as e:
            print(f"Error exporting image: {e}")

    def _on_menu_quit(self):
        try:
            gui.Application.instance.quit()
        except Exception as e:
            print(f"Error quitting application: {e}")

    def _on_menu_toggle_models_panel(self):
        try:
            self._models_panel._panel.visible = not self._models_panel._panel.visible
            gui.Application.instance.menubar.set_checked(
                App.MENU_SHOW_MODELS, self._models_panel._panel.visible
            )
            self.window.set_needs_layout()
        except Exception as e:
            print(f"Error toggling models panel: {e}")

    def _on_menu_toggle_configs_panel(self):
        try:
            self._configuration_panel._panel.visible = (
                not self._configuration_panel._panel.visible
            )
            gui.Application.instance.menubar.set_checked(
                App.MENU_SHOW_CONFIGS, self._configuration_panel._panel.visible
            )
            self.window.set_needs_layout()
        except Exception as e:
            print(f"Error toggling configs panel: {e}")

    def _on_menu_toggle_settings_panel(self):
        try:
            self._settings_panel._settings_panel.visible = (
                not self._settings_panel._settings_panel.visible
            )
            gui.Application.instance.menubar.set_checked(
                App.MENU_SHOW_SETTINGS, self._settings_panel._settings_panel.visible
            )
            self.window.set_needs_layout()
        except Exception as e:
            print(f"Error toggling settings panel: {e}")

    def _on_menu_toggle_processing_panel(self):
        try:
            self._processing_panel._panel.visible = (
                not self._processing_panel._panel.visible
            )
            gui.Application.instance.menubar.set_checked(
                App.MENU_SHOW_PCPROCESSING,
                self._processing_panel._panel.visible,
            )
            self.window.set_needs_layout()
        except Exception as e:
            print(f"Error toggling processing panel: {e}")



    def _on_menu_about(self):
        try:
            em = self.window.theme.font_size
            dlg = gui.Dialog("About")

            # Add the text
            dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
            dlg_layout.add_child(gui.Label("Reassembly Application"))

            # Add the Ok button. We need to define a callback function to handle
            # the click.
            ok = gui.Button("OK")
            ok.set_on_clicked(self._on_about_ok)

            # We want the Ok button to be an the right side, so we need to add
            # a stretch item to the layout, otherwise the button will be the size
            # of the entire row. A stretch item takes up as much space as it can,
            # which forces the button to be its minimum size.
            h = gui.Horiz()
            h.add_stretch()
            h.add_child(ok)
            h.add_stretch()
            dlg_layout.add_child(h)

            dlg.add_child(dlg_layout)
            self.window.show_dialog(dlg)
        except Exception as e:
            print(f"Error showing about dialog: {e}")

    def _on_about_ok(self):
        try:
            self.window.close_dialog()
        except Exception as e:
            print(f"Error closing about dialog: {e}")

    def _add_object_to_scene(self, path, geometry=None):
        """Adds a model or point cloud to the single 3D scene."""
        print(f"[DEBUG] _add_object_to_scene called with path: {path}, mesh: {geometry is not None}")
        if geometry is not None:
            # Use add_geometry instead of add_model, and ensure a material is provided
            import open3d.visualization.rendering as rendering
            if hasattr(self.settings, 'material') and self.settings.material is not None:
                material = self.settings.material
            else:
                material = rendering.MaterialRecord()
                material.shader = "defaultLit"
            self._scene_widget.scene.add_geometry(path, geometry, material)
            self._scene_objects[path] = {'type': 'geometry', 'geometry': geometry}
            print(f"[DEBUG] Geometry added to scene: {path}")
        else:
            print(f"[ERROR] No geometry provided for: {path}")

        # Add object to the item tree UI using the new item system
        base_model_item = self._left_panel.item_tree.add_base_model_item(
            mesh_path=path, 
            mesh=geometry, 
            label=os.path.basename(path)
        )
        
        # Store the item reference for later use
        if not hasattr(self, '_base_model_items'):
            self._base_model_items = {}
        self._base_model_items[path] = base_model_item

        # Update camera to frame all visible objects AFTER adding to item tree
        self._update_camera_bounds()
    
    def _update_camera_bounds(self):
        """Calculates the bounding box of all visible objects and adjusts the camera."""
        print(f"[DEBUG] _update_camera_bounds called with {len(self._scene_objects)} scene objects")
        bounds = None
        
        # Get all base model items from the item tree
        base_model_items = self._left_panel.item_tree.get_items_by_type('base_model')
        
        for item in base_model_items:
            if isinstance(item, BaseModelItem):
                path = item.mesh_path
                print(f"[DEBUG] Checking base model item: {path}, visible: {item.is_visible}")
                
                if item.is_visible and path in self._scene_objects:
                    print(f"[DEBUG] Object {path} is visible, calculating bounds")
                    obj_info = self._scene_objects[path]
                    
                    if obj_info['type'] == 'model':
                        geom_bounds = obj_info['mesh'].get_axis_aligned_bounding_box()
                    else: # geometry
                        geom_bounds = obj_info['geometry'].get_axis_aligned_bounding_box()

                    if bounds is None:
                        bounds = geom_bounds
                    else:
                        bounds = bounds.get_union(geom_bounds)
                else:
                    print(f"[DEBUG] Object {path} not found in scene or not visible")
        
        if bounds and bounds.volume() > 0:
            print(f"[DEBUG] Setting camera with bounds: {bounds}")
            self._scene_widget.setup_camera(60, bounds, bounds.get_center())
        else:
            print(f"[DEBUG] No valid bounds, using default camera")
            self._scene_widget.setup_camera(60, o3d.geometry.AxisAlignedBoundingBox([-1,-1,-1], [1,1,1]), [0,0,0])

    def show_geometry_viewer(self, geometries, window_name="Geometry Viewer"):
        """
        Show geometries in a separate window using the GUI framework.
        
        This method creates a new window with a scene widget to display geometries
        without using draw_geometries, avoiding threading conflicts.
        
        Args:
            geometries (list): List of Open3D geometry objects (meshes, point clouds, etc.)
            window_name (str): Title for the viewer window
            
        Returns:
            tuple: (window, scene_widget) for potential future use
        """
        try:
            print(f"[DEBUG] Creating geometry viewer window: {window_name}")
            
            # Create a new window
            viewer_window = gui.Application.instance.create_window(window_name, 800, 600)
            
            # Create scene widget
            scene_widget = gui.SceneWidget()
            scene_widget.scene = rendering.Open3DScene(viewer_window.renderer)
            scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
            
            # Set up scene
            scene_widget.scene.set_background([1, 1, 1, 1])  # White background
            scene_widget.scene.scene.set_sun_light([0, 1, 0], [1, 1, 1], 100000)
            scene_widget.scene.scene.enable_sun_light(True)
            
            # Add geometries to the scene
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            
            for i, geometry in enumerate(geometries):
                geometry_name = f"geometry_{i}"
                scene_widget.scene.add_geometry(geometry_name, geometry, material)
            
            # Set up layout
            def on_layout(layout_context):
                r = viewer_window.content_rect
                scene_widget.frame = gui.Rect(r.x, r.y, r.width, r.height)
            
            viewer_window.set_on_layout(on_layout)
            viewer_window.add_child(scene_widget)
            
            # Calculate bounds and set up camera
            bounds = None
            for geometry in geometries:
                if hasattr(geometry, 'get_axis_aligned_bounding_box'):
                    geom_bounds = geometry.get_axis_aligned_bounding_box()
                    if bounds is None:
                        bounds = geom_bounds
                    else:
                        # Use min/max to create union bounds
                        min_bound = bounds.get_min_bound()
                        max_bound = bounds.get_max_bound()
                        geom_min = geom_bounds.get_min_bound()
                        geom_max = geom_bounds.get_max_bound()
                        
                        new_min = [min(min_bound[i], geom_min[i]) for i in range(3)]
                        new_max = [max(max_bound[i], geom_max[i]) for i in range(3)]
                        bounds = o3d.geometry.AxisAlignedBoundingBox(new_min, new_max)
            
            if bounds and bounds.volume() > 0:
                scene_widget.setup_camera(60, bounds, bounds.get_center())
            else:
                # Default camera setup
                default_bounds = o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
                scene_widget.setup_camera(60, default_bounds, [0, 0, 0])
            
            print(f"[DEBUG] Geometry viewer window created successfully: {window_name}")
            return viewer_window, scene_widget
            
        except Exception as e:
            print(f"[ERROR] Failed to create geometry viewer: {e}")
            return None, None

    def show_debug_visualization(self, geometries, window_name="Debug Visualization"):
        """
        Convenience method for showing debug visualizations.
        
        Args:
            geometries (list): List of Open3D geometry objects
            window_name (str): Title for the debug window
            
        Returns:
            tuple: (window, scene_widget) for potential future use
        """
        return self.show_geometry_viewer(geometries, window_name)

    def load(self, path):
        print(f"[DEBUG] Attempting to load file: {path}")
        geometry = None
        # Try to load as mesh
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh is not None and mesh.has_vertices():
            print(f"[DEBUG] Successfully loaded mesh with {len(mesh.vertices)} vertices")
            geometry = mesh
        else:
            print(f"[DEBUG] Failed to load as mesh, trying point cloud...")
            # Try to load as point cloud
            cloud = o3d.io.read_point_cloud(path)
            if cloud is not None and cloud.has_points():
                print(f"[DEBUG] Successfully loaded point cloud with {len(cloud.points)} points")
                if not cloud.has_normals():
                    cloud.estimate_normals()
                geometry = cloud
            else:
                print(f"[DEBUG] Failed to load as point cloud")

        if geometry is not None:
            self._add_object_to_scene(path=path, geometry=geometry)
        else:
            print(f"[ERROR] Could not load any geometry or mesh from: {path}")

    def export_image(self, path, width, height):
        return

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def run(self):
        pass


def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    # Parse command line arguments for config file
    config_file = None
    if len(sys.argv) > 1:
        # Check if first argument is a config file (ends with .json)
        if sys.argv[1].endswith('.json'):
            config_file = sys.argv[1]
            # Remove config file from paths to process
            paths = sys.argv[2:]
        else:
            paths = sys.argv[1:]
    else:
        paths = []

    w = App(1024, 768, config_file=config_file)

    if paths:
        for path in paths:
            if os.path.exists(path):
                w.load(path)
            else:
                w.window.show_message_box("Error", "Could not open file '" + path + "'")
    else:
        # The app should start empty and only load files when the user imports them via the menu.
        pass

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
