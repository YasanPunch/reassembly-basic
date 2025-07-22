import os
import sys

import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore

from configuration.configuration_panel import ConfigurationPanel
from left_panel.left_panel import LeftPanel
from models.models_panel import ModelsPanel
from processing.processing_panel import ProcessingPanel
from settings.settings import Settings
from settings.settings_panel import SettingsPanel


class App:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_SHOW_MODELS = 13
    MENU_SHOW_CONFIGS = 14
    MENU_SHOW_PCPROCESSING = 12
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    def __init__(self, width, height):
        self.settings = Settings()

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

        # Add object to the item tree UI FIRST
        self._left_panel.item_tree.add_object(path, name=os.path.basename(path))

        # Update camera to frame all visible objects AFTER adding to item tree
        self._update_camera_bounds()
    
    def _update_camera_bounds(self):
        """Calculates the bounding box of all visible objects and adjusts the camera."""
        print(f"[DEBUG] _update_camera_bounds called with {len(self._scene_objects)} scene objects")
        bounds = None
        for path, obj_info in self._scene_objects.items():
            print(f"[DEBUG] Checking object: {path}, type: {obj_info['type']}")
            # Check if object is visible (get from item tree)
            obj_in_panel = next((o for o in self._left_panel.item_tree.get_all_objects() if o['path'] == path), None)
            if obj_in_panel and obj_in_panel['visible']:
                print(f"[DEBUG] Object {path} is visible, calculating bounds")
                if obj_info['type'] == 'model':
                    geom_bounds = obj_info['mesh'].get_axis_aligned_bounding_box()
                else: # geometry
                    geom_bounds = obj_info['geometry'].get_axis_aligned_bounding_box()

                if bounds is None:
                    bounds = geom_bounds
                else:
                    bounds = bounds.get_union(geom_bounds)
            else:
                print(f"[DEBUG] Object {path} not found in panel or not visible")
        
        if bounds and bounds.volume() > 0:
            print(f"[DEBUG] Setting camera with bounds: {bounds}")
            self._scene_widget.setup_camera(60, bounds, bounds.get_center())
        else:
            print(f"[DEBUG] No valid bounds, using default camera")
            self._scene_widget.setup_camera(60, o3d.geometry.AxisAlignedBoundingBox([-1,-1,-1], [1,1,1]), [0,0,0])


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

    w = App(1024, 768)

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
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
