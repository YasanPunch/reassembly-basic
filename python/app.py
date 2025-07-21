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

        self._scenes = []
        self._scenes_paths = []
        self._scenes_selected = set()

        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + App.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Reassembly", width, height
        )
        w = self.window
        self._panels_layout = gui.ScrollableVert()

        self._left_panel = LeftPanel(self)
        self._settings_panel = SettingsPanel(self)
        self._models_panel = ModelsPanel(self)
        self._configuration_panel = ConfigurationPanel(self)
        self._processing_panel = ProcessingPanel(self)

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._left_panel._db_tree_section)
        w.add_child(self._left_panel._properties_section)
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

        # Create processed scene widget
        self.load(
            "/home/pundima/dev/reassembly/data/Tombstone/Reassembled_Tombstone.obj"
        )

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        em = layout_context.theme.font_size
        width = 17 * em

        # Position the right panels layout (existing panels)
        self._panels_layout.frame = gui.Rect(
            r.get_right() - width, r.y, width, r.height
        )

        # Position the left panel
        self._left_panel._left_panel.frame = gui.Rect(
            r.x, r.y, width, r.height
        )
        
        # Calculate margins and sizes for the containers
        margin_x = width * 0.05  # 5% margin on each side (10% total)
        container_width = width * 0.9  # 90% width
        container_height = r.height * 0.5  # 50% height
        margin_y = (r.height - 2 * container_height) / 3  # Equal spacing between containers
        
        # Position the DB Tree section (upper half)
        db_tree_x = r.x + margin_x
        db_tree_y = r.y + margin_y
        self._left_panel._db_tree_section.frame = gui.Rect(
            db_tree_x, db_tree_y, container_width, container_height
        )
        
        # Position the Properties section (lower half)
        properties_x = r.x + margin_x
        properties_y = r.y + margin_y + container_height + margin_y
        self._left_panel._properties_section.frame = gui.Rect(
            properties_x, properties_y, container_width, container_height
        )

        scene_state = "normal"
        if 0 in self._scenes_selected and len(self._scenes_selected) == 1:
            scene_width = r.get_right() - 2 * width  # Account for both left and right panels
            scene_state = "processed_only"
        elif 0 in self._scenes_selected and len(self._scenes_selected) > 1:
            scene_width = (r.get_right() - 2 * width) / 2  # Account for both panels
            scene_state = "normal"
        else:
            scene_width = r.get_right() - 2 * width  # Account for both panels
            scene_state = "models_only"

        visible__models_count = 0
        for i, s in enumerate(self._scenes):
            if i not in self._scenes_selected:
                s.visible = False
                continue

            s.visible = True

            if i == 0 and scene_state == "processed_only":
                height = r.height
                s.frame = gui.Rect(r.x + width, r.y, scene_width, height)  # Start after left panel
            elif i == 0 and scene_state == "normal":
                height = r.height
                s.frame = gui.Rect(r.x + width + scene_width, r.y, scene_width, height)  # Start after left panel
            elif i != 0 and scene_state == "normal":
                height = r.height / (len(self._scenes_selected) - 1)
                start_y = r.y + (visible__models_count * height)
                s.frame = gui.Rect(r.x + width, start_y, scene_width, height)  # Start after left panel
                visible__models_count += 1
            elif i != 0 and scene_state == "models_only":
                height = r.height / len(self._scenes_selected)
                start_y = r.y + (visible__models_count * height)
                s.frame = gui.Rect(r.x + width, start_y, scene_width, height)  # Start after left panel
                visible__models_count += 1

        height = min(
            r.height,
            self._settings_panel._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._settings_panel._settings_panel.frame = gui.Rect(
            r.get_right() - width, r.y, width, height
        )

        height = min(
            r.height,
            self._configuration_panel._panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._configuration_panel._panel.frame = gui.Rect(
            r.get_right() - width, r.y, width, height
        )

        height = min(
            r.height,
            self._processing_panel._panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._processing_panel._panel.frame = gui.Rect(
            r.get_right() - 2 * width, r.get_bottom() - height, width, height
        )

    def _on_menu_open(self):
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

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(
            gui.FileDialog.SAVE, "Choose file to save", self.window.theme
        )
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        return
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_models_panel(self):
        self._models_panel._panel.visible = not self._models_panel._panel.visible
        gui.Application.instance.menubar.set_checked(
            App.MENU_SHOW_MODELS, self._models_panel._panel.visible
        )
        self.window.set_needs_layout()

    def _on_menu_toggle_configs_panel(self):
        self._configuration_panel._panel.visible = (
            not self._configuration_panel._panel.visible
        )
        gui.Application.instance.menubar.set_checked(
            App.MENU_SHOW_CONFIGS, self._configuration_panel._panel.visible
        )
        self.window.set_needs_layout()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel._settings_panel.visible = (
            not self._settings_panel._settings_panel.visible
        )
        gui.Application.instance.menubar.set_checked(
            App.MENU_SHOW_SETTINGS, self._settings_panel._settings_panel.visible
        )
        self.window.set_needs_layout()

    def _on_menu_toggle_processing_panel(self):
        self._processing_panel._panel.visible = (
            not self._processing_panel._panel.visible
        )
        gui.Application.instance.menubar.set_checked(
            App.MENU_SHOW_PCPROCESSING,
            self._processing_panel._panel.visible,
        )
        self.window.set_needs_layout()

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
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

    def _on_about_ok(self):
        self.window.close_dialog()

    # You should pass either mesh or geometry
    def create_scene_widget(self, path, mesh=None, geometry=None):
        w = self.window
        s = gui.SceneWidget()
        s.scene = rendering.Open3DScene(w.renderer)

        if mesh is not None:
            # Triangle model
            s.scene.add_model("__model__", mesh)
        else:
            pass
            # Point cloud
            s.scene.add_geometry("__model__", geometry, self.settings.material)

        bounds = s.scene.bounding_box
        s.setup_camera(60, bounds, bounds.get_center())
        s.set_on_sun_direction_changed(self._settings_panel._on_sun_dir)

        i = len(self._scenes)

        self._scenes.append(s)
        # self._scenes_selected.add(i)
        self._scenes_paths.append(path)

        self._models_panel.new_model()

        self._settings_panel._apply_settings([i])

        w.add_child(s)

        w.set_needs_layout()

    def load(self, path):
        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_model(path)
        if mesh is None:
            print("[Info]", path, "appears to be a point cloud")
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None or mesh is not None:
            try:
                self.create_scene_widget(path=path, mesh=mesh, geometry=geometry)
            except Exception as e:
                print(e)

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
        paths = [
            "C:/sem7/FYP/Tombstone/Tombstone1_low.obj",
            "C:/sem7/FYP/Tombstone/Tombstone2_low.obj",
            "C:/sem7/FYP/Tombstone/Tombstone3_low.obj",
        ]
        for path in paths:
            if os.path.exists(path):
                w.load(path)
            else:
                w.window.show_message_box("Error", "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
