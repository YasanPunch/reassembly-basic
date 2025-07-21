import open3d.visualization.gui as gui  # type: ignore
import os


class LeftPanel:
    def __init__(self, app):
        self.app = app
        w = app.window  # to make the code more concise

        # ---- Left panel ----
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Main left panel layout - using a vertical layout to stack the two sections
        self._left_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        # DB Tree Section (Upper half)
        self._db_tree_section = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )
        
        # Add a label for the DB Tree section
        self._db_tree_label = gui.Label("DB Tree")
        self._db_tree_label.text_color = gui.Color(0.2, 0.6, 1.0)  # Blue color for header
        self._db_tree_section.add_child(self._db_tree_label)
        self._db_tree_section.add_fixed(separation_height)
        
        # Add a placeholder for the tree/list of objects
        self._db_tree_list = gui.ListView()
        self._db_tree_list.set_items(["No objects loaded"])
        self._db_tree_list.set_max_visible_items(8)
        self._db_tree_section.add_child(self._db_tree_list)
        
        # Add the DB Tree section to the main panel
        self._left_panel.add_child(self._db_tree_section)
        self._left_panel.add_fixed(separation_height)

        # Properties Section (Lower half)
        self._properties_section = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )
        
        # Add a label for the Properties section
        self._properties_label = gui.Label("Properties")
        self._properties_label.text_color = gui.Color(0.2, 0.8, 0.2)  # Green color for header
        self._properties_section.add_child(self._properties_label)
        self._properties_section.add_fixed(separation_height)
        
        # Add a placeholder for properties display
        self._properties_text = gui.TextEdit()
        self._properties_text.placeholder_text = "Select an object to view properties"
        self._properties_text.enabled = False
        self._properties_section.add_child(self._properties_text)
        
        # Add the Properties section to the main panel
        self._left_panel.add_child(self._properties_section) 