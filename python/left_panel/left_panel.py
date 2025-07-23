import open3d.visualization.gui as gui  # type: ignore
import os
from .item_tree.item_tree import ItemTree
from .properties_panel import PropertiesPanel


class LeftPanel:
    def __init__(self, app):
        self.app = app
        w = app.window
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Main left panel layout
        self._left_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        # Item Tree Section (Upper half)
        self.item_tree = ItemTree(app)
        self._left_panel.add_child(self.item_tree.section)
        self._left_panel.add_fixed(separation_height)

        # Properties Section (Lower half)
        self.properties_panel = PropertiesPanel(app)
        self._left_panel.add_child(self.properties_panel.section) 