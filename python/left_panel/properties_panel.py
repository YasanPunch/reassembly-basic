import open3d.visualization.gui as gui

class PropertiesPanel:
    """
    PropertiesPanel manages the properties section UI and logic for the left panel.
    This is a basic implementation with a header and placeholder.
    """
    def __init__(self, app):
        self.app = app
        em = app.window.theme.font_size
        self.section = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.label = gui.Label("Properties")
        self.label.text_color = gui.Color(0.2, 0.8, 0.2)
        self.section.add_child(self.label)
        self.section.add_fixed(int(round(0.5 * em)))
        self.placeholder_label = gui.Label("No properties to show")
        self.section.add_child(self.placeholder_label) 
