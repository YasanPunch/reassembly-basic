import open3d.visualization.gui as gui


class ModelsPanel:
    def __init__(self, app):
        self.app = app

        w = app.window
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        # Removed the 'Models Panel' label and all model UI

    def new_model(self):
        # No longer needed, as model checkboxes are now in the left panel
        pass
