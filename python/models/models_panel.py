import open3d.visualization.gui as gui  # type: ignore
import os


class ModelsPanel:
    def __init__(self, app):
        self.app = app

        self.model_checkboxes = []

        w = app.window
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        self.label = gui.Label("Models Panel")
        self._panel.add_child(self.label)
        self._panel.add_fixed(separation_height)

        self.loaded_models = gui.CollapsableVert(
            "Loaded Models", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )

        self._panel.add_child(self.loaded_models)
        self._panel.add_fixed(separation_height)

    def new_model(self):
        i = len(self.app._scenes) - 1

        def handle_click(checked):
            self._on_cb(i, checked)

        text = os.path.basename(self.app._scenes_paths[i])

        if i == 0:
            text = "Processed"

        cb = gui.Checkbox(text)
        cb.checked = False
        cb.set_on_checked(handle_click)
        self.loaded_models.add_child(cb)

        self.model_checkboxes.append(cb)

    def _on_cb(self, index, checked):
        if checked:
            self.app._scenes_selected.add(index)
        else:
            self.app._scenes_selected.remove(index)

        self.app.window.set_needs_layout()
