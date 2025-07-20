import open3d.visualization.gui as gui  # type: ignore
import os
import json

config_file_path = (
    "C:/sem7/FYP/Reassembly/reassembly/python/configuration/configs.json"
)


class ConfigurationPanel:
    def __init__(self, app):
        self.app = app

        self.configs = self.load_config(config_file_path)
        if self.configs is None:
            print("Failed to load configurations")
            exit(1)

        # Dictionary to store references to GUI widgets
        self._widgets = {}
        # Dictionary to store the current values from the GUI
        self.current_configs = {}

        w = app.window
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        self.label = gui.Label("Configurations Panel")
        self._panel.add_child(self.label)
        self._panel.add_fixed(separation_height)

        # Iterate through each top-level section in the configs (e.g., "segmentation", "rendering")
        for section_name, items in self.configs.items():
            # Create a collapsable section for each top-level key
            section_collapsable = gui.CollapsableVert(
                section_name.replace("_", " ").title(),  # Format section name nicely
                0.25 * em,
                gui.Margins(em, 0, 0, 0),
            )
            self._panel.add_child(section_collapsable)
            self._panel.add_fixed(separation_height)

            # Add a dictionary for this section to store its widgets and values
            self._widgets[section_name] = {}
            self.current_configs[section_name] = {}

            # Iterate through each configuration item within the section
            for item in items:
                name = item["name"]
                item_type = item["type"]
                default_value = item["default"]

                # Store the default value in our current_values dictionary
                self.current_configs[section_name][name] = default_value

                # Create a horizontal layout for label and input field
                h_layout = gui.Horiz(0.25 * em)
                h_layout.add_child(
                    gui.Label(name.replace("_", " ") + ":")
                )  # Label for the setting

                widget = None
                if item_type == "text":
                    text_edit = gui.TextEdit()
                    text_edit.set_text(str(default_value))
                    text_edit.set_on_value_changed(
                        self._create_on_value_changed_callback(
                            section_name, name, text_edit
                        )
                    )
                    widget = text_edit
                elif item_type == "number":
                    number_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
                    number_edit.set_value(float(default_value))
                    number_edit.set_on_value_changed(
                        self._create_on_value_changed_callback(
                            section_name, name, number_edit
                        )
                    )
                    widget = number_edit
                else:
                    print(
                        f"Warning: Unknown input type '{item_type}' for '{name}'. Skipping."
                    )
                    continue

                if widget:
                    h_layout.add_child(widget)
                    section_collapsable.add_child(h_layout)
                    self._widgets[section_name][name] = widget

    def _create_on_value_changed_callback(self, section, name, widget):
        """
        Creates a callback function for an input widget.
        This allows us to capture the section and name for updating self.current_values.
        """

        def on_value_changed(new_value):
            self.current_configs[section][name] = new_value
            print(f"Config updated: [{section}][{name}] = {new_value}")
            # You can add more logic here, e.g., trigger a re-render or a computation
            # self.app.something_changed(section, name, new_value)

        return on_value_changed

    def load_config(self, file_path):
        """
        Loads configuration from a JSON file.
        Returns a dictionary or None if an error occurs.
        """
        if not os.path.exists(file_path):
            print(f"Error: Configuration file '{file_path}' not found.")
            return None
        try:
            with open(file_path, "r") as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in '{file_path}': {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while loading config: {e}")
            return None
