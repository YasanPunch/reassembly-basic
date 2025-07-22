import open3d.visualization.gui as gui  # type: ignore
import os

class ItemTree:
    """
    ItemTree manages the item tree (formerly db tree) UI and logic for the left panel.
    This class is currently empty and ready for new logic as per user instructions.
    """
    def __init__(self, app):
        self.app = app
        # Minimal placeholder for UI compatibility
        self.section = gui.Vert() 