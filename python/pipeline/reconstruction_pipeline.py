class ReconstructionPipeline:
    def __init__(self, app):
        self.app = app
        self.fragments_data_raw = []
        self.processed_fragments_data = []
        self.pairwise_matches = []
        self.reconstructed_model = None
        self.current_step = "idle"
        self.is_running = False

    def get_current_loaded_objects(self):
        return self.app.left_panel.item_tree.get_all_objects()
