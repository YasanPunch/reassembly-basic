import os
import sys
# Add src to path to import custom modules
module_path = os.path.abspath(os.path.join('.')) # Project root
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import visualization_utils
from src.io_utils import load_fragments_from_directory # To get original names and indices

if __name__ == '__main__':
    log_file = "data/my_run_log.pkl" # Path to your saved log
    
    # To help with coloring, load the original fragment names and their order
    # This assumes your input_fragments dir is 'data/input_fragments'
    # You might need to pass this path if it's different.
    input_frags_dir = "data/input_fragments"
    raw_frags_for_map = load_fragments_from_directory(input_frags_dir)
    
    fragments_name_to_idx_map = {}
    if raw_frags_for_map:
        for frag_info in raw_frags_for_map:
            fragments_name_to_idx_map[frag_info['name']] = frag_info['original_index']
    else:
        print(f"Warning: Could not load fragments from {input_frags_dir} to create color map.")

    if os.path.exists(log_file):
        visualization_utils.replay_visualization_log(log_file, fragments_data_map=fragments_name_to_idx_map)
    else:
        print(f"Log file not found: {log_file}")
        print("Run the main script first with --visualize_steps_file <filepath>")