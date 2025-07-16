
USE THIS COMMAND TO RUN -> python -m src.main --visualize_steps_file data/log_auto_test.pkl --num_viz_pairwise 3 --debug_pairwise_matching

# --visualize steps file is not necessary. It should log debug steps into a file. Not tested.

# --num_viz_pairwise shows the Number of top pairwise matches to visualize directly during runtime (0 for none).

# --debug_pairwise_matching displays RANSAC and ICP before and after outputs for every pairwise matching completed. 

# --top_n_matches_per_pair decides the number of top-scoring matches kept for each pair of fragments during pairwise matching step. This is 3 by default. 

SAME COMMAND WITHOUT DEBUG -> python -m src.main --visualize_steps_file data/log_auto_test.pkl --num_viz_pairwise 3 --top_n_matches_per_pair 3

