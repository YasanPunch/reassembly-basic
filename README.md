
USE THIS COMMAND TO EXECUTE -> python -m src.main --visualize_steps_file data/log_auto_test.pkl --num_viz_pairwise 3 --debug_pairwise_matching

 --num_viz_pairwise shows the Number of top pairwise matches to visualize directly during runtime (0 for none).

 --debug_pairwise_matching displays outputs before and after RANSAC and ICP for every pairwise matching completed. 

 --top_n_matches_per_pair decides the number of top-scoring matches kept for each pair of fragments during pairwise matching step. This is 3 by default. 

  --visualize steps file is not necessary. It should log debug steps into a file. Not tested.

SAME COMMAND WITHOUT DEBUG -> python -m src.main --visualize_steps_file data/log_auto_test.pkl --num_viz_pairwise 3 --top_n_matches_per_pair 3

NOTE

Overlap check in global reassembly is too strict currently. It is turned off in configuration parameters. 
Debug visualization can also be toggled on/off from config params. 

