COMPUTE_TRAINING = False
ENSEMBLE_INCLUDED = True
NUM_PROCESSES = 4

base = "flickr30k"

results_path = "results/flickr30k"
source_path = "data/flickr30k/source.txt"
reference_path = "data/flickr30k/reference.txt"
ensemble_hyps = "data/flickr30k/ensemble.txt"
log_file_path = "logs/flickr30k.log"

model_hypothesis_paths = [f"data/flickr30k/model_{i:02d}.txt" for i in range(1, 26)]

if ENSEMBLE_INCLUDED:
    model_hypothesis_paths.insert(0, ensemble_hyps)
