COMPUTE_TRAINING = False
ENSEMBLE_INCLUDED = True
NUM_PROCESSES = 4

base = "wmt14_en_de"

results_path = "results/wmt14_en_de"
source_path = "data/wmt14_en_de/source.txt"
reference_path = "data/wmt14_en_de/reference.txt"
ensemble_hyps = "data/wmt14_en_de/ensemble.txt"
log_file_path = "logs/wmt14_en_de.log"

model_hypothesis_paths = [f"data/wmt14_en_de/model_{i:02d}.txt" for i in range(1, 26)]

if ENSEMBLE_INCLUDED:
    model_hypothesis_paths.insert(0, ensemble_hyps)
