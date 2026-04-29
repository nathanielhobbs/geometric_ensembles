COMPUTE_TRAINING = False
ENSEMBLE_INCLUDED = True
NUM_PROCESSES = 4

base = "iwslt14"

results_path = "results/iwslt14"
source_path = "data/iwslt14/source.txt"
reference_path = "data/iwslt14/reference.txt"
ensemble_hyps = "data/iwslt14/ensemble.txt"
log_file_path = "logs/iwslt14.log"

model_hypothesis_paths = [f"data/iwslt14/model_{i:02d}.txt" for i in range(1, 26)]

if ENSEMBLE_INCLUDED:
    model_hypothesis_paths.insert(0, ensemble_hyps)
