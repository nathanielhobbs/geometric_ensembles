COMPUTE_TRAINING = False
ENSEMBLE_INCLUDED = True
NUM_PROCESSES = 4

base = "wmt23_cs_uk"

results_path = "results/wmt23_cs_uk"
source_path = "data/wmt23_cs_uk/source.txt"
reference_path = "data/wmt23_cs_uk/reference.txt"
ensemble_hyps = "data/wmt23_cs_uk/ensemble.txt"
log_file_path = "logs/wmt23_cs_uk.log"

model_hypothesis_paths = [f"data/wmt23_cs_uk/model_{i:02d}.txt" for i in range(1, 26)]

if ENSEMBLE_INCLUDED:
    model_hypothesis_paths.insert(0, ensemble_hyps)
