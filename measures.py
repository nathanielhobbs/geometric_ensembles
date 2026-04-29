import numpy as np


def get_f_of_g_model_idx_list(pairwise_distances, f, g):
    """For each hypothesis, apply g across models then f to select the best model index."""
    return [f(g(pairwise_distances[i], axis=1)) for i in range(len(pairwise_distances))]


def get_oracle_model_idx_by_criteria(oracle_distances, criteria_func):
    """Return the oracle-best model index per hypothesis according to criteria_func."""
    return [criteria_func(oracle_distances[i]) for i in range(len(oracle_distances))]
