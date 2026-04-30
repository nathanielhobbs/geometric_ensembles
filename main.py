import os, sys
import pickle
import h5py

if len(sys.argv) == 2:
    sys.path.append(os.getcwd() + f'/configs/{sys.argv[1]}')
else:
    sys.path.append(os.getcwd() + '/configs/wmt14_en_de')
import config
import utils
import numpy as np
import distances
import measures

print(config.base)
print('config.COMPUTE_TRAINING', config.COMPUTE_TRAINING)
print('config.ENSEMBLE_INCLUDED', config.ENSEMBLE_INCLUDED)

log_folder_path = os.path.dirname(config.log_file_path)
os.makedirs(log_folder_path, exist_ok=True) 
log_file = open(config.log_file_path, 'w')

SHORT_G = {
    'arithmetic mean': 'mean',
    'variance about arithmetic mean': 'variance',
    'arithmetic median': 'median',
    'min': 'min',
    'max': 'max',
}
F_G_PAIRS = [
    ('min', np.argmin),
    ('max', np.argmax),
]
G_PAIRS = [
    ('arithmetic mean', np.mean),
    ('variance about arithmetic mean', np.var),
    ('arithmetic median', np.median),
    ('min', np.min),
    ('max', np.max),
]


def evaluate_distances(name, pairwise_distances, models, refs, log_file, score_type='BLEU'):
    """Evaluate all f(g()) criterion combinations on pairwise_distances.

    Returns {(f_name, g_name): (model_idx_list, score)}.
    """
    results = {}
    print(f'Choosing systems based on {name}')
    log_file.write(f'Choosing systems based on {name}\n')
    for f_name, f in F_G_PAIRS:
        for g_name, g in G_PAIRS:
            idx_list = measures.get_f_of_g_model_idx_list(pairwise_distances, f, g)
            score = utils.get_score_from_model_index_list(models, idx_list, refs, score_type)
            print(f'{f_name} {g_name} {score_type} score is {score}')
            log_file.write(f'{f_name} {g_name} {score_type} score is {score}\n')
            results[(f_name, g_name)] = (idx_list, score)
    return results


def load_sub_matrix(base, results_path, train_half='', training=False):
    """Return (sub_matrix_name, unique_words_dict_path) for the given dataset."""
    if base == 'wmt23_cs_uk':
        suffix = 'train' if training else 'test'
        return (f'{results_path}/pairwise_word_sim_matrix_{suffix}.uk.h5',
                f'{results_path}/unique_words_{suffix}.uk.pkl')
    else:
        suffix = f'_training{train_half}' if training else '_training'
        return (f'{results_path}/pairwise_edit_word2vec_pretrained{suffix}_sub_matrix.pkl',
                f'{results_path}/unique_words_dict{suffix}.pkl')


# ---------------------------------------------------------------------------
# Training split analysis (COMPUTE_TRAINING = True)
# ---------------------------------------------------------------------------
if config.COMPUTE_TRAINING:
    train_refs = utils.get_refs_from_path(config.train_reference_path)
    models_training = utils.get_model_hyps(config.model_training_hypothesis_paths)
    models_training_bleu = utils.get_bleu_scores_from_path_list(config.model_training_hypothesis_paths, train_refs)

    for bleu in models_training_bleu:
        print(bleu)
        log_file.write(str(bleu) + '\n')

    num_train_hyps = len(models_training[0])
    training_pairwise_edit_distances = distances.get_distances(models_training, 'pairwise_edit', f'training{config.train_half}_')
    training_pairwise_bleu_distances = distances.get_distances(models_training, 'pairwise_bleu', f'training{config.train_half}_')
    training_oracle_bleu_distances = distances.get_distances(models_training, 'oracle_bleu', f'training{config.train_half}_')

    log_file.write('Baseline on training systems\n')
    for name, pairwise_distances in [('pairwise edit distance', training_pairwise_edit_distances),
                                      ('pairwise bleu', training_pairwise_bleu_distances)]:
        results = evaluate_distances(name, pairwise_distances, models_training, train_refs, log_file)
        if name == 'pairwise edit distance':
            edit_idx, _ = results[('min', 'arithmetic mean')]
            model_hyps = utils.get_hyps_from_model_index_list(models_training, edit_idx)
            with open(config.results_path + f'/{name}_training_hyps.txt', 'w') as my_f:
                my_f.writelines('\n'.join(model_hyps))

    # Grid search over weighted edit distance insertion/deletion cost parameters
    best_pairs_list = [(i/10, j/10) for i in range(10, 0, -1) for j in range(10, 0, -1)]
    ins_del_score_2d_array = np.zeros((len(best_pairs_list), num_train_hyps))
    ins_del_hyps_dict = {}
    ins_del_corpus_bleu_2d_array = np.zeros((10, 10))

    sub_matrix_name, unique_words_dict_path = load_sub_matrix(
        config.base, config.results_path, config.train_half, training=True)

    sub_matrix = {}
    unique_words_dict = {}
    for pair_idx, (a, b) in enumerate(best_pairs_list):
        extra_name = f"ins_{a}_del_{b}_training{config.train_half}"
        print(extra_name)

        if config.base == 'wmt23_cs_uk':
            if not (os.path.exists(sub_matrix_name) and os.path.exists(unique_words_dict_path)):
                print(f'Substitution matrix not found: {sub_matrix_name}')
                continue
            with h5py.File(sub_matrix_name, 'r') as f1, open(unique_words_dict_path, 'rb') as f2:
                sub_matrix = f1["similarities"]
                unique_words_dict = pickle.load(f2)
                if isinstance(unique_words_dict, list):
                    unique_words_dict = {word: idx for idx, word in enumerate(unique_words_dict)}
                pairwise_word2vec_edit_distances = distances.get_distances(
                    models_training, 'pairwise_edit_word2vec_pretrained', extra_name=extra_name,
                    sub_matrix=sub_matrix, unique_words_dict=unique_words_dict)
        else:
            if os.path.exists(sub_matrix_name) and len(sub_matrix) == 0:
                with open(sub_matrix_name, 'rb') as f:
                    sub_matrix = pickle.load(f)
            if len(sub_matrix) > 0 and len(unique_words_dict) > 0:
                pairwise_word2vec_edit_distances = distances.get_distances(
                    models_training, 'pairwise_edit_word2vec_pretrained', extra_name=extra_name,
                    sub_matrix=sub_matrix, unique_words_dict=unique_words_dict)
            elif len(sub_matrix) > 0:
                pairwise_word2vec_edit_distances = distances.get_distances(
                    models_training, 'pairwise_edit_word2vec_pretrained', extra_name=extra_name,
                    sub_matrix=sub_matrix)
            else:
                pairwise_word2vec_edit_distances = distances.get_distances(
                    models_training, 'pairwise_edit_word2vec_pretrained', extra_name=extra_name)

        results = evaluate_distances(
            f'pairwise edit distance (word2vec pretrained {extra_name})',
            pairwise_word2vec_edit_distances, models_training, train_refs, log_file)
        min_mean_idx, min_mean_score = results[('min', 'arithmetic mean')]

        model_hyps = utils.get_hyps_from_model_index_list(models_training, min_mean_idx)
        ins_del_hyps_dict[pair_idx] = model_hyps
        with open(config.results_path + f'/ins{a}_del{b}_training_hyps.txt', 'w') as my_f:
            my_f.writelines('\n'.join(model_hyps))
        for idx, hyp in enumerate(model_hyps):
            ins_del_score_2d_array[pair_idx][idx] = utils.get_sentence_bleu_score(hyp, train_refs[0][idx]).score
        row, col = int(a * 10), int(b * 10)
        ins_del_corpus_bleu_2d_array[row-1][col-1] = min_mean_score.score

    print(ins_del_corpus_bleu_2d_array)


# ---------------------------------------------------------------------------
# Test split evaluation
# ---------------------------------------------------------------------------
refs = utils.get_refs_from_path(config.reference_path)
length_index_dict = utils.create_length_index_dict(refs[0])
models = utils.get_model_hyps(config.model_hypothesis_paths)
models_bleu = utils.get_bleu_scores_from_path_list(config.model_hypothesis_paths, refs)

for idx, bleu in enumerate(models_bleu):
    if len(sys.argv) == 2:
        print(config.model_hypothesis_paths[idx].split('/')[-1], bleu)
    else:
        print(bleu)
    log_file.write(str(bleu) + '\n')

if config.ensemble_hyps:
    with open(config.ensemble_hyps, 'r') as f:
        ensemble_hyps = f.readlines()
    ensemble_bleu = utils.get_corpus_bleu_score(ensemble_hyps, refs)
    print(f'Ensemble BLEU to beat is: {ensemble_bleu}')
    log_file.write(f'Ensemble BLEU to beat is: {ensemble_bleu}\n')

num_hyps = len(models[0])
num_models = len(models)

# Compute pairwise distance matrices (cached to results_path as .npy files)
pairwise_edit_distances = distances.get_distances(models, 'pairwise_edit')
pairwise_bleu_distances = distances.get_distances(models, 'pairwise_bleu')
oracle_bleu_distances = distances.get_distances(models, 'oracle_bleu')

# Evaluate all f(g()) combinations and write hypothesis files for each
SHORT_DIST = {'pairwise edit distance': 'edit_distance', 'pairwise bleu': 'pw_bleu'}
os.makedirs(config.results_path, exist_ok=True)

for name, pairwise_distances in [('pairwise edit distance', pairwise_edit_distances),
                                   ('pairwise bleu', pairwise_bleu_distances)]:
    short_name = SHORT_DIST[name]
    results = evaluate_distances(name, pairwise_distances, models, refs, log_file)
    for (f_name, g_name), (idx_list, _) in results.items():
        hyps_list = [models[idx_list[i]][i] for i in range(len(idx_list))]
        out_path = f'{config.results_path}/{f_name}.{SHORT_G[g_name]}.{short_name}.hyps'
        with open(out_path, 'w') as fp:
            fp.writelines([f'{line}\n' for line in hyps_list])

# Oracle upper-bound (requires reference; not usable in deployment)
for f_name, f in F_G_PAIRS:
    oracle_idx_list = measures.get_oracle_model_idx_by_criteria(oracle_bleu_distances, f)
    score = utils.get_score_from_model_index_list(models, oracle_idx_list, refs, 'BLEU')
    print(f'{f_name} oracle BLEU score is {score}')
    log_file.write(f'{f_name} oracle BLEU score is {score}\n')
    oracle_hyps = [models[oracle_idx_list[i]][i] for i in range(num_hyps)]
    out_path = f'{config.results_path}/{f_name}.oracle.hyps'
    with open(out_path, 'w') as fp:
        fp.writelines([h + '\n' for h in oracle_hyps])

if config.ENSEMBLE_INCLUDED:
    exit()

# ---------------------------------------------------------------------------
# Weighted edit distance grid search on test set
# Requires precomputed word-pair substitution matrices (not included in repo;
# see README for details on how to generate them).
# ---------------------------------------------------------------------------
best_pairs_list = [(i/10, j/10) for i in range(10, 0, -1) for j in range(10, 0, -1)]
ins_del_score_2d_array = np.zeros((len(best_pairs_list), len(refs[0])))
ins_del_hyps_dict = {}
ins_del_corpus_bleu_2d_array = np.zeros((10, 10))

sub_matrix_name, unique_words_dict_path = load_sub_matrix(config.base, config.results_path)

sub_matrix = {}
unique_words_dict = {}
for pair_idx, (a, b) in enumerate(best_pairs_list):
    extra_name = f"ins_{a}_del_{b}"
    print(extra_name)

    if config.base == 'wmt23_cs_uk':
        if not (os.path.exists(sub_matrix_name) and os.path.exists(unique_words_dict_path)):
            print(f'Substitution matrix not found: {sub_matrix_name}')
            continue
        with h5py.File(sub_matrix_name, 'r') as f1, open(unique_words_dict_path, 'rb') as f2:
            sub_matrix = f1["similarities"]
            unique_words_dict = pickle.load(f2)
            if isinstance(unique_words_dict, list):
                unique_words_dict = {word: idx for idx, word in enumerate(unique_words_dict)}
            pairwise_word2vec_edit_distances = distances.get_distances(
                models, 'pairwise_edit_word2vec_pretrained', extra_name=extra_name,
                sub_matrix=sub_matrix, unique_words_dict=unique_words_dict)
    else:
        if os.path.exists(sub_matrix_name) and len(sub_matrix) == 0:
            with open(sub_matrix_name, 'rb') as f:
                sub_matrix = pickle.load(f)
        if len(sub_matrix) > 0:
            pairwise_word2vec_edit_distances = distances.get_distances(
                models, 'pairwise_edit_word2vec_pretrained', extra_name=extra_name, sub_matrix=sub_matrix)
        else:
            pairwise_word2vec_edit_distances = distances.get_distances(
                models, 'pairwise_edit_word2vec_pretrained', extra_name=extra_name)

    results = evaluate_distances(
        f'pairwise edit distance (word2vec pretrained {extra_name})',
        pairwise_word2vec_edit_distances, models, refs, log_file)
    min_mean_idx, min_mean_score = results[('min', 'arithmetic mean')]

    model_hyps = utils.get_hyps_from_model_index_list(models, min_mean_idx)
    ins_del_hyps_dict[pair_idx] = model_hyps
    with open(config.results_path + f'/ins{a}_del{b}_hyps.txt', 'w') as my_f:
        my_f.writelines('\n'.join(model_hyps))
    for idx, hyp in enumerate(model_hyps):
        ins_del_score_2d_array[pair_idx][idx] = utils.get_sentence_bleu_score(hyp, refs[0][idx]).score
    row, col = int(a * 10), int(b * 10)
    ins_del_corpus_bleu_2d_array[row-1][col-1] = min_mean_score.score

best_of_ins_del_by_idx = [np.argmax(ins_del_score_2d_array[:, i]) for i in range(len(refs[0]))]
best_of_ins_del_hyps = [ins_del_hyps_dict[winning_model_idx][idx]
                        for idx, winning_model_idx in enumerate(best_of_ins_del_by_idx)]
print(f'oracle from best ins/del {utils.get_corpus_bleu_score(best_of_ins_del_hyps, refs)}')
print(ins_del_corpus_bleu_2d_array)
