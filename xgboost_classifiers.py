"""XGBoost mixture-of-experts system selector for geometric ensembles.

Trains on per-sentence distance matrix features to predict which ensemble
method (edit distance, pairwise BLEU, RoBERTa embeddings, or weighted edit
distance) produces the best hypothesis for each source sentence.

Usage:
    python xgboost_classifiers.py [dataset]

    dataset: flickr30k | iwslt14 | wmt14_en_de | wmt23_cs_uk  (default: iwslt14)
"""
import sys, os, itertools
from collections import Counter

import torch
import xgboost as xgb
import numpy as np
import shelve
import utils

# Per-dataset config paths and precomputed distance file names.
DATASET_CONFIG = {
    'iwslt14': {
        'config_path': 'configs/iwslt14',
        'bert_train_file': 'bert_sentence_embeddings_distances_cross-en-de-roberta-sentence-transformer_train2.npy',
        'bert_test_file':  'bert_sentence_embeddings_distances_cross-en-de-roberta-sentence-transformer_test2.npy',
        'wed_train_file':  'ins_0.9_del_0.1_trainingpairwise_edit_word2vec_pretrained_distance.npy',
        'wed_test_file':   'ins_0.9_del_0.1pairwise_edit_word2vec_pretrained_distance.npy',
    },
    'flickr30k': {
        'config_path': 'configs/flickr30k',
        'bert_train_file': 'bert_sentence_embeddings_distances_all-mpnet-base-v2_train.npy',
        'bert_test_file':  'bert_sentence_embeddings_distances_all-mpnet-base-v2_test.npy',
        'wed_train_file':  'ins_0.7_del_0.2_trainingpairwise_edit_word2vec_pretrained_distance.npy',
        'wed_test_file':   'ins_0.7_del_0.2pairwise_edit_word2vec_pretrained_distance.npy',
    },
    'wmt14_en_de': {
        'config_path': 'configs/wmt14_en_de',
        'bert_train_file': 'bert_sentence_embeddings_distances_all-mpnet-base-v2_train2.npy',
        'bert_test_file':  'bert_sentence_embeddings_distances_all-mpnet-base-v2_test2.npy',
        'wed_train_file':  'ins_0.9_del_0.8_training1pairwise_edit_word2vec_pretrained_distance.npy',
        'wed_test_file':   'ins_0.9_del_0.8pairwise_edit_word2vec_pretrained_distance.npy',
    },
    'wmt23_cs_uk': {
        'config_path': 'configs/wmt23_cs_uk',
        'bert_train_file': 'bert_sentence_embeddings_distances_paraphrase-multilingual-MiniLM-L12-v2_train2.npy',
        'bert_test_file':  'bert_sentence_embeddings_distances_paraphrase-multilingual-MiniLM-L12-v2_test2.npy',
        'wed_train_file':  'ins_0.2_del_0.2_trainingpairwise_edit_word2vec_pretrained_distance.npy',
        'wed_test_file':   'ins_0.2_del_0.2pairwise_edit_word2vec_pretrained_distance.npy',
    },
}

dataset = sys.argv[1] if len(sys.argv) > 1 else 'iwslt14'

if dataset not in DATASET_CONFIG:
    print(f"Unknown dataset '{dataset}'. Choose from: {', '.join(DATASET_CONFIG)}")
    sys.exit(1)

num_competitors = [4]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')

file_name = f'{dataset}_classifier_info_{max(num_competitors)}_choices.pkl'
ds_cfg = DATASET_CONFIG[dataset]


def predict(booster, X):
    """Return predicted class indices from raw XGBoost margin output."""
    predt = booster.predict(X, output_margin=True)
    return np.array([np.argmax(prob_list) for prob_list in predt])


# --- Load or compute features ---

create_features_file = not os.path.isfile(f'{file_name}.dat')

if create_features_file:
    sys.path.append(os.path.join(os.getcwd(), ds_cfg['config_path']))
    import config
    import measures
    import distances

    d = shelve.open(file_name)

    train_refs = utils.get_refs_from_path(config.train_reference_path)
    models_training = utils.get_model_hyps(config.model_training_hypothesis_paths)
    models_training_bleu = utils.get_bleu_scores_from_path_list(
        config.model_training_hypothesis_paths, train_refs)
    d['train_refs'] = train_refs
    d['models_training'] = models_training
    d['models_training_bleu'] = models_training_bleu

    # PW BLEU
    train_pairwise_bleu_distances = distances.get_distances(models_training, 'pairwise_bleu', 'training_')
    d['train_pairwise_bleu_distances'] = train_pairwise_bleu_distances
    train_pw_bleu_max_mean_idx_list = measures.get_f_of_g_model_idx_list(
        train_pairwise_bleu_distances, np.argmax, np.mean)
    d['train_pw_bleu_max_mean_idx_list'] = train_pw_bleu_max_mean_idx_list
    train_pw_bleu_max_mean_hyps_list = [
        models_training[model_idx][idx]
        for idx, model_idx in enumerate(train_pw_bleu_max_mean_idx_list)]
    d['train_pw_bleu_max_mean_hyps_list'] = train_pw_bleu_max_mean_hyps_list
    print(f'pairwise bleu (max mean): {utils.get_corpus_bleu_score(train_pw_bleu_max_mean_hyps_list, train_refs)}')

    # Vanilla ED
    train_pairwise_edit_distances = distances.get_distances(models_training, 'pairwise_edit', 'training_')
    train_vanilla_pw_ed_min_mean_idx_list = measures.get_f_of_g_model_idx_list(
        train_pairwise_edit_distances, np.argmin, np.mean)
    train_vanilla_pw_ed_min_mean_hyps_list = [
        models_training[model_idx][idx]
        for idx, model_idx in enumerate(train_vanilla_pw_ed_min_mean_idx_list)]
    print(f'vanilla ed (mean): {utils.get_corpus_bleu_score(train_vanilla_pw_ed_min_mean_hyps_list, train_refs)}')
    d['train_pairwise_edit_distances'] = train_pairwise_edit_distances
    d['train_vanilla_pw_ed_min_mean_idx_list'] = train_vanilla_pw_ed_min_mean_idx_list
    d['train_vanilla_pw_ed_min_mean_hyps_list'] = train_vanilla_pw_ed_min_mean_hyps_list

    # PW RoBERTa
    train_pairwise_bert_distances = np.load(
        os.path.join(config.results_path, ds_cfg['bert_train_file']))
    train_pw_bert_max_mean_idx_list = measures.get_f_of_g_model_idx_list(
        train_pairwise_bert_distances, np.argmax, np.mean)
    train_pw_bert_max_mean_hyps_list = [
        models_training[model_idx][idx]
        for idx, model_idx in enumerate(train_pw_bert_max_mean_idx_list)]
    train_pw_bert_max_mean_score = utils.get_score_from_model_index_list(
        models_training, train_pw_bert_max_mean_idx_list, train_refs, 'BLEU')
    print(f'pairwise RoBERTa (max mean): {train_pw_bert_max_mean_score}')
    d['train_pairwise_bert_distances'] = train_pairwise_bert_distances
    d['train_pw_bert_max_mean_idx_list'] = train_pw_bert_max_mean_idx_list
    d['train_pw_bert_max_mean_hyps_list'] = train_pw_bert_max_mean_hyps_list
    d['train_pw_bert_max_mean_score'] = train_pw_bert_max_mean_score

    # Weighted ED
    train_weighted_ed_pairwise_distances = np.load(
        os.path.join(config.results_path, ds_cfg['wed_train_file']))
    train_weighted_ed_min_mean_idx_list = measures.get_f_of_g_model_idx_list(
        train_weighted_ed_pairwise_distances, np.argmin, np.mean)
    train_weighted_ed_min_mean_hyps_list = [
        models_training[model_idx][idx]
        for idx, model_idx in enumerate(train_weighted_ed_min_mean_idx_list)]
    train_weighted_ed_min_mean_score = utils.get_score_from_model_index_list(
        models_training, train_weighted_ed_min_mean_idx_list, train_refs, 'BLEU')
    print(f'weighted ed (mean): {train_weighted_ed_min_mean_score}')
    d['train_weighted_ed_pairwise_distances'] = train_weighted_ed_pairwise_distances
    d['train_weighted_ed_min_mean_idx_list'] = train_weighted_ed_min_mean_idx_list
    d['train_weighted_ed_min_mean_hyps_list'] = train_weighted_ed_min_mean_hyps_list
    d['train_weighted_ed_min_mean_score'] = train_weighted_ed_min_mean_score

    train_pw_ed_mean_sent_bleus = [
        utils.get_sentence_bleu_score(s, train_refs[0][idx]).score
        for idx, s in enumerate(train_vanilla_pw_ed_min_mean_hyps_list)]
    train_pw_bleu_max_mean_sent_bleus = [
        utils.get_sentence_bleu_score(s, train_refs[0][idx]).score
        for idx, s in enumerate(train_pw_bleu_max_mean_hyps_list)]
    train_pw_bert_max_mean_sent_bleus = [
        utils.get_sentence_bleu_score(s, train_refs[0][idx]).score
        for idx, s in enumerate(train_pw_bert_max_mean_hyps_list)]
    train_weighted_pw_ed_mean_sent_bleus = [
        utils.get_sentence_bleu_score(s, train_refs[0][idx]).score
        for idx, s in enumerate(train_weighted_ed_min_mean_hyps_list)]
    d['train_pw_ed_mean_sent_bleus'] = train_pw_ed_mean_sent_bleus
    d['train_pw_bleu_max_mean_sent_bleus'] = train_pw_bleu_max_mean_sent_bleus
    d['train_pw_bert_max_mean_sent_bleus'] = train_pw_bert_max_mean_sent_bleus
    d['train_weighted_pw_ed_mean_sent_bleus'] = train_weighted_pw_ed_mean_sent_bleus

else:
    d = shelve.open(file_name)
    train_refs = d['train_refs']
    models_training = d['models_training']
    models_training_bleu = d['models_training_bleu']

    train_pairwise_edit_distances = d['train_pairwise_edit_distances']
    train_vanilla_pw_ed_min_mean_idx_list = d['train_vanilla_pw_ed_min_mean_idx_list']
    train_vanilla_pw_ed_min_mean_hyps_list = d['train_vanilla_pw_ed_min_mean_hyps_list']
    print(f'vanilla ed (mean): {utils.get_corpus_bleu_score(train_vanilla_pw_ed_min_mean_hyps_list, train_refs)}')

    train_pairwise_bleu_distances = d['train_pairwise_bleu_distances']
    train_pw_bleu_max_mean_idx_list = d['train_pw_bleu_max_mean_idx_list']
    train_pw_bleu_max_mean_hyps_list = d['train_pw_bleu_max_mean_hyps_list']
    print(f'pairwise bleu (max mean): {utils.get_corpus_bleu_score(train_pw_bleu_max_mean_hyps_list, train_refs)}')

    train_pairwise_bert_distances = d['train_pairwise_bert_distances']
    train_pw_bert_max_mean_idx_list = d['train_pw_bert_max_mean_idx_list']
    train_pw_bert_max_mean_hyps_list = d['train_pw_bert_max_mean_hyps_list']
    train_pw_bert_max_mean_score = d['train_pw_bert_max_mean_score']
    print(f'pairwise bert (max mean): {train_pw_bert_max_mean_score}')

    train_weighted_ed_pairwise_distances = d['train_weighted_ed_pairwise_distances']
    train_weighted_ed_min_mean_idx_list = d['train_weighted_ed_min_mean_idx_list']
    train_weighted_ed_min_mean_hyps_list = d['train_weighted_ed_min_mean_hyps_list']
    train_weighted_ed_min_mean_score = d['train_weighted_ed_min_mean_score']
    print(f'weighted ed (mean): {train_weighted_ed_min_mean_score}')

    train_pw_ed_mean_sent_bleus = d['train_pw_ed_mean_sent_bleus']
    train_pw_bleu_max_mean_sent_bleus = d['train_pw_bleu_max_mean_sent_bleus']
    train_pw_bert_max_mean_sent_bleus = d['train_pw_bert_max_mean_sent_bleus']
    train_weighted_pw_ed_mean_sent_bleus = d['train_weighted_pw_ed_mean_sent_bleus']


# --- Load or compute test features ---

if create_features_file:
    refs = utils.get_refs_from_path(config.reference_path)
    models = utils.get_model_hyps(config.model_hypothesis_paths)
    d['refs'] = refs
    d['models'] = models

    pairwise_edit_distances = distances.get_distances(models, 'pairwise_edit')
    vanilla_pw_ed_min_mean_idx_list = measures.get_f_of_g_model_idx_list(
        pairwise_edit_distances, np.argmin, np.mean)
    vanilla_pw_ed_min_mean_hyps_list = [
        models[model_idx][idx] for idx, model_idx in enumerate(vanilla_pw_ed_min_mean_idx_list)]
    d['pairwise_edit_distances'] = pairwise_edit_distances
    d['vanilla_pw_ed_min_mean_idx_list'] = vanilla_pw_ed_min_mean_idx_list
    d['vanilla_pw_ed_min_mean_hyps_list'] = vanilla_pw_ed_min_mean_hyps_list
    print(f'vanilla ed (min mean): {utils.get_corpus_bleu_score(vanilla_pw_ed_min_mean_hyps_list, refs)}')

    pairwise_bleu_distances = distances.get_distances(models, 'pairwise_bleu')
    pw_bleu_max_mean_idx_list = measures.get_f_of_g_model_idx_list(
        pairwise_bleu_distances, np.argmax, np.mean)
    pw_bleu_max_mean_hyps_list = [
        models[model_idx][idx] for idx, model_idx in enumerate(pw_bleu_max_mean_idx_list)]
    d['pairwise_bleu_distances'] = pairwise_bleu_distances
    d['pw_bleu_max_mean_idx_list'] = pw_bleu_max_mean_idx_list
    d['pw_bleu_max_mean_hyps_list'] = pw_bleu_max_mean_hyps_list
    print(f'pairwise bleu (max mean): {utils.get_corpus_bleu_score(pw_bleu_max_mean_hyps_list, refs)}')

    pairwise_bert_distances = np.load(
        os.path.join(config.results_path, ds_cfg['bert_test_file']))
    pw_bert_max_mean_idx_list = measures.get_f_of_g_model_idx_list(
        pairwise_bert_distances, np.argmax, np.mean)
    pw_bert_max_mean_hyps_list = [
        models[model_idx][idx] for idx, model_idx in enumerate(pw_bert_max_mean_idx_list)]
    pw_bert_max_mean_score = utils.get_score_from_model_index_list(
        models, pw_bert_max_mean_idx_list, refs, 'BLEU')
    print(f'pairwise RoBERTa (mean): {pw_bert_max_mean_score}')
    d['pairwise_bert_distances'] = pairwise_bert_distances
    d['pw_bert_max_mean_idx_list'] = pw_bert_max_mean_idx_list
    d['pw_bert_max_mean_hyps_list'] = pw_bert_max_mean_hyps_list
    d['pw_bert_max_mean_score'] = pw_bert_max_mean_score

    weighted_ed_pairwise_distances = np.load(
        os.path.join(config.results_path, ds_cfg['wed_test_file']))
    weighted_ed_min_mean_idx_list = measures.get_f_of_g_model_idx_list(
        weighted_ed_pairwise_distances, np.argmin, np.mean)
    weighted_ed_min_mean_hyps_list = [
        models[model_idx][idx] for idx, model_idx in enumerate(weighted_ed_min_mean_idx_list)]
    weighted_ed_min_mean_score = utils.get_score_from_model_index_list(
        models, weighted_ed_min_mean_idx_list, refs, 'BLEU')
    d['weighted_ed_pairwise_distances'] = weighted_ed_pairwise_distances
    d['weighted_ed_min_mean_idx_list'] = weighted_ed_min_mean_idx_list
    d['weighted_ed_min_mean_hyps_list'] = weighted_ed_min_mean_hyps_list
    d['weighted_ed_min_mean_score'] = weighted_ed_min_mean_score
    print(f'weighted ed (min mean) {weighted_ed_min_mean_score}')

    pw_ed_mean_sent_bleus = [
        utils.get_sentence_bleu_score(s, refs[0][idx]).score
        for idx, s in enumerate(vanilla_pw_ed_min_mean_hyps_list)]
    pw_bleu_max_mean_sent_bleus = [
        utils.get_sentence_bleu_score(s, refs[0][idx]).score
        for idx, s in enumerate(pw_bleu_max_mean_hyps_list)]
    pw_bert_max_mean_sent_bleus = [
        utils.get_sentence_bleu_score(s, refs[0][idx]).score
        for idx, s in enumerate(pw_bert_max_mean_hyps_list)]
    weighted_pw_ed_mean_sent_bleus = [
        utils.get_sentence_bleu_score(s, refs[0][idx]).score
        for idx, s in enumerate(weighted_ed_min_mean_hyps_list)]
    d['pw_ed_mean_sent_bleus'] = pw_ed_mean_sent_bleus
    d['pw_bleu_max_mean_sent_bleus'] = pw_bleu_max_mean_sent_bleus
    d['pw_bert_max_mean_sent_bleus'] = pw_bert_max_mean_sent_bleus
    d['weighted_pw_ed_mean_sent_bleus'] = weighted_pw_ed_mean_sent_bleus

else:
    refs = d['refs']
    models = d['models']

    pairwise_edit_distances = d['pairwise_edit_distances']
    vanilla_pw_ed_min_mean_idx_list = d['vanilla_pw_ed_min_mean_idx_list']
    vanilla_pw_ed_min_mean_hyps_list = d['vanilla_pw_ed_min_mean_hyps_list']
    print(f'vanilla ed (min mean): {utils.get_corpus_bleu_score(vanilla_pw_ed_min_mean_hyps_list, refs)}')

    pairwise_bleu_distances = d['pairwise_bleu_distances']
    pw_bleu_max_mean_idx_list = d['pw_bleu_max_mean_idx_list']
    pw_bleu_max_mean_hyps_list = d['pw_bleu_max_mean_hyps_list']
    print(f'pairwise bleu (max mean): {utils.get_corpus_bleu_score(pw_bleu_max_mean_hyps_list, refs)}')

    pairwise_bert_distances = d['pairwise_bert_distances']
    pw_bert_max_mean_idx_list = d['pw_bert_max_mean_idx_list']
    pw_bert_max_mean_hyps_list = d['pw_bert_max_mean_hyps_list']
    pw_bert_max_mean_score = d['pw_bert_max_mean_score']
    print(f'pairwise RoBERTa (max mean): {pw_bert_max_mean_score}')

    weighted_ed_pairwise_distances = d['weighted_ed_pairwise_distances']
    weighted_ed_min_mean_idx_list = d['weighted_ed_min_mean_idx_list']
    weighted_ed_min_mean_hyps_list = d['weighted_ed_min_mean_hyps_list']
    weighted_ed_min_mean_score = d['weighted_ed_min_mean_score']
    print(f'weighted ed (min mean) {weighted_ed_min_mean_score}')

    pw_ed_mean_sent_bleus = d['pw_ed_mean_sent_bleus']
    pw_bleu_max_mean_sent_bleus = d['pw_bleu_max_mean_sent_bleus']
    pw_bert_max_mean_sent_bleus = d['pw_bert_max_mean_sent_bleus']
    weighted_pw_ed_mean_sent_bleus = d['weighted_pw_ed_mean_sent_bleus']

d.close()

# Save individual method predictions
with open(f'{dataset}.pairwise_ed.hyps', 'w') as f:
    f.writelines([f'{line}\n' for line in vanilla_pw_ed_min_mean_hyps_list])
with open(f'{dataset}.pairwise_bleu.hyps', 'w') as f:
    f.writelines([f'{line}\n' for line in pw_bleu_max_mean_hyps_list])
with open(f'{dataset}.pairwise_bert.hyps', 'w') as f:
    f.writelines([f'{line}\n' for line in pw_bert_max_mean_hyps_list])
with open(f'{dataset}.pairwise_ed_weighted.hyps', 'w') as f:
    f.writelines([f'{line}\n' for line in weighted_ed_min_mean_hyps_list])

# Flatten distance matrices into feature vectors
train_arr1_flat = train_pairwise_edit_distances.reshape(train_pairwise_edit_distances.shape[0], -1)
train_arr2_flat = train_pairwise_bleu_distances.reshape(train_pairwise_edit_distances.shape[0], -1)
train_arr3_flat = train_pairwise_bert_distances.reshape(train_pairwise_edit_distances.shape[0], -1)
train_arr4_flat = train_weighted_ed_pairwise_distances.reshape(train_pairwise_edit_distances.shape[0], -1)

arr1_flat = pairwise_edit_distances.reshape(pairwise_edit_distances.shape[0], -1)
arr2_flat = pairwise_bleu_distances.reshape(pairwise_edit_distances.shape[0], -1)
arr3_flat = pairwise_bert_distances.reshape(pairwise_edit_distances.shape[0], -1)
arr4_flat = weighted_ed_pairwise_distances.reshape(pairwise_edit_distances.shape[0], -1)

score_to_beat = max(
    utils.get_corpus_bleu_score(vanilla_pw_ed_min_mean_hyps_list, refs).score,
    utils.get_corpus_bleu_score(pw_bleu_max_mean_hyps_list, refs).score,
    utils.get_corpus_bleu_score(pw_bert_max_mean_hyps_list, refs).score,
    utils.get_corpus_bleu_score(weighted_ed_min_mean_hyps_list, refs).score)

train_features_dict = {'ed': train_arr1_flat, 'pw_bleu': train_arr2_flat,
                       'pw_bert': train_arr3_flat, 'w_ed': train_arr4_flat}
test_features_dict  = {'ed': arr1_flat, 'pw_bleu': arr2_flat,
                       'pw_bert': arr3_flat, 'w_ed': arr4_flat}
train_sent_bleu_dict = {'ed': train_pw_ed_mean_sent_bleus, 'pw_bleu': train_pw_bleu_max_mean_sent_bleus,
                        'pw_bert': train_pw_bert_max_mean_sent_bleus, 'w_ed': train_weighted_pw_ed_mean_sent_bleus}
sent_bleu_dict = {'ed': pw_ed_mean_sent_bleus, 'pw_bleu': pw_bleu_max_mean_sent_bleus,
                  'pw_bert': pw_bert_max_mean_sent_bleus, 'w_ed': weighted_pw_ed_mean_sent_bleus}
hyps_dict = {'ed': vanilla_pw_ed_min_mean_hyps_list, 'pw_bleu': pw_bleu_max_mean_hyps_list,
             'pw_bert': pw_bert_max_mean_hyps_list, 'w_ed': weighted_ed_min_mean_hyps_list}


def make_inverse_weighted_dtrain(combined_array, train_labels):
    freq = Counter(train_labels)
    n = len(train_labels)
    weights = [1 - (freq[label] / n) for label in train_labels]
    return xgb.DMatrix(data=combined_array, label=np.array(train_labels), weight=weights)


def assign_labels_4way(sent_bleu_dict_local, keys, n):
    """Assign 0–3 class label based on which of the four methods scores best."""
    one, two, three, four = keys
    labels = []
    for idx in range(n):
        if (sent_bleu_dict_local[four][idx] > sent_bleu_dict_local[three][idx] and
                sent_bleu_dict_local[four][idx] > sent_bleu_dict_local[two][idx] and
                sent_bleu_dict_local[four][idx] > sent_bleu_dict_local[one][idx]):
            labels.append(3)
        elif (sent_bleu_dict_local[three][idx] > sent_bleu_dict_local[two][idx] and
              sent_bleu_dict_local[three][idx] > sent_bleu_dict_local[one][idx] and
              sent_bleu_dict_local[three][idx] >= sent_bleu_dict_local[four][idx]):
            labels.append(2)
        elif (sent_bleu_dict_local[two][idx] > sent_bleu_dict_local[one][idx] and
              sent_bleu_dict_local[two][idx] >= sent_bleu_dict_local[three][idx] and
              sent_bleu_dict_local[two][idx] >= sent_bleu_dict_local[four][idx]):
            labels.append(1)
        else:
            labels.append(0)
    return labels


# --- 4-competitor classifier ---

if 4 in num_competitors:
    best4_dict = {'models': '', 'score': -1, 'weight_type': ''}
    combined_array = np.hstack((train_arr1_flat, train_arr2_flat, train_arr3_flat, train_arr4_flat))
    test_combined_array = np.hstack((arr1_flat, arr2_flat, arr3_flat, arr4_flat))

    for one, two, three, four in itertools.permutations(['ed', 'pw_bleu', 'pw_bert', 'w_ed'], 4):
        train_labels = assign_labels_4way(train_sent_bleu_dict, (one, two, three, four),
                                          len(train_pw_bleu_max_mean_sent_bleus))
        test_labels = assign_labels_4way(sent_bleu_dict, (one, two, three, four),
                                         len(pw_bleu_max_mean_sent_bleus))

        freq = Counter(train_labels)
        print(f'train label counts: {freq}')

        dtrain = make_inverse_weighted_dtrain(combined_array, train_labels)
        dtest = xgb.DMatrix(data=test_combined_array, label=np.array(test_labels))

        booster = xgb.train(
            {'num_class': 4, 'objective': 'multi:softmax', 'eval_metric': 'merror'},
            dtrain,
            num_boost_round=5,
        )

        y_pred = predict(booster, dtest)
        pred_hyps = [hyps_dict[one][idx] if c == 0 else
                     hyps_dict[two][idx] if c == 1 else
                     hyps_dict[three][idx] if c == 2 else
                     hyps_dict[four][idx]
                     for idx, c in enumerate(y_pred)]

        bleu_score = utils.get_corpus_bleu_score(pred_hyps, refs).score
        oracle_hyps = [hyps_dict[one][idx] if c == 0 else
                       hyps_dict[two][idx] if c == 1 else
                       hyps_dict[three][idx] if c == 2 else
                       hyps_dict[four][idx]
                       for idx, c in enumerate(test_labels)]
        oracle_bleu_score = utils.get_corpus_bleu_score(oracle_hyps, refs).score
        print(f'{one}>{two}>{three}>{four} BLEU: {bleu_score:.2f} '
              f'(+{bleu_score - score_to_beat:.3f}, oracle {oracle_bleu_score:.2f})')

        if bleu_score > best4_dict['score']:
            best4_dict['models'] = f'{one}>{two}>{three}>{four}'
            best4_dict['score'] = bleu_score
            with open(f'{dataset}.xgboost.hyps', 'w') as f:
                f.writelines([f'{line}\n' for line in pred_hyps])

# --- 3-competitor classifier ---

if 3 in num_competitors:
    best3_dict = {'models': '', 'score': -1, 'weight_type': '', 'learning_rate': None,
                  'num_boost_round': None, 'max_depth': None, 'subsample': None, 'gamma': None}

    for one, two, three in [('w_ed', 'ed', 'pw_bleu')]:
        combined_array = np.hstack((train_features_dict[one], train_features_dict[two], train_features_dict[three]))
        test_combined_array = np.hstack((test_features_dict[one], test_features_dict[two], test_features_dict[three]))

        train_labels = []
        for idx in range(len(train_pw_bleu_max_mean_sent_bleus)):
            if (train_sent_bleu_dict[three][idx] > train_sent_bleu_dict[two][idx] and
                    train_sent_bleu_dict[three][idx] > train_sent_bleu_dict[one][idx]):
                train_labels.append(2)
            elif (train_sent_bleu_dict[two][idx] > train_sent_bleu_dict[one][idx] and
                  train_sent_bleu_dict[two][idx] >= train_sent_bleu_dict[three][idx]):
                train_labels.append(1)
            else:
                train_labels.append(0)

        test_labels = []
        for idx in range(len(pw_bleu_max_mean_sent_bleus)):
            if (sent_bleu_dict[three][idx] > sent_bleu_dict[two][idx] and
                    sent_bleu_dict[three][idx] > sent_bleu_dict[one][idx]):
                test_labels.append(2)
            elif (sent_bleu_dict[two][idx] > sent_bleu_dict[one][idx] and
                  sent_bleu_dict[two][idx] >= sent_bleu_dict[three][idx]):
                test_labels.append(1)
            else:
                test_labels.append(0)

        oracle_hyps = [hyps_dict[one][idx] if c == 0 else
                       hyps_dict[two][idx] if c == 1 else
                       hyps_dict[three][idx]
                       for idx, c in enumerate(test_labels)]
        oracle_bleu_score = utils.get_corpus_bleu_score(oracle_hyps, refs).score
        dtest = xgb.DMatrix(data=test_combined_array, label=np.array(test_labels))

        freq = Counter(train_labels)
        print(f'train label counts: {freq}')

        for learning_rate, num_boost_round, max_depth, subsample, gamma in itertools.product(
                [0.3], [50], [6], [0.8], [0]):
            print(f'lr={learning_rate}, rounds={num_boost_round}, depth={max_depth}, '
                  f'subsample={subsample}, gamma={gamma}')

            params = {
                'objective': 'multi:softmax',
                'num_class': 3,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'subsample': subsample,
                'gamma': gamma,
                'eval_metric': 'merror',
                'device': device,
                'seed': 42,
            }

            dtrain = make_inverse_weighted_dtrain(combined_array, train_labels)
            booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)

            y_pred = predict(booster, dtest)
            pred_hyps = [hyps_dict[one][idx] if c == 0 else
                         hyps_dict[two][idx] if c == 1 else
                         hyps_dict[three][idx]
                         for idx, c in enumerate(y_pred)]

            bleu_score = utils.get_corpus_bleu_score(pred_hyps, refs).score
            print(f'{one}>{two}>{three} BLEU: {bleu_score:.2f} '
                  f'(+{bleu_score - score_to_beat:.3f}, oracle {oracle_bleu_score:.2f})')

            if bleu_score > best3_dict['score']:
                best3_dict.update({'models': f'{one}>{two}>{three}', 'score': bleu_score,
                                   'learning_rate': learning_rate, 'num_boost_round': num_boost_round,
                                   'max_depth': max_depth, 'subsample': subsample, 'gamma': gamma})
                with open(f'{dataset}.xgboost.hyps', 'w') as f:
                    f.writelines([f'{line}\n' for line in pred_hyps])

# --- 2-competitor classifier ---

if 2 in num_competitors:
    best2_dict = {'models': '', 'score': -1}
    combined_array = np.hstack((train_arr1_flat, train_arr2_flat))
    test_combined_array = np.hstack((arr1_flat, arr2_flat))

    train_labels = [0 if train_pw_ed_mean_sent_bleus[idx] > train_pw_bleu_max_mean_sent_bleus[idx] else 1
                    for idx in range(len(train_pw_bleu_max_mean_sent_bleus))]
    test_labels = [0 if pw_ed_mean_sent_bleus[idx] > pw_bleu_max_mean_sent_bleus[idx] else 1
                   for idx in range(len(pw_bleu_max_mean_sent_bleus))]

    freq = Counter(train_labels)
    print(f'train label counts: {freq}')

    dtrain = make_inverse_weighted_dtrain(combined_array, train_labels)
    dtest = xgb.DMatrix(data=test_combined_array, label=np.array(test_labels))

    booster = xgb.train(
        {'num_class': 2, 'objective': 'multi:softmax', 'eval_metric': 'merror'},
        dtrain,
        num_boost_round=5,
    )

    y_pred = predict(booster, dtest)
    pred_hyps = [vanilla_pw_ed_min_mean_hyps_list[idx] if c == 0 else
                 pw_bleu_max_mean_hyps_list[idx]
                 for idx, c in enumerate(y_pred)]
    bleu_score = utils.get_corpus_bleu_score(pred_hyps, refs).score
    print(f'vanilla ed vs pw bleu BLEU: {bleu_score:.2f} (+{bleu_score - score_to_beat:.2f})')
    if bleu_score > best2_dict['score']:
        best2_dict = {'models': 'vanilla ed vs pw bleu', 'score': bleu_score}
        with open(f'{dataset}.xgboost.hyps', 'w') as f:
            f.writelines([f'{line}\n' for line in pred_hyps])

# --- Summary ---

if 2 in num_competitors:
    print(f"Best 2-system: {best2_dict['models']} BLEU {best2_dict['score']:.2f} "
          f"(+{best2_dict['score'] - score_to_beat:.2f})")
if 3 in num_competitors:
    print(f"Best 3-system: {best3_dict['models']} BLEU {best3_dict['score']:.2f} "
          f"(+{best3_dict['score'] - score_to_beat:.2f})")
    print(best3_dict)
if 4 in num_competitors:
    print(f"Best 4-system: {best4_dict['models']} BLEU {best4_dict['score']:.2f} "
          f"(+{best4_dict['score'] - score_to_beat:.2f})")
