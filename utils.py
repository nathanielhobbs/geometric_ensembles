from sacrebleu.metrics import BLEU
import numpy as np
import torch
import gensim
import sys, math
from collections import Counter
import gc
import itertools
import pickle
import os


def get_bleu_scores_from_path_list(model_path_list, refs):
    """Compute corpus BLEU for each file in model_path_list against refs."""
    return [get_bleu_scores_from_path(model_path, refs) for model_path in model_path_list]

def get_bleu_scores_from_path(model_path, refs):
    with open(model_path, mode='r', encoding='utf-8') as f:
        hyps = f.read().splitlines()
    return get_corpus_bleu_score(hyps, refs)

def get_refs_from_path(refs_path):
    """Load reference file; returns list-of-lists as expected by sacrebleu."""
    with open(refs_path, mode='r', encoding='utf-8') as f:
        refs = f.read().splitlines()
    return [refs]

def get_corpus_bleu_score(hyps, refs):
    bleu = BLEU(force=True)
    return bleu.corpus_score(hyps, refs)

def get_sentence_bleu_score(hyp, ref):
    bleu = BLEU(force=True, effective_order=True)
    try:
        return bleu.sentence_score(hyp, [ref])
    except Exception:
        return bleu.sentence_score(hyp, ref)

def get_model_hyps(model_path_list):
    """Load hypothesis files; returns list of lists (one list of strings per model)."""
    models = []
    for model_path in model_path_list:
        with open(model_path, 'r', encoding='utf-8') as f:
            hyps = f.read().splitlines()
        models.append(hyps)
    return models

def softmax(x, base=None):
    if not base:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        return np.power(x, base) / np.sum(np.power(x, base), axis=0)

def get_hyps_from_model_index_list(hypothesis_matrix, model_idx_list):
    """Extract one hypothesis per sentence using per-sentence model indices."""
    return [hypothesis_matrix[model_idx_list[i]][i] for i in range(len(model_idx_list))]

def get_score_from_model_index_list(hypothesis_matrix, model_idx_list, references_list, score_type):
    """Compute corpus score for hypothesis selected per sentence by model_idx_list."""
    hypothesis_list = [hypothesis_matrix[model_idx_list[i]][i] for i in range(len(model_idx_list))]
    if score_type == 'BLEU':
        return get_corpus_bleu_score(hypothesis_list, references_list)

def get_basic_word_sub_matrix(models):
    models_to_words_list = [sent.split() for model in models for sent in model]
    word2vec_model = gensim.models.Word2Vec(sentences=models_to_words_list)
    words = word2vec_model.wv.index_to_key
    sub_matrix = {}
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            sub_matrix[(words[i], words[j])] = 1
            sub_matrix[(words[j], words[i])] = 1
    return sub_matrix

def _cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    return dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_word2vec_word_sub_matrix(models, word2vec_model=None, save_path_str=''):
    """Build word-pair cosine similarity substitution matrix using word2vec/fasttext embeddings."""
    import config
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.COMPUTE_TRAINING:
        unique_words_path = f'{config.results_path}/unique_words_training{config.train_half}.pkl'
        sub_matrix_name = f'{config.results_path}/pairwise_edit_word2vec_pretrained_training{config.train_half}_sub_matrix_distances_only_not_scaled.pkl'
    else:
        unique_words_path = f'{config.results_path}/unique_words.pkl'
        sub_matrix_name = f'{config.results_path}/pairwise_edit_word2vec_pretrained_sub_matrix_distances_only_not_scaled.pkl'

    if not os.path.exists(unique_words_path):
        print(f'{unique_words_path} not found. Creating...')
        src_models = get_model_hyps(config.model_training_hypothesis_paths if config.COMPUTE_TRAINING
                                    else config.model_hypothesis_paths)
        models_to_words_list = [sent.split() for model in src_models for sent in model]
        unique_words = get_unique_words(models_to_words_list)
        words = word2vec_model.wv.index_to_key
        missing_words = set(words) - set(unique_words)
        print(f'missing words: {len(missing_words)}/{len(unique_words)} '
              f'({len(missing_words)/len(unique_words):.3f})')
        with open(unique_words_path, 'wb') as f:
            pickle.dump(unique_words, f)
    else:
        with open(unique_words_path, 'rb') as f:
            unique_words = pickle.load(f)

    if not os.path.exists(sub_matrix_name):
        print(f'{sub_matrix_name} not found. Creating...')
        part_counter = 0
        words_done_count = 0
    else:
        with open(sub_matrix_name, 'rb') as f:
            pw_distances = pickle.load(f)
        words_done_count = len(pw_distances)
        part_counter = words_done_count // 1000
        print(f'words done: {words_done_count}, resuming from part {part_counter}')

    if words_done_count != len(unique_words):
        torch_array = torch.Tensor(np.array([word2vec_model.wv[word] for word in unique_words]))
        torch_array.to(device)
        for i in range(words_done_count, len(unique_words)):
            one_vs_all = torch.nn.functional.cosine_similarity(torch_array[i], torch_array)
            if i == 0:
                pw_distances = one_vs_all
            elif i == 1:
                pw_distances = torch.stack((pw_distances, one_vs_all), dim=0)
            else:
                pw_distances = torch.cat((pw_distances, one_vs_all.unsqueeze(0)), dim=0)

            if i % 1000 == 0 and i > 0:
                with open(sub_matrix_name, 'wb') as f:
                    try:
                        pickle.dump(pw_distances, f, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception:
                        p = pickle.Pickler(f)
                        p.fast = True
                        p.dump(pw_distances)
                print(f'saving batch {part_counter}')
                part_counter += 1

        with open(sub_matrix_name, 'wb') as f:
            try:
                pickle.dump(pw_distances, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                p = pickle.Pickler(f)
                p.fast = True
                p.dump(pw_distances)
        print(f'done saving distances, shape {pw_distances.shape}')
        print(f'negative values before scaling: {torch.sum(torch.lt(pw_distances, 0))}')
        pw_distances = torch.divide(torch.add(pw_distances, 1), 2)
        print(f'negative values after scaling: {torch.sum(torch.lt(pw_distances, 0))}')
        with open(sub_matrix_name[:-14] + '_scaled.pkl', 'wb') as f:
            try:
                pickle.dump(pw_distances, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                p = pickle.Pickler(f)
                p.fast = True
                p.dump(pw_distances)

    return {(unique_words[i], unique_words[j]): value
            for (i, j), value in np.ndenumerate(np.triu(pw_distances, k=0))}

def get_unique_words(list_of_lists):
    unique_words = sorted(set(itertools.chain.from_iterable(list_of_lists)))
    return unique_words

def get_word_freq_dict(models):
    word_counts = {}
    for model in models:
        for sent in model:
            for word in sent.split():
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def get_plurality_list_from_dict(my_dict):
    """Return per-position plurality winner across all lists in my_dict."""
    plurality_list = []
    for i in range(len(next(iter(my_dict.values())))):
        word_list = [my_dict[key][i] for key in my_dict]
        word_counts = Counter(word_list)
        plurality_list.append(word_counts.most_common(1)[0][0])
    return plurality_list

def create_length_index_dict(string_list):
    """Map sentence length (word count) to list of indices with that length."""
    length_index_dict = {}
    for index, string in enumerate(string_list):
        length = len(string.split())
        if length not in length_index_dict:
            length_index_dict[length] = []
        length_index_dict[length].append(index)
    return length_index_dict

def get_bucket_indices(length_index_dict, bucket_cutoffs=(25, 50, 75, 100)):
    """Partition sentence indices into length buckets defined by bucket_cutoffs."""
    bucket_indices_dict = {}
    for sent_len, sent_indices in length_index_dict.items():
        for idx in range(len(bucket_cutoffs)):
            if idx not in bucket_indices_dict:
                bucket_indices_dict[idx] = []
            if idx == 0:
                if sent_len <= bucket_cutoffs[idx]:
                    bucket_indices_dict[idx].extend(sent_indices)
            elif idx < len(bucket_cutoffs) - 1:
                if bucket_cutoffs[idx-1] < sent_len <= bucket_cutoffs[idx]:
                    bucket_indices_dict[idx].extend(sent_indices)
            else:
                if bucket_cutoffs[idx-1] < sent_len:
                    bucket_indices_dict[idx].extend(sent_indices)
    return bucket_indices_dict
