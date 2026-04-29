import numpy as np
import os
import editdistance
import utils
import edit_distance
import gensim
import pickle
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors, Word2Vec, FastText, fasttext
from gensim.test.utils import datapath
from gensim.scripts.glove2word2vec import glove2word2vec
from transformers import AutoTokenizer, AutoModel
import config

os.makedirs(config.results_path, exist_ok=True)


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def get_distances(models, distance_type, extra_name='', **kwargs):
    """Compute (or load cached) pairwise distance tensor for the given distance_type.

    Returns an ndarray of shape (num_hyps, num_models, num_models) for pairwise types,
    or (num_hyps, num_models) for oracle types.
    """
    save_path = config.results_path + '/{}{}_distance.npy'.format(extra_name, distance_type)
    distance_measurement_dict = {
        'pairwise_edit_bert_pretrained': edit_distance.custom_word_edit_distance,
        'pairwise_edit_word2vec': edit_distance.custom_word_edit_distance,
        'pairwise_edit_word2vec_pretrained': edit_distance.custom_word_edit_distance,
        'pairwise_edit': editdistance.eval,
        'pairwise_edit_own': edit_distance.custom_word_edit_distance,
        'pairwise_edit_avg_len_normalized': editdistance.eval,
        'pairwise_bleu': utils.get_sentence_bleu_score,
        'oracle_bleu': utils.get_sentence_bleu_score,
    }

    print(f'Does {save_path} exist? {os.path.exists(save_path)}')
    if os.path.exists(save_path):
        return np.load(save_path)

    num_hyps = len(models[0])
    num_models = len(models)

    if 'pairwise_edit' in distance_type:
        if distance_type == 'pairwise_edit':
            distance_tensor = np.zeros((num_hyps, num_models, num_models))
            for hyp_num in range(num_hyps):
                for i in range(num_models):
                    for j in range(i, num_models):
                        model_a_hyp = models[i][hyp_num].split()
                        model_b_hyp = models[j][hyp_num].split()
                        distance_tensor[hyp_num][i][j] = distance_measurement_dict[distance_type](model_a_hyp, model_b_hyp)
                        distance_tensor[hyp_num][j][i] = distance_tensor[hyp_num][i][j]

        if distance_type == 'pairwise_edit_own':
            sub_matrix = utils.get_basic_word_sub_matrix(models)
            np.save(f'{config.results_path}/{distance_type}_sub_matrix.npy', sub_matrix)
            insert_cost = 1
            delete_cost = 1

        elif distance_type == 'pairwise_edit_bert_pretrained':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained('bert-base-uncased')
            sentences = [sent for model in models for sent in model]
            encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            output = model(**encoded_input)
            print(output.last_hidden_state.shape)

        elif 'pairwise_edit_word2vec_pretrained' in distance_type and extra_name:
            if 'ins' in extra_name and 'del' in extra_name:
                insert_cost = float(extra_name.split('_')[1])
                delete_cost = float(extra_name.split('_')[3])

            if 'sub_matrix' in kwargs:
                sub_matrix = kwargs['sub_matrix']
                if 'unique_words_dict' in kwargs:
                    unique_words_dict = kwargs['unique_words_dict']
            else:
                if 'training' in extra_name:
                    sub_matrix_name = f'{config.results_path}/{distance_type}_training{config.train_half}_sub_matrix.pkl'
                else:
                    sub_matrix_name = f'{config.results_path}/{distance_type}_sub_matrix.pkl'

                if os.path.exists(sub_matrix_name):
                    with open(sub_matrix_name, 'rb') as f:
                        sub_matrix = pickle.load(f)
                else:
                    if not os.path.exists(f'{config.results_path}/word_vector_model_training{config.train_half}.pkl'):
                        print(f'{config.results_path}/word_vector_model_training{config.train_half}.pkl not found. Creating...')
                        # Use German fasttext for German-source datasets, GloVe English otherwise
                        if 'flickr30k' in save_path or 'wmt' in save_path:
                            model = fasttext.load_facebook_model(datapath("/projects/f_jsvaidya_1/voting/word_vec_stuff/facebook_german_embeddings/cc.de.300.bin"))
                            if 'training' in extra_name:
                                models_to_words_list = [sent.split() for model in models for sent in model]
                            else:
                                models_training = utils.get_model_hyps(config.model_training_hypothesis_paths)
                                models_to_words_list = [sent.split() for model in models + models_training for sent in model]
                            model.min_count = 1
                            model.build_vocab(models_to_words_list, update=True)
                            model.train(models_to_words_list, total_examples=model.corpus_count, epochs=5)
                        else:
                            models_to_words_list = [sent.split() for model in models for sent in model]
                            model = gensim.models.Word2Vec(vector_size=300, min_count=1)
                            model.build_vocab(models_to_words_list)
                            total_examples = model.corpus_count
                            pretrained_path = '/projects/f_jsvaidya_1/voting/word_vec_stuff/glove.6B.300d.txt'
                            glove_file = datapath(pretrained_path)
                            word2vec_glove_file = get_tmpfile("glove.6B.300d.word2vec.txt")
                            glove2word2vec(glove_file, word2vec_glove_file)
                            word2vec_model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
                            model.build_vocab([list(word2vec_model.index_to_key)], update=True)
                            model.wv.vectors_lockf = np.ones(len(model.wv), dtype=float)
                            model.train(models_to_words_list, total_examples=total_examples, epochs=model.epochs)
                            model.wv.intersect_word2vec_format(word2vec_glove_file, binary=False, lockf=1.0)

                        with open(f'{config.results_path}/word_vector_model_training{config.train_half}.pkl', 'wb') as f:
                            pickle.dump(model, f)
                    else:
                        print(f'{config.results_path}/word_vector_model_training{config.train_half}.pkl found. Loading...')
                        with open(f'{config.results_path}/word_vector_model_training{config.train_half}.pkl', 'rb') as f:
                            model = pickle.load(f)

                    sub_matrix = utils.get_word2vec_word_sub_matrix(models, model, sub_matrix_name)

                with open(sub_matrix_name, 'wb') as f:
                    try:
                        pickle.dump(sub_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception:
                        p = pickle.Pickler(f)
                        p.fast = True
                        p.dump(sub_matrix)

            if kwargs:
                if os.path.exists(f'{config.results_path}/word_freq.pkl'):
                    with open(f'{config.results_path}/word_freq.pkl', 'rb') as f:
                        word_freq_dict = pickle.load(f)
                else:
                    word_freq_dict = utils.get_word_freq_dict(models)
                    with open(f'{config.results_path}/word_freq.pkl', 'wb') as f:
                        pickle.dump(word_freq_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

                for key, val in kwargs.items():
                    if key == 'ins_params':
                        ins_params = val
                    elif key == 'del_params':
                        del_params = val
            elif extra_name and not kwargs:
                if 'ins' in extra_name and 'del' in extra_name:
                    insert_cost = float(extra_name.split('_')[1])
                    delete_cost = float(extra_name.split('_')[3])
            elif not extra_name:
                insert_cost = np.mean(list(sub_matrix.values()))
                delete_cost = np.mean(list(sub_matrix.values()))

            distance_tensor = np.zeros((num_hyps, num_models, num_models))
            for hyp_num in range(num_hyps):
                if hyp_num % 1000 == 0:
                    print(f'working on hyp {hyp_num} out of {num_hyps}')
                for i in range(num_models):
                    for j in range(i + 1, num_models):
                        hyp_a, hyp_b = models[i][hyp_num], models[j][hyp_num]
                        if 'unique_words_dict' in kwargs:
                            distance_tensor[hyp_num][i][j] = distance_tensor[hyp_num][j][i] = \
                                distance_measurement_dict[distance_type](
                                    hyp_a, hyp_b, sub_matrix, delete_cost, insert_cost,
                                    unique_words_dict=kwargs['unique_words_dict'])
                        else:
                            distance_tensor[hyp_num][i][j] = distance_tensor[hyp_num][j][i] = \
                                distance_measurement_dict[distance_type](
                                    hyp_a, hyp_b, sub_matrix, delete_cost, insert_cost)

        elif distance_type == 'pairwise_edit_word2vec':
            sub_matrix = utils.get_word2vec_word_sub_matrix(models)
            np.save(f'{config.results_path}/{distance_type}_sub_matrix.npy', sub_matrix)
            insert_cost = np.mean(list(sub_matrix.values()))
            delete_cost = np.mean(list(sub_matrix.values()))

    elif distance_type == 'pairwise_bleu':
        distance_tensor = np.zeros((num_hyps, num_models, num_models))
        for hyp_num in range(num_hyps):
            for i in range(num_models):
                for j in range(i, num_models):
                    model_a_hyp = models[i][hyp_num]
                    model_b_hyp = models[j][hyp_num]
                    distance_tensor[hyp_num][i][j] = distance_measurement_dict[distance_type](model_a_hyp, [model_b_hyp]).score
                    distance_tensor[hyp_num][j][i] = distance_measurement_dict[distance_type](model_b_hyp, [model_a_hyp]).score

    elif distance_type == 'oracle_bleu':
        distance_tensor = np.zeros((num_hyps, num_models))
        if 'training' in extra_name:
            refs = utils.get_refs_from_path(config.train_reference_path)
        else:
            refs = utils.get_refs_from_path(config.reference_path)
        for hyp_num in range(num_hyps):
            for i in range(num_models):
                model_hyp = models[i][hyp_num]
                ref = [refs[0][hyp_num]]
                distance_tensor[hyp_num][i] = distance_measurement_dict[distance_type](model_hyp, ref).score

    np.save(save_path, distance_tensor)
    return distance_tensor
