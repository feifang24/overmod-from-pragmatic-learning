import contextlib
import random
from collections import defaultdict
import copy
import time
import os
import json
from collections import namedtuple
import itertools
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import models
import vision
import util
from data import ShapeWorld
import data
import eval_utils
from shapeworld import SHAPES, COLORS, new_color, new_shape


def load_data(dataset):
    # keys: 'imgs', 'labels', 'langs'
    raw_data = data.load_raw_data(dataset)
    num_examples = raw_data['imgs'].shape[0]
    preprocessed_data = data.ShapeWorld(raw_data, vocab)
    dataloader = torch.utils.data.DataLoader(preprocessed_data,
                                             batch_size=num_examples,
                                             shuffle=False)

    img, y, lang = list(dataloader)[0]  # first batch of a total of 1 batch
    return img, y, lang, preprocessed_data.to_idx


def construct_utterances_and_labels(lang, text2idx_fn):
    # Types of utterances:
    # correct color + correct shape: `lang`
    # correct color (+ the word "shape"): `color_lang`
    # correct shape (no color): `shape_lang`
    # wrong color (+ the word "shape")
    # wrong shape (no color)
    # wrong color + correct shape
    # correct color + wrong shape

    num_examples, seq_len = lang.shape
    num_correct_utterance_types = 3
    num_incorrect_utterance_types = 4

    # generate incorrect utterances
    target_configs = get_target_configs(lang)

    def construct_correct_utterances(tgt_configs, feature: str):
        assert feature in ['color', 'shape', 'both']
        if feature == 'both': return [list(config) for config in tgt_configs]
        correct_configs = []
        for config in tgt_configs:
            color, shape = config
            correct_configs.append([color, 'shape'] if feature ==
                                   'color' else [shape])
        return correct_configs

    def construct_incorrect_utterances(tgt_configs, feature: str,
                                       mentions_the_other_feature: bool):
        assert feature in ['color', 'shape']
        incorrect_configs = []
        for config in tgt_configs:
            color, shape = config
            if feature == 'color':
                incorrect_feature = new_color(color)
            else:
                incorrect_feature = new_shape(shape)
            if mentions_the_other_feature:
                if feature == 'color':
                    incorrect_config = [incorrect_feature, shape]
                else:
                    incorrect_config = [color, incorrect_feature]
            else:
                if feature == 'color':
                    incorrect_config = [incorrect_feature, 'shape']
                else:
                    incorrect_config = [incorrect_feature]
            incorrect_configs.append(incorrect_config)
        return incorrect_configs

    all_raw_utterances = []
    correct_utterances = torch.zeros(
        (num_examples, num_correct_utterance_types, seq_len), dtype=lang.dtype)
    correct_utterances = correct_utterances + vocab['w2i'][
        '<PAD>']  # initialize all tokens with pad tokens
    for i, feature in enumerate(['both', 'color', 'shape']):
        curr_raw_utterances = construct_correct_utterances(target_configs,
                                                           feature=feature)
        curr_tensor = torch.from_numpy(
            text2idx_fn(np.asarray(curr_raw_utterances))[0])
        correct_utterances[:, i, :curr_tensor.shape[-1]] = curr_tensor
        all_raw_utterances.append(curr_raw_utterances)

    incorrect_utterances = torch.zeros(
        (num_examples, num_incorrect_utterance_types, seq_len),
        dtype=lang.dtype)
    incorrect_utterances = incorrect_utterances + vocab['w2i'][
        '<PAD>']  # initialize all tokens with pad tokens

    for i, (feature, mention) in enumerate(
            itertools.product(['color', 'shape'], [True, False])):
        curr_raw_utterances = construct_incorrect_utterances(
            target_configs,
            feature=feature,
            mentions_the_other_feature=mention)
        curr_tensor = torch.from_numpy(
            text2idx_fn(np.asarray(curr_raw_utterances))[0])
        incorrect_utterances[:, i, :curr_tensor.shape[-1]] = curr_tensor
        all_raw_utterances.append(curr_raw_utterances)

    all_utterances = torch.cat([correct_utterances, incorrect_utterances],
                               1)  # (bsz, num_utterance_types, seq_len)
    all_raw_utterances = np.asarray(all_raw_utterances).T.tolist()
    all_labels = np.concatenate([
        np.ones((num_examples, num_correct_utterance_types)),
        np.zeros((num_examples, num_incorrect_utterance_types))
    ],
                                axis=1).tolist()
    return all_utterances, all_raw_utterances, all_labels, target_configs


def get_target_configs(lang):
    lang_np = lang.cpu().numpy()
    combos = [
        tuple(
            eval_utils.decode_utterance(lang_per_example,
                                        include_special_tokens=False))
        for lang_per_example in lang_np
    ]
    return combos


def get_semantics(listener, imgs, utterances, seq_lengths):
    '''
    listener: torch model checkpoint
    imgs: torch tensor of shape (bsz, num_channels, width, height)
    utterances: torch tensor of shape (bsz, num_utterances_per_object, num_tokens)
    '''
    num_objects, num_utterances_per_object, num_tokens = utterances.shape
    with torch.no_grad():
        feats_emb = listener.embed_features(imgs.unsqueeze(0)).squeeze()
        lang_emb = listener.lang_model(
            F.one_hot(utterances.reshape((-1, num_tokens)),
                      num_classes=len(vocab['w2i'].keys())).float(),
            seq_lengths.flatten())
        lang_bilinear = listener.bilinear(lang_emb).reshape(
            (num_objects, num_utterances_per_object, -1))
        scores = torch.sigmoid(
            torch.einsum('ijh,ih->ij', (lang_bilinear, feats_emb)))
    return scores.cpu().numpy()


def eval_listener_semantics_on_all_images(listener, imgs, utterances,
                                          raw_utterances, labels,
                                          referent_configs):
    '''
    imgs: torch tensor of shape (bsz, num_referents_per_example, num_channels, width, height)
    '''
    results = []
    imgs = imgs.float()
    seq_lengths = torch.tensor([
        np.count_nonzero(t)
        for t in utterances.reshape(-1, utterances.shape[-1])
    ],
                               dtype=np.int).reshape(utterances.shape[:-1])
    if torch.cuda.is_available():
        imgs = imgs.cuda()
        utterances = utterances.cuda()
        seq_lengths = seq_lengths.cuda()
    semantics = get_semantics(listener, imgs, utterances,
                              seq_lengths)  # (bsz, num_utterances_per_object)

    num_examples = semantics.shape[0]

    for i, (color, shape) in enumerate(referent_configs):
        curr_semantics = semantics[i].tolist()
        curr_raw_utterances = raw_utterances[i]
        curr_labels = labels[i]
        curr_results = {'color': color, 'shape': shape, 'utterances': []}
        for sem, utterance, label in zip(curr_semantics, curr_raw_utterances,
                                         curr_labels):
            curr_results['utterances'].append({
                'utterance': ' '.join(utterance),
                'label': label,
                'sem': sem,
            })
        results.append(curr_results)
    return results


if __name__ == '__main__':

    parser = ArgumentParser(description='Eval',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--eval_dataset',
        help=
        'Name of dataset that the models are to be evaluated on (must be a subdir of data/)'
    )
    parser.add_argument(
        '--train_dataset',
        default=None,
        help=
        'Name of dataset that the models are trained on (must be a subdir of data/)'
    )

    args = parser.parse_args()
    if args.train_dataset is None:
        args.train_dataset = args.eval_dataset

    ROOT_DIR = './models/shapeworld'

    MODEL_DIR = os.path.join(ROOT_DIR, args.train_dataset, 'literal_listeners')

    # Vocab
    vocab = torch.load(os.path.join(ROOT_DIR, 'vocab.pt'))
    print(vocab)

    # ===== Specify eval data and where to store eval outputs =====

    DATA_DIR = './data'
    OUTPUT_DIR = os.path.join('./eval_results', args.train_dataset,
                              'l0-semantics')
    if args.eval_dataset != args.train_dataset:
        OUTPUT_DIR = OUTPUT_DIR + f'_eval={args.eval_dataset}'
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    context = 'both-needed'
    eval_dataset = os.path.join(
        DATA_DIR, f'{args.eval_dataset}-{context}/reference-1000-eval.npz')

    # ===== Load data and preprocess =====
    img, y, lang, encode_fn = load_data(eval_dataset)
    target_imgs = img[:, 0]
    all_utterances, all_raw_utterances, all_labels, target_configs = construct_utterances_and_labels(
        lang, encode_fn)

    # ===== Specify models =====
    listener_idx_to_exclude = [2, 10,
                               11]  # val and test listeners; see outputs.py
    literal_listener_paths = [
        os.listdir(os.path.join(MODEL_DIR, trial))
        for trial in sorted(os.listdir(MODEL_DIR))
    ]

    # ===== Run eval =====
    for trial, model_paths in enumerate(literal_listener_paths):
        for model_path in model_paths:
            listener_name = model_path.split('.')[0]
            listener_idx = listener_name.split('_')[-1]
            if int(listener_idx) in listener_idx_to_exclude: continue
            print(f'Evaluating semantic outputs from {listener_name} ...')
            listener = torch.load(
                os.path.join(MODEL_DIR, str(trial), model_path),
                map_location=torch.device('cuda')
                if torch.cuda.is_available() else torch.device('cpu'))
            listener.eval()
            sem_outputs = eval_listener_semantics_on_all_images(
                listener, target_imgs, all_utterances, all_raw_utterances,
                all_labels, target_configs)
            output_fp = os.path.join(OUTPUT_DIR,
                                     f'{trial}-{listener_name}.json')
            with open(output_fp, 'w') as output_file:
                json.dump(sem_outputs, output_file, indent=2)
