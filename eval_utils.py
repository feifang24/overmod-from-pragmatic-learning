import numpy as np
import os
import torch
import json
import pandas as pd
import random
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Union
from IPython.core.display import display

cm = sns.light_palette("green", as_cmap=True)

from shapeworld import COLORS, SHAPES, SpecificationType

ROOT_DIR = '/home/ubuntu/prag-feature-distribution' if torch.cuda.is_available(
) else '/Users/feifang/Desktop/Dev/prag-feature-distribution'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'eval_results')
VOCAB_PATH = os.path.join(ROOT_DIR, 'models', 'shapeworld', 'vocab.pt')

# eval_dataset, train_dataset

CONTEXTS = ['both-needed', 'either-okay', 'shape-needed', 'color-needed']

LISTENER_TYPES = ['train', 'val', 'test']

UTTERANCE_TYPES = ['both', 'color-only', 'shape-only']
BINARY_UTTERANCE_TYPES = ['has-color', 'has-shape']

vocab = torch.load(VOCAB_PATH)


def visualize_stats(stats, index=None, columns=None, title=''):
    df = pd.DataFrame(stats)
    if index is not None:
        df.index = index
    if columns is not None:
        df.columns = columns
    styler = df.style.format('{:.3f}').set_caption(title).background_gradient(
        cmap=cm, low=0.95, high=0.05)
    display(styler)


def visualize_binary_utterance_distribution(dist):
    df = pd.DataFrame(dist, index=BINARY_UTTERANCE_TYPES, columns=CONTEXTS)
    styler = df.style.background_gradient(cmap=cm, low=0.95, high=0.05, axis=1)
    display(styler)


def visualize_utterance_distribution(dist_per_speaker: List[np.ndarray],
                                     return_plot=True,
                                     ylim=(0, 1.1),
                                     ylabel='Proportion',
                                     figsize=(12, 6),
                                     **kwargs):
    num_examples_by_context = dist_per_speaker[0].sum(axis=0)
    num_examples_by_context = {
        context: num_examples_by_context[i]
        for i, context in enumerate(CONTEXTS)
    }

    perc_per_speaker = np.stack(
        [dist / dist.sum(axis=0) for dist in dist_per_speaker])
    aggregate_dist = perc_per_speaker.mean(axis=0)
    aggregate_std = perc_per_speaker.std(axis=0)

    aggregate_dist_df = pd.DataFrame(aggregate_dist,
                                     index=UTTERANCE_TYPES,
                                     columns=CONTEXTS).T
    aggregate_std_df = pd.DataFrame(aggregate_std,
                                    index=UTTERANCE_TYPES,
                                    columns=CONTEXTS).T

    ax = aggregate_dist_df.plot.bar(yerr=aggregate_std,
                                    rot=0,
                                    ylim=ylim,
                                    ylabel=ylabel,
                                    yticks=[0.2 * i for i in range(6)],
                                    figsize=figsize)

    # annotate height
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 9),
                    textcoords='offset points')

    # edit xticklabels to show total num examples
    xticklabels = ax.get_xticklabels()
    new_xticklabels = []
    for label in xticklabels:
        context = label.get_text()
        label.set_text(f'{context}\n$n$={num_examples_by_context[context]}')
        new_xticklabels.append(label)
    ax.set_xticklabels(new_xticklabels)

    if return_plot:
        return ax
    return aggregate_dist_df, aggregate_std_df


def visualize_stats_by_combo(stats, title=''):
    visualize_stats(stats, index=COLORS, columns=SHAPES, title=title)


def visualize_stats_by_config(stats):
    for specification_type in SpecificationType:
        data_slice = stats[specification_type.value]
        visualize_stats_by_combo(data_slice, title=specification_type.name)


def decode_utterance(utterance: np.ndarray, include_special_tokens=True):
    seq = []
    for word_index in utterance:
        if word_index < 3 and not include_special_tokens:
            continue
        seq.append(vocab['i2w'].get(word_index, '<UNK>'))
    return seq


def load_data(eval_ds, context):
    ds = '-'.join([eval_ds, context])
    data_dir = os.path.join(DATA_DIR, ds)
    return np.load(os.path.join(data_dir, 'reference-1000-eval.npz'),
                   allow_pickle=True)


def load_eval_results(eval_dataset,
                      model_type,
                      train_dataset=None,
                      listener_types=LISTENER_TYPES):
    def load_referents(eval_ds, context):
        return load_data(eval_ds, context)['all_referents']

    if train_dataset is None: train_dataset = eval_dataset
    eval_datasets = {c: '-'.join([eval_dataset, c]) for c in CONTEXTS}
    eval_dirs_per_model_trial = os.path.join(RESULTS_DIR, train_dataset,
                                             model_type)

    def load_eval_results_for_trial(trial_results_dir):
        eval_dirs = {
            context: os.path.join(trial_results_dir, eval_ds)
            for context, eval_ds in eval_datasets.items()
        }

        stats = {}
        stats['gt_lang'] = {}
        stats['all_referents'] = {}
        for context in CONTEXTS:
            stats['all_referents'][context] = load_referents(
                eval_dataset, context)

        for listener_type in listener_types:
            stats[listener_type] = defaultdict(dict)
            for context in CONTEXTS:
                context_dir = eval_dirs[context]
                listener_dir = os.path.join(context_dir, listener_type)
                if listener_type == 'test' or (model_type == 'rsa_ensemble'
                                               and listener_type != 'val'):
                    listener_ids = sorted(
                        set([
                            int(fn.split('_')[0])
                            for fn in os.listdir(listener_dir)
                        ]))
                    for listener_id in listener_ids:
                        stats[listener_type][listener_id][context] = {}
                        for fn in os.listdir(listener_dir):
                            if fn.startswith(str(listener_id)):
                                key = '_'.join(fn.split('.')[0].split('_')[1:])
                                val = np.load(os.path.join(listener_dir, fn),
                                              allow_pickle=True)
                                if key == 'gt_lang':
                                    stats[key][context] = val
                                else:
                                    stats[listener_type][listener_id][context][
                                        key] = val
                else:
                    for fn in os.listdir(listener_dir):
                        key = fn.split('.')[0]
                        val = np.load(os.path.join(listener_dir, fn),
                                      allow_pickle=True)
                        if key == 'gt_lang':
                            stats[key][context] = val
                        else:
                            stats[listener_type][context][key] = val
        return stats

    results = []
    for trial_idx in sorted(os.listdir(eval_dirs_per_model_trial)):
        trial_dir = os.path.join(eval_dirs_per_model_trial, trial_idx)
        results.append(load_eval_results_for_trial(trial_dir))

    return results


def analyze_vocab_dist(eval_summary, gt_referents, to_include_target=None):
    '''
    Takes in an eval summary (of type `dict` where the keys are contexts,
    and vals are stats for a given context.

    Returns a dict that summarizes the distribution of vocab in the generated utterances.
    '''
    mean_utterance_lengths_by_context = {}
    utterance_type_by_target = np.zeros(
        (len(COLORS), len(SHAPES), len(UTTERANCE_TYPES), len(CONTEXTS)),
        dtype=np.int32)
    num_utterances_per_context = [None] * len(CONTEXTS)
    for context, context_eval in eval_summary.items():
        targets = [tgt.split() for tgt, _, _ in gt_referents[context]]
        utterance_lengths = {
            'color': [],
            'shape': [],
            'other': [],
            'total': []
        }
        utterances = [
            decode_utterance(utterance, include_special_tokens=False)
            for utterance in context_eval['lang']
        ]
        num_utterances_per_context[CONTEXTS.index(context)] = len(utterances)
        for target, utterance in zip(targets, utterances):
            if to_include_target is None or to_include_target(target):
                target_color, target_shape = target
                num_color_tokens = len(
                    [token for token in utterance if token in COLORS])
                num_shape_tokens = len(
                    [token for token in utterance if token in SHAPES])
                num_total_tokens = len(utterance)
                num_other_tokens = num_total_tokens - num_color_tokens - num_shape_tokens
                utterance_lengths['color'].append(num_color_tokens)
                utterance_lengths['shape'].append(num_shape_tokens)
                utterance_lengths['other'].append(num_other_tokens)
                utterance_lengths['total'].append(num_total_tokens)
                if num_color_tokens > 0:
                    if num_shape_tokens > 0:
                        curr_utterance_type = 'both'
                    else:
                        curr_utterance_type = 'color-only'
                else:
                    if num_shape_tokens > 0:
                        curr_utterance_type = 'shape-only'
                utterance_type_by_target[
                    COLORS.index(target_color),
                    SHAPES.index(target_shape),
                    UTTERANCE_TYPES.index(curr_utterance_type),
                    CONTEXTS.index(context)] += 1

        mean_utterance_lengths_by_context[context] = {
            token_type: np.mean(np.asarray(lengths))
            for token_type, lengths in utterance_lengths.items()
        }
    utterance_type_by_target = utterance_type_by_target
    utterance_type_by_context = utterance_type_by_target.sum(axis=0).sum(
        axis=0)

    has_color = utterance_type_by_target.sum(
        axis=2) - utterance_type_by_target[:, :,
                                           UTTERANCE_TYPES.index('shape-only')]
    has_shape = utterance_type_by_target.sum(
        axis=2) - utterance_type_by_target[:, :,
                                           UTTERANCE_TYPES.index('color-only')]
    binary_utterance_type_by_target = np.stack((has_color, has_shape), axis=2)

    binary_utterance_type_by_context = binary_utterance_type_by_target.sum(
        axis=0).sum(axis=0)

    return {
        'mean_utterance_lengths':
        pd.DataFrame(mean_utterance_lengths_by_context),
        'utterance_type': utterance_type_by_context,
        'binary_utterance_type': binary_utterance_type_by_context,
        'utterance_type_by_target': utterance_type_by_target,
        'binary_utterance_type_by_target': binary_utterance_type_by_target,
    }


def analyze_accuracy(eval_summary, gt_referents):
    '''
    Takes in:
    1) an eval summary (of type `dict`) where the keys are contexts,
    and vals are stats for a given context.
    2) a dict where the keys are context and vals are ground truth referents in each round

    Returns a dict that summarizes the communicative accuracy.
    '''
    num_correct_by_config = np.zeros(
        (len(SpecificationType), len(COLORS), len(SHAPES)), dtype=np.int32)
    num_incorrect_by_config = np.zeros(
        (len(SpecificationType), len(COLORS), len(SHAPES)), dtype=np.int32)

    # When listener prediction is correct, what probability does the listener assign to the predicted referent?
    correct_mean_maxprob_by_config = np.zeros(
        (len(SpecificationType), len(COLORS), len(SHAPES)), dtype=np.float64)
    # When listener prediction is wrong, what probability does the listener assign to the predicted referent?
    incorrect_mean_maxprob_by_config = np.zeros(
        (len(SpecificationType), len(COLORS), len(SHAPES)), dtype=np.float64)

    for context, context_stats in eval_summary.items():
        c_idx = SpecificationType[context.split('-')[0].upper()].value
        preds = context_stats['pred'].tolist()
        probs = context_stats['score'].tolist()
        tgt_configs = [tgt.split() for tgt, _, _ in gt_referents[context]]
        for pred, prob, tgt_config in zip(preds, probs, tgt_configs):
            color, shape = tgt_config
            if pred == 0:
                num_correct_by_config[c_idx,
                                      COLORS.index(color),
                                      SHAPES.index(shape)] += 1
                correct_mean_maxprob_by_config[
                    c_idx, COLORS.index(color),
                    SHAPES.index(shape)] += max(
                        prob)  # summing here, we will average later
            else:
                num_incorrect_by_config[c_idx,
                                        COLORS.index(color),
                                        SHAPES.index(shape)] += 1
                incorrect_mean_maxprob_by_config[
                    c_idx, COLORS.index(color),
                    SHAPES.index(shape)] += max(
                        prob)  # summing here, we will average later
    # divide sum of probs by count to get mean prob
    correct_mean_maxprob_by_config /= num_correct_by_config
    incorrect_mean_maxprob_by_config /= num_incorrect_by_config

    # calculate accuracy by target config: (context, color, shape)
    num_targets_by_config = num_correct_by_config + num_incorrect_by_config
    accuracy_by_config = num_correct_by_config / num_targets_by_config

    # calculate accuracy by target combo: (color, shape)
    num_correct_by_target = num_correct_by_config.sum(axis=0)
    num_incorrect_by_target = num_incorrect_by_config.sum(axis=0)
    accuracy_by_target = num_correct_by_target / (num_correct_by_target +
                                                  num_incorrect_by_target)

    # calculate accuracy by context
    accuracy_by_context = {
        context: sum(context_stats['pred'] == 0) / context_stats['pred'].size
        for context, context_stats in eval_summary.items()
    }

    # calculate overall accuracy
    num_correct = sum(
        sum(context_stats['pred'] == 0)
        for context_stats in eval_summary.values())
    num_total = sum(context_stats['pred'].size
                    for context_stats in eval_summary.values())
    overall_accuracy = num_correct / num_total

    return {
        'overall_accuracy': overall_accuracy,
        'accuracy_by_context': accuracy_by_context,
        'accuracy_by_target': accuracy_by_target,
        'accuracy_by_config': accuracy_by_config,
        'correct_mean_maxprob_by_config': correct_mean_maxprob_by_config,
        'incorrect_mean_maxprob_by_config': incorrect_mean_maxprob_by_config,
    }


def aggregate_eval_on_multiple_listeners(eval_summary: Union[Dict, List]):
    aggregate_eval = {}
    if isinstance(eval_summary, dict):
        eval_summary = list(eval_summary.values())
    metrics = eval_summary[0].keys()
    for metric in metrics:
        aggregate_metric = [
            listener_eval[metric] for listener_eval in eval_summary
        ]
        if isinstance(aggregate_metric[0], dict) or isinstance(
                aggregate_metric[0], list):
            aggregate_metric = aggregate_eval_on_multiple_listeners(
                aggregate_metric)
        else:
            aggregate_metric = np.mean(aggregate_metric, axis=0)
        aggregate_eval[metric] = aggregate_metric
    return aggregate_eval


def qualitative_analysis(context, eval_summary, gt_referents, num_examples=20):
    context_stats = eval_summary[context]
    semantics = context_stats['sem'].round(3)
    preds = context_stats['pred']
    probs = context_stats['score'].round(3)
    utterances = [
        ' '.join(decode_utterance(utterance, include_special_tokens=False))
        for utterance in context_stats['lang']
    ]
    referents = gt_referents[context].tolist()
    examples = list(zip(preds, referents, utterances, semantics, probs))
    if num_examples > 0:
        examples = examples[:num_examples]
    return [{
        'pred': pred,
        'referents': r,
        'utterance': u,
        'sem': s,
        'probs': prob
    } for pred, r, u, s, prob in examples]
