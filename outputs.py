import contextlib
import random
from collections import defaultdict
import copy
import time
import os
from collections import namedtuple

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
from shapeworld import SHAPES, COLORS
from run import run
from train import init_metrics

SpeakerConfig = namedtuple(
    'SpeakerConfig', ['path', 'inference_type', 'activation', 'num_samples'])


def postprocess_outputs(outputs):
    for key, val in outputs.items():
        val = [v.cpu().numpy() for v in val]
        if key == 'lang':
            # pad to same seq len
            max_seq_len = max([v.shape[1] for v in val])
            for i, v in enumerate(val):
                new_v = np.zeros((v.shape[0], max_seq_len))
                new_v[:, :v.shape[1]] = v
                val[i] = new_v
        outputs[key] = np.concatenate(tuple(val), axis=0)
    return outputs


def run_eval():
    output_dir = os.path.join('./eval_results', args.train_dataset)

    # ===== Specify models =====
    speakers = {
        'naive':
        SpeakerConfig(os.path.join(model_dir, 'literal_speakers'), 'sample',
                      'gumbel', 1),
        'contextual':
        SpeakerConfig(os.path.join(model_dir, 'contextual_speakers'), 'sample',
                      'gumbel', 1),
        'rsa':
        SpeakerConfig(os.path.join(model_dir, 'literal_listeners'), 'rsa',
                      'gumbel', 1),
        'rsa_ensemble':
        SpeakerConfig(os.path.join(model_dir, 'literal_listeners'),
                      'rsa_ensemble', 'gumbel', 1),
    }

    default_listener_dir = os.path.join(model_dir, 'literal_listeners', '0')
    train_listener = 'literal_listener_0.pt'
    val_listener = 'literal_listener_1.pt'
    test_listeners = [
        fn for fn in sorted(os.listdir(model_dir))
        if fn.startswith('literal_listener_') and fn.endswith('.pt')
        and fn not in {train_listener, val_listener}
    ]

    num_test_listeners = len(test_listeners)
    num_rsa_ensemble_test_listeners = int(num_test_listeners *
                                          0.2)  # 80-20-split
    test_listeners = test_listeners[-num_rsa_ensemble_test_listeners:]
    rsa_ensemble_train_listeners = [
        train_listener
    ] + test_listeners[:-num_rsa_ensemble_test_listeners]

    listeners = {
        'val': [os.path.join(default_listener_dir, val_listener)],
        'test': [
            os.path.join(default_listener_dir, listener)
            for listener in test_listeners
        ]
    }

    for speaker_type, (speaker_dir, inference_type, activation,
                       n) in speakers.items():
        if args.speaker is not None and speaker_type != args.speaker: continue
        speaker_paths = [
            os.path.join(speaker_dir, path)
            for path in sorted(os.listdir(speaker_dir))
        ]
        if speaker_type in {'rsa', 'rsa_ensemble'}:
            # speaker comes from internal listeners
            speakers = speaker_paths
        else:
            speakers = [
                torch.load(speaker_path,
                           map_location=torch.device('cuda') if
                           torch.cuda.is_available() else torch.device('cpu'))
                for speaker_path in speaker_paths
            ]

        def run_eval_on_speaker(speaker_idx, speaker):
            metrics = init_metrics()

            # Optimization
            optimizer = None
            loss = nn.CrossEntropyLoss()

            speaker_specific_listeners = listeners.copy()
            if speaker_type == 'rsa':
                speaker_specific_listeners['train'] = [
                    os.path.join(speaker, train_listener)
                ]
            elif speaker_type == 'rsa_ensemble':
                speaker_specific_listeners['train'] = [
                    os.path.join(speaker, listener)
                    for listener in rsa_ensemble_train_listeners
                ]

            for (eval_ds, eval_datafile) in eval_datasets.items():
                for listener_type, listener_paths in speaker_specific_listeners.items(
                ):
                    if args.listener is not None and listener_type != args.listener:
                        continue
                    curr_output_dir = os.path.join(output_dir, speaker_type,
                                                   str(speaker_idx), eval_ds,
                                                   listener_type)
                    for listener_path in listener_paths:
                        listener = torch.load(
                            listener_path,
                            map_location=torch.device('cuda') if
                            torch.cuda.is_available() else torch.device('cpu'))
                        listener_index = listener_path.split('.')[-2].split(
                            '_')[-1] if len(listener_paths) > 1 else ''

                        print(
                            f"Evaluating {speaker_type} speaker {speaker_idx} on {eval_datafile} with {listener_type} listener {listener_index}..."
                        )
                        metrics, outputs = run([eval_datafile],
                                               'test',
                                               inference_type,
                                               speaker,
                                               listener,
                                               optimizer,
                                               loss,
                                               vocab,
                                               batch_size,
                                               torch.cuda.is_available(),
                                               num_samples=n,
                                               activation=activation,
                                               ci=False,
                                               dataset=args.train_dataset,
                                               debug=False)
                        outputs = postprocess_outputs(outputs)

                        # save eval outputs
                        listener_id = listener_index + '_' if len(
                            listener_paths) > 1 else ''
                        if not os.path.isdir(curr_output_dir):
                            os.makedirs(curr_output_dir)
                        np.save(
                            os.path.join(curr_output_dir,
                                         listener_id + 'acc.npy'),
                            metrics['acc'])
                        for key, val in outputs.items():
                            np.save(
                                os.path.join(curr_output_dir,
                                             listener_id + f'{key}.npy'), val)

        for i, speaker in enumerate(speakers):
            run_eval_on_speaker(i, speaker)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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
    parser.add_argument('--speaker',
                        type=str,
                        choices=[
                            'naive', 'contextual', 'srr', 'rsa',
                            'rsa_ensemble', 'reinforce', 'amortized'
                        ],
                        default=None,
                        help="Speaker to eval on")
    parser.add_argument('--listener',
                        type=str,
                        choices=['train', 'val', 'test'],
                        default=None,
                        help="Listener to eval on")

    args = parser.parse_args()
    if args.train_dataset is None:
        args.train_dataset = args.eval_dataset

    ROOT_MODEL_DIR = './models/shapeworld'
    model_dir = os.path.join(ROOT_MODEL_DIR, args.train_dataset)

    batch_size = 1000 if torch.cuda.is_available() else 100

    # ===== Specify eval data and where to store eval outputs =====

    DATA_DIR = './data'

    contexts = ['both-needed', 'either-okay', 'shape-needed', 'color-needed']
    eval_datasets = {
        f'{args.eval_dataset}-{c}':
        os.path.join(DATA_DIR,
                     f'{args.eval_dataset}-{c}/reference-1000-eval.npz')
        for c in contexts
    }

    # ===== Load vocab =====
    vocab = torch.load('./models/shapeworld/vocab.pt')
    print(vocab)

    # ===== Run eval =====
    run_eval()