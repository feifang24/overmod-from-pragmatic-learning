import contextlib
import random
from collections import defaultdict
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import vision
import util
from data import ShapeWorld
import data

from colors import ColorsInContext

from run import run, TASK, MODELS_DIR
import models
import wandb


def init_metrics():
    metrics = defaultdict(list)
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')
    metrics['best_epoch'] = 0
    return metrics


def format_log_metrics(train_m, val_m, epoch):
    metrics = {'epoch': epoch}
    metrics.update({f'{key}/train': val for key, val in train_m.items()})
    metrics.update({f'{key}/val': val for key, val in val_m.items()})
    return metrics


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Train',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset',
                        help='dataset name (must be a subdir of data/)')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=None, type=float, help='Learning rate')
    parser.add_argument(
        '--lr_warmup',
        action='store_true',
        help='Warm up learning rate during the first 10 epochs.')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--vocab',
                        action='store_true',
                        help='Generate new vocab file')
    parser.add_argument('--l0',
                        action='store_true',
                        help='Train literal listener')
    parser.add_argument('--s0',
                        action='store_true',
                        help='Train literal speaker')
    parser.add_argument('--contextual',
                        action='store_true',
                        help='Train contextual speaker')
    parser.add_argument('--amortized',
                        action='store_true',
                        help='Train amortized speaker')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Overwrite old model files')
    parser.add_argument('--activation', default=None)
    parser.add_argument('--penalty',
                        default=None,
                        help='Cost function (length)')
    parser.add_argument('--lmbd', default=0.01, help='Cost function parameter')
    parser.add_argument('--penalty_warmup',
                        action='store_true',
                        help='Warm up penalty during the first 10 epochs.')
    parser.add_argument('--tau',
                        default=1,
                        type=float,
                        help='Softmax temperature')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Print metrics on every epoch')
    parser.add_argument('--generalization', default=None)
    args = parser.parse_args()

    # sanity check type of training
    if args.l0 + args.s0 + args.contextual + args.amortized > 1:
        raise ValueError(
            'For the purpose of logging, only one of the following flags can be turned on: `l0`, `s0`, `amortized`'
        )
    elif args.l0 + args.s0 + args.contextual + args.amortized < 1:
        raise ValueError(
            'At least one of the following flags must be turned on: `l0`, `s0`, `amortized`'
        )

    # set default lr
    if args.lr == None:
        args.lr = 0.0001 if args.l0 else 0.001

    if args.amortized:
        if not args.generalization and args.activation != 'multinomial':
            # training amortized speaker (via RSA, not reinforce)
            args.penalty = 'length'

    models_subdir = os.path.join(MODELS_DIR, TASK, args.dataset)
    if args.generalization:
        models_subdir = os.path.join(models_subdir, 'generalization',
                                     args.generalization)
    if not os.path.isdir(models_subdir):
        os.makedirs(models_subdir)

    # Data
    if args.generalization == None:
        data_dir = f'./data/{args.dataset}/reference-1000-'
        pretrain_data = [
            [
                data_dir + '0.npz', data_dir + '1.npz', data_dir + '2.npz',
                data_dir + '3.npz', data_dir + '4.npz'
            ],
            [
                data_dir + '5.npz', data_dir + '6.npz', data_dir + '7.npz',
                data_dir + '8.npz', data_dir + '9.npz'
            ],
            [
                data_dir + '10.npz', data_dir + '11.npz', data_dir + '12.npz',
                data_dir + '13.npz', data_dir + '14.npz'
            ],
            [
                data_dir + '15.npz', data_dir + '16.npz', data_dir + '17.npz',
                data_dir + '18.npz', data_dir + '19.npz'
            ],
            [
                data_dir + '20.npz', data_dir + '21.npz', data_dir + '22.npz',
                data_dir + '23.npz', data_dir + '24.npz'
            ],
            [
                data_dir + '25.npz', data_dir + '26.npz', data_dir + '27.npz',
                data_dir + '28.npz', data_dir + '29.npz'
            ],
            [
                data_dir + '30.npz', data_dir + '31.npz', data_dir + '32.npz',
                data_dir + '33.npz', data_dir + '34.npz'
            ],
            [
                data_dir + '35.npz', data_dir + '36.npz', data_dir + '37.npz',
                data_dir + '38.npz', data_dir + '39.npz'
            ],
            [
                data_dir + '40.npz', data_dir + '41.npz', data_dir + '42.npz',
                data_dir + '43.npz', data_dir + '44.npz'
            ],
            [
                data_dir + '45.npz', data_dir + '46.npz', data_dir + '47.npz',
                data_dir + '48.npz', data_dir + '49.npz'
            ],
            [
                data_dir + '50.npz', data_dir + '51.npz', data_dir + '52.npz',
                data_dir + '53.npz', data_dir + '54.npz'
            ],
            [
                data_dir + '70.npz', data_dir + '71.npz', data_dir + '72.npz',
                data_dir + '73.npz', data_dir + '74.npz'
            ]
        ]
    else:
        data_dir = './data/shapeworld/generalization/' + args.generalization + '/reference-1000-'
        pretrain_data = [[
            data_dir + '0.npz', data_dir + '1.npz', data_dir + '2.npz',
            data_dir + '3.npz', data_dir + '4.npz'
        ],
                         [
                             data_dir + '5.npz', data_dir + '6.npz',
                             data_dir + '7.npz', data_dir + '8.npz',
                             data_dir + '9.npz'
                         ]]
    train_data = [
        data_dir + '60.npz', data_dir + '61.npz', data_dir + '62.npz',
        data_dir + '63.npz', data_dir + '64.npz'
    ]
    val_data = [
        data_dir + '65.npz', data_dir + '66.npz', data_dir + '67.npz',
        data_dir + '68.npz', data_dir + '69.npz'
    ]

    # Load or Generate Vocab
    if args.vocab:
        langs = np.array([])
        for files in pretrain_data:
            for file in files:
                d = data.load_raw_data(file)
                langs = np.append(langs, d['langs'])
        vocab = data.init_vocab(langs)
        torch.save(vocab, os.path.join(models_subdir, 'vocab.pt'))
    else:
        vocab = torch.load(os.path.join(MODELS_DIR, TASK, 'vocab.pt'))

    # Initialize Speaker and Listener Model
    def initialize_speaker():
        speaker_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
        speaker_vision = vision.Conv4()
        if args.s0:
            speaker = models.LiteralSpeaker(speaker_vision, speaker_embs)
        elif args.contextual:
            speaker = models.LiteralSpeaker(speaker_vision,
                                            speaker_embs,
                                            contextual=True)
        else:
            speaker = models.Speaker(speaker_vision, speaker_embs)
        if args.cuda:
            speaker = speaker.cuda()
        return speaker

    def initialize_listener():
        listener_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
        listener_vision = vision.Conv4()
        listener = models.Listener(listener_vision, listener_embs)
        if args.cuda:
            listener = listener.cuda()
        return listener

    loss = nn.CrossEntropyLoss() if not args.l0 else nn.MSELoss()

    # Initialize Metrics
    metrics = init_metrics()
    all_metrics = []

    # Pretrain Literal Listener
    if args.l0:
        num_output_files = len(pretrain_data)
        output_dir = os.path.join(models_subdir, 'literal_listeners')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        trial_dir = os.path.join(output_dir, str(len(os.listdir(output_dir))))
        os.mkdir(trial_dir)
        output_files = [
            os.path.join(trial_dir, f'literal_listener_{i}.pt')
            for i in range(num_output_files)
        ]

        for file, output_file in zip(pretrain_data, output_files):
            # Reinitialize metrics, listener model, and optimizer
            metrics = init_metrics()
            all_metrics = []
            listener = initialize_listener()
            optimizer = optim.Adam(list(listener.parameters()), lr=args.lr)

            wandb_run = wandb.init(
                project="prag-feature-distribution-l0-fix",
                job_type='train',
                group=args.dataset,
                tags=['l0'],
                name=
                f"{args.dataset}-{output_file.split('/')[-1].split('.')[0]}",
                reinit=True)
            with wandb_run:
                for epoch in range(args.epochs):
                    # Train one epoch
                    data_file = file[0:len(file) - 1]
                    train_metrics, _ = run(data_file,
                                           'train',
                                           'l0',
                                           None,
                                           listener,
                                           optimizer,
                                           loss,
                                           vocab,
                                           args.batch_size,
                                           args.cuda,
                                           debug=args.debug)

                    # Validate
                    data_file = [file[-1]]
                    val_metrics, _ = run(data_file,
                                         'val',
                                         'l0',
                                         None,
                                         listener,
                                         optimizer,
                                         loss,
                                         vocab,
                                         args.batch_size,
                                         args.cuda,
                                         debug=args.debug)

                    # Update metrics, prepending the split name
                    for metric, value in train_metrics.items():
                        metrics['train_{}'.format(metric)].append(value)
                    for metric, value in val_metrics.items():
                        metrics['val_{}'.format(metric)].append(value)
                    metrics['current_epoch'] = epoch
                    wandb.log(
                        format_log_metrics(train_metrics, val_metrics, epoch))

                    # Use validation loss to choose the best model
                    is_best = val_metrics['loss'] < metrics['best_loss']
                    if is_best:
                        metrics['best_acc'] = val_metrics['acc']
                        metrics['best_loss'] = val_metrics['loss']
                        metrics['best_epoch'] = epoch
                        best_listener = copy.deepcopy(listener)

                    if args.debug:
                        print(metrics)

                wandb.run.summary['best_val_acc'] = metrics['best_acc']
                wandb.run.summary['best_val_loss'] = metrics['best_loss']
                wandb.run.summary['best_epoch'] = metrics['best_epoch']

            # Save the best model
            literal_listener = best_listener
            torch.save(literal_listener, output_file)

    else:  # Train s0, contextual, or amortized
        # Load Literal Listener
        listener_dir = os.path.join(models_subdir, 'literal_listeners', '0')
        literal_listener = torch.load(
            os.path.join(listener_dir, 'literal_listener_0.pt'))
        literal_listener_val = torch.load(
            os.path.join(listener_dir, 'literal_listener_1.pt'))
        speaker = initialize_speaker()
        optimizer = optim.Adam(list(speaker.parameters()), lr=args.lr)

        run_kwargs = {}
        if args.amortized:
            run_kwargs = {
                'activation': args.activation,
                'dataset': args.dataset,
                'penalty': args.penalty,
                'tau': args.tau,
            }

        if args.s0:
            speaker_type = 's0'
        elif args.contextual:
            speaker_type = 'contextual'
        else:
            if args.activation == 'multinomial':
                speaker_type = 'reinforce'
            else:
                speaker_type = 'amortized'

        epoch = 0
        num_failed_inits = 0
        while (epoch < args.epochs):
            # Train one epoch
            lmbd = args.lmbd / max(10 - epoch,
                                   1) if args.penalty_warmup else args.lmbd

            if args.lr_warmup and epoch < 10:
                lr = args.lr / (10 - epoch)
                optimizer = optim.Adam(list(speaker.parameters()), lr=lr)

            train_metrics, _ = run(train_data,
                                   'train',
                                   'amortized' if args.amortized else 's0',
                                   speaker,
                                   literal_listener,
                                   optimizer,
                                   loss,
                                   vocab,
                                   args.batch_size,
                                   args.cuda,
                                   lmbd=lmbd,
                                   debug=args.debug,
                                   **run_kwargs)

            # Validate
            val_metrics, _ = run(val_data,
                                 'val',
                                 'amortized' if args.amortized else 's0',
                                 speaker,
                                 literal_listener_val,
                                 optimizer,
                                 loss,
                                 vocab,
                                 args.batch_size,
                                 args.cuda,
                                 lmbd=lmbd,
                                 debug=args.debug,
                                 **run_kwargs)
            if epoch == 0:
                if speaker_type == 'amortized' and train_metrics['acc'] <= (
                        1 / 3 + 0.01):
                    # bad initialization, start over
                    num_failed_inits += 1
                    speaker = initialize_speaker()
                    optimizer = optim.Adam(list(speaker.parameters()),
                                           lr=args.lr)
                    print(
                        f'Initialization attempt #{num_failed_inits} failed. Starting over...'
                    )
                    continue
                else:
                    # good to go, init logging
                    wandb.init(project="prag-feature-distribution",
                               job_type='train',
                               group=args.dataset,
                               tags=[speaker_type])
                    wandb.config.update(args)

            # Update metrics, prepending the split name
            for metric, value in train_metrics.items():
                metrics['train_{}'.format(metric)].append(value)
            for metric, value in val_metrics.items():
                metrics['val_{}'.format(metric)].append(value)
            metrics['current_epoch'] = epoch
            wandb.log(format_log_metrics(train_metrics, val_metrics, epoch))

            # Use validation accuracy to choose the best model
            # Break ties with validation loss
            is_best = (val_metrics['acc'] > metrics['best_acc']) or (
                val_metrics['acc'] == metrics['best_acc']
                and val_metrics['loss'] < metrics['best_loss'])
            if is_best:
                metrics['best_acc'] = val_metrics['acc']
                metrics['best_loss'] = val_metrics['loss']
                metrics['best_epoch'] = epoch
                best_speaker = copy.deepcopy(speaker)

            if args.debug:
                print(metrics)

            epoch += 1

        wandb.run.summary['best_val_acc'] = metrics['best_acc']
        wandb.run.summary['best_val_loss'] = metrics['best_loss']
        wandb.run.summary['best_epoch'] = metrics['best_epoch']
        if speaker_type == 'amortized':
            wandb.run.summary['num_failed_inits'] = num_failed_inits

        # Save the best model
        if speaker_type == 's0':
            save_dir = os.path.join(models_subdir, 'literal_speakers')
        else:
            save_dir = os.path.join(models_subdir, f'{speaker_type}_speakers')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_index = len(os.listdir(save_dir))
        save_path = os.path.join(save_dir, f'{model_index}.pt')
        torch.save(best_speaker, save_path)
        wandb.run.summary['model_path'] = save_path