# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the License
# located at the root directory.
# ---------------------------------------------------------------

import json
import os, sys
import pickle
from os.path import join
from copy import deepcopy
from pathlib import Path
from sys import argv
import random

import dataclasses
from munch import Munch

import useful_utils
import offline_early_stop
import COSMO_utils
from useful_utils import SummaryWriter_withCSV, wandb_myinit

import torch.nn
import torch
import numpy as np

import signal

from data import CompDataFromDict
from params import CommandlineArgs, TrainCfg, ExperimentCfg
from train import train, alternate_training


def set_random_seeds(base_seed):
    random.seed(base_seed)
    np.random.seed(base_seed+7205)  # 7205 is a base seed that was randomly chosen
    torch.random.manual_seed(base_seed+1000)
    torch.cuda.manual_seed(base_seed+1001)
    torch.backends.cudnn.deterministic = True


def main(args: CommandlineArgs):
    train_cfg: TrainCfg = args.train
    exp_cfg: ExperimentCfg = args.exp
    set_random_seeds(base_seed=train_cfg.seed)

    # init logging
    writer = init_logging(args)

    # load data
    test_dataset, train_dataset, valid_dataset = load_data(args)
    train_cfg.set_n_iter(len(train_dataset.data))

    # training
    with useful_utils.profileblock('complete training'):
        if train_cfg.alternate_ys == 0:
            ### train both heads jointly ##
            train(args, train_dataset, valid_dataset, test_dataset, writer)
        else:
            ### alternate between heads ###
            alternate_training(args, train_dataset, valid_dataset, test_dataset, writer)

    # ----- Finalizing ------
    # dump log to csv
    writer.dump_to_csv(verbose=1, float_format=f'%.{args.exp.csv_precision}f')
    writer.close()

    # Indicate run has complete
    COSMO_utils.run_bash(f'touch {join(exp_cfg.output_dir, "completed_training.touch")}')

    # Process offline early stopping according to results at output_dir
    early_stop_results_dict = process_offline_early_stopping(exp_cfg)

    # Print results
    print_results(early_stop_results_dict, exp_cfg)

    # Delete temporary artifacts from output dir
    clear_output_dir(exp_cfg)

    print('Done.\n')


def print_results(early_stop_results_dict, exp_cfg):
    from munch import munchify
    early_stop_results_dict = munchify(early_stop_results_dict)
    print('\n\n####################################')
    if exp_cfg.report_imbalanced_metrics:
        # E.g. Zappos
        U = 100 * early_stop_results_dict.open_H_IMB_valid.metrics.unseen_open_acc_test
        S = 100 * early_stop_results_dict.open_H_IMB_valid.metrics.seen_open_acc_test
        H = 100 * early_stop_results_dict.open_H_IMB_valid.metrics.open_H_IMB_test
        closed = 100 * early_stop_results_dict.AUC_open_valid.metrics.closed_acc_test
        AUC = 100 * early_stop_results_dict.AUC_open_valid.metrics.AUC_open_test
        print('Reporting IMbalanced metrics')
        print(f'Unseen={U:.1f}, Seen={S:.1f}, Harmonic={H:.1f}, Closed={closed:.1f}, AUC={AUC:.1f}')

    else:
        # e.g. AO-CLEVr
        U = 100 * early_stop_results_dict.open_H_valid.metrics.open_balanced_unseen_acc_test
        S = 100 * early_stop_results_dict.open_H_valid.metrics.open_balanced_seen_acc_test
        H = 100 * early_stop_results_dict.open_H_valid.metrics.open_H_test
        closed = 100 * early_stop_results_dict.closed_balanced_acc_valid.metrics.closed_balanced_acc_test
        print('Reporting Balanced metrics')
        print(f'Unseen={U:.1f}, Seen={S:.1f}, Harmonic={H:.1f}, Closed={closed:.1f}')
    print('####################################\n\n')


def init_logging(args):
    exp_cfg: ExperimentCfg = args.exp
    output_dir = Path(exp_cfg.output_dir)
    if not exp_cfg.ignore_existing_output_contents and len(list(output_dir.iterdir())) > 0:
        raise ValueError(f'Output directory {output_dir} is not empty')

    args_dict = dataclasses.asdict(args)
    if exp_cfg.use_wandb:
        import wandb
        wandb_myinit(project_name=exp_cfg.project_name, experiment_name=exp_cfg.experiment_name,
                     instance_name=exp_cfg.instance_name, config=args_dict, workdir=exp_cfg.output_dir,
                     username=exp_cfg.wandb_user)
    # printing starts here - after initializing w&b
    print('commandline was:')
    print(' '.join(argv))
    print(vars(args))
    writer = SummaryWriter_withCSV(log_dir=exp_cfg.output_dir, suppress_tensorboard=True, wandb=exp_cfg.use_wandb)
    writer.set_print_options(pandas_max_columns=500, pandas_max_width=200)
    to_json(args_dict, exp_cfg.output_dir, filename='args.json')
    return writer


def clear_output_dir(exp_cfg):
    # Always delete dumped (per-epoch) logits when done, because it takes a lot of space
    delete_dumped_logits(exp_cfg.output_dir)

    # Delete dumped (per epoch) decisions if required
    if exp_cfg.delete_dumped_preds:
        print('Delete logging of per-epoch dumped predictions')
        cmd = f'rm -rf {join(exp_cfg.output_dir, "dump_preds")}'
        print(cmd)
        COSMO_utils.run_bash(cmd)


def process_offline_early_stopping(exp_cfg: ExperimentCfg):
    cfg_offline_early_stop = Munch()
    cfg_offline_early_stop.dir = exp_cfg.output_dir
    cfg_offline_early_stop.early_stop_metrics = 'open_H_valid,closed_balanced_acc_valid,open_H_IMB_valid,AUC_open_valid'
    early_stop_results_dict = offline_early_stop.main(cfg_offline_early_stop)
    if exp_cfg.use_wandb:
        # dump each early_stop result to currents project
        offline_early_stop.early_stop_results_to_wandb_summary(early_stop_results_dict)
        # and save the dumped predictions at its epoch
    offline_early_stop.dump_preds_at_early_stop(early_stop_results_dict, exp_cfg.output_dir, use_wandb=exp_cfg.use_wandb)
    return early_stop_results_dict


def load_data(args: CommandlineArgs):
    if args.data.metadata_from_pkl:
        train_dataset, valid_dataset, test_dataset = load_pickled_metadata(args)
        print('load data from PKL')
    else:
        train_dataset, valid_dataset, test_dataset = load_TMN_data(args)
        print('load data using TMN project')
    return test_dataset, train_dataset, valid_dataset


def to_json(args_dict, log_dir, filename):
    args_json = os.path.join(log_dir, filename)
    with open(args_json, 'w') as f:
        json.dump(args_dict, f)
        print(f'\nDump configuration to JSON file: {args_json}\n\n')


def SIGINT_KeyboardInterrupt_handler(sig, frame):
        raise KeyboardInterrupt()


def load_TMN_data(args: CommandlineArgs):
    import sys
    sys.path.append('taskmodularnets')
    import taskmodularnets.data.dataset as tmn_data

    dict_data = dict()
    for subset in ['train', 'val', 'test']:
        dTMN = tmn_data.CompositionDatasetActivations(root=args.data.data_dir,
                                               phase=subset,
                                               split='compositional-split-natural')


        # Add class attributes according to the current project API
        dTMN.all_open_pairs, dTMN.seen_pairs = \
            dTMN.pairs, dTMN.train_pairs

        # Get TMN unseen pairs, because val/test_pairs include both seen and unseen pairs
        dTMN.unseen_closed_val_pairs = list(set(dTMN.val_pairs).difference(dTMN.seen_pairs))
        dTMN.unseen_closed_test_pairs = list(set(dTMN.test_pairs).difference(dTMN.seen_pairs))

        dTMN.closed_unseen_pairs = dict(
            train=[],
            val=dTMN.unseen_closed_val_pairs,
            test=dTMN.unseen_closed_test_pairs)[subset]

        dict_data[f'{subset}'] = deepcopy(vars(dTMN))

    train_dataset = CompDataFromDict(dict_data['train'], data_subset='train_data', data_dir=args.data.data_dir)
    valid_dataset = CompDataFromDict(dict_data['val'], data_subset='val_data', data_dir=args.data.data_dir)
    test_dataset = CompDataFromDict(dict_data['test'], data_subset='test_data', data_dir=args.data.data_dir)

    print('Seen   (train)  pairs: ', train_dataset.seen_pairs)
    print('Unseen (val)    pairs: ', train_dataset.unseen_closed_val_pairs)
    print('Unseen (test)   pairs: ', train_dataset.unseen_closed_test_pairs)

    return train_dataset, valid_dataset, test_dataset


def load_pickled_metadata(args: CommandlineArgs):
    data_cfg = args.data
    dataset_name = deepcopy(data_cfg['dataset_name'])
    dataset_variant = deepcopy(data_cfg['dataset_variant'])
    meta_path = Path(f"{data_cfg['data_dir']}/metadata_pickles")
    random_state_path = Path(f"{data_cfg['data_dir']}/np_random_state_pickles")
    meta_path = meta_path.expanduser()

    dict_data = dict()
    seen_seed = args.train.seed
    for subset in ['train', 'valid', 'test']:
        metadata_full_filename = meta_path / f"metadata_{dataset_name}__{dataset_variant}__comp_seed_{data_cfg['num_split']}__seen_seed_{seen_seed}__{subset}.pkl"
        dict_data[f'{subset}'] = deepcopy(pickle.load(open(metadata_full_filename, 'rb')))

    np_rnd_state_fname = random_state_path / f"np_random_state_{dataset_name}__{dataset_variant}__comp_seed_{data_cfg['num_split']}__seen_seed_{seen_seed}.pkl"
    np_seed_state = pickle.load(open(np_rnd_state_fname, 'rb'))
    np.random.set_state(np_seed_state)

    train_dataset = CompDataFromDict(dict_data['train'], data_subset='train_data', data_dir=data_cfg['data_dir'])
    valid_dataset = CompDataFromDict(dict_data['valid'], data_subset='val_data', data_dir=data_cfg['data_dir'])
    test_dataset = CompDataFromDict(dict_data['test'], data_subset='test_data', data_dir=data_cfg['data_dir'])

    print('Seen   (train)  pairs: ', train_dataset.seen_pairs)
    print('Unseen (val)    pairs: ', train_dataset.unseen_closed_val_pairs)
    print('Unseen (test)   pairs: ', train_dataset.unseen_closed_test_pairs)

    return train_dataset, valid_dataset, test_dataset


def delete_dumped_logits(logdir):
    # Delete dumped logits
    f"find {join(logdir, 'dump_preds')} -name 'logits*' -delete"


if __name__ == '__main__':
    args = CommandlineArgs.get_args()
    main(args)

