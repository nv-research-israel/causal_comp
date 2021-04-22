# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the License
# located at the root directory.
# ---------------------------------------------------------------

# from loguru import logger
import argparse
import glob
import os
import tempfile
import warnings
from collections import OrderedDict
from copy import deepcopy
from os.path import join
import json
from shutil import copyfile
from datetime import datetime
import pandas as pd
import sys
import numpy as np
from scipy import signal

from useful_utils import comma_seperated_str_to_list, wandb_myinit, slice_dict_to_dict, \
    fill_missing_by_defaults
from COSMO_utils import run_bash


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dir', required=True, type=str)
parser.add_argument('--early_stop_metrics', required=False, default=None, type=str)
parser.add_argument('--ignore_skipping', default=True, action="store_true")
parser.add_argument('--infer_on_incompleted', default=False, action="store_true")
parser.add_argument('--smooth_val_curve', default=False, type=bool)
parser.add_argument('--smooth_window_width', default=6, type=int)
parser.add_argument('--use_wandb', default=False, type=bool)
parser.add_argument('--wandb_project_name', default=None, type=str)
parser.add_argument('--wandb_subset_metrics', default=False, type=bool)
parser.add_argument('--eval_on_last_epoch', default=False, type=bool)


def main(args):
    if isinstance(args, dict):
        args = fill_missing_by_defaults(args, parser)

    files = glob.glob(args.dir + "/*")
    # results_files = [file for file in files if "results" in basename(file)]
    print("###############")
    print("Starting offline_early_stop")
    print(f"running on '{args.dir}'")
    # if args.create_results_json:
    #     if args.metric is None:
    #         print("if creating empty results.json, must give specific metric")
    #     with open(join(args.dir, "results.json"), 'w') as f:
    #         json.dump({"metrics": {}, "train_cfg": {}, "meta_cfg": {}}, f)
    if args.early_stop_metrics is None:
        assert ValueError('--early_stop_metrics is required')
    if not args.infer_on_incompleted:
        assert (os.path.exists(join(args.dir, 'completed_training.touch')) or os.path.exists(join(args.dir, 'results.json')))
    if join(args.dir, "summary.csv") not in files:
        raise (RuntimeError("no summary.csv file!\n"))

    if not args.ignore_skipping and os.path.exists(join(args.dir, "lock")):
        print("this folder was already processed, skipping!\n")
        sys.exit(0)
    else:
        with open(join(args.dir, "lock"), "w") as f:
            f.write("0")
    summary_csv = pd.read_csv(join(args.dir, 'summary.csv'), sep='|')

    def smooth_validation_curve(validation_curve):
        if args.smooth_val_curve:
            win = np.hanning(args.smooth_window_width)
            validation_curve = signal.convolve(validation_curve, win, mode='same',
                                               method='direct') / sum(win)
            validation_curve = pd.Series(validation_curve)

        return validation_curve

    es_metric_list = comma_seperated_str_to_list(args.early_stop_metrics)
    # get run arguments

    args_dict = json.load(open(join(args.dir, "args.json"), "r"))
    early_stop_results_dict = OrderedDict()
    for i, primary_early_stop_metric in enumerate(es_metric_list):
        metric_index = i+1
        results = deepcopy(args_dict)
        print('')

        new_results_json_file = join(args.dir, f"results{metric_index}.json")
        if os.path.exists(new_results_json_file):
            backup_file_name = new_results_json_file.replace(".json",
                                                             f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
            copyfile(new_results_json_file, backup_file_name)
            print(f"backed up '{new_results_json_file}' => '{backup_file_name}'")

        print(f"creating new file: {new_results_json_file}")

        try:
            validation_curve = summary_csv[primary_early_stop_metric].copy()
            validation_curve = smooth_validation_curve(validation_curve)
            best_epoch = validation_curve.idxmax()
            if np.isnan(best_epoch):
                continue
            if args.eval_on_last_epoch:
                best_epoch = len(validation_curve) -1
            best_epoch_summary = summary_csv.iloc[[best_epoch]]
            best_epoch_test_score = best_epoch_summary[primary_early_stop_metric.replace("valid", "test")]
            best_epoch_summary = best_epoch_summary.to_dict(orient='index')[best_epoch]
            print(f"best epoch is: {best_epoch}")
            print(f"test score: {best_epoch_test_score}")

            results['metrics'] = best_epoch_summary
            results['train']['primary_early_stop_metric'] = primary_early_stop_metric
            json.dump(results, open(new_results_json_file, "w"))
            early_stop_results_dict[primary_early_stop_metric] = results
        except KeyError as e:
            warnings.warn(repr(e))

    if args.use_wandb:
        import wandb
        offline_log_to_wandb(args.wandb_project_name, args_dict, early_stop_results_dict, summary_csv,
                             workdir=args.dir,
                             wandb_log_subset_of_metrics=args.wandb_subset_metrics)
    print("done offline_early_stop!\n")

    return early_stop_results_dict


def offline_log_to_wandb(project_name, args_dict, early_stop_results_dict, summary_df, workdir=None,
                         wandb_log_subset_of_metrics=False):

    if project_name is None:
        project_name = args_dict['exp']['project_name'] + '_offline'
        if wandb_log_subset_of_metrics:
            project_name += '_subset'
    print(f'Writing to W&B project {project_name}')

    curve_metric_names = None
    if wandb_log_subset_of_metrics:
        curve_metric_names = get_wandb_curve_metrics()

    print(f'Start dump results to W&B project: {project_name}')
    wandb_myinit(project_name=project_name, experiment_name=args_dict['exp']['experiment_name'],
                 instance_name=args_dict['exp']['instance_name'], config=args_dict, workdir=workdir)


    global_step_name = 'epoch'
    summary_df = summary_df.set_index(global_step_name)
    print(f'Dump run curves')
    first_iter = True
    for global_step, step_metrics in summary_df.iterrows():
        if first_iter:
            first_iter = False
            if curve_metric_names is not None:
                for metric in curve_metric_names:
                    if metric not in step_metrics:
                        warnings.warn(f"Can't log '{metric}'. It doesn't exists.")

        if wandb_log_subset_of_metrics:
            metrics_to_log = slice_dict_to_dict(step_metrics.to_dict(), curve_metric_names, ignore_missing_keys=True)
        else:
            # log all metrics
            metrics_to_log = step_metrics.to_dict()

        metrics_to_log[global_step_name] = global_step
        wandb.log(metrics_to_log)

    early_stop_results_to_wandb_summary(early_stop_results_dict)
    dump_preds_at_early_stop(early_stop_results_dict, workdir, use_wandb=True)

    # terminate nicely offline w&b run
    wandb.join()

def dump_preds_at_early_stop(early_stop_results_dict, workdir, use_wandb):
    print(f'Save to the dumped predictions at early stop epochs')
    # dirpath = tempfile.mkdtemp()
    for es_metric, results_dict in early_stop_results_dict.items():
        for phase_name in ('valid', 'test'):
            target_fname_preds = join(workdir, f'preds__{es_metric}_{phase_name}.npz')

            epoch = results_dict['metrics']['epoch']
            fname = join(workdir, 'dump_preds', f'epoch_{epoch}', f'dump_preds_{phase_name}.npz')
            if os.path.exists(fname):
                run_bash(f'cp {fname} {target_fname_preds}')
                if use_wandb:
                    import wandb
                    wandb.save(target_fname_preds)
                print(f'Saved {target_fname_preds}')


def early_stop_results_to_wandb_summary(early_stop_results_dict):
    print(f'Dump early stop results')
    wandb_summary = OrderedDict()
    for es_metric, results_dict in early_stop_results_dict.items():
        wandb_summary[f'res__{es_metric}'] = results_dict['metrics']
    import wandb
    wandb.run.summary.update(wandb_summary)




def get_wandb_curve_metrics():
    eval_metric_names = comma_seperated_str_to_list(
        'y_joint_loss_mean, y1_loss_mean, y2_loss_mean'
        ', closed_balanced_acc'
        ', open_balanced_unseen_acc, open_balanced_seen_acc, open_H'
        ', y1_balanced_acc_unseen, y2_balanced_acc_unseen'
        ', y1_balanced_acc_seen, y2_balanced_acc_seen'
        ', closed_acc'
        ', unseen_open_acc, seen_open_acc, open_H_IMB'
        ', y1_acc_unseen, y2_acc_unseen'
    )

    train_metric_names = comma_seperated_str_to_list('y1_loss, y2_loss, y_loss'#, d_loss'
                                                     ', hsic_loss, total_loss'#, d_fool_loss'
                                                     ', y1_acc, y2_acc'#, ds1_acc, ds2_acc, current_alpha'
                                                     ', HSIC_cond1, HSIC_cond2'
                                                     ', loss, leplus_loss, tloss_feat, tloss_ao_emb'
                                                     ', tloss_a, tloss_o, loss_aux'
                                                     ', loss_aux_disjoint_attr, loss_aux_disjoint_obj')


    logged_metrics = []
    for metric in eval_metric_names:
        logged_metrics.append(metric + '_valid')
        logged_metrics.append(metric + '_test')

    for metric in train_metric_names:
        logged_metrics.append(metric + '_mean')

    return logged_metrics


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
