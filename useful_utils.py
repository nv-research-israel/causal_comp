# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the License
# located at the root directory.
# ---------------------------------------------------------------
import re
from collections import OrderedDict, defaultdict
import time
from copy import deepcopy
import os

import torch

from tensorboardX import SummaryWriter
import tensorboardX
import pandas as pd
import numpy as np

from COSMO_utils import run_bash


class SummaryWriter_withCSV(SummaryWriter):
    """ A wrapper for tensorboard SummaryWriter that is based on pandas,
    and writes to CSV and optionally to W&B"""
    def __init__(self, log_dir, *args, **kwargs):

        global_step_name = 'epoch'
        if 'global_step_name' in kwargs:
            global_step_name = kwargs['global_step_name']
            kwargs.pop('global_step_name')

        self.global_step_name = global_step_name

        self.log_wandb = False
        if 'wandb' in kwargs:
            self.log_wandb = kwargs['wandb']
            kwargs.pop('wandb')

        if self.log_wandb:
            import wandb

        self.suppress_tensorboard = kwargs.get('suppress_tensorboard', False)

        if not self.suppress_tensorboard:
            super(SummaryWriter_withCSV, self).__init__(log_dir, *args, **kwargs)

        self.df = pd.DataFrame()
        self.df.index.name = self.global_step_name
        self._log_dir = log_dir
        self.last_global_step = -1

    def add_scalar(self, tag, scalar_value, global_step=None):
        if global_step is not None:
            self.df.loc[global_step, tag] = scalar_value

            # if finalized last step, dump its metrics to wandb
            if self.last_global_step < global_step:
                if self.log_wandb and self.last_global_step >= 0:
                    with ns_profiling_label('wandb_log'):
                        import wandb
                        wandb.log(self.df.loc[self.last_global_step, :].to_dict(), sync=False)
                self.last_global_step = global_step

        if not self.suppress_tensorboard:
            super().add_scalar(tag, scalar_value, global_step=global_step)

    def add_summary(self, summary, global_step=None):
        summary_proto = tensorboardX.summary.Summary()

        if isinstance(summary, bytes):
            summary_list = [value for value in summary_proto.FromString(summary).value]
        else:
            summary_list = [value for value in summary.value]
        for val in summary_list:
            self.add_scalar(val.tag, val.simple_value, global_step=global_step)


    def set_print_options(self, pandas_max_columns=500, pandas_max_width=1000):
        pd.set_option('display.max_columns', pandas_max_columns)
        pd.set_option('display.width', pandas_max_width)

    def last_results_as_string(self, regex_filter_out=None):
        if regex_filter_out is not None:
            s = self.df.loc[:, ~self.df.columns.str.match(regex_filter_out)].iloc[-1, :]
        else:
            s = self.df.iloc[-1, :]
        string = ' '.join([f'{key}={s[key]:.2g}' for key in s.keys()])
        return string

    def dump_to_csv(self, fname='summary.csv', sep='|', verbose=0, **kwargs_df_to_csv):
        fullfname = os.path.join(self._log_dir, fname)
        self.df.to_csv(fullfname, sep=sep, **kwargs_df_to_csv)
        if verbose>0:
            print(f'Dump history to CSV file: {fullfname}')
            if verbose == 2:
                print('')
                with open(fullfname) as f:
                    print(f.read())
    def close(self):

        # dump last step results to wandb
        if self.log_wandb:
            with ns_profiling_label('wandb_log'):
                import wandb
                wandb.log(self.df.loc[self.last_global_step, :].to_dict(), sync=False)

        if not self.suppress_tensorboard:
            super().close()


def slice_dict_to_dict(d, keys, returned_keys_prefix='', returned_keys_postfix='', ignore_missing_keys=False):
    """ Returns a tuple from dictionary values, ordered and slice by given keys
        keys can be a list, or a CSV string
    """
    if isinstance(keys, str):
        keys = keys[:-1] if keys[-1] == ',' else keys
        keys = re.split(', |[, ]', keys)

    if returned_keys_prefix != '' or returned_keys_postfix != '':
        return OrderedDict((returned_keys_prefix + k + returned_keys_postfix, d[k]) for k in keys)

    if ignore_missing_keys:
        return OrderedDict((k, d[k]) for k in keys if k in d)
    else:
        return OrderedDict((k, d[k]) for k in keys)


def clone_model(model):
    # clone a pytorch model
    return deepcopy(model)


def is_uncommited_git_repo(list_ignored_regex_pattern_filenames=None, ignore_untracked_files=True):
    """ Check if there are uncommited files in workdir.
        Can ignore specific file names or regex patterns
    """

    uncommitted_files_list = get_uncommitted_files(list_ignored_regex_pattern_filenames, ignore_untracked_files)

    if uncommitted_files_list:
        return True
    else:
        return False


def get_uncommitted_files(list_ignored_regex_pattern_filenames, ignore_untracked_files):
    ignore_untracked_files_str = ''
    if ignore_untracked_files:
        ignore_untracked_files_str = ' -uno'
    if list_ignored_regex_pattern_filenames is None:
        list_ignored_regex_pattern_filenames = []
    git_status = run_bash('git status --porcelain' + ignore_untracked_files_str)
    uncommitted_files = []
    for line in git_status.split('\n'):
        ignore_current_file = False
        if line:
            fname = re.split(' ?\w +', line)[1]
            for ig_file_regex_pattern in list_ignored_regex_pattern_filenames:
                if re.match(ig_file_regex_pattern, fname):
                    ignore_current_file = True
                    break
            if ignore_current_file:
                continue
            else:
                uncommitted_files.append(fname)
    return uncommitted_files


def list_to_2d_tuple(l):
    return tuple(tuple(tup) for tup in l)


def categorical_histogram(data, labels_list, plot=True, frac=True, plt_show=False):
    import matplotlib.pyplot as plt
    s_counts = pd.Series(data).value_counts()
    s_frac = s_counts/s_counts.sum()
    hist_dict = s_counts.to_dict()
    if frac:
        hist_dict = s_frac.to_dict()
    hist = []
    for ix, _ in enumerate(labels_list):
        hist.append(hist_dict.get(ix, 0))

    if plot:
        pd.Series(hist, index=labels_list).plot(kind='bar')
        if frac:
            plt.ylim((0,1))
        if plt_show:
            plt.show()
    else:
        return np.array(hist, dtype='float32')


def to_torch(array, device):
    array = np.asanyarray(array) # cast to array if not an array. othrwise do nothing
    return torch.from_numpy(array).to(device)


def comma_seperated_str_to_list(comma_seperated_str, regex_sep=r', |[, ]'):
    return re.split(regex_sep, comma_seperated_str)



class profileblock(object):
    """
    Usage example:
        with profileblock(label='abc'):
            time.sleep(0.1)

    """
    def __init__(self, label=None, disable=False):
        self.disable = disable
        self.label = ''
        if label is not None:
            self.label = label + ': '

    def __enter__(self):
        if self.disable:
            return
        self.tic = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if self.disable:
            return
        elapsed = np.round(time.time() - self.tic, 2)
        print(f'{self.label} Elapsed {elapsed} sec')

###########


##################

from contextlib import contextmanager
from torch.cuda import nvtx


# @contextmanager
# def ns_profiling_label(label, disable=False):
#     """
#     Wraps a code block with a label for profiling using "nsight-systems"
#     Usage example:
#     with ns_profiling_label('epoch %d'%epoch):
#         << CODE FOR TRAINING AN EPOCH >>
#
#     """
#     if not disable:
#         nvtx.range_push(label)
#     try:
#         yield None
#     finally:
#         if not disable:
#             nvtx.range_pop()


@contextmanager
def ns_profiling_label(label):
    """
    A do nothing version of ns_profiling_label()

    """
    try:
        yield None
    finally:
        pass


def torch_nans(*args, **kwargs):
    return np.nan*torch.zeros(*args, **kwargs)



class batch_torch_logger():
    def __init__(self, cs_str_args=None, num_batches=None, nanmean_args_cs_str=None, device=None):
        """
        cs_str_args: arguments list as a comma separated string
        """
        assert(num_batches is not None)
        self.device = device
        self.loggers = {}
        if cs_str_args is not None:
            args_list = comma_seperated_str_to_list(cs_str_args)
            self.loggers = dict((arg_name, torch_nans(num_batches, device=device)) for arg_name in args_list)

        self.nanmean_args_list = []
        if nanmean_args_cs_str is not None:
            self.nanmean_args_list = comma_seperated_str_to_list(nanmean_args_cs_str)


        self.cnt = -1
        
    def new_batch(self):
        self.cnt += 1
        
    def log(self, locals_dict):
        for arg in self.loggers.keys():
            self.log_arg(arg, locals_dict.get(arg, torch.tensor(-9999.).to(self.device)))
        
    def log_arg(self, arg, value):
        with ns_profiling_label(f'log arg'):
            try:
                if type(value) == float or isinstance(value, np.number):
                    value = torch.FloatTensor([value])
                self.loggers[arg][self.cnt] = value.detach()
            except AttributeError:
                print(f'Error: arg name = {arg}')
                raise

    def mean(self, arg):
        if arg in self.nanmean_args_list:
            return torch_nanmean(self.loggers[arg][:(self.cnt + 1)]).detach().item()
        else:
            return self.loggers[arg][:(self.cnt+1)].mean().detach().item()
    
    def get_means(self):
        return OrderedDict((arg + '_mean', self.mean(arg)) for arg in self.loggers.keys())

def torch_nanmean(x):
    return x[~torch.isnan(x)].mean()


def wandb_myinit(project_name, experiment_name, instance_name, config, workdir=None, wandb_output_dir='/tmp/wandb',
                 reinit=True, username='user'):

    import wandb
    tags = [experiment_name]
    if experiment_name.startswith('qa_'):
        tags.append('qa')
    config['workdir'] = workdir
    wandb.init(project=project_name, name=instance_name, config=config, tags=tags, dir=wandb_output_dir, reinit=reinit,
               entity=username)


def get_all_argparse_keys(parser):
    return [action.dest for action in parser._actions]


def fill_missing_by_defaults(args_dict, argparse_parser):
    all_arg_keys = get_all_argparse_keys(argparse_parser)
    for key in all_arg_keys:
        if key not in args_dict:
            args_dict[key] = argparse_parser.get_default(key)
    return args_dict


def get_and_update_num_calls(func_ptr):
    try:
        get_and_update_num_calls.num_calls_cnt[func_ptr] += 1
    except AttributeError as e:
        if 'num_calls_cnt' in repr(e):
            get_and_update_num_calls.num_calls_cnt = defaultdict(int)
        else:
            raise

    return get_and_update_num_calls.num_calls_cnt[func_ptr]


def duplicate(x, times):
    return tuple(deepcopy(x) for _ in range(times))


def to_simple_parsing_args(some_dataclass_type):
    """
        Add this as a classmethod to some dataclass in order to make its arguments accessible from commandline
        Example:

        @classmethod
        def get_args(cls):
            args: cls = to_simple_parsing_args(cls)
            return args

    """
    from simple_parsing import ArgumentParser, ConflictResolution
    parser = ArgumentParser(conflict_resolution=ConflictResolution.NONE)
    parser.add_arguments(some_dataclass_type, dest='cfg')
    args: some_dataclass_type = parser.parse_args().cfg
    return args
