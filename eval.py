# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the License
# located at the root directory.
# ---------------------------------------------------------------

import os

import torch
from torch import Tensor
import numpy as np

from model import CompModel
from data import CompDataFromDict
from useful_utils import ns_profiling_label, batch_torch_logger, list_to_2d_tuple, slice_dict_to_dict, to_torch
from COSMO_utils import calc_cs_ausuc, per_class_balanced_accuracy


def evaluation_step(model, valid_loader, test_loader, loss_funcs, writer, epoch, n_epochs, curr_epoch_metrics,
                    early_stop_metric_name, best_ES_metric_value, calc_AUC=False):
    """
    Forward pass validation and test data, updated the metrics and finds best epoch according to validation metric.
    """
    model.eval()
    # a path for dumping logits and predictions of each epoch.
    dump_preds_path = os.path.join(writer._log_dir, 'dump_preds', f'epoch_{epoch}')

    # get metrics on validation set
    with ns_profiling_label('eval val set'):
        metrics_valid = eval_model_with_dataloader(model, valid_loader, loss_funcs, phase_name='valid',
                                                   calc_AUC=calc_AUC, dump_to_fs_basedir=dump_preds_path)
        metrics_valid['epoch'] = epoch  # log epoch number as a metric
        # update metrics dictionary with validation metrics
        curr_epoch_metrics.update(metrics_valid)

    # get metrics on test set
    with ns_profiling_label('eval test set'):
        metrics_test = eval_model_with_dataloader(model, test_loader, loss_funcs, phase_name='test', calc_AUC=calc_AUC,
                                                  dump_to_fs_basedir=dump_preds_path)
        # update metrics dictionary with test metrics
        curr_epoch_metrics.update(metrics_test)


    # Early Stop (ES) monitoring
    current_ES_metric_value = metrics_valid[early_stop_metric_name.metric]
    ES_metric_polarity = 2 * (early_stop_metric_name.polarity == 'max') - 1
    is_best = False
    if best_ES_metric_value*ES_metric_polarity <= current_ES_metric_value*ES_metric_polarity:
        is_best = True
        best_ES_metric_value = current_ES_metric_value

    model.train()

    return best_ES_metric_value, is_best


def eval_model_with_dataloader(model, data_loader, loss_funcs, phase_name,
                            calc_AUC=False,
                            dump_to_fs_basedir=None):
    if dump_to_fs_basedir is not None:
        os.makedirs(dump_to_fs_basedir, exist_ok=True)
        print(f'\n\n\nEvaluating metrics for {phase_name} phase:')

    # get model logits and ground-truth for evaluation data
    with ns_profiling_label('forward_pass_data'):
        logits_pred_ys_1, logits_pred_ys_2, logits_ys_joint, y1, y2, ys_joint, logger_means, logits_filenames \
            = forward_pass_data(model, data_loader, loss_funcs)

    preds_dump_dict, results = calc_metrics_from_logits(data_loader.dataset, logits_ys_joint, ys_joint, y1, y2,
                                                        logits_filenames, phase_name, logger_means, calc_AUC)

    ### dump predictions to filesystem
    if dump_to_fs_basedir is not None:
        # cast to numpy
        preds_dump_dict.update(
            dict((k, v.detach().cpu().numpy()) for k, v in preds_dump_dict.items() if isinstance(v, torch.Tensor)))
        fname = os.path.join(dump_to_fs_basedir, 'dump_preds' + f'_{phase_name}') + '.npz'
        np.savez_compressed(fname, **preds_dump_dict)
        print(f'dumped predictions to: {fname}')

    return results


def forward_pass_data(model: CompModel, data_loader, loss_funcs):
    device = model.device
    with torch.no_grad():
        num_batches = len(data_loader)
        batch_metrics_logger = batch_torch_logger(num_batches=num_batches,
                                                cs_str_args='y1_loss, y2_loss, y_sum_loss',
                                                  device=device)

        # predefine lists that aggregates data across batches
        ys_1 = [None] * num_batches
        ys_2 = [None] * num_batches
        fname_list = [None] * num_batches
        logits_pred_ys_1 = [None] * num_batches
        logits_pred_ys_2 = [None] * num_batches
        ys_joint = [None] * num_batches
        logits_ys_joint = [None] * num_batches
        data_iter = iter(data_loader)

        # iterate on all the data - forward-pass it through the model and aggregate the ground-truth labels,
        # the logits, and losses
        for i in range(num_batches):
            X_batch, y2_batch, y1_batch, _, _, fname_batch = next(data_iter)

            with ns_profiling_label('process batch'):
                batch_metrics_logger.new_batch()

                with torch.no_grad():

                    with ns_profiling_label('copy to gpu'):
                        X_batch = X_batch.float().to(device)
                        y1_batch = y1_batch.long().to(device)
                        y2_batch = y2_batch.long().to(device)

                    with ns_profiling_label('fwd pass'):
                        y1_pred, y2_pred, _, _, joint_pred = model(X_batch)
                    y1_pred = y1_pred.detach()
                    y2_pred = y2_pred.detach()

                    # calc a joint label
                    y_joint = data_loader.dataset.to_joint_label(y1_batch, y2_batch).detach()

                    # evaluate the loss for the current eval batch
                    if loss_funcs is not None:
                        y1_loss = loss_funcs.y1(y1_pred, y1_batch).detach()
                        y2_loss = loss_funcs.y2(y2_pred, y2_batch).detach()
                        y_sum_loss = (y1_loss + y2_loss).detach()

                batch_metrics_logger.log(locals_dict=locals())

            # aggregate labels and logits of current batch.
            ys_1[i] = y1_batch
            ys_2[i] = y2_batch
            ys_joint[i] = y_joint

            fname_list[i] = fname_batch

            logits_pred_ys_1[i] = y1_pred
            logits_pred_ys_2[i] = y2_pred
            logits_ys_joint[i] = joint_pred

        # calc the mean of each metric across all batches
        logger_means = batch_metrics_logger.get_means()

        # concat per-batch ground-truth labels list to tensors
        y1 = torch.cat(ys_1).detach()
        y2 = torch.cat(ys_2).detach()
        ys_joint = torch.cat(ys_joint).detach()

        # cocant per-batch filenames list to a single tuple
        fname = sum(list_to_2d_tuple(fname_list), tuple())

        # cocant per-batch logits list to tensors
        logits_pred_ys_1 = torch.cat(logits_pred_ys_1).detach()
        logits_pred_ys_2 = torch.cat(logits_pred_ys_2).detach()
        logits_ys_joint = torch.cat(logits_ys_joint).detach()
        return logits_pred_ys_1, logits_pred_ys_2, logits_ys_joint, y1, y2, ys_joint, logger_means, fname


def calc_metrics_from_logits(dataset, logits_ys_joint, ys_joint, y1, y2, logits_fnames, phase_name, logger_means={},
                             calc_AUC=False, device=None):

    """
    Notes:  1. This function assumes that the logits contain all (cartesian product) combinations of attribute and object
                i.e. including combinations that are not included in the open-set.
            2. calc_AUC is optional, because it slows down evaluation step
    """

    # Prepare variables
    fname = logits_fnames  # filenames


    # Get:
    # (1) logits based scores for objects and for attributes
    # (2) pairs indexing variables
    logits_pred_ys_1, logits_pred_ys_2 = \
        prepare_logits_for_metrics_calc(logits_ys_joint, dataset)


    shape_obj_attr = dataset.shape_obj_attr
    flattened_seen_pairs_mask = dataset.flattened_seen_pairs_mask
    flattened_closed_unseen_pairs_mask = dataset.flattened_closed_unseen_pairs_mask
    flattened_all_open_pairs_mask = dataset.flattened_all_open_pairs_mask

    # get indices of seen samples and of unseen samples
    seen_pairs_joint_class_ids = dataset.seen_pairs_joint_class_ids
    ids_seen_samples = np.where(np.in1d(ys_joint.cpu(), seen_pairs_joint_class_ids[0]))[0]
    ids_unseen_samples = np.where(~np.in1d(ys_joint.cpu(), seen_pairs_joint_class_ids[0]))[0]

    # ======== Open-Set Accuracy Metrics ====================

    # filter-out logits of irrelevant pairs by setting their scores to -inf
    logits_ys_open = logits_ys_joint.clone()
    logits_ys_open[:, ~flattened_all_open_pairs_mask] = -np.inf

    # Calc standard (imbalanced) accuracy metrics for open set
    unseen_open_acc = acc_from_logits(logits_ys_open, ys_joint, ids_unseen_samples)
    seen_open_acc = acc_from_logits(logits_ys_open, ys_joint, ids_seen_samples)
    open_H_IMB = 2 * (unseen_open_acc * seen_open_acc) / (unseen_open_acc + seen_open_acc + 1e-7)

    # Calc balanced accuracy metrics for open set
    num_class_joint_unseen = len(np.unique(ys_joint[ids_unseen_samples].cpu()))
    num_class_joint_seen = len(ys_joint[ids_seen_samples].unique())
    pred_open_unseen = logits_ys_open.argmax(1)[ids_unseen_samples]
    open_balanced_unseen_acc = per_class_balanced_accuracy(ys_joint[ids_unseen_samples].cpu().numpy(),
                                                           pred_open_unseen.cpu().numpy(), num_class_joint_unseen)
    pred_open_seen = logits_ys_open.argmax(1)[ids_seen_samples]
    open_balanced_seen_acc = per_class_balanced_accuracy(ys_joint[ids_seen_samples].cpu().numpy(),
                                                         pred_open_seen.cpu().numpy(), num_class_joint_seen)
    # harmonic accuracy
    open_H = 2 * (open_balanced_unseen_acc * open_balanced_seen_acc) / (
                open_balanced_unseen_acc + open_balanced_seen_acc)
    logits_ys_open = None  # release memory


    # ======== Closed-Set Accuracy Metrics ====================

    # filter-out logits of irrelevant pairs by setting their scores to -inf
    logits_ZS_ys_closed = logits_ys_joint.clone()
    logits_ZS_ys_closed[:, ~flattened_closed_unseen_pairs_mask] = -np.inf

    # Calc standard (imbalanced) accuracy metrics for closed set
    closed_acc = acc_from_logits(logits_ZS_ys_closed, ys_joint, ids_unseen_samples)
    # Calc balanced accuracy metrics for closed set
    pred_closed_unseen = logits_ZS_ys_closed.argmax(1)[ids_unseen_samples]
    closed_balanced_acc = per_class_balanced_accuracy(ys_joint[ids_unseen_samples].cpu().numpy(),
                                                      pred_closed_unseen.cpu().numpy(), num_class_joint_unseen)
    closed_balanced_acc_random = calc_random_balanced_baseline_by_logits_neginf(logits_ZS_ys_closed)
    pred_closed_unseen, logits_ZS_ys_closed = None, None  # release memory

    # ===== Unseen Accuracy metrics for objects (y1) or attributes (y2) =====
    # Note 'unseen' indicates that y1 resides in an unseen *combination* of (y1,y2), not that y1 is unseen
    # (and similarly for y2)
    pred_y_1_unseen = logits_pred_ys_1.argmax(1)[ids_unseen_samples]
    pred_y_2_unseen = logits_pred_ys_2.argmax(1)[ids_unseen_samples]
    pred_joint_unseen = logits_ys_joint.argmax(1)[ids_unseen_samples]
    y1_acc_unseen = (pred_y_1_unseen == y1[ids_unseen_samples]).sum().float() / len(y1[ids_unseen_samples])
    y2_acc_unseen = (pred_y_2_unseen == y2[ids_unseen_samples]).sum().float() / len(y2[ids_unseen_samples])

    # balanced accuracy metric.
    # NOTE !!: balanced accuracy metric ignores classes that don't participate in the set (relevant to val set)
    num_class_y1_unseen = len(y1[ids_unseen_samples].unique())
    y1_balanced_acc_unseen = per_class_balanced_accuracy(y1[ids_unseen_samples], pred_y_1_unseen, num_class_y1_unseen)
    y1_balanced_acc_unseen_random = calc_random_balanced_baseline_by_logits_neginf(logits_pred_ys_1)
    num_class_y2_unseen = len(y2[ids_unseen_samples].unique())
    y2_balanced_acc_unseen = per_class_balanced_accuracy(y2[ids_unseen_samples], pred_y_2_unseen, num_class_y2_unseen)
    y2_balanced_acc_unseen_random = calc_random_balanced_baseline_by_logits_neginf(logits_pred_ys_2)

    ##
    # Init a dictionary (preds_dump_dict) that holds variables to dump to file system
    preds_dump_dict =  slice_dict_to_dict(locals(), 'flattened_seen_pairs_mask, seen_pairs_joint_class_ids, shape_obj_attr, '
                                     'y1, y2, ys_joint, fname, '
                                     'ids_seen_samples, ids_unseen_samples, '
                                     'flattened_closed_unseen_pairs_mask, '
                                     'num_class_joint_unseen, num_class_joint_seen, '
                                     'flattened_all_open_pairs_mask, pred_open_unseen, pred_open_seen, '
                                     'pred_closed_unseen, '
                                     'pred_y_1_unseen, pred_y_2_unseen, pred_joint_unseen, num_class_y1_unseen, '
                                     'num_class_y2_unseen')

    if calc_AUC:
        AUC_open, AUC_open_balanced = calc_ausuc_from_logits(preds_dump_dict, logits_ys_joint, dataset, device)

    results = build_results_dict(locals(), logger_means, phase_name, calc_AUC)
    return preds_dump_dict, results


def build_results_dict(local_vars_dict, logger_means, phase_name, calc_AUC):
    _phase_name = f'_{phase_name}'
    results = slice_dict_to_dict(
        local_vars_dict,
        ['closed_acc',
         'closed_balanced_acc',
         'open_H',
         'open_H_IMB',
         'open_balanced_seen_acc',
         'open_balanced_unseen_acc',
         'seen_open_acc',
         'unseen_open_acc',
         'y1_acc_unseen',
         'y1_balanced_acc_unseen',
         'y1_balanced_acc_unseen_random',
         'y2_acc_unseen',
         'y2_balanced_acc_unseen',
         'y2_balanced_acc_unseen_random'],
        returned_keys_postfix=_phase_name)
    logger_means_with_postfix = slice_dict_to_dict(logger_means, logger_means.keys(),
                                                   returned_keys_postfix=_phase_name)
    results.update(logger_means_with_postfix)
    # cast torch & numpy scalars to ordinary python scalar (in order to be compatibility with json.dump() )
    results.update(dict((k, v.cpu().detach().item()) for k, v in results.items() if isinstance(v, torch.Tensor)))
    results.update(dict((k, v.item()) for k, v in results.items() if isinstance(v, np.number)))
    if calc_AUC:
        results['AUC_open' + _phase_name] = local_vars_dict['AUC_open']
        results['AUC_open_balanced' + _phase_name] = local_vars_dict['AUC_open_balanced']
    return results



def prepare_logits_for_metrics_calc(logits_ys_joint, dataset):
    """ Returns:
        (1) logits based scores for objects and for attributes
        (2) pairs indexing variables
    """
    unflattened_logits = logits_ys_joint.view((-1, dataset.num_objs, dataset.num_attrs))
    logits_pred_ys_1 = unflattened_logits.max(dim=2)[0].detach()
    logits_pred_ys_2 = unflattened_logits.max(dim=1)[0].detach()
    return logits_pred_ys_1, logits_pred_ys_2


def accuracy(predictions: Tensor, labels: Tensor, return_tensor=False):
    acc = (predictions == labels).float().mean()
    if not return_tensor:
        acc = acc.item()
    return acc


def acc_from_logits(logits: Tensor, labels: Tensor, subset_ids=None, return_tensor=False):
    predictions = logits.argmax(1)
    if subset_ids is None:
        return accuracy(predictions, labels, return_tensor=return_tensor)
    else:
        if len(subset_ids) == 0:
            return np.nan
        return accuracy(predictions[subset_ids], labels[subset_ids], return_tensor=return_tensor)


def calc_ausuc_from_logits(preds_dump_dict, logits_ys_joint, dataset: CompDataFromDict, device):
    # AUSUC when evaluating on open pairs.
    # We use the code from https://github.com/yuvalatzmon/COSMO/blob/master/src/metrics.py (CVPR 2019)
    # For that we first represent our logits and labels to match that API

    ys_joint = preds_dump_dict['ys_joint']
    seen_pairs_joint_class_ids = preds_dump_dict['seen_pairs_joint_class_ids']
    flattened_closed_unseen_pairs = preds_dump_dict['flattened_closed_unseen_pairs_mask']
    flattened_all_open_pairs_mask = preds_dump_dict['flattened_all_open_pairs_mask']
    # shape_obj_attr = preds_dump_dict['shape_obj_attr']

    # filter-in only the logits of the open pairs
    ids_open_pairs = np.where(flattened_all_open_pairs_mask)[0]
    logits_open = logits_ys_joint[:, ids_open_pairs]
    logits_ys_joint = None  # release memory

    # get a mapping from indices of cartesian-product for y1,y2 to open-pairs
    inv_ids_open_pairs = (0 * flattened_all_open_pairs_mask)  # init an array of zeros
    inv_ids_open_pairs[ids_open_pairs] = np.array(list(range(len(ids_open_pairs))))

    if not isinstance(logits_open, torch.Tensor):
        logits_open = to_torch(logits_open, device)
    logits_open = logits_open.to(device)

    if isinstance(ys_joint, torch.Tensor):
        ys_joint = ys_joint.cpu()

    ys_open = to_torch(inv_ids_open_pairs[ys_joint], device)
    seen_pairs = inv_ids_open_pairs[seen_pairs_joint_class_ids[0]]
    unseen_pairs = inv_ids_open_pairs[np.where(flattened_closed_unseen_pairs)[0]]

    AUSUC_open = calc_cs_ausuc(logits_open, ys_open, seen_pairs, unseen_pairs, use_balanced_accuracy=False)
    AUSUC_open_balanced = calc_cs_ausuc(logits_open, ys_open, seen_pairs, unseen_pairs, use_balanced_accuracy=True)

    return AUSUC_open, AUSUC_open_balanced


def calc_random_balanced_baseline_by_logits_neginf(logits):
    num_active = (logits[0, :] != -np.inf).sum().item()
    if num_active == 0:
        return np.nan
    else:
        return 1./num_active



