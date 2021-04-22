# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the License
# located at the root directory.
# ---------------------------------------------------------------

from collections import OrderedDict
from typing import NamedTuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import one_hot

from HSIC import HSIC
from model import get_model

from useful_utils import to_torch, batch_torch_logger, ns_profiling_label, profileblock, clone_model
from data import get_data_loaders
from pprint import pprint
from eval import evaluation_step, acc_from_logits, eval_model_with_dataloader

from params import EarlyStopMetric, CommandlineArgs
from model import CompModel


def train(args: CommandlineArgs, train_dataset, valid_dataset, test_dataset, writer, model: CompModel = None):
    # Init
    train_cfg = args.train

    best_metrics = {}
    epoch = -1
    start_epoch = 0
    device = args.device
    if len(writer.df) > 0:
        start_epoch = writer.df.index.max()

    # Get pytorch data loaders
    test_loader, train_loader, valid_loader = get_data_loaders(train_dataset, valid_dataset, test_dataset,
                                                               train_cfg.batch_size, train_cfg.num_workers,
                                                               test_batchsize=train_cfg.test_batchsize,
                                                               shuffle_eval_set=train_cfg.shuffle_eval_set)

    if model is None:
        model: CompModel = get_model(args, train_dataset)
    best_model = clone_model(model)

    ## NOTE:
    # y1 refer to object labels
    # y2 refer to attribute labels
    num_classes1 = train_dataset.num_objs
    num_classes2 = train_dataset.num_attrs

    class NLLLossFuncs(NamedTuple):
        y1: nn.NLLLoss
        y2: nn.NLLLoss

    nll_loss_funcs = NLLLossFuncs(y1=nn.NLLLoss(), y2=nn.NLLLoss())
    if train_cfg.balanced_loss:
        nll_loss_funcs=NLLLossFuncs(y1=nn.NLLLoss(weight=to_torch(1 / train_dataset.y1_freqs, device)),
                                    y2=nn.NLLLoss(weight=to_torch(1 / train_dataset.y2_freqs, device)))

    itr_per_epoch = len(train_loader)
    n_epochs = train_cfg.n_iter // itr_per_epoch

    best_primary_metric = np.inf * (2 * (train_cfg.primary_early_stop_metric.polarity == 'min') - 1)

    optimizer = get_optimizer(train_cfg.optimizer_name, train_cfg.lr, train_cfg.weight_decay, model, args)

    epoch_range = range(start_epoch + 1, start_epoch + n_epochs + 1)
    data_iterator = iter(train_loader)

    for epoch in epoch_range:
        with profileblock(label='Epoch train step'):
            # Select which tensors to log. Taking an average on all batches per epoch.
            logger = batch_torch_logger(num_batches=len(train_loader),
                                        cs_str_args='y1_loss, y2_loss, y_loss, '
                                                    'L_rep, '
                                                    'y1_acc, y2_acc, '
                                                    'HSIC_cond1, HSIC_cond2, '
                                                    'pairwise_dist_cond1_repr1, '
                                                    'pairwise_dist_cond1_repr2, '
                                                    'pairwise_dist_cond2_repr1, '
                                                    'pairwise_dist_cond2_repr2, '
                                                    'HSIC_label_cond1, HSIC_label_cond2',
                                        nanmean_args_cs_str = 'pairwise_dist_cond1_repr1, '
                                                    'pairwise_dist_cond1_repr2, '
                                                    'pairwise_dist_cond2_repr1, '
                                                    'pairwise_dist_cond2_repr2, '
                                                    'tloss_a, tloss_o, tloss_g_imgfeat, '
                                                    'loss_inv_core, loss_inv_g_hidden, loss_inv_g_imgfeat',
                                        device=device
                                        )


            for batch_cnt in range(len(train_loader)):
                logger.new_batch()

                optimizer.zero_grad()

                with ns_profiling_label('fetch batch'):
                    try:
                        batch = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(train_loader)
                        batch = next(data_iterator)

                with ns_profiling_label('send to gpu'):
                    X, y2, y1 = batch[0], batch[1], batch[2]
                    neg_attrs, neg_objs = batch[3].to(device), batch[4].to(device)
                    X = X.float().to(device)  # images
                    y1 = y1.long().to(device)  # object labels
                    y2 = y2.long().to(device)  # attribute labels

                with ns_profiling_label('forward pass'):
                    # y1_scores, y2_scores are logits of negative-squared-distances at the embedding space
                    # repr1, repr2 are phi_hat1, phi_hat2 at the paper
                    y1_scores, y2_scores, repr1, repr2, _ = \
                        model(X, freeze_class1=train_cfg.freeze_class1,
                              freeze_class2=train_cfg.freeze_class2)

                y1_loss = nll_loss_funcs.y1(y1_scores, y1)
                y2_loss = nll_loss_funcs.y2(y2_scores, y2)
                y_loss = y1_loss * train_cfg.Y12_balance_coeff + y2_loss * (1 - train_cfg.Y12_balance_coeff)
                L_data = train_cfg.lambda_CE * y_loss

                L_invert = 0.
                if not args.model.VisProd:
                    # pair embedding losses
                    tloss_g_hidden, tloss_g_imgfeat, loss_inv_core, loss_inv_g_hidden, loss_inv_g_imgfeat = \
                        model.eval_pair_embed_losses(args, X, model.last_feature_common, y2, y1, neg_attrs,
                                                     neg_objs, nll_loss_funcs)

                    # aggregate triplet loss into L_data
                    L_data += train_cfg.lambda_ao_emb * tloss_g_hidden
                    L_data += train_cfg.lambda_feat * tloss_g_imgfeat

                    # aggregate components of L_invert
                    L_invert += train_cfg.lambda_aux_disjoint * loss_inv_core
                    L_invert += train_cfg.lambda_aux * loss_inv_g_hidden
                    L_invert += train_cfg.lambda_aux_img * loss_inv_g_imgfeat


                ys = (y1, y2)
                L_rep, HSIC_rep_loss_terms, HSIC_mean_of_median_pairwise_dist_terms = \
                    conditional_indep_losses(repr1, repr2, ys, train_cfg.HSIC_coeff, indep_coeff2=train_cfg.HSIC_coeff,
                                             num_classes1=num_classes1,
                                             num_classes2=num_classes2, log_median_pairwise_distance=False,
                                             device=device)

                ohy1 = one_hot(y1, num_classes1)
                ohy2 = one_hot(y2, num_classes2)
                L_oh1, HSIC_oh_loss_terms1, _ = \
                    conditional_indep_losses(ohy2, repr1, ys, train_cfg.alphaH, indep_coeff2=0, num_classes1=num_classes1,
                                             num_classes2=num_classes2, log_median_pairwise_distance=False,
                                             device=device)

                L_oh2, HSIC_oh_loss_terms2, _ = \
                    conditional_indep_losses(ohy1, repr2, ys, 0, indep_coeff2=train_cfg.alphaH, num_classes1=num_classes1,
                                             num_classes2=num_classes2, log_median_pairwise_distance=False,
                                             device=device)

                L_indep = L_rep + L_oh1 + L_oh2

                loss = L_data + L_indep + L_invert


                with ns_profiling_label('loss and update'):
                    loss.backward()
                    optimizer.step()

                # log the metrics
                with ns_profiling_label('log batch'):

                    # extract indep loss terms from lists for logging
                    HSIC_cond1, HSIC_cond2, pairwise_dist_cond1_repr1, pairwise_dist_cond1_repr2, \
                    pairwise_dist_cond2_repr1, pairwise_dist_cond2_repr2 = \
                        HSIC_logging_terms(HSIC_rep_loss_terms, HSIC_mean_of_median_pairwise_dist_terms)

                    HSIC_label_cond1 = HSIC_oh_loss_terms1[0]
                    HSIC_label_cond2 = HSIC_oh_loss_terms2[1]

                    with ns_profiling_label('calc y1 train acc'):
                        y1_acc = acc_from_logits(y1_scores, y1, return_tensor=True).detach()
                    with ns_profiling_label('calc y2 train acc'):
                        y2_acc = acc_from_logits(y2_scores, y2, return_tensor=True).detach()

                    logger.log(locals_dict=locals())

            curr_epoch_metrics = OrderedDict()
            curr_epoch_metrics.update(logger.get_means())
        with profileblock(label='Evaluation step'):

            best_primary_metric, is_best = evaluation_step(model, valid_loader, test_loader, nll_loss_funcs, writer, epoch,
                                                           n_epochs, curr_epoch_metrics,
                                                           early_stop_metric_name=train_cfg.primary_early_stop_metric,
                                                           best_ES_metric_value=best_primary_metric,
                                                           calc_AUC=train_cfg.metrics.calc_AUC)

            # write current epoch metrics to metrics logger
            with ns_profiling_label('write eval step metrics'):
                for metric_key, value in curr_epoch_metrics.items():
                    writer.add_scalar(f'{metric_key}', value, epoch)
            # dump collected metrics to csv
            writer.dump_to_csv()

            # print all columns
            last_results_as_string = writer.last_results_as_string()
            last_results_as_string = '\n       '.join(last_results_as_string.split('\n'))
            if train_cfg.verbose:
                print('\n[%d/%d]' % (epoch, n_epochs), last_results_as_string)

            if is_best:
                best_model = clone_model(model)
                best_metrics = writer.df.iloc[-1, :].to_dict()
                best_metrics['epoch'] = int(writer.df.iloc[-1, :].name)
                if train_cfg.verbose:
                    print(f'Best! (@epoch {epoch})')


    model = best_model
    model.eval()

    print('Best epoch was: ', best_metrics['epoch'])
    print(f'Primary early stop monitor was {train_cfg.primary_early_stop_metric}')

    val_metrics = eval_model_with_dataloader(model, valid_loader, nll_loss_funcs, phase_name='valid')
    best_metrics.update(val_metrics)
    print('Val metrics on best val epoch :')
    pprint([(k, v) for k, v in val_metrics.items()])

    test_metrics = eval_model_with_dataloader(model, test_loader, nll_loss_funcs, phase_name='test')
    best_metrics.update(test_metrics)
    print('\n\nTest metrics on best val epoch :')
    pprint([(k, v) for k, v in test_metrics.items()])

    # cast numpy items to their original type
    for k, v in best_metrics.items():
        if isinstance(v, np.number):
            best_metrics[k] = v.item()

    #### two redundant calls to align random-number-generator with original training script
    # check: (to delete?)
    _ = eval_model_with_dataloader(model, valid_loader, nll_loss_funcs, phase_name='valid')
    _ = eval_model_with_dataloader(model, test_loader, nll_loss_funcs, phase_name='test')


    return model, best_metrics


def alternate_training(args, train_dataset, valid_dataset, test_dataset, writer):
    train_cfg = args.train
    ### alternate between heads ###
    ## train first head #
    # Save 'HSIC_coeff' for usage during step2. For step1, set HSIC_coeff to 0,
    HSIC_coeff_step2 = train_cfg.HSIC_coeff
    train_cfg.HSIC_coeff = 0
    if train_cfg.alternate_ys == 12:
        train_cfg.Y12_balance_coeff = 1
        train_cfg.primary_early_stop_metric = EarlyStopMetric('y1_balanced_acc_unseen_valid', 'max')
        train_cfg.freeze_class1 = False
        train_cfg.freeze_class2 = True
    elif train_cfg.alternate_ys == 21:
        train_cfg.Y12_balance_coeff = 0
        train_cfg.primary_early_stop_metric = EarlyStopMetric('y2_balanced_acc_unseen_valid', 'max')
        train_cfg.freeze_class1 = True
        train_cfg.freeze_class2 = False
    else:
        raise ValueError("train_cfg.alternate_ys = ", train_cfg.alternate_ys)
    print(
        f"first iter ay={train_cfg.alternate_ys}, primary_early_stop_metric={train_cfg.primary_early_stop_metric.metric}")
    model, best_metrics_dict1 = train(args, train_dataset, valid_dataset, test_dataset, writer)
    print('step1 best metrics epoch = ', best_metrics_dict1['epoch'])

    ## train 2nd head #
    model.train()
    train_cfg.set_n_iter(len(train_dataset.data), (1 + train_cfg.max_epoch_step2))
    train_cfg.lr = train_cfg.lr_step2
    if train_cfg.alternate_ys == 12:
        train_cfg.Y12_balance_coeff = 0
        train_cfg.primary_early_stop_metric = EarlyStopMetric('epoch', 'max')
        train_cfg.freeze_class1 = True
        train_cfg.freeze_class2 = False
    elif train_cfg.alternate_ys == 21:
        train_cfg.Y12_balance_coeff = 1
        train_cfg.primary_early_stop_metric = EarlyStopMetric('epoch', 'max')
        train_cfg.freeze_class1 = False
        train_cfg.freeze_class2 = True
    train_cfg.HSIC_coeff = HSIC_coeff_step2
    train(args, train_dataset, valid_dataset, test_dataset, writer, model=model)



def get_optimizer(optimizer_name, lr, weight_decay, model, args: CommandlineArgs):
    """ returns an optimizer instance """

    # list the weights to optimize
    obj_related_weights = list(model.g_inv_O.parameters()) + list(model.emb_cf_O.parameters())
    attr_related_weights = list(model.g_inv_A.parameters()) + list(model.emb_cf_A.parameters())
    pair_related_weights = []
    if not args.model.VisProd:
        obj_related_weights += list(model.obj_inv_core_logits.parameters())
        attr_related_weights += list(model.attr_inv_core_logits.parameters())

        obj_related_weights += list(model.obj_inv_g_hidden_logits.parameters())
        attr_related_weights += list(model.attr_inv_g_hidden_logits.parameters())

        obj_related_weights += list(model.obj_inv_g_imgfeat_logits.parameters())
        attr_related_weights += list(model.attr_inv_g_imgfeat_logits.parameters())

        pair_related_weights = list(model.g1_emb_to_hidden_feat.parameters()) + list(model.g2_feat_to_image_feat.parameters())
    all_weights = obj_related_weights + attr_related_weights + pair_related_weights + list(model.ECommon.parameters())

    # set optimizer hyper param
    optimizer_kwargs = dict(lr=lr, weight_decay=weight_decay)
    if optimizer_name.lower() == 'nest':
        optimizer_kwargs.update(momentum=0.9, nesterov=True)

    # choose optimizer
    optimizer_class = dict(adam=optim.Adam, sgd=optim.SGD, nest=optim.SGD)[optimizer_name.lower()]

    # initialize an optimizer instance
    optimizer = optimizer_class(all_weights, **optimizer_kwargs)

    return optimizer


def conditional_indep_losses(repr1, repr2, ys, indep_coeff, indep_coeff2=None, num_classes1=None, num_classes2=None,
                             Hkernel='L', Hkernel_sigma_obj=None, Hkernel_sigma_attr=None,
                             log_median_pairwise_distance=False, device=None):
    # check readability

    normalize_to_mean = (num_classes1, num_classes2)

    if indep_coeff2 is None:
        indep_coeff2 = indep_coeff

    HSIC_loss_terms = []
    HSIC_mean_of_median_pairwise_dist_terms = []
    with ns_profiling_label('HSIC/d loss calc'):
        # iterate on both heads
        for m, num_class in enumerate((num_classes1, num_classes2)):
            with ns_profiling_label(f'iter m={m}'):
                HSIC_tmp_loss = 0.
                HSIC_median_pw_y1 = []
                HSIC_median_pw_y2 = []

                labels_in_batch_sorted, indices = torch.sort(ys[m])
                unique_ixs = 1 + (labels_in_batch_sorted[1:] - labels_in_batch_sorted[:-1]).nonzero()
                unique_ixs = [0] + unique_ixs.flatten().cpu().numpy().tolist() + [len(ys[m])]

                for j in range(len(unique_ixs)-1):
                    current_class_indices = unique_ixs[j], unique_ixs[j + 1]
                    count = current_class_indices[1] - current_class_indices[0]
                    if count < 2:
                        continue
                    curr_class_slice = slice(*current_class_indices)
                    curr_class_indices = indices[curr_class_slice].sort()[0]

                    with ns_profiling_label(f'iter j={j}'):
                        HSIC_kernel = dict(G='Gaussian', L='Linear')[Hkernel]
                        with ns_profiling_label('HSIC call'):
                            hsic_loss_i, median_pairwise_distance_y1, median_pairwise_distance_y2 = \
                                HSIC(repr1[curr_class_indices, :].float(), repr2[curr_class_indices, :].float(),
                                     kernelX=HSIC_kernel, kernelY=HSIC_kernel,
                                     sigmaX=Hkernel_sigma_obj, sigmaY=Hkernel_sigma_attr,
                                     log_median_pairwise_distance=log_median_pairwise_distance)
                        HSIC_tmp_loss += hsic_loss_i
                        HSIC_median_pw_y1.append(median_pairwise_distance_y1)
                        HSIC_median_pw_y2.append(median_pairwise_distance_y2)

                HSIC_tmp_loss = HSIC_tmp_loss / normalize_to_mean[m]
                HSIC_loss_terms.append(HSIC_tmp_loss)
                HSIC_mean_of_median_pairwise_dist_terms.append([np.mean(HSIC_median_pw_y1), np.mean(HSIC_median_pw_y2)])

    indep_loss = torch.tensor(0.).to(device)
    if indep_coeff > 0:
        indep_loss = (indep_coeff * HSIC_loss_terms[0] + indep_coeff2 * HSIC_loss_terms[1]) / 2
    return indep_loss, HSIC_loss_terms, HSIC_mean_of_median_pairwise_dist_terms




def HSIC_logging_terms(HSIC_loss_terms, HSIC_mean_of_median_pairwise_dist_terms):
    """ This is just a utility function for naming monitored values of HSIC loss """
    HSIC_cond1, HSIC_cond2, pairwise_dist_cond1_repr1, pairwise_dist_cond1_repr2, \
    pairwise_dist_cond2_repr1, pairwise_dist_cond2_repr2 = [np.nan] * 6

    if HSIC_loss_terms:
        HSIC_cond1 = HSIC_loss_terms[0]
        HSIC_cond2 = HSIC_loss_terms[1]
        pairwise_dist_cond1_repr1 = HSIC_mean_of_median_pairwise_dist_terms[0][0]
        pairwise_dist_cond1_repr2 = HSIC_mean_of_median_pairwise_dist_terms[0][1]
        pairwise_dist_cond2_repr1 = HSIC_mean_of_median_pairwise_dist_terms[1][0]
        pairwise_dist_cond2_repr2 = HSIC_mean_of_median_pairwise_dist_terms[1][1]

    return HSIC_cond1, HSIC_cond2, pairwise_dist_cond1_repr1, pairwise_dist_cond1_repr2, \
        pairwise_dist_cond2_repr1, pairwise_dist_cond2_repr2
